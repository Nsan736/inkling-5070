// dspark.cu - Inkling 2 Mini's DSpark drafter (the design and where it comes from: dspark.hpp).
//
// The safetensors reader, the small kernels (rms, mix_rms, rope_neox, silu_mul_bf16, rnn_inputs, rnn_gates and the
// fused argmax over the Markov head's fold) and the drafter's steps are ik2's (Strata pull request #6506,
// engine/src/ik2/ik2_draft.cu and ik2_kernels.cu; MIT License, Copyright (c) 2026 Peasant Smith, OscarSheng,
// alesha-pro, cobra91 and the Project Maya contributors).  The context caches, their attention and the fold are this
// fork's (dspark.hpp says how they differ).
#include "dspark.hpp"

#include "strata/kernels/native_mmvq.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fcntl.h>

#ifdef _WIN32
#include <BaseTsd.h>
#include <io.h>
typedef SSIZE_T ssize_t;
#define O_RDONLY_BIN (_O_RDONLY | _O_BINARY)
static inline ssize_t pread_at(int fd, void* buf, size_t count, int64_t offset) {
    if (_lseeki64(fd, offset, SEEK_SET) == -1) return -1;
    return _read(fd, buf, (unsigned int) std::min<size_t>(count, size_t(1) << 30));
}
#define open_file _open
#define close_file _close
#else
#include <unistd.h>
#define O_RDONLY_BIN O_RDONLY
static inline ssize_t pread_at(int fd, void* buf, size_t count, int64_t offset) { return ::pread(fd, buf, count, offset); }
#define open_file ::open
#define close_file ::close
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <vector>

namespace strata::poly::dspark {
namespace {

void cu(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("drafter: ") + what + ": " + cudaGetErrorString(e));
}
void cb(cublasStatus_t e, const char* what) {
    if (e != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string("drafter: ") + what + ": cuBLAS " + std::to_string((int) e));
}
float bf(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// ---- safetensors (ik2's reader) ----
struct StTensor {
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t begin = 0, end = 0;
};
struct StFile {
    std::string path;
    uint64_t data0 = 0;
    std::map<std::string, StTensor> t;
};

StFile st_open(const std::string& path) {
    StFile f;
    f.path = path;
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) throw std::runtime_error("drafter: cannot open " + path);
    uint64_t n = 0;
    if (std::fread(&n, 8, 1, fp) != 1 || n > (64u << 20)) {
        std::fclose(fp);
        throw std::runtime_error("drafter: not a safetensors file");
    }
    std::string h(n, '\0');
    const bool ok = std::fread(&h[0], 1, n, fp) == n;
    std::fclose(fp);
    if (!ok) throw std::runtime_error("drafter: short header");
    f.data0 = 8 + n;
    size_t i = h.find('{') + 1;
    while (true) {
        const size_t q0 = h.find('"', i);
        if (q0 == std::string::npos) break;
        const size_t q1 = h.find('"', q0 + 1);
        const std::string name = h.substr(q0 + 1, q1 - q0 - 1);
        const size_t ob = h.find('{', q1), oe = h.find('}', ob);
        const std::string body = h.substr(ob, oe - ob);
        i = oe + 1;
        if (name == "__metadata__") continue;
        StTensor t;
        const size_t d0 = body.find("\"dtype\"");
        const size_t d1 = body.find('"', body.find(':', d0) + 1), d2 = body.find('"', d1 + 1);
        t.dtype = body.substr(d1 + 1, d2 - d1 - 1);
        auto ints = [&](const char* key) {
            std::vector<int64_t> v;
            const size_t a = body.find(key);
            const size_t b0 = body.find('[', a), b1 = body.find(']', b0);
            const std::string list = body.substr(b0 + 1, b1 - b0 - 1);
            for (const char* p = list.c_str(); *p;) {
                while (*p == ' ' || *p == ',') ++p;
                if (!*p) break;
                v.push_back(std::strtoll(p, (char**) &p, 10));
            }
            return v;
        };
        t.shape = ints("\"shape\"");
        const auto off = ints("\"data_offsets\"");
        if (off.size() != 2) throw std::runtime_error("drafter: bad entry " + name);
        t.begin = (uint64_t) off[0], t.end = (uint64_t) off[1];
        f.t[name] = t;
    }
    return f;
}

std::vector<uint16_t> st_bf16(const StFile& f, const std::string& name, const std::vector<int64_t>& shape) {
    auto it = f.t.find(name);
    if (it == f.t.end()) throw std::runtime_error("drafter: no tensor " + name);
    if (it->second.dtype != "BF16") throw std::runtime_error("drafter: " + name + " is " + it->second.dtype);
    if (it->second.shape != shape) throw std::runtime_error("drafter: " + name + ": unexpected shape");
    const size_t n = (it->second.end - it->second.begin) / 2;
    std::vector<uint16_t> v(n);
    const int fd = open_file(f.path.c_str(), O_RDONLY_BIN);
    if (fd < 0) throw std::runtime_error("drafter: cannot open " + f.path);
    size_t done = 0;
    while (done < n * 2) {
        const ssize_t r = pread_at(fd, (char*) v.data() + done, n * 2 - done, (int64_t) (f.data0 + it->second.begin + done));
        if (r <= 0) {
            close_file(fd);
            throw std::runtime_error("drafter: read failed");
        }
        done += (size_t) r;
    }
    close_file(fd);
    return v;
}

// ---- kernels (ik2's, but the attention and the cache stores) ----
constexpr int kBlk = 256;

__device__ float bsum(float v, float* sh) {
    const int t = threadIdx.x;
    sh[t] = v;
    __syncthreads();
    for (int o = blockDim.x >> 1; o > 0; o >>= 1) {
        if (t < o) sh[t] += sh[t + o];
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}
__device__ __forceinline__ uint16_t f2bf_rn(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
__device__ __forceinline__ float bf2f(uint16_t b) { return __uint_as_float((uint32_t) b << 16); }

// y = x / rms(x) * w, a block a row of D (in place allowed)
__global__ void rms_kernel(const float* x, const float* w, int D, float eps, float* y) {
    __shared__ float sh[kBlk];
    const float* xr = x + (size_t) blockIdx.x * D;
    float* yr = y + (size_t) blockIdx.x * D;
    float a = 0.f;
    for (int d = threadIdx.x; d < D; d += kBlk) a += xr[d] * xr[d];
    const float rs = rsqrtf(bsum(a, sh) / (float) D + eps);
    for (int d = threadIdx.x; d < D; d += kBlk) yr[d] = xr[d] * rs * w[d];
}
__global__ void add_kernel(const float* a, const float* b, size_t n, float* y) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}
__global__ void to_bf16_kernel(const float* x, size_t n, uint16_t* y) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = f2bf_rn(x[i]);
}
__global__ void to_f16_kernel(const float* x, size_t n, __half* y) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half_rn(x[i]);
}
// a row: the taps mixed with alpha, then RMS-normed with w (the taps are four separate arrays here)
struct Taps {
    const float* p[kTaps];
};
__global__ void mix_rms_kernel(Taps taps, int C, const float* alpha, const float* w, float eps, float* out) {
    __shared__ float sh[kBlk];
    const int row = blockIdx.x;
    float a = 0.f;
    for (int c = threadIdx.x; c < C; c += kBlk) {
        float m = 0.f;
#pragma unroll
        for (int t = 0; t < kTaps; ++t) m += alpha[t] * taps.p[t][(size_t) row * C + c];
        out[(size_t) row * C + c] = m;
        a += m * m;
    }
    const float r = rsqrtf(bsum(a, sh) / (float) C + eps);
    for (int c = threadIdx.x; c < C; c += kBlk) out[(size_t) row * C + c] *= r * w[c];
}
__global__ void rope_neox_kernel(float* x, int heads, int64_t pos0, float theta) {
    const int row = blockIdx.x, h = blockIdx.y, i = threadIdx.x;   // i < 64
    const float inv = 1.0f / powf(theta, (float) (2 * i) / 128.0f);
    const float ang = (float) (pos0 + row) * inv;
    float sn, cs;
    sincosf(ang, &sn, &cs);
    float* p = x + ((size_t) row * heads + h) * 128;
    const float a = p[i], b = p[i + 64];
    p[i] = a * cs - b * sn;
    p[i + 64] = b * cs + a * sn;
}
__global__ void silu_mul_bf16_kernel(const float* gu, int T, int F, uint16_t* out) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) T * F) return;
    const size_t t = i / F, f = i % F;
    const float g = gu[t * 2 * F + f], u = gu[t * 2 * F + F + f];
    out[i] = f2bf_rn(g / (1.0f + expf(-g)) * u);
}
__global__ void silu_mul_f32_kernel(const float* gu, int T, int F, float* out) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) T * F) return;
    const size_t t = i / F, f = i % F;
    const float g = gu[t * 2 * F + f], u = gu[t * 2 * F + F + f];
    out[i] = g / (1.0f + expf(-g)) * u;
}
__global__ void rnn_inputs_kernel(const float* state, const uint16_t* emb, const int* prev, const float* hidden,
                                  const float* f, int R, int C, uint16_t* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int W = 2 * R + 2 * C;
    if (i >= W) return;
    float v;
    if (i < R) v = state[i];
    else if (i < 2 * R) v = bf2f(emb[(size_t) (*prev) * R + (i - R)]);
    else if (i < 2 * R + C) v = hidden[i - 2 * R];
    else v = f[i - 2 * R - C];
    out[i] = f2bf_rn(v);
}
__global__ void rnn_gates_kernel(const float* proj, const float* bias, int R, float* state, float* latent) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= R) return;
    const float gr = proj[i] + bias[i], cr = proj[R + i] + bias[R + i], orw = proj[2 * R + i] + bias[2 * R + i];
    const float gate = 1.0f / (1.0f + expf(-gr));
    state[i] = gate * state[i] + (1.0f - gate) * tanhf(cr);
    latent[i] = tanhf(orw);
}
// F32 rows [n][KD] into the fp16 cache at positions pos0 .. (ring: row pos % ring)
__global__ void store_rows_kernel(const float* src, int KD, int64_t pos0, int ring, __half* cache) {
    const int r = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= KD) return;
    const int64_t p = pos0 + r;
    const int64_t row = ring > 0 ? p % ring : p;
    cache[(size_t) row * KD + i] = __float2half_rn(src[(size_t) r * KD + i]);
}
__global__ void init_block_kernel(const float* anchor_h, const float* mask, float* res, int* toks, int C, int anchor) {
    const int t = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (t == 0 && c == 0) toks[0] = anchor;
    if (c < C) res[(size_t) t * C + c] = (t == 0 ? anchor_h : mask)[c];
}
__global__ void pack_f16_kernel(const float* src, size_t n, __half* dst) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half_rn(src[i]);
}

// The block's attention for one kv group: its 24 query rows (8 positions x 3 heads) against the cached keys of
// [k_lo, k_hi) (the block's own rows are in the cache at P .. P+7 by then), a block a group and a split of the keys.
// Each key row is read once for all 24 rows: a tile of 32 keys in shared memory, the scores a thread a (row, key)
// pair, an online softmax a row, and a thread a value dimension for the 24 rows' sums.
constexpr int kQ = kBlock * 3, kTile = 32, kAttnThreads = 128;
__global__ void __launch_bounds__(kAttnThreads) attn_split_kernel(const float* q, const __half* kc, const __half* vc,
                                                                  int64_t k_lo, int64_t k_hi, int ring, int keys_per,
                                                                  float scale, float* part) {
    __shared__ float qs[kQ][128];
    __shared__ __half kt[kTile][128];
    __shared__ __half vt[kTile][128];
    __shared__ float sc[kQ][kTile];
    __shared__ float m_s[kQ], l_s[kQ], al_s[kQ];
    const int s = blockIdx.x, g = blockIdx.y, tid = threadIdx.x;
    const int H = 24, KD = 8 * 128;
    for (int j = 0; j < kQ; ++j) {   // row j: position j / 3, head g * 3 + j % 3
        const int qi = j / 3, h = g * 3 + j % 3;
        qs[j][tid] = q[((size_t) qi * H + h) * 128 + tid];
    }
    if (tid < kQ) m_s[tid] = -INFINITY, l_s[tid] = 0.f;
    float o[kQ];
#pragma unroll
    for (int j = 0; j < kQ; ++j) o[j] = 0.f;
    const int64_t a0 = k_lo + (int64_t) s * keys_per, a1 = a0 + keys_per < k_hi ? a0 + keys_per : k_hi;
    __syncthreads();
    for (int64_t t0 = a0; t0 < a1; t0 += kTile) {
        const int nt = (int) (a1 - t0 < kTile ? a1 - t0 : kTile);
        for (int r = 0; r < nt; ++r) {
            const int64_t p = t0 + r;
            const int64_t row = ring > 0 ? p % ring : p;
            kt[r][tid] = kc[(size_t) row * KD + g * 128 + tid];
            vt[r][tid] = vc[(size_t) row * KD + g * 128 + tid];
        }
        __syncthreads();
        for (int pr = tid; pr < kQ * kTile; pr += kAttnThreads) {
            const int j = pr / kTile, r = pr % kTile;
            float a = -INFINITY;
            if (r < nt) {
                a = 0.f;
#pragma unroll 8
                for (int d = 0; d < 128; ++d) a += qs[j][d] * __half2float(kt[r][d]);
                a *= scale;
            }
            sc[j][r] = a;
        }
        __syncthreads();
        if (tid < kQ) {   // the row's online softmax over this tile
            const int j = tid;
            float mx = m_s[j];
            for (int r = 0; r < nt; ++r) mx = fmaxf(mx, sc[j][r]);
            const float al = m_s[j] == -INFINITY ? 0.f : expf(m_s[j] - mx);
            float l = l_s[j] * al;
            for (int r = 0; r < nt; ++r) {
                const float e = expf(sc[j][r] - mx);
                sc[j][r] = e;
                l += e;
            }
            m_s[j] = mx, l_s[j] = l, al_s[j] = al;
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < kQ; ++j) {
            float acc = o[j] * al_s[j];
            for (int r = 0; r < nt; ++r) acc += sc[j][r] * __half2float(vt[r][tid]);
            o[j] = acc;
        }
        __syncthreads();
    }
    // the split's partial: m, l and o of each row
    float* pp = part + ((size_t) s * 8 + g) * kQ * 130;
    for (int j = 0; j < kQ; ++j) pp[(size_t) j * 130 + 2 + tid] = o[j];
    if (tid < kQ) pp[(size_t) tid * 130] = m_s[tid], pp[(size_t) tid * 130 + 1] = l_s[tid];
}
// the splits merged: a block a (group, row), a thread a dimension
__global__ void attn_merge_kernel(const float* part, int splits, float* out) {
    const int g = blockIdx.x, j = blockIdx.y, tid = threadIdx.x;
    float M = -INFINITY;
    for (int s = 0; s < splits; ++s) M = fmaxf(M, part[(((size_t) s * 8 + g) * kQ + j) * 130]);
    float L = 0.f, acc = 0.f;
    for (int s = 0; s < splits; ++s) {
        const float* pp = part + (((size_t) s * 8 + g) * kQ + j) * 130;
        if (pp[0] == -INFINITY) continue;
        const float w = expf(pp[0] - M);
        L += pp[1] * w;
        acc += pp[2 + tid] * w;
    }
    const int qi = j / 3, h = g * 3 + j % 3;
    out[((size_t) qi * 24 + h) * 128 + tid] = acc / L;
}

// ik2's fused argmax over base + fold_w . x + fold_b (fold_w fp16 [R][ld], the vocabulary its columns)
constexpr int kArgmaxMaxPartials = 1024;
__device__ float g_argmax_v[kArgmaxMaxPartials];
__device__ int g_argmax_i[kArgmaxMaxPartials];
__device__ __forceinline__ void argmax_better(float v, int i, float& best, int& idx) {
    if (v > best || (v == best && i < idx)) best = v, idx = i;
}
__device__ __forceinline__ void argmax_block_reduce(float& best, int& idx) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ob = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
        argmax_better(ob, oi, best, idx);
    }
    __shared__ float bv[32];
    __shared__ int bi[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    if (lane == 0) bv[warp] = best, bi[warp] = idx;
    __syncthreads();
    if (warp == 0) {
        best = lane < nw ? bv[lane] : -INFINITY;
        idx = lane < nw ? bi[lane] : 0x7fffffff;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
            argmax_better(ob, oi, best, idx);
        }
    }
}
__global__ void argmax_gemv_part_kernel(const __half* W, int ld, const float* x, const float* base, const __half* bias,
                                        int R, int n, int* out) {
    extern __shared__ float sx[];
    for (int r = threadIdx.x; r < R; r += blockDim.x) sx[r] = x[r];
    __syncthreads();
    float best = -INFINITY;
    int idx = 0x7fffffff;
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < n; v += gridDim.x * blockDim.x) {
        float d = 0.f;
        const __half* wr = W + v;
        for (int r = 0; r < R; ++r, wr += ld) d = fmaf(__half2float(*wr), sx[r], d);
        d += base[v] + __half2float(bias[v]);
        argmax_better(d, v, best, idx);
    }
    argmax_block_reduce(best, idx);
    if (threadIdx.x == 0) {
        if (gridDim.x == 1) *out = idx;
        else g_argmax_v[blockIdx.x] = best, g_argmax_i[blockIdx.x] = idx;
    }
}
__global__ void argmax_final_kernel(int n_partials, int* out) {
    float best = -INFINITY;
    int idx = 0x7fffffff;
    for (int j = threadIdx.x; j < n_partials; j += blockDim.x) argmax_better(g_argmax_v[j], g_argmax_i[j], best, idx);
    argmax_block_reduce(best, idx);
    if (threadIdx.x == 0) *out = idx;
}

unsigned grid(size_t n) { return (unsigned) ((n + kBlk - 1) / kBlk); }

}  // namespace

struct DLayer {
    bool local = false;
    float *in_norm, *post_norm, *dflare, *q_norm, *k_norm, *k_norm_ctx;
    uint16_t *wq, *wk, *wv, *wo, *wkc, *wvc, *wgu, *wd;   // BF16 [out][in]
    uint8_t *q8q = nullptr, *q8k = nullptr, *q8v = nullptr, *q8o = nullptr, *q8gu = nullptr, *q8d = nullptr;   // Q8_0
    __half *kc, *vc;                                      // fp16 caches: [ctx][KV*128], a window layer's a ring
    int ring = 0;
};

struct Drafter {
    int C = 2304, H = 24, KV = 8, F = 9216, R = 256, V = 0;
    int64_t ctx = 0;
    float eps = 1e-6f, theta = 1e6f;
    std::vector<DLayer> L;
    float *alpha, *norm, *mask, *mk_dflare, *rnn_b;
    uint16_t *w1, *rnn_w;
    float* fold_in = nullptr;            // [R + 1][C]
    __half *fold_w = nullptr, *fold_b = nullptr;
    int n_rows = 0;
    cublasHandle_t blas = nullptr;
    cudaStream_t s = nullptr;
    size_t vram = 0;
    std::vector<void*> allocs;
    float *x, *res, *h, *q, *kk, *vv, *o, *gu, *f_rnn, *state, *latent, *proj, *mix, *part;
    uint16_t *hb, *hb2, *rnn_in;
    int* toks;
    int* h_out = nullptr;
    int max_splits = 128;
    bool q8 = false;              // INKLING2_DSPARK_Q8: the block's projections in Q8_0 blocks (engine's mmvq)
    int64_t window_g = 0;         // INKLING2_DSPARK_CTX: the whole-context layer's reach (0: all of it)
    uint8_t* xq = nullptr;        // Q8_1 rows of the block's inputs
    float* act = nullptr;         // the MLP's activation, F32 (Q8 path)

    template <class T> T* alloc(size_t n) {
        T* p = nullptr;
        cu(cudaMalloc(&p, std::max<size_t>(1, n) * sizeof(T)), "cudaMalloc");
        allocs.push_back(p);
        vram += std::max<size_t>(1, n) * sizeof(T);
        return p;
    }
    uint16_t* up16(const std::vector<uint16_t>& v) {
        uint16_t* p = alloc<uint16_t>(v.size());
        cu(cudaMemcpy(p, v.data(), v.size() * 2, cudaMemcpyHostToDevice), "upload");
        return p;
    }
    float* up32(const std::vector<uint16_t>& v) {
        std::vector<float> f(v.size());
        for (size_t i = 0; i < v.size(); ++i) f[i] = bf(v[i]);
        float* p = alloc<float>(f.size());
        cu(cudaMemcpy(p, f.data(), f.size() * 4, cudaMemcpyHostToDevice), "upload");
        return p;
    }
};

namespace {

constexpr int kMixRows = 512;   // write_context's rows a pass

// BF16 [N][K] to Q8_0 blocks (ggml's: an fp16 scale, the largest |value| / 127, and 32 int8 codes), row by row
std::vector<uint8_t> to_q8_0(const std::vector<uint16_t>& w, int64_t N, int64_t K) {
    const int64_t nb = K / 32;
    std::vector<uint8_t> out((size_t) (N * nb * 34));
    for (int64_t r = 0; r < N; ++r)
        for (int64_t b = 0; b < nb; ++b) {
            float v[32], amax = 0.f;
            for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(v[i] = bf(w[(size_t) (r * K + b * 32 + i)])));
            const float dsc = amax / 127.f, inv = dsc > 0.f ? 1.f / dsc : 0.f;
            uint8_t* blk = out.data() + (size_t) ((r * nb + b) * 34);
            const __half h = __float2half_rn(dsc);
            std::memcpy(blk, &h, 2);
            for (int i = 0; i < 32; ++i) blk[2 + i] = (uint8_t) (int8_t) std::lround(v[i] * inv);
        }
    return out;
}
// Y [T][N] = X [T][K] F32 . W [N][K] (Q8_0), through the engine's matvec of T columns
void mm_q8(Drafter* d, const uint8_t* W, int N, int K, const float* X, int T, float* Y) {
    strata::kernels::native_quantize_q8_1(X, d->xq, K, T, d->s);
    strata::kernels::native_mmvq(8, W, d->xq, Y, K, N, T, d->s);
}

// Y [T][N] (ldy) = X [T][K] BF16 . W [N][K]^T BF16, F32 sums
void gemm_bf(Drafter* d, const uint16_t* W, int N, int K, const uint16_t* X, int T, float* Y, int ldy) {
    const float one = 1.0f, zero = 0.0f;
    cb(cublasGemmEx(d->blas, CUBLAS_OP_T, CUBLAS_OP_N, N, T, K, &one, W, CUDA_R_16BF, K, X, CUDA_R_16BF, K, &zero, Y,
                    CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
       "gemm_bf");
}
void rms(Drafter* d, const float* x, const float* w, int rows, int D, float* y) {
    rms_kernel<<<rows, kBlk, 0, d->s>>>(x, w, D, d->eps, y);
}
void to_bf16(Drafter* d, const float* x, size_t n, uint16_t* y) { to_bf16_kernel<<<grid(n), kBlk, 0, d->s>>>(x, n, y); }
void rope(Drafter* d, float* x, int rows, int heads, int64_t pos0) {
    rope_neox_kernel<<<dim3(rows, heads), 64, 0, d->s>>>(x, heads, pos0, d->theta);
}
void store(Drafter* d, const DLayer& l, const float* k, const float* v, int rows, int64_t pos0) {
    const int KD = d->KV * 128;
    store_rows_kernel<<<dim3((KD + 255) / 256, rows), 256, 0, d->s>>>(k, KD, pos0, l.ring, l.kc);
    store_rows_kernel<<<dim3((KD + 255) / 256, rows), 256, 0, d->s>>>(v, KD, pos0, l.ring, l.vc);
}

void layer_attn(Drafter* d, DLayer& l, int64_t P, float* out) {
    const int T = kBlock, HD = d->H * 128, KD = d->KV * 128;
    if (d->q8) {
        mm_q8(d, l.q8q, HD, d->C, d->h, T, d->q);
        mm_q8(d, l.q8k, KD, d->C, d->h, T, d->kk);
        mm_q8(d, l.q8v, KD, d->C, d->h, T, d->vv);
    } else {
        to_bf16(d, d->h, (size_t) T * d->C, d->hb);
        gemm_bf(d, l.wq, HD, d->C, d->hb, T, d->q, HD);
        gemm_bf(d, l.wk, KD, d->C, d->hb, T, d->kk, KD);
        gemm_bf(d, l.wv, KD, d->C, d->hb, T, d->vv, KD);
    }
    rms(d, d->q, l.q_norm, T * d->H, 128, d->q);
    rms(d, d->kk, l.k_norm, T * d->KV, 128, d->kk);
    rope(d, d->q, T, d->H, P);
    rope(d, d->kk, T, d->KV, P);
    // the block's own rows go in the cache at P .. P+7 (the context write puts the committed ones there again)
    store(d, l, d->kk, d->vv, T, P);
    // every row of the block sees the context (a window layer's 1023 positions before P) and the whole block
    const int64_t k_lo = l.local ? std::max<int64_t>(0, P - 1023)
                         : d->window_g > 0 ? std::max<int64_t>(0, P - d->window_g + 1) : 0, k_hi = P + T;
    const int64_t keys = k_hi - k_lo;
    int splits = (int) std::min<int64_t>(d->max_splits, (keys + 1023) / 1024);
    splits = std::max(1, splits);
    const int per = (int) ((keys + splits - 1) / splits);
    attn_split_kernel<<<dim3(splits, d->KV), kAttnThreads, 0, d->s>>>(d->q, l.kc, l.vc, k_lo, k_hi, l.ring, per,
                                                                      1.0f / std::sqrt(128.0f), d->part);
    attn_merge_kernel<<<dim3(d->KV, kQ), 128, 0, d->s>>>(d->part, splits, d->o);
    if (d->q8) {
        mm_q8(d, l.q8o, d->C, HD, d->o, T, out);
        return;
    }
    to_bf16(d, d->o, (size_t) T * HD, d->hb);
    gemm_bf(d, l.wo, d->C, HD, d->hb, T, out, d->C);
}

void layer_mlp(Drafter* d, DLayer& l, float* out) {
    const int T = kBlock;
    if (d->q8) {
        mm_q8(d, l.q8gu, 2 * d->F, d->C, d->h, T, d->gu);
        silu_mul_f32_kernel<<<grid((size_t) T * d->F), kBlk, 0, d->s>>>(d->gu, T, d->F, d->act);
        mm_q8(d, l.q8d, d->C, d->F, d->act, T, out);
        return;
    }
    to_bf16(d, d->h, (size_t) T * d->C, d->hb);
    gemm_bf(d, l.wgu, 2 * d->F, d->C, d->hb, T, d->gu, 2 * d->F);
    silu_mul_bf16_kernel<<<grid((size_t) T * d->F), kBlk, 0, d->s>>>(d->gu, T, d->F, d->hb2);
    gemm_bf(d, l.wd, d->C, d->F, d->hb2, T, out, d->C);
}

}  // namespace

Drafter* load(const std::string& path, int C, int ctx, void* stream) {
    const StFile f = st_open(path);
    auto* d = new Drafter;
    d->C = C, d->ctx = ctx, d->s = (cudaStream_t) stream;
    cb(cublasCreate(&d->blas), "cublasCreate");
    cb(cublasSetStream(d->blas, d->s), "cublasSetStream");
    const auto& fw_t = f.t.at("fuse_weight");
    if (fw_t.shape != std::vector<int64_t>{4, 4}) throw std::runtime_error("drafter: fuse_weight is not [4, 4]");
    d->V = (int) f.t.at("markov_head.markov_w1.weight").shape[0];
    d->R = (int) f.t.at("markov_head.markov_w1.weight").shape[1];
    d->H = (int) (f.t.at("layers.0.self_attn.q_proj.weight").shape[0] / 128);
    d->KV = (int) (f.t.at("layers.0.self_attn.k_proj.weight").shape[0] / 128);
    d->F = (int) f.t.at("layers.0.mlp.gate_proj.weight").shape[0];
    if (d->H != 24 || d->KV != 8) throw std::runtime_error("drafter: expected 24 heads in 8 groups");
    d->q8 = std::getenv("INKLING2_DSPARK_Q8") != nullptr;
    if (const char* w = std::getenv("INKLING2_DSPARK_CTX")) d->window_g = std::max<int64_t>(0, std::atoll(w));
    const int64_t H = d->H, KV = d->KV, F = d->F, R = d->R;
    {   // fuse_weight [4][4] -> softmax over each row
        const auto fw = st_bf16(f, "fuse_weight", {4, 4});
        std::vector<float> a(16);
        for (int r = 0; r < 4; ++r) {
            float m = -1e30f, sum = 0.f;
            for (int t = 0; t < 4; ++t) m = std::max(m, bf(fw[r * 4 + t]));
            for (int t = 0; t < 4; ++t) sum += (a[r * 4 + t] = std::exp(bf(fw[r * 4 + t]) - m));
            for (int t = 0; t < 4; ++t) a[r * 4 + t] /= sum;
        }
        d->alpha = d->alloc<float>(16);
        cu(cudaMemcpy(d->alpha, a.data(), 64, cudaMemcpyHostToDevice), "alpha");
    }
    d->L.resize(3);
    for (int i = 0; i < 3; ++i) {
        DLayer& l = d->L[(size_t) i];
        const std::string b = "layers." + std::to_string(i) + ".";
        l.local = i != 1;   // layers 0 and 2 see a window of 1024, layer 1 the whole context
        l.in_norm = d->up32(st_bf16(f, b + "input_layernorm.weight", {C}));
        l.post_norm = d->up32(st_bf16(f, b + "post_attention_layernorm.weight", {C}));
        l.dflare = d->up32(st_bf16(f, b + "dflare_norm.weight", {C}));
        l.q_norm = d->up32(st_bf16(f, b + "self_attn.q_norm.weight", {128}));
        l.k_norm = d->up32(st_bf16(f, b + "self_attn.k_norm.weight", {128}));
        l.k_norm_ctx = d->up32(st_bf16(f, b + "self_attn.k_norm_ctx.weight", {128}));
        const auto up8 = [&](const std::vector<uint16_t>& w, int64_t N, int64_t K) {
            const auto qv = to_q8_0(w, N, K);
            uint8_t* p = d->alloc<uint8_t>(qv.size());
            cu(cudaMemcpy(p, qv.data(), qv.size(), cudaMemcpyHostToDevice), "upload");
            return p;
        };
        if (d->q8) {
            l.q8q = up8(st_bf16(f, b + "self_attn.q_proj.weight", {H * 128, C}), H * 128, C);
            l.q8k = up8(st_bf16(f, b + "self_attn.k_proj.weight", {KV * 128, C}), KV * 128, C);
            l.q8v = up8(st_bf16(f, b + "self_attn.v_proj.weight", {KV * 128, C}), KV * 128, C);
            l.q8o = up8(st_bf16(f, b + "self_attn.o_proj.weight", {C, H * 128}), C, H * 128);
        } else {
            l.wq = d->up16(st_bf16(f, b + "self_attn.q_proj.weight", {H * 128, C}));
            l.wk = d->up16(st_bf16(f, b + "self_attn.k_proj.weight", {KV * 128, C}));
            l.wv = d->up16(st_bf16(f, b + "self_attn.v_proj.weight", {KV * 128, C}));
            l.wo = d->up16(st_bf16(f, b + "self_attn.o_proj.weight", {C, H * 128}));
        }
        l.wkc = d->up16(st_bf16(f, b + "self_attn.k_proj_ctx.weight", {KV * 128, C}));
        l.wvc = d->up16(st_bf16(f, b + "self_attn.v_proj_ctx.weight", {KV * 128, C}));
        {
            auto g = st_bf16(f, b + "mlp.gate_proj.weight", {F, C});
            const auto u = st_bf16(f, b + "mlp.up_proj.weight", {F, C});
            g.insert(g.end(), u.begin(), u.end());   // gate's rows, then up's
            if (d->q8) l.q8gu = up8(g, 2 * F, C);
            else l.wgu = d->up16(g);
        }
        if (d->q8) l.q8d = up8(st_bf16(f, b + "mlp.down_proj.weight", {C, F}), C, F);
        else l.wd = d->up16(st_bf16(f, b + "mlp.down_proj.weight", {C, F}));
        l.ring = l.local ? 1024 + 2 * kBlock : d->window_g > 0 ? (int) d->window_g + 2 * kBlock : 0;
        const size_t rows = l.ring > 0 ? (size_t) l.ring : (size_t) ctx + kBlock;
        l.kc = d->alloc<__half>(rows * KV * 128);
        l.vc = d->alloc<__half>(rows * KV * 128);
    }
    d->norm = d->up32(st_bf16(f, "norm.weight", {C}));
    d->mask = d->up32(st_bf16(f, "mask_embed", {C}));
    d->mk_dflare = d->up32(st_bf16(f, "markov_head.dflare_norm.weight", {C}));
    d->w1 = d->up16(st_bf16(f, "markov_head.markov_w1.weight", {d->V, R}));
    d->rnn_w = d->up16(st_bf16(f, "markov_head.rnn_proj.weight", {3 * R, 2 * R + 2 * C}));
    d->rnn_b = d->up32(st_bf16(f, "markov_head.rnn_proj.bias", {3 * R}));
    {   // the fold's inputs: hidden_out's columns, then its bias
        const auto hw = st_bf16(f, "markov_head.hidden_out.weight", {C, R});
        const auto hb = st_bf16(f, "markov_head.hidden_out.bias", {C});
        std::vector<float> in((size_t) (R + 1) * C);
        for (int64_t c = 0; c < C; ++c)
            for (int64_t r = 0; r < R; ++r) in[(size_t) r * C + c] = bf(hw[(size_t) c * R + r]);
        for (int64_t c = 0; c < C; ++c) in[(size_t) R * C + c] = bf(hb[(size_t) c]);
        d->fold_in = d->alloc<float>(in.size());
        cu(cudaMemcpy(d->fold_in, in.data(), in.size() * 4, cudaMemcpyHostToDevice), "fold inputs");
    }
    const size_t T = kBlock;
    d->x = d->alloc<float>(T * C), d->res = d->alloc<float>(T * C), d->h = d->alloc<float>(T * C);
    d->q = d->alloc<float>(T * H * 128), d->o = d->alloc<float>(T * H * 128);
    d->kk = d->alloc<float>(std::max<size_t>(T, kMixRows) * KV * 128);
    d->vv = d->alloc<float>(std::max<size_t>(T, kMixRows) * KV * 128);
    d->gu = d->alloc<float>(T * 2 * F);
    d->hb = d->alloc<uint16_t>(std::max<size_t>(T * std::max<int64_t>(C, H * 128), (size_t) kMixRows * C));
    d->hb2 = d->alloc<uint16_t>(T * F);
    if (d->q8) {
        d->xq = d->alloc<uint8_t>(strata::kernels::native_q8_1_bytes((int) std::max<int64_t>(F, H * 128), (int) T));
        d->act = d->alloc<float>(T * F);
    }
    d->part = d->alloc<float>((size_t) d->max_splits * 8 * kQ * 130);
    d->mix = d->alloc<float>((size_t) kMixRows * C);
    d->f_rnn = d->alloc<float>(C);
    d->state = d->alloc<float>(R), d->latent = d->alloc<float>(R), d->proj = d->alloc<float>(3 * R);
    d->rnn_in = d->alloc<uint16_t>(2 * R + 2 * C);
    d->toks = d->alloc<int>(kBlock);
    cu(cudaHostAlloc(&d->h_out, (kBlock - 1) * sizeof(int), cudaHostAllocDefault), "h_out");
    cu(cudaMemset(d->f_rnn, 0, (size_t) C * 4), "f_rnn");
    return d;
}

void destroy(Drafter* d) {
    if (!d) return;
    cudaStreamSynchronize(d->s);
    for (void* p : d->allocs) cudaFree(p);
    if (d->h_out) cudaFreeHost(d->h_out);
    if (d->blas) cublasDestroy(d->blas);
    delete d;
}

size_t bytes(const Drafter* d) { return d ? d->vram : 0; }
int markov_rank(const Drafter* d) { return d->R; }
const float* fold_inputs(Drafter* d) { return d->fold_in; }

void fold(Drafter* d, const float* products, int n_rows) {
    d->n_rows = n_rows;
    const size_t nw = (size_t) d->R * n_rows;
    d->fold_w = d->alloc<__half>(nw);
    d->fold_b = d->alloc<__half>((size_t) n_rows);
    pack_f16_kernel<<<grid(nw), kBlk, 0, d->s>>>(products, nw, d->fold_w);
    pack_f16_kernel<<<grid((size_t) n_rows), kBlk, 0, d->s>>>(products + nw, (size_t) n_rows, d->fold_b);
    cu(cudaStreamSynchronize(d->s), "fold");
}

void write_context(Drafter* d, const float* const taps[kTaps], int m, int64_t pos0) {
    const int C = d->C, KD = d->KV * 128;
    for (int a = 0; a < m; a += kMixRows) {
        const int n = std::min(kMixRows, m - a);
        Taps t;
        for (int k = 0; k < kTaps; ++k) t.p[k] = taps[k] + (size_t) a * C;
        for (size_t li = 0; li < d->L.size(); ++li) {
            DLayer& l = d->L[li];
            mix_rms_kernel<<<n, kBlk, 0, d->s>>>(t, C, d->alpha + li * kTaps, l.dflare, d->eps, d->mix);
            to_bf16(d, d->mix, (size_t) n * C, d->hb);
            gemm_bf(d, l.wkc, KD, C, d->hb, n, d->kk, KD);
            gemm_bf(d, l.wvc, KD, C, d->hb, n, d->vv, KD);
            rms(d, d->kk, l.k_norm_ctx, n * d->KV, 128, d->kk);
            rope(d, d->kk, n, d->KV, pos0 + a);
            store(d, l, d->kk, d->vv, n, pos0 + a);
        }
    }
}

void set_f_rnn(Drafter* d, const float* const taps[kTaps], int row) {
    Taps t;
    for (int k = 0; k < kTaps; ++k) t.p[k] = taps[k] + (size_t) row * d->C;
    mix_rms_kernel<<<1, kBlk, 0, d->s>>>(t, d->C, d->alpha + d->L.size() * kTaps, d->mk_dflare, d->eps, d->f_rnn);
}

const float* block(Drafter* d, int anchor, const float* anchor_h, int64_t P) {
    if (P + kBlock > d->ctx + kBlock) throw std::runtime_error("drafter: the block is outside the context");
    const int C = d->C, T = kBlock;
    init_block_kernel<<<dim3((C + 255) / 256, T), 256, 0, d->s>>>(anchor_h, d->mask, d->res, d->toks, C, anchor);
    const size_t TC = (size_t) T * C;
    for (size_t i = 0; i < d->L.size(); ++i) {   // pre-norm layers, the residual carried apart
        DLayer& l = d->L[i];
        if (i > 0) add_kernel<<<grid(TC), kBlk, 0, d->s>>>(d->x, d->res, TC, d->res);
        rms(d, d->res, l.in_norm, T, C, d->h);
        layer_attn(d, l, P, d->x);
        add_kernel<<<grid(TC), kBlk, 0, d->s>>>(d->x, d->res, TC, d->res);
        rms(d, d->res, l.post_norm, T, C, d->h);
        layer_mlp(d, l, d->x);
    }
    add_kernel<<<grid(TC), kBlk, 0, d->s>>>(d->x, d->res, TC, d->res);
    rms(d, d->res, d->norm, T, C, d->h);   // the draft hidden; rows 1 .. 7 are used
    return d->h + C;
}

void markov(Drafter* d, const float* base, int n_rows, int n_vocab, int32_t* out) {
    const int C = d->C, R = d->R, T = kBlock;
    cu(cudaMemsetAsync(d->state, 0, (size_t) R * 4, d->s), "state");
    const int blocks = std::min(kArgmaxMaxPartials, std::max(1, (n_vocab + 255) / 256 / 4));
    for (int t = 0; t < T - 1; ++t) {
        rnn_inputs_kernel<<<grid((size_t) 2 * R + 2 * C), kBlk, 0, d->s>>>(d->state, d->w1, d->toks + t,
                                                                          d->h + (size_t) (t + 1) * C, d->f_rnn, R, C,
                                                                          d->rnn_in);
        gemm_bf(d, d->rnn_w, 3 * R, 2 * R + 2 * C, d->rnn_in, 1, d->proj, 3 * R);
        rnn_gates_kernel<<<grid((size_t) R), kBlk, 0, d->s>>>(d->proj, d->rnn_b, R, d->state, d->latent);
        argmax_gemv_part_kernel<<<blocks, 256, (size_t) R * 4, d->s>>>(d->fold_w, d->n_rows, d->latent,
                                                                      base + (size_t) t * n_rows, d->fold_b, R,
                                                                      n_vocab, d->toks + t + 1);
        if (blocks > 1) argmax_final_kernel<<<1, 256, 0, d->s>>>(blocks, d->toks + t + 1);
    }
    cu(cudaMemcpyAsync(d->h_out, d->toks + 1, (T - 1) * sizeof(int), cudaMemcpyDeviceToHost, d->s), "tokens");
    cu(cudaStreamSynchronize(d->s), "propose");
    for (int i = 0; i < T - 1; ++i) out[i] = d->h_out[i];
}

}  // namespace strata::poly::dspark
