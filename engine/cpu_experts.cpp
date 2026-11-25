// src/poly/cpu_experts.cpp - see the header.
#include "cpu_experts.hpp"
#include "cpu_lowbit.hpp"
#include "cpu_nvfp4.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <intrin.h>
#include <cstdlib>
#include <mutex>
#include <numeric>

namespace strata::poly {

// inkling2 fork, INKLING2_FAST_PICK: exp of 8 floats (Cephes' expf: 2^n * a degree-5 polynomial, 2 ulp or so)
static inline __m256 exp8(__m256 x) {
    x = _mm256_max_ps(x, _mm256_set1_ps(-87.3f));
    const __m256 fx = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)),
                                      _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693359375f), x);
    r = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.12194440e-4f), r);
    __m256 y = _mm256_set1_ps(1.9875691500e-4f);
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.3981999507e-3f));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(8.3334519073e-3f));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(4.1665795894e-2f));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.6666665459e-1f));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(5.0000001201e-1f));
    y = _mm256_fmadd_ps(y, _mm256_mul_ps(r, r), _mm256_add_ps(r, _mm256_set1_ps(1.f)));
    const __m256i n = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(fx), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(n));
}

double softmax_blocks(const float* logits, int n, float inv_t, float* p, double* blk) {
    __m256 m8 = _mm256_set1_ps(-INFINITY);
    int i = 0;
    for (; i + 8 <= n; i += 8) m8 = _mm256_max_ps(m8, _mm256_loadu_ps(logits + i));
    float mt[8];
    _mm256_storeu_ps(mt, m8);
    float mx = mt[0];
    for (int j = 1; j < 8; ++j) mx = std::max(mx, mt[j]);
    for (; i < n; ++i) mx = std::max(mx, logits[i]);
    const __m256 vm = _mm256_set1_ps(mx), vt = _mm256_set1_ps(inv_t);
    double total = 0.0;
    for (int a = 0; a < n; a += 256) {
        const int z = std::min(n, a + 256);
        __m256 acc = _mm256_setzero_ps();
        int k = a;
        for (; k + 8 <= z; k += 8) {
            const __m256 e = exp8(_mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(logits + k), vm), vt));
            _mm256_storeu_ps(p + k, e);
            acc = _mm256_add_ps(acc, e);
        }
        float at[8];
        _mm256_storeu_ps(at, acc);
        double b = 0.0;
        for (int j = 0; j < 8; ++j) b += at[j];
        for (; k < z; ++k) b += p[k] = std::exp((logits[k] - mx) * inv_t);
        blk[a / 256] = b;
        total += b;
    }
    return total;
}
namespace {

// Rows of gate and up, and of down, a task.  With fewer tasks than several a thread, the last round left threads
// idle at the end of every layer: 64 rows gave Tiel's 512-row experts 8 tasks each, and its down products 32 tasks.
constexpr int kRowsPerTask = 32;
constexpr int kRowsPerDownTask = 16;
constexpr int kMaxFusedRuns = 256;   // fused_runs(): a call's runs of uses (kCpuRows tokens' experts)

// inkling2 fork: NVFP4 rows against Q8_0 activations through cpu_nvfp4.hpp (AVX-VNNI, two rows at once with a
// prefetch ahead), bit for bit ggml-cpu's sums.  On by default on a CPU with AVX-VNNI (INKLING2_CPU_VNNI=0: ggml's):
// alone it reads 72 GB/s where ggml's reads 59 (12 threads of a 265K); the engine's calls of 1 to 8 experts at random
// (inkling2_cpu_calls) 61-71 GB/s where ggml's 51-62, and at 64K a written token 43.0 -> 48.1 tok/s (prose) and
// 49.0 -> 52.9 (code), each the second answer of two runs; at 8K, where fewer experts are the CPU's, no change.
constexpr int kNvfp4 = 40, kQ8_0 = 8, kNvfp4Prefetch = 2048;
bool vnni_rows() {
    static const bool on = [] {
        const char* v = std::getenv("INKLING2_CPU_VNNI");
        if (v && std::atoi(v) == 0) return false;
        int r[4] = {};
        __cpuidex(r, 7, 1);
        return (r[0] & (1 << 4)) != 0;   // CPUID.(EAX=7,ECX=1):EAX[4] AVX-VNNI
    }();
    return on;
}
// inkling2 fork: the pause instructions an idle worker spins for before it sleeps.  Upstream's 20,000 can end on a
// Core Ultra 7 265K before the card's next visit comes back to the CPU; 200,000 keeps the workers awake through a
// token's visits (+5% at 8K and at 128K, about 3 to 4 more cores busy while it writes, none once it is done).
// INKLING2_SPIN=N sets it.
constexpr int kSpinsBeforeSleep = 200000;
// inkling2 fork: a call's gate and up products, the quantizations of their outputs and the down products in one
// Workers::run(): the task that ends an expert's gate and up quantizes its outputs, and a down task waits for every
// expert's.  The same arithmetic in the same order (the same bits), with one hand-out where there were three or four.
// Off by default (no change measured at 8K on the Core Ultra 7 265K); INKLING2_CPU_FUSED=1 turns it on.
bool fused_runs() {
    static const bool on = [] {
        const char* v = std::getenv("INKLING2_CPU_FUSED");
        return v && std::atoi(v) > 0;
    }();
    return on;
}

int spins_before_sleep() {
    static const int n = [] {
        const char* v = std::getenv("INKLING2_SPIN");
        return v ? std::max(1, std::atoi(v)) : kSpinsBeforeSleep;
    }();
    return n;
}

const ggml_type_traits_cpu* traits(int type) { return ggml_get_type_traits_cpu((ggml_type) type); }

void init_once() {
    static std::once_flag once;
    std::call_once(once, [] { ggml_cpu_init(); });
}

thread_local bool in_task = false;   // this thread is running a task of a batch

}  // namespace

Workers::Workers(int threads) {
    for (int i = 1; i < threads; ++i) th_.emplace_back([this] { loop(); });
}

Workers::~Workers() {
    quit_.store(true);
    posted_.fetch_add(1);
    posted_.notify_all();
    for (auto& t : th_) t.join();
}

// Whether it ran a task.  next is read before it is counted up, so it stops near n however often a worker comes
// back to a batch whose tasks are taken.
bool Workers::drain(Batch& b) {
    bool ran = false;
    while (b.next.load(std::memory_order_relaxed) < b.n) {
        const int i = b.next.fetch_add(1, std::memory_order_relaxed);
        if (i >= b.n) break;
        in_task = true;
        (*b.fn)(i);
        in_task = false;
        b.left.fetch_sub(1, std::memory_order_release);
        ran = true;
    }
    return ran;
}

// A worker with nothing to do only reads posted_: a write by each of them, every turn of the loop, kept run() from
// ever seeing users_ at zero for long.  When a batch has been posted since the worker last looked, it counts itself
// in users_ and reads cur_ again before it uses it, so run() can tell when nobody holds its Batch any more.
//
// A batch stays posted after its last task is taken, until run() takes it down, and run() cannot while it has no
// CPU.  That happens with more threads than the system runs: a laptop that parks cores under load makes that of
// any count.  So a worker that found no task in what is posted leaves it alone: it pauses, then sleeps, until
// posted_ moves.  Going round the finished batch instead, the workers kept run() off the CPU for the rest of
// their time slices, every batch.
void Workers::loop() {
    int idle = 0;
    uint32_t seen = posted_.load() - 1;   // the count at which this worker last found nothing to do
    while (!quit_.load(std::memory_order_relaxed)) {
        const uint32_t now = posted_.load(std::memory_order_acquire);
        if (now != seen) {
            // cur_ is set before posted_ counts: the batch of `now` is there, or a later one, or it is done
            bool ran = false;
            if (cur_.load(std::memory_order_acquire)) {
                users_.fetch_add(1);
                Batch* b = cur_.load();
                if (b) ran = drain(*b);
                users_.fetch_sub(1);
            }
            if (ran) {
                idle = 0;
                continue;
            }
            seen = now;
        }
        if (++idle < spins_before_sleep()) {
            _mm_pause();
            continue;
        }
        // run() counts posted_ up before it reads sleepers_, and wait() returns at once when posted_ has moved
        sleepers_.fetch_add(1);
        if (!quit_.load()) posted_.wait(seen);
        sleepers_.fetch_sub(1);
        idle = 0;
    }
}

void Workers::run(int n, const std::function<void(int)>& fn) {
    if (n <= 0) return;
    // One task is nothing to share out.  And a task that shares out work of its own does it alone: there is one
    // batch at a time, and a worker that waited below for the others to let go of theirs would wait for itself.
    if (n == 1 || in_task) {
        for (int i = 0; i < n; ++i) fn(i);
        return;
    }
    const std::lock_guard<std::mutex> one(one_);
    Batch b;
    b.fn = &fn;
    b.n = n;
    b.left.store(n);
    cur_.store(&b);
    posted_.fetch_add(1);
    if (sleepers_.load() > 0) posted_.notify_all();
    drain(b);
    cur_.store(nullptr);   // every task is taken: a worker that comes now has nothing to find
    while (b.left.load(std::memory_order_acquire) != 0) _mm_pause();
    while (users_.load() != 0) _mm_pause();
}

bool cpu_expert_types_ok(int gu_type, int d_type, int n_embd, int n_ff, std::string& err) {
    init_once();
    for (const int type : {gu_type, d_type}) {
        const ggml_type_traits_cpu* t = traits(type);
        const ggml_type_traits_cpu* a = t && t->vec_dot ? traits(t->vec_dot_type) : nullptr;
        if (a == nullptr || a->from_float == nullptr) {
            err = "ggml-cpu has no dot product for tensor type " + std::to_string(type);
            return false;
        }
        const int n = type == gu_type ? n_embd : n_ff;
        if (n % ggml_blck_size((ggml_type) type) || n % ggml_blck_size(t->vec_dot_type)) {
            err = "a row of " + std::to_string(n) + " values is not whole blocks of tensor type " + std::to_string(type);
            return false;
        }
    }
    return true;
}

void (*g_visit_trace)(int tag) = nullptr;

void cpu_experts(Workers& workers, const CpuExpert* experts, int n, int rows, int gu_type, int u_type, int d_type,
                 int n_embd, int n_ff, const float* x, float limit, bool gelu, float* out,
                 const std::function<void()>* meanwhile, const uint8_t* late, const std::function<void(int)>* wait,
                 const Situ* situ, bool lowbit) {
    init_once();
    const ggml_type_traits_cpu* tg = traits(gu_type);
    const ggml_type_traits_cpu* tu = traits(u_type);
    const ggml_type_traits_cpu* td = traits(d_type);
    const size_t gu_row = ggml_row_size((ggml_type) gu_type, n_embd);
    const size_t u_row = ggml_row_size((ggml_type) u_type, n_embd);
    const size_t d_row = ggml_row_size((ggml_type) d_type, n_ff);
    const size_t a_bytes = ggml_row_size(tg->vec_dot_type, n_embd);
    const size_t au_bytes = ggml_row_size(tu->vec_dot_type, n_embd);
    const size_t h_bytes = ggml_row_size(td->vec_dot_type, n_ff);
    const bool own_act = tu->vec_dot_type != tg->vec_dot_type;   // up's dot product takes another quantization
    // the dot products: with `lowbit` cpu_lowbit.cpp's for Q2_0 and IQ1_S (on the same activations), else ggml's
    const auto dot_of = [lowbit](int type, const ggml_type_traits_cpu* t) {
        const ggml_vec_dot_t f = lowbit ? lowbit_vec_dot(type) : nullptr;
        return f ? f : t->vec_dot;
    };
    const ggml_vec_dot_t gdot = dot_of(gu_type, tg), udot = dot_of(u_type, tu), ddot = dot_of(d_type, td);

    // The workers get these through the pointers below: a thread_local named in their code would be their own.
    static thread_local std::vector<uint8_t> act, act_u, hq;
    static thread_local std::vector<float> h;
    static thread_local std::vector<int> by, run;
    act.resize((size_t) rows * a_bytes);
    if (own_act) act_u.resize((size_t) rows * au_bytes);
    hq.resize((size_t) n * h_bytes);
    h.resize((size_t) n * n_ff);
    // ggml's quantizers for the activations are scalar: microseconds a row, which on one thread kept the others
    // waiting, here and before the down products
    uint8_t* actp = act.data();
    const auto act_from = traits(tg->vec_dot_type)->from_float;
    workers.run(rows, [&](int t) { act_from(x + (size_t) t * n_embd, actp + (size_t) t * a_bytes, n_embd); });
    uint8_t* actup = actp;
    if (own_act) {
        actup = act_u.data();
        const auto au_from = traits(tu->vec_dot_type)->from_float;
        workers.run(rows, [&](int t) { au_from(x + (size_t) t * n_embd, actup + (size_t) t * au_bytes, n_embd); });
    }
    const size_t au_step = own_act ? au_bytes : a_bytes;
    namespace nf = strata::poly::nvfp4cpu;
    const bool vnni_gu = vnni_rows() && !lowbit && gu_type == kNvfp4 && u_type == kNvfp4 && !own_act &&
                         (int) tg->vec_dot_type == kQ8_0 && n_embd % 64 == 0;
    const bool vnni_d = vnni_rows() && !lowbit && d_type == kNvfp4 && (int) td->vec_dot_type == kQ8_0 && n_ff % 64 == 0;
    static thread_local std::vector<int32_t> ysum, ysum_h;
    if (vnni_gu) {
        ysum.resize((size_t) rows * n_embd / 4);
        int32_t* ys = ysum.data();
        workers.run(rows, [&](int t) {
            nf::nvfp4_ysum4((const nf::BlockQ8_0*) (actp + (size_t) t * a_bytes), n_embd, ys + (size_t) t * n_embd / 4);
        });
    }
    const int32_t* ysp = ysum.data();

    // The uses of one expert next to each other (by), and where each expert's run of them starts (run).
    by.resize((size_t) n);
    std::iota(by.begin(), by.end(), 0);
    std::stable_sort(by.begin(), by.end(), [&](int a, int b) { return experts[a].gate < experts[b].gate; });
    run.clear();
    for (int i = 0; i < n; ++i)
        if (i == 0 || experts[by[(size_t) i]].gate != experts[by[(size_t) i - 1]].gate) run.push_back(i);
    const int n_run = (int) run.size();
    run.push_back(n);
    // the runs in the order their products are taken: the late experts' last, so their reads go on meanwhile
    static thread_local std::vector<int> order;
    order.resize((size_t) n_run);
    std::iota(order.begin(), order.end(), 0);
    if (late)
        std::stable_partition(order.begin(), order.end(), [&](int u) { return !late[by[(size_t) run[(size_t) u]]]; });

    float* hp = h.data();
    const int* byp = by.data();
    const int* runp = run.data();
    const int* orderp = order.data();
    const int gu_tasks = (n_ff + kRowsPerTask - 1) / kRowsPerTask;
    const int first = meanwhile ? 1 : 0;   // task 0, so it is taken at once
    std::atomic<bool> traced{false};
    const auto gu_task = [&](int t) {
        if (g_visit_trace && !traced.exchange(true, std::memory_order_relaxed)) g_visit_trace(14);
        const int u = orderp[t / gu_tasks], i0 = runp[u], i1 = runp[u + 1];
        if (late && late[byp[i0]]) (*wait)(byp[i0]);
        const CpuExpert& e = experts[byp[i0]];
        const int r0 = (t % gu_tasks) * kRowsPerTask, r1 = std::min(n_ff, r0 + kRowsPerTask);
        for (int r = r0; r < r1; ++r) {
            const uint8_t* gate = e.gate + (size_t) r * gu_row;
            const uint8_t* up = e.up + (size_t) r * u_row;
            for (int i = i0; i < i1; ++i) {
                const int j = byp[i];
                const uint8_t* a = actp + (size_t) experts[j].row * a_bytes;
                const uint8_t* au = actup + (size_t) experts[j].row * au_step;
                float g = 0.f, u = 0.f;
                if (vnni_gu) {
                    nf::nvfp4_dot2(n_embd, (const nf::BlockNvfp4*) gate, (const nf::BlockNvfp4*) up,
                                   (const nf::BlockQ8_0*) a, ysp + (size_t) experts[j].row * n_embd / 4, &g, &u,
                                   kNvfp4Prefetch);
                } else {
                    gdot(n_embd, &g, 0, gate, 0, a, 0, 1);
                    udot(n_embd, &u, 0, up, 0, au, 0, 1);
                }
                g *= e.gate_scale;
                u *= e.up_scale;
                if (situ) {
                    const float gated = situ->beta * std::tanh(g / situ->beta) / (1.f + std::exp(-g));
                    const float lin = situ->lin_beta > 0.f ? situ->lin_beta * std::tanh(u / situ->lin_beta) : u;
                    hp[(size_t) j * n_ff + r] = gated * lin;
                    continue;
                }
                if (gelu) {
                    hp[(size_t) j * n_ff + r] =
                        0.5f * g * (1.f + std::tanh(0.7978845608f * g * (1.f + 0.044715f * g * g))) * u;
                    continue;
                }
                g = std::min(g, limit);
                u = std::clamp(u, -limit, limit);
                hp[(size_t) j * n_ff + r] = g / (1.f + std::exp(-g)) * u;
            }
        }
    };
    uint8_t* hqp = hq.data();
    const auto hq_from = traits(td->vec_dot_type)->from_float;
    if (vnni_d) ysum_h.resize((size_t) n * n_ff / 4);
    int32_t* yh = ysum_h.data();
    const int32_t* yhp = yh;
    const auto hq_one = [&](int j) { hq_from(hp + (size_t) j * n_ff, hqp + (size_t) j * h_bytes, n_ff); };
    const auto yh_one = [&](int j) {
        nf::nvfp4_ysum4((const nf::BlockQ8_0*) (hqp + (size_t) j * h_bytes), n_ff, yh + (size_t) j * n_ff / 4);
    };

    const int d_tasks = (n_embd + kRowsPerDownTask - 1) / kRowsPerDownTask;
    const auto down_task = [&](int t) {
        static thread_local std::vector<float> one, two;   // an output value of every use (rows r and r + 1)
        one.resize((size_t) n);
        two.resize((size_t) n);
        float* s = one.data();
        float* s2 = two.data();
        const int r0 = t * kRowsPerDownTask, r1 = std::min(n_embd, r0 + kRowsPerDownTask);
        if (vnni_d) {   // two rows at once, each summed for its tokens as below
            for (int r = r0; r < r1; r += 2) {
                const bool pair = r + 1 < r1;
                for (int u = 0; u < n_run; ++u) {
                    const uint8_t* down = experts[byp[runp[u]]].down + (size_t) r * d_row;
                    for (int i = runp[u]; i < runp[u + 1]; ++i) {
                        const int j = byp[i];
                        const nf::BlockQ8_0* hj = (const nf::BlockQ8_0*) (hqp + (size_t) j * h_bytes);
                        if (pair)
                            nf::nvfp4_dot2(n_ff, (const nf::BlockNvfp4*) down, (const nf::BlockNvfp4*) (down + d_row),
                                           hj, yhp + (size_t) j * n_ff / 4, s + j, s2 + j, kNvfp4Prefetch);
                        else
                            s[j] = nf::nvfp4_dot(n_ff, (const nf::BlockNvfp4*) down, hj, yhp + (size_t) j * n_ff / 4);
                    }
                }
                for (int k = 0; k < (pair ? 2 : 1); ++k) {
                    const float* sv = k ? s2 : s;
                    int j = 0;
                    for (int row = 0; row < rows; ++row) {
                        float acc = 0.f;
                        for (; j < n && experts[j].row == row; ++j) acc += experts[j].weight * sv[j];
                        out[(size_t) row * n_embd + r + k] = acc;
                    }
                }
            }
            return;
        }
        for (int r = r0; r < r1; ++r) {
            for (int u = 0; u < n_run; ++u) {
                const uint8_t* down = experts[byp[runp[u]]].down + (size_t) r * d_row;
                for (int i = runp[u]; i < runp[u + 1]; ++i)
                    ddot(n_ff, s + byp[i], 0, down, 0, hqp + (size_t) byp[i] * h_bytes, 0, 1);
            }
            // a token's sum in the order its experts were given: the order it has in a call of its own
            int j = 0;
            for (int row = 0; row < rows; ++row) {
                float acc = 0.f;
                for (; j < n && experts[j].row == row; ++j) acc += experts[j].weight * s[j];
                out[(size_t) row * n_embd + r] = acc;
            }
        }
    };

    if (fused_runs() && n_run <= kMaxFusedRuns) {
        std::atomic<int> left[kMaxFusedRuns];   // per run of uses: its gate and up tasks not done yet
        for (int u = 0; u < n_run; ++u) left[u].store(gu_tasks, std::memory_order_relaxed);
        std::atomic<int> runs_done{0};
        const int n_gu = n_run * gu_tasks;
        // the tasks are taken in their order, so a down task is only taken once every gate and up task is
        workers.run(first + n_gu + d_tasks, [&](int t) {
            if (t < first) return (*meanwhile)();
            t -= first;
            if (t < n_gu) {
                gu_task(t);
                const int u = orderp[t / gu_tasks];
                if (left[u].fetch_sub(1, std::memory_order_acq_rel) == 1) {   // the run's last: its outputs are all there
                    for (int i = runp[u]; i < runp[u + 1]; ++i) {
                        hq_one(byp[i]);
                        if (vnni_d) yh_one(byp[i]);
                    }
                    runs_done.fetch_add(1, std::memory_order_release);
                }
                return;
            }
            while (runs_done.load(std::memory_order_acquire) < n_run) _mm_pause();
            down_task(t - n_gu);
        });
        return;
    }
    workers.run(first + n_run * gu_tasks, [&](int t) {
        if (t < first) return (*meanwhile)();
        gu_task(t - first);
    });
    workers.run(n, hq_one);
    if (vnni_d) workers.run(n, yh_one);
    workers.run(d_tasks, down_task);
}

}  // namespace strata::poly
