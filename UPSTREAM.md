# Where the code comes from

| Here | From | License |
|---|---|---|
| `engine/` | PolyStrata pull request #200 (`src/poly`, head `6e610541`), itself built on Strata. File by file: `engine/UPSTREAM.md` | MIT |
| `engine/dspark.cu`, `engine/dspark.hpp` | Strata pull request #6506, `engine/src/ik2/ik2_draft.cu` and parts of `ik2_kernels.cu` | MIT |
| `third_party/llama.cpp/ggml` | llama.cpp `3cf03257` (unchanged; the pin PolyStrata uses) | MIT |
| `server/serve`, `server/tools/strata_tokenizer.py`, `server/tools/gguf_reader.py` | PolyStrata `serve/` and `tools/` at `6e610541`, without its tests; with thinking off the reply opens as a text block (after Strata's #6612 / #6627) | MIT |
| `tools/vision` (strata-vision, the image encoder) | PolyStrata `tools/vision` at `6e610541`, unchanged; built on llama.cpp `690b87d88`'s mtmd (CPU) | MIT |
| everything else | this project | MIT |

## This project's changes to the engine

Each one is a commit in the development history; the engine's defaults are listed with what they were measured on
(RTX 5070 12 GB, Core Ultra 7 265K, DDR5-5600 32 GB, Windows 11). The opt-in ones stay off unless their
environment variable is set.

- Builds and runs on Windows (MSVC, CUDA 13), on its own beside the rest of Strata (`engine/CMakeLists.txt`).
- Defaults chosen by measurement: KV cache q8_0, prompt batches of 2,048 tokens, 400 MiB of VRAM left free,
  a fixed RAM budget for the experts outside VRAM, context lookup drafting with a window of 8.
- The sampler takes a fast path when top_k, top_p and min_p are off.
- The router's read-ahead scores are written to pinned memory by a kernel instead of a copy (`INKLING2_ZC_AHEAD`):
  on Windows the copy engine's start delayed every layer (+13% at 8K).
- The CPU expert threads spin longer before sleeping (`INKLING2_SPIN`, default 200,000): +5% at 8K and 128K, with
  3 to 4 more cores busy while it writes.
- Opt-in: one synchronization per CPU expert call (`INKLING2_CPU_FUSED`), routing on the GPU with only the misses sent to
  the host (`INKLING2_CARD_SPLIT`, `INKLING2_CARD_PIPE`), the DSpark drafter (`INKLING2_DSPARK`), a pinned expert
  pool, merged CUDA graphs, flag waits and timing reports (`INKLING2_TIMELINE`, `INKLING2_PREFILL_TIMES`,
  `INKLING2_REPORT_CACHE`).
- In the engine but not offered by setup: Strata's `.strata` v1.1 SQ builds (SQ-10, SQ-15), read
  by `engine/include/strata/sq/` and run by Strata's own kernels (copied whole, MIT; NOTICE lists them): decode
  by its dp4a S_K dot products (a token's S_K experts in one launch) and its half-precision SF4 kernels
  (`INKLING2_SQ_SF4_FLOAT=1`: float ones), prompt batches by its int8 / fp16 tensor-core kernels with the experts outside
  VRAM staged a round at a time (`INKLING2_SQ_STAGE_MIB`, default 64), the CPU's experts by its host evaluator (S2_K / S3_K unpacked by AVX2, bit-identical); the trunk of a
  prompt batch by MMQ.
  Opt-in: FP4 activations for the SF4 experts of a prompt (`INKLING2_SQ_W4A4=1`, Strata's sm_120a kernels), and
  FP16 sums in the prompt's attention (`INKLING2_ATTN_F16=1`: P V within a tile of 32 keys; `=2`: Q K^T too).
- 0.2.0: `--skip-renorm` and `--skip-in-prompt` beside PolyStrata's `--skip-weight` / `--tail-skip` (both
  approximations, off by default; the skipping no longer applies to a prompt unless `--skip-in-prompt`), the experts
  left out and the hit rate in the per-answer cache line, the skip options in `inkling2_logits_dump`, and
  `INKLING2_ZERO_ORDER` (opt-in: the order of the experts no token has used yet when the cache is filled; the idea of
  favoring the loop's layers is kuraneko1/Strata's, MIT, no code taken; measured slower, so the default is unchanged).
- 0.3.2 (each the same bits as before unless noted; speeds measured warm, after one answer in the same engine):
  - The prompt attention's codes to halves without int-to-float conversions (`INKLING2_ATTN_CVT=0` turns it off):
    256K read +2-3%.
  - A read's token embeddings on the CPU threads (`INKLING2_EMBED_MT=0` off).
  - A long prompt is read in fewer passes (`--read-mib`, by default by the machine's RAM: 2048 MiB with 30 GiB or
    more, 1024 with 15, else 512, at most a quarter of what is free), so the experts outside VRAM are copied to the
    card fewer times; setup writes it.  The idea of staging only the experts VRAM lacks follows kuraneko1/Strata (MIT).
  - A prompt batch's F32 products with 64 or fewer outputs (the loop's gates) by the tiled kernel
    (`INKLING2_GEMM_TILE=0` off). With the two above: 4K / 32K / 64K reads +5% / +9% / +11%.
  - The CPU experts' AVX-VNNI NVFP4 dot product by default on a CPU with AVX-VNNI (`INKLING2_CPU_VNNI=0` off; it
    matches ggml's bit for bit): 64K writing +8-12%.
  - `--window` up to 64 (default still 8); per-window-size tables that were written past their end from 9 tokens on
    are sized for it.
  - Opt-in: `--kv k8v4` (8-bit keys, 4-bit values with a per-32 Walsh-Hadamard rotation; the ideas of NInfer's
    `rk8v4` (Apache-2.0) and TurboQuant (MIT), no code taken): more experts in VRAM at 256K (673 -> 927) and
    writing +18% there, quality at 128K a little past this project's bar (KLD 0.0075 -> 0.0124).
  - Measurement switches: `INKLING2_VISIT_TRACE` (a written token's visits stamped on the card and the host),
    `inkling2_cpu_calls`, `INKLING2_READ_MIB`.
- 0.3.3:
  - `--loops N` (experimental, 1 to the file's 3): the loop's layers (6 to 17) run N times, an approximation; a
    session file saved with another count is bound to it.
  - `sys_at=POS` (the server sends it unless `STRATA_SYS_CHECKPOINT=0`): a checkpoint where the prompt's first turn
    ends, kept when the others are thinned and written to a session file, so a compacted conversation goes on from
    its system prompt and tools (60K: 6,977 tokens read -> 62).
  - setup writes `--window 32` (8 before).
  - Measurement switch: `INKLING2_ROUTE_LOG` (a written token's picks, a line a visit).
- `inkling2_logits_dump`, `inkling2_cpu_bandwidth` and `inkling2_pcie_bandwidth` for the quality checks and
  measurements.

## Server

PolyStrata's server checks the engine's file name to know which commands it has; `strata-inkling2` is added to
those checks (two lines of `server/serve/server.py`). The other changes:

- 0.2.0: with thinking off, the prompt opens the reply's text block (an empty thinking block with tools), after
  Strata's #6612 / #6627.
- 0.3.1: with tools, the thinking closed early to leave room for the answer leaves it half of `max_tokens` (a
  quarter without tools, as before); a tool call cut off by `max_tokens` sends the arguments written so far, with
  `finish_reason` `length` (they were sent empty); one line per request with the sampling the client sent and the
  one used, the thinking level and the limits, on the screen and in the engine log (`STRATA_LOG_SAMPLING=0` turns
  it off).
