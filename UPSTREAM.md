# Where the code comes from

| Here | From | License |
|---|---|---|
| `engine/` | PolyStrata pull request #200 (`src/poly`, head `6e610541`), itself built on Strata. File by file: `engine/UPSTREAM.md` | MIT |
| `engine/dspark.cu`, `engine/dspark.hpp` | Strata pull request #6506, `engine/src/ik2/ik2_draft.cu` and parts of `ik2_kernels.cu` | MIT |
| `third_party/llama.cpp/ggml` | llama.cpp `3cf03257` (unchanged; the pin PolyStrata uses) | MIT |
| `server/serve`, `server/tools/strata_tokenizer.py`, `server/tools/gguf_reader.py` | PolyStrata `serve/` and `tools/` at `6e610541`, without its tests | MIT |
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
- Opt-in: an AVX-VNNI NVFP4 dot product for the CPU experts that matches ggml's bit for bit (`INKLING2_CPU_VNNI`),
  one synchronization per CPU expert call (`INKLING2_CPU_FUSED`), routing on the GPU with only the misses sent to
  the host (`INKLING2_CARD_SPLIT`, `INKLING2_CARD_PIPE`), the DSpark drafter (`INKLING2_DSPARK`), a pinned expert
  pool, merged CUDA graphs, flag waits and timing reports (`INKLING2_TIMELINE`, `INKLING2_PREFILL_TIMES`,
  `INKLING2_REPORT_CACHE`).
- `inkling2_logits_dump`, `inkling2_cpu_bandwidth` and `inkling2_pcie_bandwidth` for the quality checks and
  measurements.

## Server

PolyStrata's server checks the engine's file name to know which commands it has; `strata-inkling2` is added to
those checks (two lines of `server/serve/server.py`). Nothing else is changed.
