# Measurements

All on one PC: RTX 5070 12 GB (250 W, PCIe 5.0 x16), Core Ultra 7 265K, DDR5-5600 32 GB, Windows 11, NVIDIA
driver 610.60, November 22, 2026. One request at a time; writing speed over 200 tokens after the first 16;
`bench\bench.py` for this engine, `bench\ninfer_bench.py` for NInfer, the same token sequences.

## This engine

Settings: `--max-context 132096 --threads 12 --window 8 --lookup 3 --prompt-batch 2048 --vram-reserve-mib 400
--ram-experts-mib 9216 --kv q8_0` (what setup chooses for 12 GB / 32 GB). About 1,500 of the 3,072 routed experts
fit in VRAM; the others are computed on the CPU from RAM.

| | Plain, greedy | Plain, temperature 1.0 | Lookup, greedy | Lookup, temperature 1.0 | Reading |
|---|---|---|---|---|---|
| 8K chat prompt | 68.9 tok/s | 67.1 tok/s | 69.7 tok/s | 66.2 tok/s | 2,247 tok/s (8,192 tokens) |
| 128K code context | 42.9 tok/s | 44.8 tok/s | 45.2 tok/s | 44.5 tok/s | 1,313 tok/s (the last 8K), 1,738 tok/s (all 128K) |

- Temperature 1.0 is `temperature=1.0 top_k=0 top_p=1 min_p=0 seed=1234` (the whole vocabulary).
- Run to run these move by about ±3% (thermal state, what else runs, the expert cache).
- While writing, the engine keeps 7 to 8 of the CPU's 20 threads busy (its expert threads spin between layers).
- The release build (all GPU generations, AVX2 CPU code) wrote 62.4-65.3 tok/s at 8K, temperature 1.0, against
  64.4-64.8 for a build for this PC only: the same within the noise.

Quality, against llama.cpp 690b87d's logits (KV fp16) for 1,023 positions of a 2,048-token text: mean KLD 0.00491,
top-1 agreement 97.36%, perplexity 3.518 against llama.cpp's 3.516.

## 0.2.0: smaller models, VRAM budgets, 256K (November 23, 2026)

Same PC and settings, temperature 1.0. The 8 GB and 6 GB budgets are the RTX 5070 with `--vram-reserve-mib 4435` /
`6483` (what setup leaves free for such a card, with 504 MiB held by other programs), not real 8 GB / 6 GB cards.
Prompt batches of 2,048. Long contexts: the last 8K read after a saved context, then writing.

| Model | VRAM | 8K read / write | 64K read / write | 128K read / write |
|---|---|---|---|---|
| IQ4_XS | 12 GB | 2,162 / 66.7 | 1,725 / 51.2 | 1,327 / 43.8 |
| IQ4_XS | 8 GB | 1,902 / 51.4 | 1,541 / 34.3 | 1,214 / 25.3 |
| IQ4_XS | 6 GB | 1,795 / 35.0 | 1,446 / 19.4 | does not fit |
| Q3_K_XL | 12 GB | 2,186 / 69.0 | 1,741 / 55.4 | 1,340 / 45.1 |
| Q3_K_XL | 8 GB | 2,003 / 57.6 | 1,650 / 38.8 | 1,253 / 26.1 |
| Q3_K_XL | 6 GB | 1,949 / 39.0 | 1,532 / 22.2 | does not fit |
| Q2_K_XL | 12 GB | 2,604 / 74.0 | 1,962 / 53.7 | 1,432 / 45.4 |
| Q2_K_XL | 8 GB | 2,325 / 67.0 | 1,811 / 43.5 | 1,351 / 27.0 |
| Q2_K_XL | 6 GB | 2,272 / 43.9 | 1,710 / 23.9 | does not fit |
| IQ2_XXS | 12 GB | 2,870 / 79.9 | 1,966 / 63.0 | 1,509 / 51.6 |
| IQ2_XXS | 8 GB | 2,362 / 66.7 | 1,841 / 45.7 | 1,397 / 26.3 |
| IQ2_XXS | 6 GB | 2,388 / 44.5 | 1,755 / 21.5 | does not fit |

- 12 GB at 64K: one run each. "does not fit": at 6 GB the 128K KV cache alone went 0.7-0.9 GB over the budget.
- 2K quality against Unsloth's Q8_K_XL (llama.cpp): IQ4_XS mean KLD 0.0026 (top-1 98.4%), Q3_K_XL 0.0090 (97.2%),
  Q2_K_XL 0.0535 (93.5%), IQ2_XXS 0.0687 (90.8%).

256K on 12 GB (`--max-context 263168`, the last 8K read after 245,758 tokens). IQ2_XXS, one run each:

| Prompt batch | Left free | Threads | Experts in VRAM | Write | Read the last 8K |
|---|---|---|---|---|---|
| 2,048 | 400 MiB | 12 | 1,485 | 31.9 | 1,008 |
| 1,024 | 400 MiB | 12 | 1,844 | 35.3 | 954 |
| 512 | 400 MiB | 12 | 2,026 | 36.1 | 758 |
| 1,024 | 200 MiB | 12 | 1,903 | 36.9 | 956 |
| 1,024 | 800 MiB | 12 | 1,654 | 34.6 | 944 |
| 1,024 | 400 MiB | 8 / 10 | 1,844 | 32.2 / 34.1 | 956 / 948 |

With 1,024 / 200 MiB (what setup uses for 256K): IQ4_XS 33.6 / 848, Q2_K_XL 38.1 / 942; reading all 253,950 tokens
from the start 1,355 tok/s (IQ2_XXS). At 128K the same settings gave IQ4_XS 45.4 / 1,195 against 43.7 / 1,343 with
2,048 / 400 MiB, so the 128K default stays.

Expert skipping (`--skip-weight`, an approximation), IQ4_XS at 128K: none 43.0, 0.06 45.1, 0.08 46.7, 0.12 50.5 tok/s;
2K KLD vs Q8 0.0026, 0.0031, 0.0038, 0.0071 (with the cache cut to its 128K size: 0.0047 at 0.08, 0.0106 at 0.12).
At 8K it changes little (95% of the experts a token uses are in VRAM). `--skip-renorm` was as fast and worse at 2K.

A conversation's turn 2, thinking off (the start it shares with turn 1 reused / read again from the start): a short
one 164 / 402 ms, a 32K one 229 ms / 15.6 s.

## NInfer 2026-11-22.2

`ninfer-serve --expert-residency host --host-cache-mib 1024 --spec none`, KV MXFP8 (its default), the model
`WaveCut/Inkling-2-Mini-UD-IQ4_XS-NInfer-v3`, Windows CUDA 13.4 package. NInfer's top_k is at most 20.

| | Greedy | Temperature 1.0 | Reading |
|---|---|---|---|
| 8K chat prompt | 51.5 tok/s | 49.7 tok/s | 2,171 tok/s |
| 128K code context | 26.3 tok/s | 26.2 tok/s | 663 tok/s (all 131,072 tokens) |

More, and what it does differently: [ninfer_comparison.md](ninfer_comparison.md).
