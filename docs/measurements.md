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

## NInfer 2026-11-22.2

`ninfer-serve --expert-residency host --host-cache-mib 1024 --spec none`, KV MXFP8 (its default), the model
`WaveCut/Inkling-2-Mini-UD-IQ4_XS-NInfer-v3`, Windows CUDA 13.4 package. NInfer's top_k is at most 20.

| | Greedy | Temperature 1.0 | Reading |
|---|---|---|---|
| 8K chat prompt | 51.5 tok/s | 49.7 tok/s | 2,171 tok/s |
| 128K code context | 26.3 tok/s | 26.2 tok/s | 663 tok/s (all 131,072 tokens) |

More, and what it does differently: [ninfer_comparison.md](ninfer_comparison.md).
