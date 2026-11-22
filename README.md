# inkling2-mini

[日本語](README.ja.md)

> **Unofficial.** An independent fork built on [Strata](https://github.com/Niko1221/Strata) and
> [PolyStrata](https://github.com/simplexai-labs/PolyStrata); it is not part of either project and is not supported
> by their authors.

Inkling 2 Mini (Thinking Machines, 26.4B total / 3.4B active, looped MoE) on one NVIDIA GPU plus system RAM, with
the experts that do not fit in VRAM computed on the CPU.

- Model: Unsloth's `Inkling-2-Mini-UD-IQ4_XS.gguf` (15.7 GB). An online VRAM expert cache keeps the most used
  experts on the GPU.
- Up to 128K context on a 12 GB card (q8_0 KV).
- Exact: every routed expert is computed (no expert skipping); context-lookup drafting is verified.
- Engine: PolyStrata's `strata-poly` (on Strata) cut down to Inkling 2 and tuned on Windows. Server: PolyStrata's.
  See [UPSTREAM.md](UPSTREAM.md) and [NOTICE](NOTICE).

Tested on one PC only: RTX 5070 12 GB, Core Ultra 7 265K, DDR5-5600 32 GB, Windows 11, driver 610.60.

## Speed on that PC

One request; 200 tokens after the first 16. 128K = a 120K-token code context plus 8K more.

| | This engine | NInfer 2026-11-22.2 (Windows) |
|---|---|---|
| 8K, temperature 1.0 | 67.1 tok/s | 49.7 tok/s |
| 8K, greedy | 68.9 tok/s | 51.5 tok/s |
| 8K, greedy + lookup drafting | 69.7 tok/s | |
| 8K prompt | 2,247 tok/s | 2,171 tok/s |
| 128K, temperature 1.0 | 44.8 tok/s | 26.2 tok/s |
| 128K, greedy | 42.9 tok/s | 26.3 tok/s |
| 128K prompt (whole context) | 1,738 tok/s | 663 tok/s |
| CPU while writing | 7-8 cores | ~10 cores |

- NInfer: `--expert-residency host --host-cache-mib 1024 --spec none`, MXFP8 KV, same tokens. Its top_k is capped
  at 20 (this engine samples from the whole vocabulary). Its docs report 83.5 tok/s on an RTX 5070 under Linux;
  on Windows it disables CUDA graphs for this model. [docs/ninfer_comparison.md](docs/ninfer_comparison.md)
- Against llama.cpp's logits (fp16 KV), 2,048 tokens: mean KLD 0.0049, top-1 97.4%.
- More: [docs/measurements.md](docs/measurements.md)

## Requirements

- Windows 10/11 x64
- NVIDIA GPU, compute capability 7.5+ (RTX 20 to 50), 8 GB VRAM or more (12 GB for 128K). Built for 7.5, 8.6, 8.9,
  12.0 + PTX.
- Driver 580+ (CUDA 13)
- RAM 16 GB+, 32 GB recommended (up to 9 GB of experts in RAM; with less, experts are read from the SSD and
  writing slows down)
- ~20 GB free disk
- Python 3.10+ (setup creates a venv and installs the server's packages into it)

## Use

1. Unpack the release zip (or build, below).
2. `setup.bat`: checks the GPU/driver/RAM/disk, downloads the model (resumable) and checks its SHA-256, creates
   `.venv` with the server's Python packages, and writes the settings to `config\inkling.json`.
3. `run.bat`: an OpenAI-compatible server (Chat Completions, Responses, Anthropic Messages, a chat page) on
   `http://127.0.0.1:8080`, local only.

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "inkling-2-mini", "messages": [{"role": "user", "content": "Hello"}], "reasoning_effort": "medium"}'
```

| Option | |
|---|---|
| `setup.bat --port N` | server port |
| `setup.bat --limit-ram-gb 16 --limit-vram-gb 8` | settings for a smaller budget |
| `setup.bat --context N` | context length |
| `setup.bat --configure-only` | choose the settings again, no download |
| `run.bat --api-key KEY` | require a key |

Settings setup chooses by usable VRAM: 128K context / 2,048-token prompt batches from 12 GB, 64K / 1,024 from 10 GB,
32K / 1,024 from 8 GB; experts in RAM = RAM - 14 GB (max 9 GB); up to 12 CPU threads. Only 12 GB / 32 GB was measured.

## Known limits

- Output varies slightly between runs, greedy included: which experts run on the GPU or the CPU follows the cache,
  and the two sum in different orders.
- 7-8 CPU cores busy while writing (the expert threads spin between layers; idle between requests).
- ~12 GB of RAM for the engine with the default 9 GB expert budget, plus the OS file cache.
- Tested on Windows, on one PC.
- Quality at 128K context has not been checked yet (only 2K against llama.cpp so far).
- No image input in this package.

## Build

Visual Studio 2022+ (C++ desktop), CUDA Toolkit 13.x, Python 3.10+.

```bat
tools\build.bat            :: compute capability 7.5 / 8.6 / 8.9 / 12.0 (slow)
tools\build.bat --native   :: this PC only
```

Output: `build\bin` (setup finds it). `python tools\package.py --version X` writes the release zip to `dist\`.

## Measuring

`bench\` (notes in Japanese, `bench\README.md`): `bench.py` (speed), `quality.py` (2K logits vs llama.cpp),
`quality_all.py` (2K and 128K, unattended), `ninfer_bench.py` (NInfer on the same tokens).

## License

MIT ([LICENSE](LICENSE)); components in [NOTICE](NOTICE). The model (Apache-2.0) is downloaded separately.
