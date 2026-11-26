# inkling-5070

[日本語](README.md)

> **Unofficial.** An independent fork built on [Strata](https://github.com/Niko1221/Strata) and
> [PolyStrata](https://github.com/simplexai-labs/PolyStrata); it is not part of either project and is not supported
> by their authors.

> **Status (0.3.5)**
> - The default IQ4_XS against llama.cpp's logits (IQ4_XS, fp16 KV): mean KLD 0.0051, top-1 agreement 97.5% at a 2K
>   context; mean KLD 0.0079, top-1 97.0% at a 128K context (0.3.4: 0.0053 / 97.7%, 0.0081 / 96.7%).
> - Against Unsloth's Q8_K_XL (llama.cpp) at 2K: mean KLD 0.0027, top-1 98.4% (0.3.4: 0.0030 / 98.0%).
> - The release zip, unpacked into an empty folder, set up and started, held a 30-turn conversation through the server.
> - The smaller models (Q3_K_XL / Q2_K_XL / IQ2_XXS) were checked at 2K only.
> - Pictures: a photo, a screenshot and a chart were described correctly (no benchmark).
> - Tested on one PC only (RTX 5070 / Windows 11). The 8 GB and 6 GB figures were measured on that RTX 5070 with its
>   VRAM limited; no real 8 GB or 6 GB card has been tried.
> - setup's model download has not been tried end to end yet (the SHA-256 check has been).

Inkling 2 Mini (Thinking Machines, 26.4B total / 3.4B active, looped MoE) on one NVIDIA GPU plus system RAM, with
the experts that do not fit in VRAM computed on the CPU.

- **GPUs:** tuned and tested on an RTX 5070. It is built to run on the RTX 20 to 50 series, but no other GPU has been
  tried.
- **How it differs from other 12 GB projects:** those quantize to low bits so that the whole model fits in VRAM; this
  one keeps the 4-bit IQ4_XS weights and computes the experts that do not fit on the CPU, up to a 128K context
  (256K if asked for).
- Model: Unsloth's `Inkling-2-Mini-UD-IQ4_XS.gguf` (15.7 GB) by default; smaller, faster ones can be chosen (below).
  An online VRAM expert cache keeps the most used experts on the GPU.
- Exact by default: every routed expert is computed; context-lookup drafting is verified. Expert skipping is an
  opt-in approximation (below).
- Engine: PolyStrata's `strata-poly` (on Strata) cut down to Inkling 2 and tuned on Windows. Server: PolyStrata's.
  See [UPSTREAM.md](UPSTREAM.md) and [NOTICE](NOTICE).

Much of the code and documentation was written with the help of AI.

Measured on: RTX 5070 12 GB, Core Ultra 7 265K, DDR5-5600 32 GB, Windows 11, driver 610.60.

**Keep Windows' Hardware-accelerated GPU scheduling (HAGS) on.** With it off, the GPU took 5-6 times as long to start what the host sent it, and writing was 10-20% slower (62-64 -> 54 tok/s at 8K).

## What to use, by VRAM

Writing speed (temperature 1.0, tok/s). The 12 GB IQ4_XS row is 0.3.4 (256K: 0.3.2), new code written after a source-code context, measured after one warm-up answer in the same engine (the first answer after a start is about 10% slower, while the expert cache settles). The other rows are 0.3.1's (after an 8K prompt, or after reading the last 8K of a saved long context).

| VRAM | Choice | 8K | 64K | 128K | 256K |
|---|---|---|---|---|---|
| 12 GB | IQ4_XS (default), 128K | 84.7 | 68.6 | 51.2 | 36.6 (`--context 256k`, 0.3.2) |
| 8 GB | IQ4_XS (quality) | 51.4 | 34.3 | 25.3 | does not fit |
| 8 GB | Q3_K_XL (speed) | 57.6 | 38.8 | 26.1 | does not fit |
| 6 GB | IQ4_XS or Q3_K_XL, up to 64K | 35.0 / 39.0 | 19.4 / 22.2 | does not fit | does not fit |

- The figures vary by about ±2% between measuring sessions.
- 12 GB: code contexts (100K: 61.6). 256K was measured with the bench settings (2,048-token prompt batches, 400 MiB of VRAM left free).
- **8 GB / 6 GB:** measured on this RTX 5070 with its VRAM limited by `--vram-reserve-mib` (not on real cards), with
  2,048-token prompt batches. setup picks 1,024 / 512 for those cards, which leaves more VRAM for experts, so writing
  should be faster than this (not measured).
- **Context:** setup picks 32K on 8 GB and 16K on 6 GB; for the lengths above use e.g. `setup.bat --context 128k`.
- **Long contexts narrow the gaps:** the KV cache takes the experts' VRAM and most of them run on the CPU.

![Writing speed](docs/charts/decode_en.png)

![Reading speed](docs/charts/reading_en.png)

![Where a token's time goes at 100K](docs/charts/breakdown_en.png)

The charts are drawn from `docs/charts/measurements.json` by `python docs/charts/make_charts.py`. The "0.3.2" bars are this engine with 0.3.4's three changes set to 0 and `--window 8` (0.3.2's decode), measured the same day.

### Quality and speed of each model

Quality: 2K context against Unsloth's Q8_K_XL logits (llama.cpp); a lower KLD is closer to Q8.

| Model (`--model`) | File | Mean KLD | Top-1 | 12 GB writing (8K / 128K) |
|---|---|---|---|---|
| `iq4_xs` (default) | 15.7 GB | 0.0030 | 98.0% | 84.7 / 51.2 (0.3.4) |
| `q3_k_xl` | 13.6 GB | 0.0090 | 97.2% | 69.0 / 45.1 |
| `q2_k_xl` | 9.1 GB | 0.0535 | 93.5% | 74.0 / 45.4 |
| `iq2_xxs` | 8.5 GB | 0.0687 | 90.8% | 79.9 / 51.6 |

- Speeds vary by about ±2% between measuring sessions. All but IQ4_XS are 0.3.1's (their quality 0.3.1-0.3.2's; 0.3.4's
  changes were checked with IQ4_XS).
- Q2_K_XL and IQ2_XXS are fast, but their measured quality drop is large (KLD about 20-26x IQ4_XS's); setup says so
  when one is chosen.
- The smaller models come from the same Unsloth repository and revision, SHA-256 checked.

![Quality](docs/charts/quality_en.png)

### 256K (12 GB)

`setup.bat --context 256k` (setup then uses 1,024-token prompt batches and leaves 200 MiB free). After reading the
last 8K of a ~245K-token context (0.3.1):

| Model | Experts in VRAM | Writing (temperature 1.0) | Reading the last 8K |
|---|---|---|---|
| IQ4_XS | 898 / 3072 | 33.6 tok/s | 848 tok/s |
| Q2_K_XL | 1776 / 3072 | 38.1 tok/s | 942 tok/s |
| IQ2_XXS | 1903 / 3072 | 36.9 tok/s | 956 tok/s |

At 256K the three are close, so IQ4_XS (the best quality) is the recommendation. Reading 256K from the start: 1,355
tok/s (IQ2_XXS).

## Speed on that PC (IQ4_XS, 12 GB)

One request; 200 tokens after the first 16. This engine's 8K and 128K temperature-1.0 writing are 0.3.5, after one warm-up answer on the same prompt, without lookup (128K with lookup: 65.9 tok/s); its 128K read is 0.3.4's; the greedy rows, the 8K read and NInfer are 0.3.1's. 128K writing: after a 128K code context.

| | This engine | NInfer 2026-11-22.2 (Windows) |
|---|---|---|
| 8K, temperature 1.0 | 91.3 tok/s | 49.7 tok/s |
| 8K, greedy | 68.9 tok/s | 51.5 tok/s |
| 8K, greedy + lookup drafting | 69.7 tok/s | |
| 8K prompt | 2,247 tok/s (0.3.1; 0.3.2: 2,565 at 4K, 2,475 at 32K) | 2,171 tok/s |
| 128K, temperature 1.0 | 61.3 tok/s | 26.2 tok/s |
| 128K, greedy | 42.9 tok/s | 26.3 tok/s |
| 128K prompt (whole context) | 1,978 tok/s | 663 tok/s |
| CPU while writing | 7-8 cores | ~10 cores |

- This engine's figures vary by about ±2% between measuring sessions.
- NInfer: `--expert-residency host --host-cache-mib 1024 --spec none`, MXFP8 KV, same tokens. Its top_k is capped
  at 20 (this engine samples from the whole vocabulary). Its docs report 83.5 tok/s on an RTX 5070 under Linux;
  on Windows it disables CUDA graphs for this model. [docs/ninfer_comparison.md](docs/ninfer_comparison.md)
- Against llama.cpp's logits (IQ4_XS, fp16 KV; this engine's KV is q8_0):
  - 2K (2,048 tokens): mean KLD 0.0053, top-1 97.7% (0.3.2: 0.0049 / 97.4%)
  - 128K (the last 1,023 positions of a 124,928-token context): mean KLD 0.0081, top-1 96.7% (0.3.2: 0.0075 / 96.9%)
  - Against Q8_K_XL at 2K: mean KLD 0.0030, top-1 98.0% (0.3.2: 0.0026 / 98.4%)
- A conversation's next turn reuses the start it shares with the last one: turn 2 of a 32K conversation read in
  0.23 s (15.6 s read again from the start).
  - After an agent compacts the conversation, the system prompt and the tool definitions are not read again either
    (a checkpoint that is never dropped sits where the first turn ends). In a 60K conversation the turn after a
    compaction read 62 tokens instead of 6,977 (0.4 s instead of 4.2 s). `STRATA_SYS_CHECKPOINT=0` turns this off.
    The same holds after a saved session (`/slots`) is restored: the 60K conversation read 62 tokens.
  - Send the tool definitions from the first turn on: adding them later changes the start, which is then read again.
- When an answer copies text from the conversation (a file being edited), the engine guesses up to 31 tokens of what
  follows and checks them at once (`--window 32`; 8 up to 0.3.2). Edit requests at 64K and 100K wrote 5-14% faster
  than with 8 (search/replace blocks alone: ±2%). Text that copies little is as fast as before.
  - Window 32 places slightly different experts in VRAM than window 8, so the output may not match it exactly (the
    same kind of variation that happens between starts at the default).
- More: [docs/measurements.md](docs/measurements.md)

### What changed in 0.3.5 (on by default; `=0` turns each off)

Ten changes, most of them a token's small kernels done as fewer, are on by default.

| Change | Off |
|---|---|
| The experts' weight scales applied in the sum of the card's part | `INKLING2_FUSE=0` |
| The mixing step over a cluster of 6 blocks | `INKLING2_MIX_CLUSTER=0` |
| In a token's CUDA graph, a kernel launched while the one before it ends (PDL) | `INKLING2_PDL=0` |
| The same, the next kernel launched once every block of the one before runs | `INKLING2_PDL_EARLY=0` |
| The same, a kernel's weights asked into L2 while it waits | `INKLING2_PDL_PREFETCH=0` |
| While writing, the scratch holds 256 rows and the expert cache takes the rest (a read widens it) | `INKLING2_NARROW_WRITE=0` |
| A window's (the checked guesses') attention with its tokens side by side | `INKLING2_ATTN_TOKEN_FAST=0` |
| The streams' row copies by a kernel | `INKLING2_COPY_KERNEL=0` |
| A window's logits copied to the host only as far as they are used | `INKLING2_LAZY_LOGITS=0` |
| A window's (up to 32 tokens) experts routed on the card too | `INKLING2_CARD_SPLIT=0` |

Measured (0.3.4's defaults -> the nine of them other than CARD_SPLIT, window 32, temperature 1.0):

| | 0.3.4 | 0.3.5 | Change |
|---|---|---|---|
| 8K writing, no lookup (2 runs each) | 82.4 / 82.6 | 90.4 / 92.2 | +10.7% |
| 100K writing, no lookup (2 runs each) | 60.0 / 59.9 | 65.3 / 66.4 | +9.9% |
| 128K writing, no lookup (mean of 4) | 53.5 | 61.3 | +14.8% |
| 128K writing, lookup (mean of 4) | 54.0 | 65.9 | +22% |
| 128K read | 2,044 / 2,060 | 2,009 / 2,014 | -1.9% |
| 64K edit requests fix / code / blocks (1 run each) | 94.2 / 91.5 / 77.2 | 95.2 / 101.6 / 88.2 | +1% / +11% / +14% |
| 100K edit requests fix / code / blocks | 79.8 / 81.9 / 64.3 | 87.1 / 89.1 / 74.6 | +9% / +9% / +16% |

- **CARD_SPLIT=2:** edit requests that copy the conversation's text write faster still.
  - +2-9% at 64K, +2-8% at 100K.
  - Measured with the others on, against `INKLING2_CARD_SPLIT=1` (windows of up to 4 tokens routed on the card), 2 runs
    each, alternating.
- **Output:** the model is the same. The sums' order and the experts' places change, so the output is not exactly 0.3.4's;
  the quality is the Status above (about 0.3.4's).
- **Reads:** NARROW_WRITE widens the scratch for each read: a long read is about 2% slower (+30-60 ms a short read).
- **Fixed in 0.3.5** (each could happen with these changes on):
  - After a read, the scratch was narrowed while a copy on the card still read the VRAM it gave back (a rare illegal memory
    access).
  - During a read, a stale table pointed at expert slots that had been given back (an illegal memory access with CARD_SPLIT).
  - A STOP that came just after the next request was taken could be lost.
  - When widening the scratch failed, the expert cache stayed smaller.
  - LAZY_LOGITS: a window's logits overwritten before they were copied are an error now, not silently used.
- **Not in (still opt-in):**
  - Prompt-read kernels: `INKLING2_QSCATTER=1`, `INKLING2_SITU_MMQ=1`, `INKLING2_CTRL_TC=1` (the loop control on TF32),
    `STRATA_PREFILL_NVFP4=w4a4` (experts as FP4 x FP4; changes the output). Their gains were within the run-to-run noise
    (about 8%).
  - `INKLING2_GPU_WINDOW` (a window as one CUDA graph too) still has a bug.

### What changed in 0.3.4 (on by default; an environment variable turns each off)

| Change | Effect (100K, code) | Off with |
|---|---|---|
| A token on the card alone: its 48 visits are one CUDA graph, the experts routed on the card; only when a pick is outside VRAM does a kernel tell the CPU through mapped host memory, and wait in a kernel for its sum | a token without lookup: 21.3 -> 18.3 ms | `INKLING2_GPU_TOKEN=0` |
| Where experts go: layers 20-23, of other types (Q5_K / Q4_K / IQ4_XS), get their slots first, so they keep them when a growing context cuts the cache (at 100K their picks were 93-98% on the CPU) | +6% without lookup | `INKLING2_SLOT_ORDER=0` |
| Sampling: the whole vocabulary's probabilities in AVX2 | 0.78 -> 0.12 ms a token | `INKLING2_FAST_PICK=0` |

- Together, writing at 100K with lookup went from about 50 to 60 tok/s on prose (+21%) and from about 55 to 62.5 on code
  (+14%), two runs each, alternating.
- None changes the model; the order of the sums and where the experts sit change, so the output is not bit for bit
  0.3.2's. The quality is the Status above.

## For long contexts: a 4-bit V cache (a little lower quality)

`--kv k8v4` keeps the attention's values in 4 bits (each block of 32 rotated by a Walsh-Hadamard transform before it is quantized) and its keys in 8 bits. The KV cache is about a fifth smaller than the default (`q8_0`), and the VRAM goes to experts. Off by default; to use it, set `--kv` to `k8v4` in `args` in `config\inkling.json`.

| | default (q8_0) | k8v4 |
|---|---|---|
| KV a token | 17,408 bytes | 13,312 bytes |
| Experts in VRAM at 256K | 673 | 927 |
| Writing at 256K (temperature 1.0) | 30.4 tok/s | 35.8 tok/s |
| 2K KLD vs Q8 | 0.0026 | 0.0049 |
| 128K KLD vs llama.cpp / top-1 | 0.0075 / 96.9% | 0.0124 / 96.1% |

- The 256K speeds were measured in 0.3.1's time (the first answer after a start); they compare runs of one session.
- At 128K its quality drop is a little past this project's bar (1.5x the default's KLD). It is a choice for speed at 256K.
- **With 0.3.4:** faster at long contexts (100K without lookup: 59.9 -> 63.3 tok/s, +5.7%), but at 128K the quality falls to
  mean KLD 0.0112 (1.5x the default's) and top-1 95.5% (1.4 points below the default). The default stays q8_0.

## Faster, approximate: expert skipping

`--skip-weight W` leaves out a routed expert outside VRAM whose router weight is under W of its token's total (the
experts in VRAM are always computed). The output changes; off by default. Add it to `args` in `config\inkling.json`.

| `--skip-weight` | 128K writing (IQ4_XS, 12 GB) | 2K KLD vs Q8 |
|---|---|---|
| none (default) | ~43-45 tok/s (43.0 in this table's session) | 0.0026 |
| 0.06 | 45.1 tok/s (+5%) | 0.0031 |
| 0.08 | 46.7 tok/s (+9%) | 0.0038 (0.0047 with a 128K-sized cache) |
| 0.12 | 50.5 tok/s (+17%) | 0.0071 (0.0106) |

The percentages compare runs of one session; the figures vary by about ±2% between sessions. Little gain where most
experts are in VRAM (8K). Quality at 128K was not measured.

### Experimental: loop count (`--loops N`)

`--loops N` (1 to 3, default 3) runs the loop layers (6 to 17) fewer times. It is faster (8K writing: about 1.2x with
2, about 1.8x with 1), but the answers break down, so it is not for everyday use (2K KLD: 3 loops 0.0026 / 2 loops 0.70
/ 1 loop 0.87). A session saved with one loop count does not load with another. It is an experiment, for a model
trained with a different loop count.

## Requirements

- Windows 10/11 x64
- NVIDIA GPU, compute capability 7.5+ (RTX 20 to 50), 8 GB VRAM or more (6 GB measured only with the VRAM limited).
  Built for 7.5, 8.6, 8.9, 12.0 + PTX.
- Driver 580+ (CUDA 13)
- RAM 16 GB+, 32 GB recommended (up to 9 GB of experts in RAM; with less, experts are read from the SSD and
  writing slows down)
  Up to 2 GB of host memory is pinned for reading long prompts (Task Manager shows it as shared GPU memory).
- ~20 GB free disk for IQ4_XS and the image encoder (10-15 GB for the smaller models)
- Python 3.10+ (setup creates a venv and installs the server's packages into it)

## Use

1. Unpack the release zip (or build, below).
2. `setup.bat`: checks the GPU/driver/RAM/disk, downloads the model (resumable) and the image encoder's weights (0.35 GB; `--vision off`: none) and checks their SHA-256, creates
   `.venv` with the server's Python packages, and writes the settings to `config\inkling.json`. A model you already
   have: type its path when setup asks before the download, or `setup.bat --model-file PATH` (below). Existing
   settings are overwritten only if you say so (`--yes`: without asking). At the end it prints
   estimates for the choice (usable VRAM, KV cache size, experts in VRAM) and tables of other context lengths and
   models with the speeds measured above.
3. `run.bat`: without settings (`config\inkling.json`) it runs `setup.bat` first; with them it starts as they say
   and downloads nothing (a model file the settings name that is not there: it stops and says what to do). It is an
   OpenAI-compatible server (Chat Completions, Responses, Anthropic Messages, a chat page) on
   `http://127.0.0.1:8080`, local only. After the start and after each request it prints the experts in VRAM, their
   share and the hit rate (`--quiet` leaves that out).

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "inkling-2-mini", "messages": [{"role": "user", "content": "Hello"}], "reasoning_effort": "medium"}'
```

`"reasoning_effort": "none"` (or `"chat_template_kwargs": {"enable_thinking": false}`) answers without thinking.
A request that sends no thinking level thinks at `medium` (`setup.bat --thinking`). For an agent that writes files,
`medium` or lower is recommended: at `high` the model often drafts the files in its thinking and reaches `max_tokens`.

| Option | |
|---|---|
| `setup.bat --model NAME` | `iq4_xs` (default) / `q3_k_xl` / `q2_k_xl` / `iq2_xxs` |
| `setup.bat --model-file PATH` | a GGUF you have (no download; other quantizations too) |
| `setup.bat --yes` | overwrite existing settings without asking |
| `setup.bat --vision off` | no pictures (default `cpu`: the image encoder on the CPU) |
| `setup.bat --vision-max-tokens N` | the most tokens a picture takes (default 1,024; larger ones are scaled down) |
| `setup.bat --thinking LEVEL` | the thinking level of requests that send none: `low` / `medium` (default) / `high` |
| `setup.bat --context N` | context length (`64k` / `128k` / `256k` too) |
| `setup.bat --port N` | server port |
| `setup.bat --limit-ram-gb 16 --limit-vram-gb 8` | settings for a smaller budget |
| `setup.bat --configure-only` | choose the settings again, no download |
| `run.bat --api-key KEY` | require a key |
| `run.bat --quiet` | no lines about the experts and speed |

Settings setup chooses by usable VRAM: 128K context / 2,048-token prompt batches from 12 GB (256K with
`--context 256k`: 1,024), 64K / 1,024 from 10 GB, 32K / 1,024 from 8 GB, 16K / 512 below; experts in RAM = RAM - 14 GB
(max 9 GB); up to 12 CPU threads.

### Using a model you have

- Point setup at the GGUF: `setup.bat --model-file C:\models\Inkling-2-Mini-UD-IQ4_XS.gguf`.
  - Nothing is downloaded, and the file stays where it is (the settings name its path).
  - A file on setup's list (same name and size) gets its SHA-256 checked (a minute or two).
  - Plain `setup.bat` asks for a path before it downloads (Enter downloads).
- After moving the file: `setup.bat --model-file NEW_PATH --yes`.
- Once set with `--model-file`, `setup.bat --yes` (to change the context, say) keeps using that file.
- Without the image encoder's weights in `models\`, pictures are turned off (`setup.bat --vision cpu` downloads them).

### Another quantization

- A GGUF of a quantization setup does not list (IQ4_NL, say) works with `setup.bat --model-file PATH`.
  - setup reads the file and checks it is Inkling 2 and made only of types the engine reads.
  - It warns that the quantization is not verified: its quality and speed were not measured (nor do setup's estimates
    apply).
  - A layer whose experts are of a type without GPU kernels computes them on the CPU (slow); setup says which.
- Types the engine reads: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K-Q6_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS,
  IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, MXFP4, NVFP4, Q2_0. With another (TQ1_0, ...), setup and the engine stop and
  name it.
- Back to the default model: `setup.bat --model iq4_xs --yes` (no download if it is in `models\`).

### Updating (`update.bat`, from 0.3.5)

1. Stop the server (close `run.bat`'s window, or Ctrl+C). While it runs, update.bat changes nothing.
2. Run `update.bat`.
   - It asks GitHub for [the newest release of Nsan736/inkling-5070](https://github.com/Nsan736/inkling-5070/releases/latest)
     and downloads its zip if it is newer. It downloads from this repository's releases only.
   - The zip's SHA-256 must be the one in the release's `SHA256SUMS.txt`; otherwise it stops with nothing changed.
   - Only the program is replaced (`bin\`, `server\`, `scripts\`, `docs\`, the bat files, the READMEs, ...). The model
     (`models\`), the settings (`config\`), `.venv`, the logs (`logs\`) and the tokenizer (`data\`) stay.
   - Settings a new version adds go into `config\inkling.json` with your values kept, and are shown (the old file is
     `config\inkling.json.before-update`).
   - A version whose Python packages changed updates `.venv`.
   - The program as it was goes to `old\<version>\`.
- `update.bat --check` only says whether a newer release is out.
- **Going back:** `update.bat --rollback` puts back the newest version in `old\` (the current program goes to
  `old\<version>.replaced\`).
  - By hand: stop the server, delete `bin\`, `server\`, `scripts\`, `docs\`, `licenses\`, the bat files and the READMEs
    in the folder, and move what is in `old\<version>\` into it.
- 0.3.4 and before have no `update.bat`. Copy `update.bat` and `scripts\update.ps1` from the 0.3.5 zip to the same places
  in your folder, then run it to update from 0.3.4.

## Privacy (what 0.3.5's code was read for)

- **Listening:** `run.bat` starts the server on `127.0.0.1` (this PC only).
  - Other devices reach it only if you pass a host yourself, e.g. `run.bat --host 0.0.0.0` (then add `--api-key`).
  - The engine opens no network socket.
- **Outbound:** the engine, the server and the chat page connect to nothing outside (the page's scripts and fonts are
  local). The exceptions:
  - `setup.bat`: Python packages (PyPI), the model and the image encoder's weights (Hugging Face).
  - `update.bat`: GitHub's newest release, to check and to download.
  - A request that holds an image URL (`http(s)://`): the server fetches that image.
  - An MCP server you add in the settings: the connection to it.
- **Logs:** `logs\engine.log` gets numbers only (tokens read, speed, the sampling values); no request or answer text.
  Only with `STRATA_DEBUG=1` does the console show part of a request and the answer.
- **Where files go:** the settings, the logs and the record of reused conversation prefixes (token ids) go inside the
  unpacked folder (`config\`, `logs\`). A picture's temporary files and the like go to the folder's `tmp\`, where
  `run.bat` points them (0.3.4 and before: the user's `%TEMP%`).
- **Browser:** the chat page keeps its conversation and the API key in the browser's localStorage; mind that on a shared PC.

## Known limits

- Output varies slightly between runs, greedy included: which experts run on the GPU or the CPU follows the cache,
  and the two sum in different orders.
- 7-8 CPU cores busy while writing (the expert threads spin between layers; idle between requests).
- ~12 GB of RAM for the engine with the default 9 GB expert budget, plus the OS file cache.
- Tested on Windows, on one PC.
- A little further from the reference at 128K: against llama.cpp, mean KLD rises from 0.0053 at 2K to 0.0081 (top-1 97.7% to
  96.7%). The reference uses fp16 KV; this engine uses q8_0.
- With tools and thinking off, the model may still think briefly before its call.
- Pictures are on by default (`setup.bat --vision off` turns them off). The image encoder runs on the CPU and takes
  no VRAM: about 0.5 GB of RAM, started by the first picture and stopped after two idle minutes. A 640 x 400 picture
  takes about 0.7 s and 160 tokens, 1920 x 1200 about 5.8 s and 1,000 tokens (Core Ultra 7 265K, at most 1,024
  tokens; `--vision-max-tokens 300` brings large ones to about 1.3 s and 273 tokens). With pictures on, the experts in
  VRAM and the writing speed stay the same (checked at 8K and 128K). OpenAI `image_url` parts (base64 or URL) and
  Anthropic image blocks.

## Build

Visual Studio 2022+ (C++ desktop), CUDA Toolkit 13.x, Python 3.10+.

```bat
tools\build.bat            :: compute capability 7.5 / 8.6 / 8.9 / 12.0 (slow)
tools\build.bat --native   :: this PC only
```

Output: `build\bin` (setup finds it). The image encoder (`strata-vision.exe`, CPU): `tools\build_vision.bat` with
`LLAMA_DIR` set to a llama.cpp checkout (690b87d88 or later), into `build-vision\bin`. `python tools\package.py --version X` writes the release zip to `dist\`.

## Measuring

`bench\` (notes in Japanese, `bench\README.md`): `bench.py` (speed; `--model` another model, `--text code256` 256K),
`quality.py` (2K logits vs llama.cpp), `quality_2k.py` (settings against the saved 2K references, IQ4_XS and
Q8_K_XL), `quality_all.py` (2K and 128K, unattended), `ninfer_bench.py` (NInfer on the same tokens).

## License

MIT ([LICENSE](LICENSE)); components in [NOTICE](NOTICE). The model (Apache-2.0) is downloaded separately.
