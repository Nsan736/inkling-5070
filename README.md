# inkling-5070

[English](README.en.md)

> **非公式です。** [Strata](https://github.com/Niko1221/Strata) と [PolyStrata](https://github.com/simplexai-labs/PolyStrata)
> をもとにした独立のフォークで、どちらのプロジェクトの一部でもなく、作者によるサポートもありません。

> **確認の状況（0.1.0）**
> - 品質は llama.cpp（IQ4_XS、KV fp16）のロジットと照合しました。2K の文脈で KLD 平均 0.0049、上位 1 位の一致 97.4%、
>   128K の文脈で KLD 平均 0.0075、上位 1 位の一致 96.9% です。
> - Unsloth の Q8_K_XL（llama.cpp）を基準にすると、2K で KLD 平均 0.0026、上位 1 位の一致 98.4% です。
> - setup のモデルのダウンロードは、最初から最後までの通しではまだ試していません（SHA-256 の確認は動作を確かめ済み）。
> - 動作を確かめたのは 1 台の PC（RTX 5070 / Windows 11）だけです。

Thinking Machines の Inkling 2 Mini（全体 264 億、1 トークンで使うのは 34 億パラメーター、ループ構造の MoE）を、
NVIDIA の GPU 1 枚と PC のメモリで動かすエンジンです。

- **対象の GPU:** RTX 5070 で調整し、動作を確かめました。作りとしては RTX 20〜50 シリーズ（VRAM 8 GB 以上）で動きますが、
  ほかの GPU では確かめていません。
- **ほかの 12 GB 向けプロジェクトとの違い:** よくある方式は、低ビットに量子化してモデル全体を VRAM に載せます。
  このエンジンは 4 ビット（IQ4_XS）のまま、VRAM に入らないエキスパートを CPU で計算して、128K の文脈まで使います。
- **モデル:** Unsloth の `Inkling-2-Mini-UD-IQ4_XS.gguf`（15.7 GB）
  - よく使うエキスパートは、VRAM のキャッシュに置いて生成しながら入れ替えます。
- **文脈:** 12 GB の GPU で最大 128K（KV は q8_0）
- **厳密さ:** 選ばれたエキスパートは全部計算します（エキスパートを飛ばしません）。会話内の繰り返しによる推測は、検証してから採用します。
- **由来:**
  - エンジン: PolyStrata の `strata-poly`（Strata が土台）を Inkling 2 用に絞り、Windows で調整したもの
  - サーバー: PolyStrata のもの
  - 詳しくは [UPSTREAM.md](UPSTREAM.md) と [NOTICE](NOTICE)

計測した PC: RTX 5070 12 GB、Core Ultra 7 265K、DDR5-5600 32 GB、Windows 11、ドライバー 610.60。

## その PC での速さ

- リクエストは 1 つずつ、最初の 16 トークンを除いた 200 トークンで測りました。
- 128K は、120K トークンのコードの文脈に 8K を足したあとの値です。

| | このエンジン | NInfer 2026-11-22.2（Windows） |
|---|---|---|
| 8K 生成（温度 1.0） | 67.1 tok/s | 49.7 tok/s |
| 8K 生成（greedy） | 68.9 tok/s | 51.5 tok/s |
| 8K 生成（greedy・推測あり） | 69.7 tok/s | |
| 8K 読み込み | 2,247 tok/s | 2,171 tok/s |
| 128K 生成（温度 1.0） | 44.8 tok/s | 26.2 tok/s |
| 128K 生成（greedy） | 42.9 tok/s | 26.3 tok/s |
| 128K 読み込み（最初から） | 1,738 tok/s | 663 tok/s |
| 生成中の CPU | 7〜8 コア分 | 約 10 コア分 |

- **NInfer の条件:**
  - 設定は `--expert-residency host --host-cache-mib 1024 --spec none`、KV は MXFP8 で、同じトークン列を使いました。
  - top_k の上限が 20 です（このエンジンは語彙全体から選びます）。
  - NInfer の文書には、RTX 5070・Linux で 83.5 tok/s とあります。Windows では、このモデルの CUDA グラフを使いません。
  - 詳しくは [docs/ninfer_comparison.md](docs/ninfer_comparison.md) にあります。
- **品質:** llama.cpp（IQ4_XS、KV fp16）のロジットと照合しました（このエンジンの KV は q8_0）。
  - 2K（2,048 トークン）: KLD 平均 0.0049、上位 1 位の一致 97.4%
  - 128K（124,928 トークンの文脈の最後の 1,023 位置）: KLD 平均 0.0075、上位 1 位の一致 96.9%
  - Q8_K_XL を基準にした 2K: KLD 平均 0.0026、上位 1 位の一致 98.4%
- ほかの計測値は [docs/measurements.md](docs/measurements.md) にあります。

## 必要な環境

| 項目 | 必要なもの |
|---|---|
| OS | Windows 10 / 11（64 ビット） |
| GPU | NVIDIA、compute capability 7.5 以上（RTX 20〜50）、VRAM 8 GB 以上（128K なら 12 GB） |
| ドライバー | 580 以降（CUDA 13） |
| RAM | 16 GB 以上、32 GB 推奨 |
| ディスク | 空き 約 20 GB |
| Python | 3.10 以降 |

- **GPU:** ビルドの対象は 7.5、8.6、8.9、12.0 と、それより新しい世代向けの PTX です。
- **RAM:** エキスパートを最大 9 GB 置きます。RAM が少ないと、エキスパートを SSD から読むので、生成が遅くなります。
- **Python:** setup が venv を作り、サーバーの依存パッケージをそこに入れます。

## 使い方

1. 配布 zip を展開します（自分でビルドする場合は下の「ビルド」へ）。
2. `setup.bat` を実行します。
   - GPU、ドライバー、RAM、ディスクを確かめます。
   - モデルを取得し（途中から再開できます）、SHA-256 を確かめます。
   - `.venv` を作り、サーバーの Python パッケージを入れます。
   - 設定を `config\inkling.json` に書きます。
3. `run.bat` を実行します。
   - OpenAI 互換のサーバーが `http://127.0.0.1:8080` で起動します（この PC からだけ接続できます）。
   - Chat Completions、Responses、Anthropic Messages、チャット画面が使えます。

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "inkling-2-mini", "messages": [{"role": "user", "content": "こんにちは"}], "reasoning_effort": "medium"}'
```

| オプション | 内容 |
|---|---|
| `setup.bat --port N` | サーバーのポート |
| `setup.bat --limit-ram-gb 16 --limit-vram-gb 8` | 小さな予算向けの設定 |
| `setup.bat --context N` | 文脈の長さ |
| `setup.bat --configure-only` | ダウンロードせずに設定だけ決め直す |
| `run.bat --api-key KEY` | API キーを必須にする |

setup が選ぶ設定（使える VRAM による）:

| 使える VRAM | 文脈 | プロンプトの読み込み単位 |
|---|---|---|
| 12 GB 以上 | 128K | 2,048 トークン |
| 10 GB 以上 | 64K | 1,024 トークン |
| 8 GB 以上 | 32K | 1,024 トークン |

- RAM に置くエキスパートは「RAM − 14 GB」（最大 9 GB）、CPU のスレッドは最大 12 です。
- 計測したのは「VRAM 12 GB・RAM 32 GB」の場合だけです。

## 既知の制約

- **出力は実行ごとにわずかに変わります**（greedy でも同じ）。
  - エキスパートを GPU と CPU のどちらで計算するかは、キャッシュに合わせて変わります。
  - 両者は足し算の順序が違うので、結果が少しずれます。
- **生成中は CPU を 7〜8 コア分使います。** エキスパートのスレッドが層の合間も待機し続けるためです。リクエストがないときは休みます。
- **RAM の使用量:** 既定（エキスパート 9 GB）で、エンジンが約 12 GB を使います。これとは別に、OS のファイルキャッシュも使います。
- **確かめた環境:** Windows の 1 台だけです。
- **128K では参照とのずれが少し大きくなります。** llama.cpp との照合で、KLD 平均が 2K の 0.0049 から 0.0075 に上がります
  （上位 1 位の一致は 97.4% から 96.9%）。参照の KV は fp16、このエンジンは q8_0 です。
- **画像入力:** この配布物では使えません。

## ビルド

必要なもの:
- Visual Studio 2022 以降（C++ デスクトップ開発）
- CUDA Toolkit 13.x
- Python 3.10 以降

```bat
tools\build.bat            :: compute capability 7.5 / 8.6 / 8.9 / 12.0（時間がかかる）
tools\build.bat --native   :: この PC 専用
```

- できたものは `build\bin` に入ります。setup はそこを見つけて使います。
- `python tools\package.py --version X` で、配布 zip が `dist\` にできます。

## 計測

`bench\` にスクリプトがあります（説明は `bench\README.md`）。

| スクリプト | 内容 |
|---|---|
| `bench.py` | 速さ |
| `quality.py` | 2K のロジットを llama.cpp と照合 |
| `quality_all.py` | 2K と 128K の品質確認を無人で実行 |
| `ninfer_bench.py` | 同じトークン列で NInfer を測る |

## ライセンス

- MIT です（[LICENSE](LICENSE)）。
- 含まれる部品とそのライセンスは [NOTICE](NOTICE) にあります。
- モデル（Apache-2.0）は別にダウンロードします。
