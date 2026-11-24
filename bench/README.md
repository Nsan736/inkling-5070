# Inkling 2 Mini の計測

| やること | コマンド | かかる時間 |
|---|---|---|
| ビルド (変更したファイルだけ再コンパイル) | 公開用の配置: `tools\build.bat --native`、開発用の配置: `tools\build_inkling2.cmd` | 数秒〜数分 |
| 速度の計測 (普段はこれだけ) | `python -I bench/bench.py --label メモ` | 約 20 秒 (読み込み直し時は約 40 秒) |
| 128K の計測 (指示があったときだけ) | `python -I bench/bench.py --long` | 初回は 120K を読んで保存するため数分 |
| 品質の確認 (採用を記録する前に 1 回) | `python -I bench/quality.py --label メモ` | 約 1 分 |
| 常駐サーバーを止める | `python -I bench/bench.py --stop` | |
| 品質確認タスク (無人、2K + 128K) | `python -I bench/quality_all.py` (`--estimate` で目安だけ) | 初回は参照作りを含め約 2 時間 10 分 |
| 文脈の長さごとの生成 (ウォームアップ後) | `python -I bench/dspark_probe.py --ctx 8k 64k --modes base` | 8K で約 1 分、64K で約 3 分 |
| 最初から読む速さ | `python -I bench/full_reads.py --len 4k 32k 64k 128k` | 約 5 分 |
| 入力を写す編集の依頼 (lookup の窓) | `python -I bench/copy_window.py --windows 8 16` | 約 5 分 |
| 生成の 1 回の訪問の時刻 | `INKLING2_VISIT_TRACE=ファイル` で記録し `python -I bench/visit_trace.py ファイル` | |

速度の比較は、起動直後の 1 回目ではなく、同じエンジンで 1 回答えさせてから測る (1 回目はエキスパートの
キャッシュが落ち着く前で 1 割ほど遅い)。bench.py と上のスクリプトはそうしている。

- 品質確認タスク: 構成は `bench/data/quality_configs.json` (名前、kv、環境変数 env、エンジンの設定 extra)。暫定採用の構成を
  足して実行する。q8_0 は比較の基準として必ず入る。進み具合は `bench/quality_runs/main/run.log` (最後の行が「完了」か
  「失敗」)、結果は `bench/quality_runs/main/report.md`。止まったら同じコマンドで続きから (`--fresh` で最初から)。
  128K の参照は llama.cpp の llama-perplexity に `tools/llamacpp_ppl_tail.patch` (ローカル変更: `LLAMA_PPL_TAIL=N` で
  チャンクの最後の N 位置だけ保存) を当てたもので作る。`--smoke` は 4K の小さな参照で同じ手順を通す確認 (約 2 分)。

- エンジンの既定 (bench.py): KV q8_0、`--prompt-batch 2048`、`--ram-experts-mib 9216` (VRAM の外のエキスパートを
  全部 RAM に。エンジンの RAM は約 11.7 GB)。RAM を固定しているので、他のアプリの RAM 使用で数字が揺れない。
- 計測中のマシンの状態 (GPU / CPU のクロック、温度、電力、クロック低下の理由、他の処理の CPU 使用、エンジンの共有
  GPU メモリ) も CSV に残る (`bench/telemetry.py`)。CPU の温度は管理者権限がないと読めないので空欄。
- 時間の内訳 (エンジンの opt-in。`--env` で渡すと読み込み直す):
  `--env INKLING2_TIMELINE=1` (生成 1 トークン: CPU エキスパート / GPU のエキスパート / GPU の CPU 待ち / 重なり)、
  `--env INKLING2_PREFILL_TIMES=1` (読み込み: アテンション / その他のヘッド / エキスパート / 転送。段階ごとに同期するので
  全体は少し遅くなる)。結果は `bench/logs/engine.log` に出る。
- 品質の判定は `bench/data/quality_baseline.json` (2026-11-22 の新既定: KLD 0.00491、上位1一致 97.36%) から
  KLD 2 倍以内、上位1一致の低下 0.5 ポイント以内。`quality.py` が合否を表示する。
- 理論値との比較: `strata\build-inkling2\inkling2_kernels_test.exe decode` (アテンション単独と KV の読み出し帯域)、
  `strata\build-inkling2\inkling2_cpu_bandwidth.exe 4 8 12` (メモリと NVFP4 内積の帯域)。
- 既定は KV q8_0 (採用済み)。`--kv fp16` で元の形式。128K の比較は `--restart --long --reps 3` で揃える
  (常駐中は使用回数が溜まってだんだん速くなるため、読み込み直してから測る)。
- その他: `--extra="--prompt-batch 2048"` (エンジンの設定を足す)、`--env K=V` (環境変数。読み込み直す)、
  `--lookup` (会話内の繰り返しによる推測。CSV の decode 欄が lookup になり、素の速度 plain と別に残る)。
- 生成だけの短い比較: `--quick` (素・温度 1.0 を `--reps` 回。greedy / lookup / 読み込みの計測を省く)。`--long --quick`
  は保存した 120K の状態を 1 回復元して後ろの 8K を読み、続けて生成する (1 条件 約 45 秒)。`--long --no-full` は
  128K の全読みだけを省く。
- エンジンの CPU 使用 (計測中のコア数) は CSV の `engine_cpu_cores` / `engine_cpu_pct` に残る。
- CPU エキスパートの opt-in: `INKLING2_SPIN=N` (待機中のスピン回数、既定 200000)、`INKLING2_CPU_VNNI=1` (NVFP4 の
  AVX-VNNI 内積)、`INKLING2_CPU_FUSED=1` (1 ビジットの同期を 1 回に)、`INKLING2_CARD_SPLIT=1` (VRAM のエキスパートを
  GPU で選んでそのまま計算し、残りの番号だけをホストへ)。
- 生成 1 トークンの内訳 (CPU のエキスパートの時間、ルーター待ち、CPU が計算したエキスパート数) は
  `bench/logs/engine.log` の `strata-inkling2: expert cache ...` 行に出る。
- 結果は `bench/results.csv` (速度) と `bench/quality.csv` (品質) に、Strata フォークのコミット ID 付きで 1 行ずつ追記されます。
  未コミットの変更があると ID の末尾に `-dirty` が付きます。
- エンジンは `bench/server.py` が 127.0.0.1:18457 で常駐させ、モデルを読み直さずに使い回します。ビルドし直すと
  (exe の更新時刻が変わると) 次の計測で自動的に読み込み直します。
- 生成: greedy、EOS で止めない、最初の 16 トークンを除いた 200 トークン。時刻はエンジン内でトークンを出した時刻 (TT 行)。
- 読み込み: 8192 トークン。エンジンが計った読み込み時間 (DONE 行) から。毎回先頭のトークンを変えて再利用を防ぐ。
- エンジンの設定: `--max-context 132096 --threads 12 --window 1 --lookup 0` (会話内の繰り返しを使う推測は切ってある)。
  エキスパートを飛ばす近似 (`--tail-skip`, `--skip-weight`, `STRATA_ROUTE_TAIL_SKIP`) は既定で無効のまま。
- 入力のトークン列は `bench/make_inputs.py` で一度だけ作ったもの (`bench/data/*.ids`)。reasoning effort は medium。
- 参照ロジット `bench/data/ref_logits.bin` は llama.cpp 690b87d (#32694 と、NVFP4 スケールの修正 #32704 を含む) で
  Mainline_Support 版の GGUF から一度だけ作ったもの (`bench/data/ref_logits.log`)。
