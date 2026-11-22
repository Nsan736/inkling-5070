"""品質 (正しさ) の確認: 速くなった変更を「採用」として記録する前に 1 回だけ回す。

llama.cpp (690b87d = #32694 + #32704) で一度だけ作った参照ロジット bench/data/ref_logits.bin (2048 トークン、
後半 1023 位置) と、inkling2_logits_dump が同じトークンで出したロジットを照合する。
エンジンは先頭 1024 トークンをプロンプトとして読み、残りを 1 トークンずつ流してロジットを書く。
常駐サーバーとは別にモデルを読むので、VRAM を空けるため常駐サーバーは先に止める。

判定: bench/data/quality_baseline.json (--set-baseline で更新) から KLD 平均 2 倍以内、上位1一致の低下 0.5 ポイント以内
なら合格。基準は 2026-11-22 の新しい既定 (KV q8_0 / prompt-batch 2048 / ram 9216 / VRAM 空き 400MB) の値。

使い方: python bench/quality.py [--label 文字列]   結果は bench/quality.csv に 1 行追記
"""
import argparse
import csv
import json
import datetime
import pathlib
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bench  # noqa: E402

ROOT = bench.ROOT
DUMP = bench.BUILD / "inkling2_logits_dump.exe"
QUALITY = ROOT / "bench" / "quality.csv"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="")
    ap.add_argument("--kv", choices=["fp16", "q8_0"], default="q8_0")
    ap.add_argument("--set-baseline", action="store_true", help="この結果を判定の基準にする")
    a = ap.parse_args()
    bench.stop_server()
    with tempfile.TemporaryDirectory() as out:
        decode = (ROOT / "bench" / "data" / "ref_decode.txt").read_text().strip()
        subprocess.run([str(DUMP), str(bench.MODEL), "--tokens-file", str(ROOT / "bench/data/ref_prompt1024.txt"),
                        "--decode", decode, "--vocab", "200058", "--out", out, "--threads", str(bench.THREADS),
                        "--kv", a.kv, "--prompt-batch", "2048", "--ram-experts-mib", "9216",
                        "--vram-reserve-mib", "400"],   # bench.py と同じ
                       check=True, timeout=900, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        res = subprocess.run([sys.executable, "-I", str(ROOT / "bench/compare_logits.py"), "--ref",
                              str(ROOT / "bench/data/ref_logits.bin"), "--engine", str(pathlib.Path(out) / "decode_logits.f32"),
                              "--engine-vocab", "200058", "--engine-first", "1024"],
                             check=True, capture_output=True, text=True, encoding="utf-8").stdout
    print(res, end="")
    kld = float(re.search(r"KLD 平均 ([0-9.]+)", res).group(1))
    top1 = float(re.search(r"上位1一致 ([0-9.]+)", res).group(1))
    ppl = re.search(r"PPL 参照 ([0-9.]+)  エンジン ([0-9.]+)", res)
    new = not QUALITY.exists()
    with open(QUALITY, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "commit", "label", "kld_mean", "top1_pct", "ppl_ref", "ppl_engine", "kv"])
        w.writerow([datetime.datetime.now().isoformat(timespec="seconds"), bench.commit(), a.label, kld, top1,
                    ppl.group(1), ppl.group(2), a.kv])
    base_path = ROOT / "bench" / "data" / "quality_baseline.json"
    if a.set_baseline:
        base_path.write_text(json.dumps({"kld_mean": kld, "top1_pct": top1, "commit": bench.commit(),
                                         "label": a.label}, ensure_ascii=False), encoding="utf-8")
        print(f"判定の基準にした: KLD {kld}, 上位1一致 {top1}%")
    elif base_path.exists():
        b = json.loads(base_path.read_text(encoding="utf-8"))
        ok = kld <= 2.0 * b["kld_mean"] and top1 >= b["top1_pct"] - 0.5
        print(f"判定: {'合格' if ok else '不合格'} (基準 KLD {b['kld_mean']} の 2 倍まで、上位1一致 {b['top1_pct']}% から "
              f"0.5 ポイント低下まで; 今回 KLD {kld / b['kld_mean']:.2f} 倍、上位1一致 {top1 - b['top1_pct']:+.2f})")
    print(f"記録: {QUALITY}")


if __name__ == "__main__":
    main()
