"""2K の品質だけを、作成済みの 2 つの参照 (llama.cpp の IQ4_XS: bench/data/ref_logits.bin、unsloth Q8_K_XL:
bench/data/ref_q8_logits.bin) と照合する。参照のモデル (GGUF) は要らない。

  python -I bench/quality_2k.py 構成の JSON [--run 名前]

構成の JSON は quality_configs.json と同じ形 (name, model, kv, env, extra)。ref_same (同じモデルの llama.cpp の
参照ロジット) があればそれとも照合する。結果は bench/quality_runs/<名前>/ の
state.json と report.md。終わった構成は飛ばす (同じ名前で再開できる)。
"""
import argparse
import datetime
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import quality_all as q  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("configs")
    ap.add_argument("--run", default="q2k")
    a = ap.parse_args()
    cs = json.loads(pathlib.Path(a.configs).read_text(encoding="utf-8"))
    d = q.ROOT / "bench" / "quality_runs" / a.run
    d.mkdir(parents=True, exist_ok=True)
    sp = d / "state.json"
    state = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}

    class R:   # what q.dump() needs of a run
        dir = d
    decode = (q.DATA / "ref_decode.txt").read_text().strip()
    for c in cs:
        if c["name"] in state:
            print(f"{c['name']}: 済み")
            continue
        out = d / f"work_2k_{q.safe(c['name'])}"
        out.mkdir(exist_ok=True)
        t0 = datetime.datetime.now()
        try:
            q.dump(R, c, q.DATA / "ref_prompt1024.txt", decode, out, 4096, 1800)
        except Exception as e:   # a model the engine cannot read: its log says why
            state[c["name"]] = {"error": str(e)[:300]}
            print(f"{c['name']}: 失敗 ({e}); {out / 'dump.log'}")
            sp.write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding="utf-8")
            continue
        r = q.compare(q.DATA / "ref_logits.bin", out / "decode_logits.f32", 1024, 0)
        r["q8"] = q.compare(q.REF_Q8, out / "decode_logits.f32", 1024, 0)
        if c.get("ref_same"):   # the same model's own llama.cpp logits (the engine's error alone)
            r["same"] = q.compare(c["ref_same"], out / "decode_logits.f32", 1024, 0)
        r["seconds"] = (datetime.datetime.now() - t0).total_seconds()
        (out / "decode_logits.f32").unlink()
        state[c["name"]] = r
        sp.write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{c['name']}: Q8 参照 KLD {r['q8']['kld']:.5f} 上位1 {r['q8']['top1']:.2f}%  "
              f"IQ4_XS 参照 KLD {r['kld']:.5f} 上位1 {r['top1']:.2f}%  ({r['seconds']:.0f} 秒)", flush=True)
    lines = [f"# 2K の品質 ({a.run}、{datetime.datetime.now().isoformat(timespec='seconds')}、エンジン {q.bench.commit()})", "",
             "2,048 トークン (前半 1,024 を読み、後半 1,024 を 1 つずつ) のロジットを llama.cpp の参照 (KV fp16) と照合。", "",
             "| 構成 | Q8_K_XL 参照 KLD | KLD 99% | 上位1一致 | PPL (参照 / エンジン) | IQ4_XS 参照 KLD | 上位1一致 |",
             "|---|---|---|---|---|---|---|"]
    for c in cs:
        r = state.get(c["name"])
        if not r or "error" in r:
            lines.append(f"| {c['name']} | 失敗: {(r or {}).get('error', '未実行')[:80]} | | | | | |")
            continue
        x = r["q8"]
        lines.append(f"| {c['name']} | {x['kld']:.5f} | {x['kld99']:.5f} | {x['top1']:.2f}% | "
                     f"{x['ppl_ref']:.4f} / {x['ppl_engine']:.4f} | {r['kld']:.5f} | {r['top1']:.2f}% |")
        if "same" in r:
            y = r["same"]
            lines.append(f"| ({c['name']} を llama.cpp で動かした参照との差) | {y['kld']:.5f} | {y['kld99']:.5f} | "
                         f"{y['top1']:.2f}% | {y['ppl_ref']:.4f} / {y['ppl_engine']:.4f} | | |")
    lines += ["", "構成の中身:", ""] + [f"- {c['name']}: モデル {q.model_of(c).name}、kv {c['kv']}、環境変数 "
                                     f"{c.get('env') or 'なし'}、設定 {' '.join(c.get('extra', [])) or 'なし'}。"
                                     f"{c.get('note', '')}" for c in cs]
    (d / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(d / "report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
