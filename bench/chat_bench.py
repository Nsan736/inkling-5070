"""OpenAI 互換のチャット API だけを持つサーバー (Strata 本家 v0.2.19 など) を、bench.py に近い入力で測る (計測のみ)。

トークン列 (bench/data/*.ids) を setup が取り出したトークナイザー (data/tokenizer) で文字に戻し、ユーザーの発言として送る。
チャットのテンプレートが付くので、プロンプトのトークン数は bench.py とわずかに違う (応答の usage に出る)。
  8K    生成: gen_prompt.txt に 216 トークンまで答える (温度 1.0 と greedy)。読み込み: prefill_8k.ids の文 (毎回先頭を変える)
  128K  long_code_base + long_code_tail の文を読み (1 回目)、同じ発言にもう一度答えて生成を測る
起動済みのサーバーに送るだけで、起動も停止もしない。

使い方: python -I bench/chat_bench.py --port 18471 --model inkling-2-mini-sq-10 [--long] [--reps 2] [--label メモ]
結果は bench/chat_results.csv に追記
"""
import argparse
import csv
import datetime
import json
import pathlib
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
PUB = ROOT if (ROOT / "server").exists() else ROOT / "public"   # 公開用の配置か、開発用の配置 (public/ の下)
sys.path.insert(0, str(PUB / "server" / "tools"))
import strata_tokenizer  # noqa: E402

DATA = ROOT / "bench" / "data"
OUT = ROOT / "bench" / "chat_results.csv"
T1 = {"temperature": 1.0, "top_p": 1.0, "seed": 1234}
GREEDY = {"temperature": 0.0}


def ids(name):
    return [int(x) for x in (DATA / name).read_text().replace(",", " ").split()]


def chat(port, model, text, max_tokens, sampling, timeout=7200):
    body = {"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
            "ignore_eos": True, **sampling}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    if "timings" not in d:
        raise SystemExit(f"timings がない: {json.dumps(d)[:300]}")
    return d


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--long", action="store_true")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--label", default="")
    ap.add_argument("--no-greedy", action="store_true")
    a = ap.parse_args()
    tok = strata_tokenizer.Tokenizer.from_dir(PUB / "data" / "tokenizer")
    res = {"t1": [], "greedy": [], "prefill": [], "notes": []}
    if not a.long:
        gen = (DATA / "gen_prompt.txt").read_text(encoding="utf-8")
        pre = tok.decode(ids("prefill_8k.ids"))
        chat(a.port, a.model, gen, 32, GREEDY)   # ウォームアップ
        for r in range(a.reps):
            d = chat(a.port, a.model, gen, 216, T1)
            res["t1"].append(d["timings"]["predicted_per_second"])
            res["notes"].append(f"t1 {d['timings']['predicted_n']} tok")
            if not a.no_greedy:
                d = chat(a.port, a.model, gen, 216, GREEDY)
                res["greedy"].append(d["timings"]["predicted_per_second"])
                res["notes"].append(f"greedy {d['timings']['predicted_n']} tok draft {d['timings'].get('draft_n_accepted')}"
                                    f"/{d['timings'].get('draft_n')}")
            d = chat(a.port, a.model, f"[{time.time_ns()}]\n" + pre, 1, GREEDY)
            res["prefill"].append(d["timings"]["prompt_per_second"])
            res["notes"].append(f"prefill {d['timings']['prompt_n']} tok")
    else:
        # 131,072 の文脈に答えの 216 トークンとテンプレートが入るように、先頭から 130,400 トークン
        text = tok.decode((ids("long_code_base.ids") + ids("long_code_tail.ids"))[:130400])
        t0 = time.time()
        d = chat(a.port, a.model, text, 1, GREEDY)
        res["prefill"].append(d["timings"]["prompt_per_second"])
        res["notes"].append(f"prefill {d['timings']['prompt_n']} tok in {time.time() - t0:.0f} s")
        for r in range(a.reps):
            d = chat(a.port, a.model, text, 216, T1)
            res["t1"].append(d["timings"]["predicted_per_second"])
            res["notes"].append(f"t1 cache {d['timings'].get('cache_n')} prompt {d['timings']['prompt_n']}")
            if not a.no_greedy:
                d = chat(a.port, a.model, text, 216, GREEDY)
                res["greedy"].append(d["timings"]["predicted_per_second"])
                res["notes"].append(f"greedy draft {d['timings'].get('draft_n_accepted')}/{d['timings'].get('draft_n')}")
    mean = lambda xs: round(sum(xs) / len(xs), 2) if xs else ""
    row = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "label": a.label,
           "mode": "long" if a.long else "short", "model": a.model, "gen_tps_t1": mean(res["t1"]),
           "gen_tps_t1_runs": " ".join(f"{x:.2f}" for x in res["t1"]), "gen_tps_greedy": mean(res["greedy"]),
           "gen_tps_greedy_runs": " ".join(f"{x:.2f}" for x in res["greedy"]), "prefill_tps": mean(res["prefill"]),
           "notes": "; ".join(res["notes"])}
    new = not OUT.exists()
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"[{row['mode']}] 温度 1.0 {row['gen_tps_t1']} tok/s ({row['gen_tps_t1_runs']})  greedy {row['gen_tps_greedy']} "
          f"tok/s ({row['gen_tps_greedy_runs']})  読み込み {row['prefill_tps']} tok/s\n  {row['notes']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
