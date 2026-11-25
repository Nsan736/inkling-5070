"""複数ターンの会話で、ターンごとに読み直しているトークン数を測る (開発用、計測のみ)。

起動済みの OpenAI 互換サーバー (scripts/run.py) に、合成した会話を順に送る。題材はこのスクリプトで作る一般的なもの
(会話の記録などは使わない)。システムの指示 + 前置きの資料 (bench/data の long_code、約 --ctx トークン) のあと:
  1  ファイルを見せて、要約を頼む
  2  前の答えを履歴に入れ、関数を 1 つ直すよう頼む (ツールの定義あり、ツールを呼べる)
  3  ツールの結果を返して続けさせる
  4  もう 1 つ頼む
  5  圧縮の後: システムの指示 + 前置き + 要約 1 つ + 新しい依頼 (履歴の途中が要約に置き換わる)
各ターンの後に、エンジンの記録 (config の log) の「prompt N tokens = R reused + X read」を読む。
結果は bench/logs/turns_probe.jsonl に足す。

使い方: python -I bench/turns_probe.py --port 18092 --log public/logs/engine.log [--ctx 60000] [--think none] [--tools-always]
"""
import argparse
import json
import pathlib
import re
import sys
import time
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import copy_window as C     # noqa: E402
import upstream_bench as U  # noqa: E402

REQ = re.compile(r"strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read in ([\d.]+) ms")
OUT = C.B.ROOT / "bench" / "logs" / "turns_probe.jsonl"
TOOLS = [{"type": "function", "function": {
    "name": "write_file", "description": "Write a whole file.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                   "required": ["path", "content"]}}}]


def post(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18092)
    ap.add_argument("--log", required=True)
    ap.add_argument("--ctx", type=int, default=60000)
    ap.add_argument("--think", default="none")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--label", default="")
    ap.add_argument("--tools-always", action="store_true", help="the tools in every turn (as an agent sends them)")
    ap.add_argument("--save", default="", help="after turn 4, save the conversation (/slots/0, slot_save_path) and stop")
    ap.add_argument("--restore", default="", help="restore this saved conversation, then only turn 5")
    a = ap.parse_args()
    tok = C.strata_tokenizer.Tokenizer.from_dir(C.PUB / "data" / "tokenizer")
    filler = tok.decode((U.ids("long_code_base.ids") + U.ids("long_code_tail.ids"))[:a.ctx])
    system = ("You are a careful coding assistant working in a repository. Answer briefly. Reference material "
              "follows.\n\n" + filler)
    code = C.code_file()
    log = pathlib.Path(a.log)
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": "Here is inventory.py:\n\n```python\n" + code + "\n```\n\nSummarize it in "
                                        "three sentences."}]
    asks = [None,
            "Fix `value_of_hinges` so that it skips items with a negative price. Write the whole file with write_file.",
            None,
            "Now add a function `low_items(items)` that returns the items at or below their reorder level. Show only "
            "the new function."]

    def turn(i, messages, tools=None):
        at = log.stat().st_size
        body = {"model": "inkling-2-mini", "messages": messages, "max_tokens": a.max_tokens, "temperature": 0,
                "reasoning_effort": a.think}
        if tools or a.tools_always:
            body["tools"] = TOOLS
        t0 = time.time()
        r = post(a.port, body)
        sec = time.time() - t0
        time.sleep(0.5)
        m = REQ.findall(log.read_text(encoding="utf-8", errors="replace")[at:])
        n, reused, read, ms = (int(m[-1][0]), int(m[-1][1]), int(m[-1][2]), float(m[-1][3])) if m else (0, 0, 0, 0)
        row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "label": a.label, "turn": i, "prompt": n, "reused": reused,
               "read": read, "read_ms": ms, "seconds": round(sec, 1), "think": a.think,
               "finish": r["choices"][0]["finish_reason"]}
        print(f"turn {i}: prompt {n}, reused {reused}, read {read} ({ms / 1000:.1f} s), whole {sec:.1f} s, "
              f"finish {row['finish']}", flush=True)
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        return r["choices"][0]["message"]

    def slot(action, name):
        req = urllib.request.Request(f"http://127.0.0.1:{a.port}/slots/0?action={action}",
                                     json.dumps({"filename": name}).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            print(f"{action}: {r.read().decode()[:300]}", flush=True)

    if a.restore:
        slot("restore", a.restore)
    else:
        reply = turn(1, msgs)
        msgs.append(reply)
        msgs.append({"role": "user", "content": asks[1]})
        reply = turn(2, msgs, TOOLS)
        msgs.append(reply)
        calls = reply.get("tool_calls") or []
        if calls:
            for c in calls:
                msgs.append({"role": "tool", "tool_call_id": c.get("id", "call_0"), "content": "written"})
        else:
            msgs.append({"role": "user", "content": "Thanks."})
        reply = turn(3, msgs, TOOLS)
        msgs.append(reply)
        msgs.append({"role": "user", "content": asks[3]})
        turn(4, msgs, TOOLS)
        if a.save:
            slot("save", a.save)
            return
    # after a compaction: the middle of the conversation becomes one summary
    turn(5, [msgs[0], {"role": "user", "content": "Summary of the conversation so far: we read inventory.py, fixed "
                                                    "value_of_hinges to skip negative prices and added low_items."},
             {"role": "user", "content": "Now make `report` print the items sorted by name instead of by sku. "
                                         "Show only the new function."}], TOOLS)


if __name__ == "__main__":
    main()
