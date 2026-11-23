"""長い出力 (ファイル作成) の依頼を温度と思考レベルを変えて送り、上限到達とループの起きやすさを比べる (計測のみ)。

OpenAI 互換の /v1/chat/completions に stream で送り、思考 (reasoning_content) と本文を分けて受け取る。
起動済みのサーバーに送るだけで、起動も停止もしない。

使い方: python -I bench/loop_probe.py --port 8079 --temps 0,0.6,1.0 --efforts low,medium,high --reps 1,2,3
  --reps は温度ごとの回数 (温度 0 は毎回同じになるので 1 回)。
結果: bench/logs/loop_probe.jsonl に 1 回 1 行、本文と思考は bench/logs/loop_probe/ に保存
"""
import argparse
import collections
import json
import pathlib
import re
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "bench" / "logs"

PROMPTS = {
    "app_py": (
        "Write server.py for a small book-lending tracker for a reading club, run on one PC and opened from other "
        "devices on the LAN. Use only the Python standard library (http.server, json, threading). Features: books "
        "with a title, an author, a shelf and a state (on the shelf, lent, lost, being repaired) with a free-text note; "
        "a list of loans with the book, the borrower, the date lent and the date due; a cover image upload for each "
        "book (stored under data/images, png/jpg/webp only, at most 2 MB); all data saved to data/state.json with an "
        "atomic write; a JSON API (GET /api/state, POST /api/books, PATCH /api/books/<id>, DELETE /api/books/<id>, "
        "POST /api/loans, DELETE /api/loans/<id>, POST /api/books/<id>/cover); static files served from static/; "
        "input validation with clear error messages; it must listen on the LAN address only, print the URL to open, "
        "and never be exposed to the internet. Write the complete file in one ```python block, with no explanation "
        "before or after it."),
}


def coverage(text, width=12):
    """12 語の並びが 3 回以上出る部分が全体に占める割合 (サーバーの reasoning_repeat_coverage と同じ数え方、全体で)。"""
    words = re.findall(r"\w+|[^\w\s]", text.lower())
    if len(words) < width * 3:
        return 0.0
    grams = [tuple(words[i:i + width]) for i in range(len(words) - width + 1)]
    counts = collections.Counter(grams)
    covered = set()
    for i, g in enumerate(grams):
        if counts[g] >= 3:
            covered.update(range(i, i + width))
    return len(covered) / len(words)


def longest_line_loop(text):
    """連続して同じ行のまとまりが繰り返される最長の部分: (周期の行数, 繰り返し回数, その行数)。空行は数えない。"""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    best = (0, 0, 0)
    n = len(lines)
    for p in range(1, 41):
        i = 0
        while i + 2 * p <= n:
            r = 1
            while i + (r + 1) * p <= n and lines[i + r * p:i + (r + 1) * p] == lines[i:i + p]:
                r += 1
            if r >= 2 and p * r > best[2]:
                best = (p, r, p * r)
            i += max(1, (r - 1) * p) if r >= 2 else 1
    return best


WRITE_TOOL = [{"type": "function", "function": {
    "name": "write", "description": "Write a file (the whole content).",
    "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}},
                   "required": ["file_path", "content"]}}}]


def run_one(port, model, prompt, temp, effort, max_tokens, timeout, llamacpp=False, tools=False):
    if tools:   # ファイルはツールで書かせる (ファイルを書くエージェントと同じ形)
        prompt = prompt.replace("Write the complete file in one ```python block, with no explanation before or after it.",
                                "Create the complete file with the write tool.")
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": True,
            "max_tokens": max_tokens, "temperature": temp, "reasoning_effort": effort,
            "stream_options": {"include_usage": True}}
    if tools:
        body["tools"] = WRITE_TOOL
    if llamacpp:   # llama-server はテンプレートの変数として渡す。サンプラーは温度だけにそろえる (このサーバーの既定と同じ)
        body.update({"chat_template_kwargs": {"reasoning_effort": effort}, "top_k": 0, "top_p": 1.0, "min_p": 0.0})
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    reasoning, content, finish, usage, args = [], [], None, {}, []
    n_reason = n_content = 0
    t0 = time.time()
    t_first_content = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("reasoning_content"):
                    reasoning.append(d["reasoning_content"])
                    n_reason += 1
                if d.get("content"):
                    content.append(d["content"])
                    n_content += 1
                    if t_first_content is None:
                        t_first_content = time.time()
                for tc in d.get("tool_calls") or []:
                    args.append((tc.get("function") or {}).get("arguments") or "")
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    t1 = time.time()
    return {"reasoning": "".join(reasoning), "content": "".join(content) + "".join(args), "args_chars": len("".join(args)), "finish": finish, "usage": usage,
            "reason_chunks": n_reason, "content_chunks": n_content, "secs": round(t1 - t0, 1),
            "think_secs": round((t_first_content or t1) - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8079)
    ap.add_argument("--model", default="inkling-2-mini")
    ap.add_argument("--prompt", default="app_py", choices=sorted(PROMPTS))
    ap.add_argument("--temps", default="0,0.6,1.0")
    ap.add_argument("--efforts", default="low,medium,high")
    ap.add_argument("--reps", default="1,2,3", help="温度ごとの回数 (--temps と同じ並び)")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--label", default="")
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--llamacpp", action="store_true", help="送り先が llama.cpp の llama-server")
    ap.add_argument("--tools", action="store_true", help="write ツールを渡し、ファイルをツールで書かせる")
    a = ap.parse_args()
    temps = [float(x) for x in a.temps.split(",")]
    reps = [int(x) for x in a.reps.split(",")]
    efforts = a.efforts.split(",")
    keep = OUT / "loop_probe"
    keep.mkdir(parents=True, exist_ok=True)
    plan = [(t, e, k) for t, r in zip(temps, reps) for e in efforts for k in range(r)]
    for t, e, k in plan:
        res = run_one(a.port, a.model, PROMPTS[a.prompt], t, e, a.max_tokens, a.timeout, a.llamacpp, a.tools)
        tag = f"{time.strftime('%H%M%S')}_t{t}_{e}_{k}"
        (keep / f"{tag}_reasoning.txt").write_text(res["reasoning"], encoding="utf-8")
        (keep / f"{tag}_content.txt").write_text(res["content"], encoding="utf-8")
        row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "label": a.label, "prompt": a.prompt, "server": "llama.cpp" if a.llamacpp else "strata", "temp": t,
               "effort": e, "rep": k, "max_tokens": a.max_tokens, "finish": res["finish"], "tools": a.tools, "args_chars": res["args_chars"],
               "completion_tokens": res["usage"].get("completion_tokens"),
               "reasoning_tokens": (res["usage"].get("completion_tokens_details") or {}).get("reasoning_tokens"),
               "reason_chunks": res["reason_chunks"], "content_chunks": res["content_chunks"],
               "reason_chars": len(res["reasoning"]), "content_chars": len(res["content"]),
               "closed_block": res["content"].count("```") >= 2,
               "cov_reasoning": round(coverage(res["reasoning"]), 3), "cov_content": round(coverage(res["content"]), 3),
               "line_loop_reasoning": longest_line_loop(res["reasoning"]),
               "line_loop_content": longest_line_loop(res["content"]),
               "secs": res["secs"], "think_secs": res["think_secs"], "files": tag}
        with open(OUT / "loop_probe.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
