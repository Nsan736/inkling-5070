"""入力を写す編集の依頼で、lookup の窓の大きさを比べる (開発用、計測のみ)。

依頼は 2 つ。題材はこのスクリプトで決まった形に作る一般的なもの (会話の記録などは使わない)。
  code   Python のモジュール (在庫の管理) の関数名を 1 つ変え、ファイル全体を返す
  fix    同じモジュールの関数の中身を 2 か所直し、ファイル全体を返す
  blocks 同じモジュールへの 3 つの変更を、置き換えのブロック (古い行の写しと新しい行) で返す
  jsonl  JSONL (本の一覧) の 1 行の値を変え、全体を返す
文脈の長さ: 8k は依頼だけ (約 8K トークンになるよう前置きの資料を足す)、64k は前置きの資料を約 64K トークンにする。
前置きの資料は bench/data の long_code (公開のソースコード) をトークナイザーで文字に戻したもの。
普通の文章: gen_prompt.txt に 216 トークン答える。

窓ごとにエンジンを起動し直し (--window W)、温度 0 で書かせる。記録するもの: 生成 tok/s、1 回の計算で確定した
トークン数 (書いたトークン数 / 計算の回数)、書いたトークン列 (窓 8 との比較用)。
結果は bench/logs/copy_window.jsonl に 1 行ずつ足す。

使い方: python -I bench/copy_window.py [--windows 8 16 32 64] [--ctx 8k 64k] [--tasks code jsonl text] [--env K=V ...]
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench as B           # noqa: E402
import upstream_bench as U  # noqa: E402

PUB = B.ROOT if (B.ROOT / "server").exists() else B.ROOT / "public"   # the public layout, or the development one
sys.path.insert(0, str(PUB / "server" / "tools"))
sys.path.insert(0, str(PUB / "server"))
import strata_tokenizer  # noqa: E402

LOG = "copy_window_engine.log"
OUT = B.ROOT / "bench" / "logs" / "copy_window.jsonl"
CTX = {"8k": 8192, "64k": 65536, "100k": 102400}


def code_file():
    """在庫の管理のモジュール (約 2,000 トークン)。"""
    rnd = random.Random(7)
    parts = ['"""A small inventory module: items, stock levels and reorder rules."""', "import json", "import time",
             "from dataclasses import dataclass, field", "", ""]
    parts.append("@dataclass\nclass Item:\n    sku: str\n    name: str\n    price: float\n    stock: int = 0\n"
                 "    reorder_at: int = 5\n    tags: list = field(default_factory=list)\n\n")
    kinds = ["bolt", "washer", "bracket", "hinge", "spring", "clamp", "pulley", "gasket", "rivet", "valve"]
    for k in kinds:
        n = rnd.randint(3, 9)
        parts.append(
            f"def restock_{k}(items, amount={n}):\n"
            f"    \"\"\"Add {n} (or `amount`) to every {k} whose stock is at or below its reorder level.\"\"\"\n"
            f"    changed = []\n"
            f"    for item in items:\n"
            f"        if '{k}' in item.tags and item.stock <= item.reorder_at:\n"
            f"            item.stock += amount\n"
            f"            changed.append(item.sku)\n"
            f"    return changed\n\n")
        parts.append(
            f"def value_of_{k}s(items):\n"
            f"    \"\"\"The stock value of the {k}s, rounded to cents.\"\"\"\n"
            f"    total = sum(item.price * item.stock for item in items if '{k}' in item.tags)\n"
            f"    return round(total, 2)\n\n")
    parts.append("def load_items(path):\n    with open(path, encoding='utf-8') as f:\n"
                 "        return [Item(**row) for row in json.load(f)]\n\n")
    parts.append("def save_items(items, path):\n    rows = [item.__dict__ for item in items]\n"
                 "    tmp = path + '.tmp'\n    with open(tmp, 'w', encoding='utf-8') as f:\n"
                 "        json.dump(rows, f, indent=2)\n    import os\n    os.replace(tmp, path)\n\n")
    parts.append("def report(items):\n    lines = []\n    for item in sorted(items, key=lambda i: i.sku):\n"
                 "        flag = ' LOW' if item.stock <= item.reorder_at else ''\n"
                 "        lines.append(f'{item.sku:10} {item.name:24} {item.stock:5}{flag}')\n"
                 "    stamp = time.strftime('%Y-%m-%d %H:%M')\n    return f'inventory at {stamp}\\n' + '\\n'.join(lines)\n")
    return "\n".join(parts)


def jsonl_file():
    """本の一覧の JSONL (約 2,500 トークン)。"""
    rnd = random.Random(11)
    words = ["river", "stone", "garden", "winter", "harbor", "lantern", "meadow", "summit", "orchard", "compass",
             "willow", "ember", "falcon", "glacier", "island", "marble"]
    rows = []
    for i in range(1, 61):
        title = " ".join(w.capitalize() for w in rnd.sample(words, 2))
        rows.append(json.dumps({"id": i, "title": f"The {title}", "shelf": f"{rnd.choice('ABCDEF')}{rnd.randint(1, 9)}",
                                "year": rnd.randint(1950, 2024), "copies": rnd.randint(1, 4),
                                "state": rnd.choice(["on the shelf", "lent", "being repaired"])}, ensure_ascii=False))
    return "\n".join(rows)


def task_text(task):
    if task == "fix":   # an edit inside a function, the whole file back (most of it copied)
        return ("In `value_of_hinges`, skip items whose price is negative, and in `report`, also print the price with "
                "two decimals after the stock. Change nothing else. Return the complete file in one ```python block, "
                "with no explanation.\n\n```python\n" + code_file() + "\n```")
    if task == "blocks":   # search / replace blocks: the old lines copied, then the new ones
        return ("Make these three changes to the file below: rename `restock_clamp` to `restock_clamps`; in "
                "`save_items`, write with `indent=4`; in `load_items`, skip rows whose `stock` is missing. For each "
                "change write one block in this form, with the old lines copied exactly and enough lines around them "
                "to be unique:\n<<<<<<< SEARCH\n(old lines)\n=======\n(new lines)\n>>>>>>> REPLACE\n"
                "Write only the blocks.\n\n```python\n" + code_file() + "\n```")
    if task == "code":
        return ("Rename the function `restock_spring` to `restock_springs` (only the definition; nothing calls it). "
                "Change nothing else. Return the complete file in one ```python block, with no explanation.\n\n"
                "```python\n" + code_file() + "\n```")
    return ("In the JSONL below, change the `state` of the record with id 37 to \"lost\". Change nothing else. "
            "Return all the records, one per line, in one ```jsonl block, with no explanation.\n\n"
            "```jsonl\n" + jsonl_file() + "\n```")


def prompt_ids(tok, tmpl, task, ctx):
    """前置きの資料 + 依頼。前置きは文脈の長さに合わせて切る。"""
    body = task_text(task)
    own = len(tok.encode(body))
    room = CTX[ctx] - own - 600
    filler_ids = (U.ids("long_code_base.ids") + U.ids("long_code_tail.ids"))[:max(0, room)]
    filler = tok.decode(filler_ids)
    content = ("Here is some reference source code from the project, for context only:\n\n" + filler +
               "\n\n----\n\n" + body) if room > 0 else body
    text = tmpl.render([{"role": "user", "content": content}], reasoning_effort="none")
    return tok.encode(text, parse_special=True)


def gen_tokens(e, n, prompt):
    e.p.stdin.write(f"GEN {n} temperature=0 {','.join(map(str, prompt))}\n".encode())
    e.p.stdin.flush()
    toks = []
    while True:
        _, line = e.get(3600)
        if line.startswith("T "):
            toks.append(int(line.split()[1]))
        elif line.startswith("DONE"):
            return toks, line.split()
        elif line.startswith("ERR"):
            raise RuntimeError(line)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", nargs="+", type=int, default=[8, 16, 32, 64])
    ap.add_argument("--ctx", nargs="+", default=["8k", "64k"])
    ap.add_argument("--tasks", nargs="+", default=["code", "jsonl", "text"])
    ap.add_argument("--max-new", type=int, default=3000)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--label", default="")
    ap.add_argument("--extra", default="", help='engine flags to add (e.g. "--lookup 2")')
    ap.add_argument("--warm", type=int, default=256, help="tokens of a warm-up answer to the same prompt (0: none)")
    a = ap.parse_args()
    for kv in a.env:
        k, _, v = kv.partition("=")
        os.environ[k] = v
    tok = strata_tokenizer.Tokenizer.from_dir(PUB / "data" / "tokenizer")
    from serve.frontend import ChatTemplate   # noqa: E402
    tmpl = ChatTemplate(PUB / "data" / "tokenizer" / "chat_template.jinja")
    B.stop_server()
    for ctx in a.ctx:
        prompts = {t: (prompt_ids(tok, tmpl, t, ctx) if t != "text" else U.ids("gen.ids")) for t in a.tasks}
        for w in a.windows:
            flags = list(B.BASE_FLAGS)
            flags[flags.index("--max-context") + 1] = str(CTX[ctx] + a.max_new + 1024)
            flags[flags.index("--window") + 1] = str(w)
            e = U.Engine([str(B.EXE), "--model", str(B.MODEL), "--serve"] + flags + ["--kv", "q8_0"] + a.extra.split(), log=LOG)
            try:
                e.gen(16, "temperature=0", U.ids("gen.ids"))   # warm-up (not counted)
                for t in a.tasks:
                    n = 216 if t == "text" else a.max_new
                    if a.warm:   # the same prompt read and answered first (not counted): the expert cache settles
                        gen_tokens(e, a.warm, prompts[t])
                    toks, d = gen_tokens(e, n, prompts[t])
                    produced, decode_ms, right = int(d[1]), float(d[4]), int(d[6])
                    passes = max(1, produced - right)
                    row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "label": a.label, "ctx": ctx, "task": t,
                           "extra": a.extra, "env": a.env, "ms": round(decode_ms, 1),
                           "window": w, "prompt": len(prompts[t]), "produced": produced,
                           "tps": round(1000.0 * produced / decode_ms, 2), "per_pass": round(produced / passes, 2),
                           "right": right, "offered": int(d[7]), "tokens": toks}
                    print(f"[{ctx} {t} w{w}] {produced} トークン {row['tps']} tok/s、1 回の計算で {row['per_pass']} "
                          f"トークン (当たり {right} / 下書き {d[7]})", flush=True)
                    with open(OUT, "a", encoding="utf-8") as f:
                        f.write(json.dumps(row) + "\n")
            finally:
                e.quit()


if __name__ == "__main__":
    main()
