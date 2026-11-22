"""計測に流すトークン列を一度だけ作る (bench/data/*.ids)。エンジンにはこの ID をそのまま渡すので、計測のたびに
トークナイザーを動かさずに済み、どのエンジンにも同じ入力が入る。

  gen.ids        生成の計測: チャットテンプレート (reasoning effort = medium) で包んだ短い質問
  prefill_8k.ids 読み込みの計測: 8192 トークンの文章 (チャットテンプレートで包む)
  long_base.ids  128K 計測 (指示があったときだけ使う): 先頭 120K トークン
  long_tail.ids  128K 計測: その続きの 8K トークン

文章は llama.cpp のドキュメント (MIT License, ref/llama.cpp/docs, tools/*/README.md) をつなげたもの。
トークン化は llama.cpp (690b87d) の llama-tokenize で、モデルの GGUF の語彙を使う。

使い方: python bench/make_inputs.py
"""
import ast
import json
import pathlib
import re
import subprocess
import sys

import jinja2

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "bench" / "data"
TOKENIZE = ROOT / "ref" / "llama.cpp" / "build" / "bin" / "llama-tokenize.exe"
VOCAB_GGUF = ROOT / "models" / "Mainline_Support" / "Inkling-2-Mini-UD-IQ4_XS.gguf"
HEADER_JSON = ROOT / "models" / "hdr" / "root.json"

GEN_QUESTION = ("Explain in detail how a mixture-of-experts language model decides which experts to run for each "
                "token, and what that means for running it on a computer whose graphics card cannot hold every "
                "expert. Use several paragraphs.")
PREFILL_QUESTION = "Summarize the following documentation in five bullet points.\n\n"
CODE_TASK = ("\n\nUsing the engine code above, write a new CUDA kernel and its host launcher that fuses the RMS norm "
             "and the Q8_1 quantization of several rows into one launch, in the style of this code base. Then explain "
             "how you would test it against the existing two-launch path.")
N_PREFILL = 8192
N_LONG_BASE = 120 * 1024
N_LONG_TAIL = 8 * 1024


def tokenize(text):
    tmp = DATA / "_tokenize_input.txt"
    tmp.write_text(text, encoding="utf-8", newline="")
    out = subprocess.run([str(TOKENIZE), "-m", str(VOCAB_GGUF), "-f", str(tmp), "--ids", "--log-disable"],
                         capture_output=True, text=True, check=True).stdout
    tmp.unlink()
    return ast.literal_eval(out.strip().splitlines()[-1])


def render(template, user_text):
    env = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"])

    def raise_exception(msg):
        raise RuntimeError(msg)

    env.globals["raise_exception"] = raise_exception
    # {% generation %} は transformers 独自のタグ (学習用の印) で、出力される文字は変わらないので外す
    template = re.sub(r"\{%-?\s*(end)?generation\s*-?%\}", "", template)
    t = env.from_string(template)
    return t.render(messages=[{"role": "user", "content": user_text}], add_generation_prompt=True,
                    reasoning_effort="medium", bos_token="", eos_token="")


def corpus():
    docs = ROOT / "ref" / "llama.cpp"
    files = sorted(docs.glob("docs/*.md")) + sorted(docs.glob("docs/*/*.md")) + sorted(docs.glob("tools/*/README.md"))
    return "\n\n".join(f.read_text(encoding="utf-8", errors="replace") for f in files)


def code_corpus():
    """このフォーク自身のエンジンのソース (engine/src の .cpp .cu .hpp .cuh .h)、ファイル名を前に付けて。"""
    src = ROOT / "strata" / "engine" / "src"
    files = sorted(f for f in src.rglob("*") if f.suffix in (".cpp", ".cu", ".hpp", ".cuh", ".h"))
    return "\n\n".join(f"// ===== {f.relative_to(src).as_posix()} =====\n" + f.read_text(encoding="utf-8", errors="replace")
                       for f in files)


def save(name, ids):
    (DATA / name).write_text(",".join(str(int(i)) for i in ids), encoding="ascii")
    print(f"{name}: {len(ids)} トークン")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    DATA.mkdir(parents=True, exist_ok=True)
    template = json.loads(HEADER_JSON.read_text(encoding="utf-8"))["kv"]["tokenizer.chat_template"]

    gen_text = render(template, GEN_QUESTION)
    if "Thinking effort level: 0.7" not in gen_text:
        raise SystemExit("reasoning effort medium (0.7) がテンプレートに入っていない")
    (DATA / "gen_prompt.txt").write_text(gen_text, encoding="utf-8", newline="")
    save("gen.ids", tokenize(gen_text))

    body = tokenize(corpus())
    print(f"文章: {len(body)} トークン")
    head = tokenize(render(template, PREFILL_QUESTION).split(PREFILL_QUESTION)[0] + PREFILL_QUESTION)
    save("prefill_8k.ids", (head + body)[:N_PREFILL])
    if len(body) < N_LONG_BASE + N_LONG_TAIL:
        raise SystemExit(f"128K 用の文章が足りない: {len(body)}")
    long_ids = head + body
    save("long_base.ids", long_ids[:N_LONG_BASE])
    save("long_tail.ids", long_ids[N_LONG_BASE:N_LONG_BASE + N_LONG_TAIL])

    # code: the engine's own sources, then a task to write code; the prompt ends as the template ends a turn
    marker = "@@CODE@@"
    full = render(template, marker + CODE_TASK)
    pre, post = full.split(marker)
    pre_ids, post_ids = tokenize(pre), tokenize(post)
    code = tokenize(code_corpus())
    total = N_LONG_BASE + N_LONG_TAIL
    need = total - len(pre_ids) - len(post_ids)
    if len(code) < need:
        raise SystemExit(f"code is too short: {len(code)} < {need}")
    ids = pre_ids + code[:need] + post_ids
    save("long_code_base.ids", ids[:N_LONG_BASE])
    save("long_code_tail.ids", ids[N_LONG_BASE:])


if __name__ == "__main__":
    main()
