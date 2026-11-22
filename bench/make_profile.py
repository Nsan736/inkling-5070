"""--expert-profile 用のエキスパート使用回数ファイルを作る。

計測に使う文章 (コーパスの先頭 131K トークン) とは別の部分 (140K トークン目から 32K) を読み、別の質問で 200 トークン
生成させて、その間の使用回数をエンジンに書かせる。エンジンを止めたときに書き出される。

使い方: python bench/make_profile.py 出力ファイル [--kv q8_0]
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bench  # noqa: E402
import make_inputs  # noqa: E402

QUESTION = ("Write a careful technical comparison of three ways to cache data between a slow disk and a fast "
            "processor, with their trade-offs, in several paragraphs.")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--kv", default="q8_0")
    a = ap.parse_args()
    out = pathlib.Path(a.out).resolve()
    if out.exists():
        out.unlink()
    import json
    template = json.loads(make_inputs.HEADER_JSON.read_text(encoding="utf-8"))["kv"]["tokenizer.chat_template"]
    body = make_inputs.tokenize(make_inputs.corpus())[140000:140000 + 32768]
    gen = make_inputs.tokenize(make_inputs.render(template, QUESTION))
    bench.ENGINE_FLAGS.extend(["--kv", a.kv, "--expert-profile", str(out)])
    bench.ensure_server(True)
    c = bench.Conn()
    c.send(f"GEN 1 temperature=0 {','.join(map(str, body))}")
    c.read_until(("DONE",), 600)
    c.send(bench.gen_line(216, gen))
    c.read_until(("DONE",), 600)
    c.close()
    bench.stop_server()
    print(f"書き出した: {out} ({out.stat().st_size} バイト)")


if __name__ == "__main__":
    main()
