"""GGUF のヘッダー (メタデータとテンソル一覧) だけを読む。テンソル本体は不要なので先頭の一部でも読める。

使い方: python tools/gguf_header.py ファイル [--kv] [--tensors] [--json 出力先]
"""
import argparse
import collections
import json
import struct
import sys

GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
    22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
    29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4",
}


class Reader:
    def __init__(self, data):
        self.d = data
        self.p = 0

    def take(self, fmt):
        v = struct.unpack_from("<" + fmt, self.d, self.p)
        self.p += struct.calcsize("<" + fmt)
        return v[0] if len(v) == 1 else v

    def string(self):
        n = self.take("Q")
        s = self.d[self.p:self.p + n].decode("utf-8", "replace")
        self.p += n
        return s

    def value(self, t):
        scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
        if t in scalar:
            return self.take(scalar[t])
        if t == 8:
            return self.string()
        if t == 9:
            et = self.take("I")
            n = self.take("Q")
            return [self.value(et) for _ in range(n)]
        raise ValueError(f"unknown kv type {t}")


def parse(path):
    with open(path, "rb") as f:
        data = f.read()
    r = Reader(data)
    if r.take("4s") != b"GGUF":
        raise SystemExit("not a GGUF file")
    version = r.take("I")
    n_tensors = r.take("Q")
    n_kv = r.take("Q")
    kv = {}
    for _ in range(n_kv):
        k = r.string()
        t = r.take("I")
        kv[k] = r.value(t)
    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        nd = r.take("I")
        dims = [r.take("Q") for _ in range(nd)]
        t = r.take("I")
        off = r.take("Q")
        tensors.append({"name": name, "dims": dims, "type": GGML_TYPES.get(t, f"type{t}"), "offset": off})
    return version, kv, tensors


def short(v):
    if isinstance(v, list):
        return f"[{len(v)} items] " + repr(v[:8])[:200]
    return repr(v)[:200]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--kv", action="store_true")
    ap.add_argument("--tensors", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    version, kv, tensors = parse(a.file)
    print(f"GGUF v{version}, {len(kv)} kv, {len(tensors)} tensors")
    if a.kv:
        for k, v in kv.items():
            if not k.startswith("tokenizer.ggml."):
                print(f"  {k} = {short(v)}")
    by_type = collections.Counter(t["type"] for t in tensors)
    print("type counts:", dict(by_type))
    if a.tensors:
        for t in tensors:
            print(f"  {t['type']:8s} {str(t['dims']):28s} {t['name']}")
    if a.json:
        kv2 = {k: v for k, v in kv.items() if not k.startswith("tokenizer.ggml.")}
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"version": version, "kv": kv2, "tensors": tensors}, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
