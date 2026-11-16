#!/usr/bin/env python3
"""Trim the output projection of an MTP draft GGUF to the K most frequent tokens.

`output.weight` keeps K rows (those of the tokens `order[:K]`) and a tensor `d2t` (I64, K) is added with the token
id of each row. The engine (src/models/qwen4exp.cpp) scatters those logits over the full vocabulary and leaves the
rest at -inf. The order comes from `mtp_vocab_order.py` (JSON {"order": [...]}). Q4_0 and Q8_0 drafts are supported.

    gguf_mtp_d2t.py SOURCE.gguf DEST.gguf ORDER.json [K=65536]
"""
import json
import struct
import sys

GGML_TYPE_I64 = 27
ROW_BYTES_PER_BLOCK = {2: 18, 8: 34}  # Q4_0 and Q8_0: bytes per block of 32 weights
SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
T_STRING, T_ARRAY = 8, 9


def main():
    src, dst, order_json = sys.argv[1:4]
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 65536
    order = json.load(open(order_json))["order"]

    f = open(src, "rb")

    def rd(fmt):
        return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))

    def rstr():
        return f.read(rd("Q")[0])

    def skip(t):
        if t == T_STRING:
            rstr()
        elif t == T_ARRAY:
            item, n = rd("IQ")
            if item == T_STRING:
                for _ in range(n):
                    rstr()
            else:
                f.seek(n * SCALAR[item], 1)
        else:
            f.seek(SCALAR[t], 1)

    assert f.read(4) == b"GGUF"
    version, n_tensors, n_kv = rd("IQQ")
    alignment = 32
    kv_begin = f.tell()
    for _ in range(n_kv):
        key = rstr()
        t = rd("I")[0]
        if key == b"general.alignment":
            alignment = rd("I")[0]
        else:
            skip(t)
    kv_end = f.tell()

    tensors = []
    for _ in range(n_tensors):
        name = rstr()
        dims = rd("Q" * rd("I")[0])
        ty, off = rd("IQ")
        tensors.append([name, dims, ty, off])
    data_start = (f.tell() + alignment - 1) // alignment * alignment
    f.seek(0, 2)
    data_size = f.tell() - data_start

    # the size of each tensor (with its padding) is the distance to the next one
    by_off = sorted(tensors, key=lambda t: t[3])
    size = {t[0]: (by_off[i + 1][3] if i + 1 < len(by_off) else data_size) - t[3] for i, t in enumerate(by_off)}
    assert b"d2t" not in size, "the file already has d2t"

    def pad(n):
        return (alignment - n % alignment) % alignment

    out_infos, plan, pos = [], [], 0
    for name, dims, ty, off in tensors:
        if name == b"output.weight":
            assert ty in ROW_BYTES_PER_BLOCK and dims[0] % 32 == 0, (dims, ty)
            assert dims[1] == len(order) and sorted(order) == list(range(dims[1])) and 0 < k < dims[1]
            row = dims[0] // 32 * ROW_BYTES_PER_BLOCK[ty]
            f.seek(data_start + off)
            old = f.read(row * dims[1])
            blob = b"".join(old[t * row:(t + 1) * row] for t in order[:k])
            dims = (dims[0], k)
            plan.append(blob)
        else:
            blob = None
            plan.append((off, size[name]))
        n = len(blob) if blob is not None else size[name]
        out_infos.append((name, dims, ty, pos))
        pos += n + pad(n)
    assert any(isinstance(p, bytes) for p in plan), "no output.weight"
    d2t = struct.pack(f"<{k}q", *order[:k])
    out_infos.append((b"d2t", (k,), GGML_TYPE_I64, pos))
    plan.append(d2t)

    with open(dst, "wb") as g:
        g.write(b"GGUF" + struct.pack("<IQQ", version, n_tensors + 1, n_kv))
        f.seek(kv_begin)
        g.write(f.read(kv_end - kv_begin))
        for name, dims, ty, off in out_infos:
            g.write(struct.pack("<Q", len(name)) + name + struct.pack("<I", len(dims)))
            g.write(struct.pack("<" + "Q" * len(dims), *dims) + struct.pack("<IQ", ty, off))
        g.write(b"\0" * pad(g.tell()))
        for item in plan:
            if isinstance(item, bytes):
                g.write(item)
                g.write(b"\0" * pad(len(item)))
            else:
                off, n = item
                f.seek(data_start + off)
                while n:
                    chunk = f.read(min(n, 1 << 24))
                    assert chunk
                    g.write(chunk)
                    n -= len(chunk)
                g.write(b"\0" * pad(g.tell()))
    print(f"ok {dst}: output.weight {k} rows of {len(order)}, d2t added")


if __name__ == "__main__":
    main()
