#!/usr/bin/env python3
"""Rank a model's vocabulary by how often each token is used, to trim the output of its MTP draft.

Tokenizes text corpora with a running llama-server that has the target model loaded, adds up the relative
frequency of every token with a weight per corpus, and writes {"order": [ids, most to least frequent], "stats": ...}.
Use text that looks like what you generate: language, code, chat. Control tokens go first so they are always kept.
`gguf_mtp_d2t.py` takes the result.

    mtp_vocab_order.py OUT.json --n-vocab 248320 [--first-special 248000] [--server URL] [--model NAME]
                       [--held-out FILE] WEIGHT:PATH [WEIGHT:PATH ...]

PATH is a text file or a directory (every file under it is read). --held-out is text that is not used for the
ranking: the script prints which share of its tokens the first K of the order cover.
"""
import argparse
import collections
import json
import os
import urllib.request


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("corpora", nargs="+", metavar="WEIGHT:PATH")
    ap.add_argument("--n-vocab", type=int, required=True)
    ap.add_argument("--first-special", type=int, help="ids from this one on are control tokens")
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--model", help="model name, when the server is a router")
    ap.add_argument("--held-out")
    args = ap.parse_args()

    def tokenize(text):
        out = []
        for i in range(0, len(text), 300000):
            body = {"content": text[i:i + 300000]}
            if args.model:
                body["model"] = args.model
            req = urllib.request.Request(args.server + "/tokenize", headers={"Content-Type": "application/json"},
                                         data=json.dumps(body).encode())
            out += json.loads(urllib.request.urlopen(req, timeout=600).read())["tokens"]
        return out

    def read(path):
        if os.path.isdir(path):
            paths = sorted(os.path.join(d, f) for d, _, fs in os.walk(path) for f in fs)
        else:
            paths = [path]
        return "\n".join(open(p, errors="ignore").read() for p in paths)

    score, stats = collections.Counter(), {}
    for item in args.corpora:
        weight, path = item.split(":", 1)
        tokens = tokenize(read(path))
        if not tokens:
            ap.error(f"{path}: no text")
        count = collections.Counter(tokens)
        stats[path] = {"tokens": len(tokens), "distinct": len(count), "weight": float(weight)}
        for token, n in count.items():
            score[token] += float(weight) * n / len(tokens)
        print(path, stats[path], flush=True)

    order = [t for t, _ in score.most_common()]
    special = [i for i in range(args.first_special, args.n_vocab) if i not in score] if args.first_special else []
    seen = set(order) | set(special)
    order = special + order + [i for i in range(args.n_vocab) if i not in seen]
    assert sorted(order) == list(range(args.n_vocab)), "--n-vocab does not match the model"
    json.dump({"order": order, "stats": stats, "n_scored": len(score)}, open(args.out, "w"))
    print("ids with a score:", len(score), "of", args.n_vocab)

    if args.held_out:
        tokens = tokenize(read(args.held_out))
        for k in (16384, 32768, 49152, 65536, 98304):
            if k < args.n_vocab:
                top = set(order[:k])
                print(f"K={k}: covers {sum(t in top for t in tokens) / len(tokens) * 100:.2f} % of the held-out tokens"
                      f"   (first {k} ids: {sum(t < k for t in tokens) / len(tokens) * 100:.2f} %)")


if __name__ == "__main__":
    main()
