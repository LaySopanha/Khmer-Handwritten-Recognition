import argparse
import json
from collections import Counter


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_vocab(train_jsonl: str, valid_jsonl: str | None, min_freq: int = 1):
    counter = Counter()
    for path in filter(None, [train_jsonl, valid_jsonl]):
        for obj in iter_jsonl(path):
            text = obj.get("text", "")
            counter.update(list(text))
    # filter by min_freq
    chars = [ch for ch, c in counter.items() if c >= min_freq]
    # sort for reproducibility (keep space near others by codepoint)
    chars = sorted(set(chars))
    # ensure space present if used in data later
    if " " not in chars:
        chars.append(" ")
    # special tokens first
    vocab = ["<pad>", "<bos>", "<eos>", "<unk>"] + chars
    return {
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "vocab": vocab,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--valid_jsonl", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_freq", type=int, default=1)
    args = ap.parse_args()

    spec = build_vocab(args.train_jsonl, args.valid_jsonl, args.min_freq)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

