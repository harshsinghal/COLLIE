#!/usr/bin/env python3
"""Scale draw: +3,000 docs from the TRAIN registers only (enron, finepdfs,
chat, ghcode). apache + loghub are permanently held out as OOD registers and
must never enter training. Dedupes against manifest_2k. Writes
data/manifest_more.jsonl with ids continuing after the first manifest.
"""
import gzip, hashlib, json, os

DATA = os.path.expanduser("~/workspace/ai_soc/dlp_bench/sources/data")
HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

QUOTAS = {
    "enron.jsonl.gz": (1100, "enron"),
    "finepdfs_english_diverse_10k.jsonl.gz": (1100, "finepdfs"),
    "chat_like_public_raw/chat_like_public_9k.jsonl.gz": (500, "chat"),
    "github_code_clean_code.jsonl.gz": (300, "ghcode"),
}

def hkey(s):
    return int(hashlib.sha256(s.encode()).hexdigest(), 16)

def sha(r):
    return r.get("text_sha256") or r.get("id") or \
        hashlib.sha256(json.dumps(r, sort_keys=True)[:2000].encode()).hexdigest()

def get_text(r, src):
    if src == "chat":
        return r.get("body_text", "")
    return r.get("text", "")

def clean_ok(t):
    return t and 200 <= len(t) <= 20000

def main():
    prev = [json.loads(l) for l in gzip.open(f"{HERE}/manifest_2k.jsonl.gz", "rt", encoding="utf-8")]
    seen = {p["id"] for p in prev}
    next_i = max(p["i"] for p in prev) + 1
    picked = []
    for fn, (quota, src) in QUOTAS.items():
        rows = []
        for line in gzip.open(os.path.join(DATA, fn), "rt", encoding="utf-8"):
            r = json.loads(line)
            t = get_text(r, src)
            if not clean_ok(t) or r.get("id") in seen or sha(r) in seen:
                continue
            rows.append((r, t))
        rows.sort(key=lambda rt: hkey("more-" + sha(rt[0])))
        for r, t in rows[:quota]:
            seen.add(sha(r))
            picked.append((src, r, t))
    with open(f"{HERE}/manifest_more.jsonl", "w", encoding="utf-8") as f:
        for k, (src, r, t) in enumerate(picked):
            f.write(json.dumps({"i": next_i + k, "src": src,
                                "id": r.get("id", f"{src}-more-{k}"),
                                "tier": "base", "text": t[:4000]}) + "\n")
    from collections import Counter
    print(f"new docs: {len(picked)} starting at i={next_i}")
    print("by source:", dict(Counter(s for s, _, _ in picked)))

if __name__ == "__main__":
    main()
