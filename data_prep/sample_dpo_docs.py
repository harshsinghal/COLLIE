#!/usr/bin/env python3
"""Sample documents for DPO preference collection.

Must be disjoint from BOTH the SFT training corpus (manifest_2k +
manifest_more) and the faithfulness eval corpus (faith_manifest) — otherwise
we would tune on the test set. Train registers only (apache/loghub stay held
out). Writes data/dpo_manifest.jsonl.
"""
import gzip, hashlib, json, os, sys

DATA = os.path.expanduser("~/workspace/ai_soc/dlp_bench/sources/data")
HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sample_more import get_text, clean_ok, sha  # noqa: E402

QUOTAS = {
    "enron.jsonl.gz": (450, "enron"),
    "finepdfs_english_diverse_10k.jsonl.gz": (450, "finepdfs"),
    "chat_like_public_raw/chat_like_public_9k.jsonl.gz": (200, "chat"),
    "github_code_clean_code.jsonl.gz": (100, "ghcode"),
}


def hkey(s):
    return int(hashlib.sha256(s.encode()).hexdigest(), 16)


def main():
    used = set()
    for l in gzip.open(f"{HERE}/manifest_2k.jsonl.gz", "rt", encoding="utf-8"):
        used.add(json.loads(l)["id"])
    for fn in ("manifest_more.jsonl", "faith_manifest.jsonl"):
        for l in open(f"{HERE}/{fn}", encoding="utf-8"):
            used.add(json.loads(l).get("id", ""))

    picked = []
    for fn, (quota, src) in QUOTAS.items():
        rows = []
        for line in gzip.open(os.path.join(DATA, fn), "rt", encoding="utf-8"):
            r = json.loads(line)
            t = get_text(r, src)
            if not clean_ok(t) or r.get("id") in used:
                continue
            rows.append((r, t))
        rows.sort(key=lambda rt: hkey("dpo-" + sha(rt[0])))
        for r, t in rows[:quota]:
            picked.append((src, r.get("id", ""), t))

    with open(f"{HERE}/dpo_manifest.jsonl", "w", encoding="utf-8") as f:
        for k, (src, rid, t) in enumerate(picked):
            f.write(json.dumps({"i": k, "src": src, "id": rid, "text": t[:4000]},
                               ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"dpo docs: {len(picked)}  by src: {dict(Counter(s for s, _, _ in picked))}")


if __name__ == "__main__":
    main()
