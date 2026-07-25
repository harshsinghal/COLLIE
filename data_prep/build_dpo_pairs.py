#!/usr/bin/env python3
"""Turn judge rankings into DPO preference pairs.

Keeps a pair only when the judge's score gap is >= MARGIN — near-ties are noise
and hurt DPO more than they help. Both sides must be well-formed cards, so the
model is never taught to prefer malformed output. Writes data/dpo_pairs.jsonl
with prompt / chosen / rejected.
"""
import argparse, json, os, re, sys
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "data")
sys.path.insert(0, os.path.join(ROOT, "train"))
from collie_reward import parse_card  # noqa: E402

H = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
MARGIN = 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="dpo")
    ap.add_argument("--cands", required=True)
    a = ap.parse_args()
    st = json.load(open(f"{HERE}/state_ov.json"))
    b = requests.get(f"https://api.openai.com/v1/batches/{st[f'rank_{a.run}']['batch_id']}",
                     headers=H, timeout=30).json()
    if b.get("status") != "completed":
        c = b.get("request_counts", {})
        print(f"rank batch: {b.get('status')} {c.get('completed',0)}/{c.get('total',0)}")
        return
    raw = requests.get(f"https://api.openai.com/v1/files/{b['output_file_id']}/content",
                       headers=H, timeout=600).text
    cands = {json.loads(l)["i"]: json.loads(l)["candidates"]
             for l in open(a.cands, encoding="utf-8")}
    man = {json.loads(l)["i"]: json.loads(l)
           for l in open(f"{HERE}/dpo_manifest.jsonl", encoding="utf-8")}
    system = open(f"{HERE}/student_prompt.txt", encoding="utf-8").read()

    pairs, skipped = [], {"parse": 0, "margin": 0, "malformed": 0}
    gaps = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        i = int(r["custom_id"].split("-")[1])
        try:
            content = r["response"]["body"]["choices"][0]["message"]["content"]
            d = json.loads(re.search(r'\{.*\}', content, re.S).group(0))
            scores = [float(s) for s in d["scores"]]
        except Exception:
            skipped["parse"] += 1; continue
        cs = cands.get(i, [])
        if len(scores) != len(cs):
            skipped["parse"] += 1; continue
        best = max(range(len(cs)), key=lambda k: scores[k])
        worst = min(range(len(cs)), key=lambda k: scores[k])
        gap = scores[best] - scores[worst]
        if gap < MARGIN:
            skipped["margin"] += 1; continue
        if parse_card(cs[best]) is None or parse_card(cs[worst]) is None:
            skipped["malformed"] += 1; continue
        gaps.append(gap)
        pairs.append({"prompt": [{"role": "system", "content": system},
                                 {"role": "user", "content": "Document:\n" + man[i]["text"]}],
                      "chosen": [{"role": "assistant", "content": cs[best]}],
                      "rejected": [{"role": "assistant", "content": cs[worst]}],
                      "i": i, "gap": gap})
    with open(f"{HERE}/dpo_pairs.jsonl", "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"pairs kept: {len(pairs)}  skipped: {skipped}")
    if gaps:
        gaps.sort()
        print(f"score gap: min {gaps[0]:.1f}  median {gaps[len(gaps)//2]:.1f}  max {gaps[-1]:.1f}")


if __name__ == "__main__":
    main()
