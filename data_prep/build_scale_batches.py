#!/usr/bin/env python3
"""Submit the two round-5 batches (both gpt-5.4-mini):

A) scale: label the 3,000 new train-register docs, same 5 anchor regimes as
   round 4 (anchor saved per doc for SFT conditioning).
B) anchor-OOD gold: relabel the 340 existing eval docs (140 in-dist + 200
   register-OOD) under NEVER-TRAINED anchor vocabularies from
   anchor_pool_holdout.json — the gold for the unseen-catalog test. Anchor
   assignment is deterministic per doc.
"""
import gzip, hashlib, json, os, requests

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
H = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_openvocab_batches import anchor_for, sys_prompt  # same regimes + prompt

HOLD = json.load(open(f"{HERE}/anchor_pool_holdout.json"))["vocabularies"]

def hkey(s):
    return int(hashlib.sha256(s.encode()).hexdigest(), 16)

def submit(path):
    fid = requests.post("https://api.openai.com/v1/files", headers=H,
                        files={"file": (os.path.basename(path), open(path, "rb"))},
                        data={"purpose": "batch"}).json()["id"]
    b = requests.post("https://api.openai.com/v1/batches", headers=H,
                      json={"input_file_id": fid, "endpoint": "/v1/chat/completions",
                            "completion_window": "24h"}).json()
    return b.get("id"), b.get("status")

def main():
    # A) scale batch
    docs = [json.loads(l) for l in open(f"{HERE}/manifest_more.jsonl", encoding="utf-8")]
    with open(f"{HERE}/anchors_more.jsonl", "w", encoding="utf-8") as fa, \
         open(f"{HERE}/scale_batch.jsonl", "w", encoding="utf-8") as fb:
        for d in docs:
            regime, anchor = anchor_for(d["i"])
            fa.write(json.dumps({"i": d["i"], "regime": regime, "anchor": anchor}) + "\n")
            fb.write(json.dumps({
                "custom_id": f"SC-{d['i']:05d}", "method": "POST", "url": "/v1/chat/completions",
                "body": {"model": "gpt-5.4-mini",
                         "messages": [{"role": "system", "content": sys_prompt(anchor)},
                                      {"role": "user", "content": "Document:\n" + d["text"]}],
                         "max_completion_tokens": 700}}) + "\n")

    # B) anchor-OOD gold: existing eval docs under holdout vocabularies
    evals = []
    for split in ("eval_id", "eval_ood"):
        evals += [(split, json.loads(l)) for l in open(f"{HERE}/ov_gold_{split}.jsonl", encoding="utf-8")]
    with open(f"{HERE}/anchors_holdout.jsonl", "w", encoding="utf-8") as fa, \
         open(f"{HERE}/anchorood_batch.jsonl", "w", encoding="utf-8") as fb:
        for split, g in evals:
            v = HOLD[hkey(f"hold-{g['i']}") % len(HOLD)]
            fa.write(json.dumps({"i": g["i"], "split": split, "domain": v["domain"],
                                 "anchor": v["topics"]}) + "\n")
            fb.write(json.dumps({
                "custom_id": f"AO-{g['i']:05d}", "method": "POST", "url": "/v1/chat/completions",
                "body": {"model": "gpt-5.4-mini",
                         "messages": [{"role": "system", "content": sys_prompt(v["topics"])},
                                      {"role": "user", "content": "Document:\n" + g["text"]}],
                         "max_completion_tokens": 700}}) + "\n")

    st = json.load(open(f"{HERE}/state_ov.json"))
    for name, path in (("scale", f"{HERE}/scale_batch.jsonl"),
                       ("anchorood", f"{HERE}/anchorood_batch.jsonl")):
        bid, status = submit(path)
        n = sum(1 for _ in open(path))
        print(f"{name} ({n} reqs): {bid} {status}")
        st[name] = {"batch_id": bid, "model": "gpt-5.4-mini", "n": n}
    json.dump(st, open(f"{HERE}/state_ov.json", "w"), indent=2)

if __name__ == "__main__":
    main()
