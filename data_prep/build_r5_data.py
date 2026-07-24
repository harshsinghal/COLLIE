#!/usr/bin/env python3
"""Round-5 data assembly, run after the scale + anchorood batches complete.

1) Harvest the scale batch (3,000 new train-register docs) with the same gate
   as round 4; merge with round-4 survivors into enlarged train sets
   (register holdout unchanged: apache/loghub never in train).
2) Harvest the anchorood batch (340 eval docs relabeled under never-trained
   vocabularies) -> ov_gold_eval_anchor.jsonl + per-variant eval prompt files
   (student sees the SAME holdout anchor the teacher saw).
Also folds in the stale gpt-5.5 boost batch if it has completed.
"""
import json, os, sys, requests
from collections import Counter

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harvest_openvocab import extract  # same filter gate
from build_openvocab_sft import sys_prompt  # same student prompt

H = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}

def fetch(batch_key, st):
    b = requests.get(f"https://api.openai.com/v1/batches/{st[batch_key]['batch_id']}",
                     headers=H, timeout=30).json()
    if b.get("status") != "completed" or not b.get("output_file_id"):
        print(f"{batch_key}: {b.get('status')} — skipped")
        return None
    return requests.get(f"https://api.openai.com/v1/files/{b['output_file_id']}/content",
                        headers=H, timeout=300).text

def main():
    st = json.load(open(f"{HERE}/state_ov.json"))

    # ---- 1) scale harvest -> enlarged train sets ----
    import gzip
    man = {json.loads(l)["i"]: json.loads(l)
           for l in open(f"{HERE}/manifest_more.jsonl", encoding="utf-8")}
    anch = {json.loads(l)["i"]: json.loads(l)
            for l in open(f"{HERE}/anchors_more.jsonl", encoding="utf-8")}
    new_rows, rej = [], Counter()
    raw = fetch("scale", st)
    if raw:
        for line in raw.splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            i = int(r["custom_id"].split("-")[1])
            try:
                content = r["response"]["body"]["choices"][0]["message"]["content"] or ""
            except Exception:
                rej["no_response"] += 1; continue
            think, parsed = extract(content)
            if think is None:
                rej[parsed] += 1; continue
            d, a = man[i], anch[i]
            new_rows.append({"i": i, "src": d["src"], "anchor": a["anchor"],
                             "regime": a["regime"], "text": d["text"],
                             "think": think, **parsed})
    print(f"scale survivors: {len(new_rows)}  rejects: {dict(rej.most_common(5))}")

    # append to existing train files (round-4 rows already there)
    for variant, reason in (("reason", True), ("direct", False)):
        with open(f"{HERE}/ov_{variant}_train.jsonl", "a", encoding="utf-8") as f:
            for r in new_rows:
                lj = json.dumps({"topics": r["topics"], "tags": r["tags"]}, ensure_ascii=False)
                asst = f"<think>{r['think']}</think>\n{lj}" if reason else lj
                f.write(json.dumps({"messages": [
                    {"role": "system", "content": sys_prompt(r["anchor"], reason)},
                    {"role": "user", "content": "Document:\n" + r["text"]},
                    {"role": "assistant", "content": asst}],
                    "src": r["src"], "i": r["i"], "regime": r["regime"]},
                    ensure_ascii=False) + "\n")
        n = sum(1 for _ in open(f"{HERE}/ov_{variant}_train.jsonl"))
        print(f"ov_{variant}_train.jsonl now {n} rows")

    # ---- 2) anchor-OOD gold + eval prompt files ----
    hold = {json.loads(l)["i"]: json.loads(l)
            for l in open(f"{HERE}/anchors_holdout.jsonl", encoding="utf-8")}
    texts = {}
    for split in ("eval_id", "eval_ood"):
        for l in open(f"{HERE}/ov_gold_{split}.jsonl", encoding="utf-8"):
            g = json.loads(l)
            texts[g["i"]] = g["text"]
    raw = fetch("anchorood", st)
    ao, rej2 = [], Counter()
    if raw:
        for line in raw.splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            i = int(r["custom_id"].split("-")[1])
            try:
                content = r["response"]["body"]["choices"][0]["message"]["content"] or ""
            except Exception:
                rej2["no_response"] += 1; continue
            think, parsed = extract(content)
            if think is None:
                rej2[parsed] += 1; continue
            ao.append({"i": i, "domain": hold[i]["domain"], "anchor": hold[i]["anchor"],
                       "text": texts[i], **parsed})
    print(f"anchor-OOD gold: {len(ao)}  rejects: {dict(rej2.most_common(5))}")
    with open(f"{HERE}/ov_gold_eval_anchor.jsonl", "w", encoding="utf-8") as f:
        for g in sorted(ao, key=lambda x: x["i"]):
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    for variant, reason in (("reason", True), ("direct", False)):
        with open(f"{HERE}/ov_{variant}_eval_anchor.jsonl", "w", encoding="utf-8") as f:
            for g in sorted(ao, key=lambda x: x["i"]):
                f.write(json.dumps({"messages": [
                    {"role": "system", "content": sys_prompt(g["anchor"], reason)},
                    {"role": "user", "content": "Document:\n" + g["text"]},
                    {"role": "assistant", "content": ""}],
                    "i": g["i"]}, ensure_ascii=False) + "\n")
    print("wrote ov_{reason,direct}_eval_anchor.jsonl")

if __name__ == "__main__":
    main()
