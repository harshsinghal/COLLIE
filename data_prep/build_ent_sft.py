#!/usr/bin/env python3
"""Enterprise-librarian SFT assembly: anchor-free faceted cards.

Drops remaining compound-value entries (5.7%); register holdout intact
(apache/loghub never in train). Student prompt = teacher prompt minus the
think requirement. Also writes the faithfulness inference file (single
prompt for all docs — no conditioning modes anymore).
"""
import json, os, sys

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_enterprise_batch import SYS_ENTERPRISE

TRAIN_SRCS = {"enron", "finepdfs", "chat", "ghcode"}
TIME_OK = {"historical", "current", "forward_looking", None}

STUDENT = SYS_ENTERPRISE.replace(
    "You MUST reason first inside <think>...</think> — REQUIRED for every document, even short "
    "ones. Keep it CONCISE: 40-120 words, citing the words/context that resolve each facet. "
    "Then, AFTER </think>, output STRICT JSON on its own line:\n",
    "Output STRICT JSON on its own line:\n")

def clean(e):
    if any(e[k] and "_and_" in e[k] for k in ("type", "audience", "purpose")):
        return False
    return e["time"] in TIME_OK

def main():
    rows = [json.loads(l) for l in open(f"{HERE}/ent_clean.jsonl", encoding="utf-8")]
    kept = dropped = 0
    with open(f"{HERE}/ent_direct_train.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            if r["src"] not in TRAIN_SRCS:
                continue
            if not clean(r["entry"]):
                dropped += 1; continue
            f.write(json.dumps({"messages": [
                {"role": "system", "content": STUDENT},
                {"role": "user", "content": "Document:\n" + r["text"]},
                {"role": "assistant", "content": json.dumps(r["entry"], ensure_ascii=False)}],
                "src": r["src"], "i": r["i"]}, ensure_ascii=False) + "\n")
            kept += 1
    print(f"train: kept={kept} dropped_compound={dropped}")
    n = 0
    with open(f"{HERE}/ent_faith_infer.jsonl", "w", encoding="utf-8") as out:
        for l in open(f"{HERE}/faith_manifest.jsonl", encoding="utf-8"):
            d = json.loads(l)
            out.write(json.dumps({"messages": [
                {"role": "system", "content": STUDENT},
                {"role": "user", "content": "Document:\n" + d["text"]},
                {"role": "assistant", "content": ""}],
                "i": d["i"]}, ensure_ascii=False) + "\n")
            n += 1
    print(f"faith inference: {n}")

if __name__ == "__main__":
    main()
