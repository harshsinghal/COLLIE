#!/usr/bin/env python3
"""Faceted-librarian SFT assembly (direct variant — the settled recipe).

- Drops entries with compound single-facet values ("_and_", multi-orientation
  time) — one answer per facet is the contract; compounds teach compounding.
- Register holdout intact: apache + loghub NEVER in train.
- Student prompt mirrors the teacher's facet spec + per-doc anchor (no think).
- Also writes the faceted inference file for the 3,705-doc faithfulness corpus.
"""
import json, os, re, sys

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_facet_batches import sys_prompt_facets

TRAIN_SRCS = {"enron", "finepdfs", "chat", "ghcode"}
TIME_OK = {"historical", "current", "forward_looking", None}

def student_prompt(anchor):
    # teacher prompt minus the mandatory-think block
    p = sys_prompt_facets(anchor)
    p = p.replace(
        "You MUST reason first inside <think>...</think> — REQUIRED for every document, even short "
        "ones. Keep it CONCISE: 40-120 words, citing the words/context that resolve each facet. "
        "Then, AFTER </think>, output STRICT JSON on its own line:\n", "Output STRICT JSON on its own line:\n")
    return p

def clean(entry):
    for k in ("type", "audience", "purpose"):
        if entry[k] and "_and_" in entry[k]:
            return False
    t = entry["time"]
    if t not in TIME_OK:
        return False
    return True

def main():
    rows = [json.loads(l) for l in open(f"{HERE}/facet_clean.jsonl", encoding="utf-8")]
    kept = dropped = 0
    with open(f"{HERE}/facet_direct_train.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            if r["src"] not in TRAIN_SRCS:
                continue
            if not clean(r["entry"]):
                dropped += 1; continue
            f.write(json.dumps({"messages": [
                {"role": "system", "content": student_prompt(r["anchor"])},
                {"role": "user", "content": "Document:\n" + r["text"]},
                {"role": "assistant", "content": json.dumps(r["entry"], ensure_ascii=False)}],
                "src": r["src"], "i": r["i"]}, ensure_ascii=False) + "\n")
            kept += 1
    print(f"train: kept={kept} dropped_compound={dropped}")

    n = 0
    with open(f"{HERE}/facet_faith_infer.jsonl", "w", encoding="utf-8") as out:
        for l in open(f"{HERE}/faith_manifest.jsonl", encoding="utf-8"):
            d = json.loads(l)
            out.write(json.dumps({"messages": [
                {"role": "system", "content": student_prompt(d["anchor"])},
                {"role": "user", "content": "Document:\n" + d["text"]},
                {"role": "assistant", "content": ""}],
                "i": d["i"]}, ensure_ascii=False) + "\n")
            n += 1
    print(f"faith inference file: {n}")

if __name__ == "__main__":
    main()
