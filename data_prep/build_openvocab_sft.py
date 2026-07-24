#!/usr/bin/env python3
"""Assemble open-vocab SFT sets with a REGISTER holdout (the OOD test).

Train registers: enron, finepdfs, chat, ghcode (minus an in-dist eval slice).
Held-out registers (never trained): apache (JIRA tickets), loghub (system logs)
— the out-of-distribution eval measuring whether the librarian *procedure*
transfers to registers the student never saw.

Student prompts mirror the teacher's anchor conditioning exactly (same per-doc
anchor, same soft-preference wording). Two variants over identical docs:
  ov_reason: assistant = <think>{trace}</think>\n{topics,tags json}
  ov_direct: assistant = {topics,tags json}
Gold files carry topics/tags for the LLM-judge scorer.
"""
import argparse, hashlib, json, os
from collections import defaultdict

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SRC = f"{HERE}/ov_clean.jsonl"
TRAIN_SRCS = {"enron", "finepdfs", "chat", "ghcode"}
OOD_SRCS = {"apache", "loghub"}

TAG_HINT = ("individual, group, aggregate, org_wide, public, internal, restricted, historical, "
            "current, forward_looking, named, figures, general, report, negotiation, decision, "
            "directive, request, mention, speculation")

def sys_prompt(anchor, reason):
    base = ("You are COLLIE, a librarian for enterprise documents. Read the document and catalog it: "
            "identify the topic(s) it genuinely discusses and assign descriptive tags. Describe, do "
            "not judge sensitivity.\n\n")
    if reason:
        base += ("Reason first inside <think>...</think>: name candidate topics, cite the specific "
                 "words/context that confirm or reject each, and choose tags from the evidence. Then, "
                 "AFTER </think>, output STRICT JSON on its own line: {\"topics\":[...],\"tags\":[...]}.\n\n")
    else:
        base += "Output STRICT JSON on its own line: {\"topics\":[...],\"tags\":[...]}.\n\n"
    base += "topics: 1-4 short snake_case noun phrases naming the subjects discussed. "
    if anchor:
        base += ("Prefer these catalog topics when they genuinely fit: " + ", ".join(anchor) +
                 ". If the content is not covered by the catalog, coin your own coherent topic "
                 "instead of forcing a bad fit.\n")
    else:
        base += "Coin coherent topics yourself; there is no fixed catalog.\n"
    base += ("tags: 2-8 short descriptors characterizing how the topics are discussed (who/scope, "
             "audience/publicity, time orientation, specificity, speech act). Prefer these when they "
             f"fit: {TAG_HINT}. Coin others when needed.")
    return base

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indist-eval-per-source", type=int, default=35)
    ap.add_argument("--ood-eval-cap", type=int, default=100, help="per OOD source")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]
    rows.sort(key=lambda r: hashlib.sha256(str(r["i"]).encode()).hexdigest())
    by_src = defaultdict(list)
    for r in rows:
        by_src[r["src"]].append(r)

    split = {}
    for src, rs in by_src.items():
        if src in TRAIN_SRCS:
            for k, r in enumerate(rs):
                split[r["i"]] = "eval_id" if k < a.indist_eval_per_source else "train"
        else:
            for k, r in enumerate(rs):
                split[r["i"]] = "eval_ood" if k < a.ood_eval_cap else "drop"

    out = {("reason", s): open(f"{HERE}/ov_reason_{s}.jsonl", "w", encoding="utf-8")
           for s in ("train", "eval_id", "eval_ood")}
    out.update({("direct", s): open(f"{HERE}/ov_direct_{s}.jsonl", "w", encoding="utf-8")
                for s in ("train", "eval_id", "eval_ood")})
    gold = {s: open(f"{HERE}/ov_gold_{s}.jsonl", "w", encoding="utf-8")
            for s in ("eval_id", "eval_ood")}

    from collections import Counter
    counts = Counter()
    for r in sorted(rows, key=lambda x: x["i"]):
        s = split[r["i"]]
        if s == "drop":
            counts["drop"] += 1; continue
        lj = json.dumps({"topics": r["topics"], "tags": r["tags"]}, ensure_ascii=False)
        u = "Document:\n" + r["text"]
        for variant, reason in (("reason", True), ("direct", False)):
            asst = f"<think>{r['think']}</think>\n{lj}" if reason else lj
            out[(variant, s)].write(json.dumps({"messages": [
                {"role": "system", "content": sys_prompt(r["anchor"], reason)},
                {"role": "user", "content": u},
                {"role": "assistant", "content": asst}],
                "src": r["src"], "i": r["i"], "regime": r["regime"]}, ensure_ascii=False) + "\n")
        if s in gold:
            gold[s].write(json.dumps({"i": r["i"], "src": r["src"], "regime": r["regime"],
                                      "anchor": r["anchor"], "text": r["text"],
                                      "topics": r["topics"], "tags": r["tags"]},
                                     ensure_ascii=False) + "\n")
        counts[s] += 1
    for f in list(out.values()) + list(gold.values()):
        f.close()
    print(dict(counts))

if __name__ == "__main__":
    main()
