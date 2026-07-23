#!/usr/bin/env python3
"""Assemble COLLIE SFT sets (reason vs direct) from filtered teacher labels.

Reads reason_clean_v2.jsonl (survivors of the multi-teacher filter) and emits
two parallel datasets over the SAME docs and the SAME held-out eval split:

  - collie_reason: assistant = <think>{trace}</think>\n{labels_json}
  - collie_direct: assistant = {labels_json}          (labels-only baseline)

The only variable between them is the presence of the reasoning trace, so a
score gap is attributable to reasoning-first, not to different data. Eval is a
stratified hold-out (by source) shared by both, and also written as a gold file
for collie_eval.py. `--max-think-words` optionally truncates long CoT for
distillation into the 0.6B student (0 = keep full).
"""
import argparse, gzip, json, os, re
from collections import defaultdict

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SRC = f"{HERE}/reason_clean_gpt.jsonl"

TOPICS = ("compensation, workforce, mergers_acquisitions, financials, pricing, legal, security, "
          "credentials, product, strategy, competition, personnel, regulatory, customer_data")
FACET_SPEC = ("scope=individual|group|aggregate|org_wide; publicity=public|internal|restricted; "
              "temporality=historical|current|forward_looking; specificity=named|figures|both|general; "
              "register_facet=report|negotiation|decision|directive|request|mention|speculation")

SYS_REASON = (
    "You are COLLIE, a document cataloger. Read the document and decide which topics it genuinely "
    "discusses and how, using ONLY the ontology below. Describe, do not judge sensitivity.\n\n"
    "Reason first inside <think>...</think>: name the candidate subjects, cite the words/context that "
    "confirm or reject each, and resolve each facet from evidence. Then, AFTER </think>, output STRICT "
    "JSON on its own line: {\"labels\":[{\"topic\":\"<t>\",\"scope\":\"<v>\",\"publicity\":\"<v>\","
    "\"temporality\":\"<v>\",\"specificity\":\"<v>\",\"register_facet\":\"<v>\"}]} (empty list if none).\n\n"
    f"Topics: {TOPICS}\nFacets (per topic): {FACET_SPEC}")

SYS_DIRECT = (
    "You are COLLIE, a document cataloger. Read the document and decide which topics it genuinely "
    "discusses and how, using ONLY the ontology below. Describe, do not judge sensitivity.\n\n"
    "Output STRICT JSON on its own line: {\"labels\":[{\"topic\":\"<t>\",\"scope\":\"<v>\","
    "\"publicity\":\"<v>\",\"temporality\":\"<v>\",\"specificity\":\"<v>\",\"register_facet\":\"<v>\"}]} "
    "(empty list if none).\n\n"
    f"Topics: {TOPICS}\nFacets (per topic): {FACET_SPEC}")

FACET_KEYS = ["scope", "publicity", "temporality", "specificity", "register_facet"]

SYS_REASON_FLAT = (
    "You are COLLIE, a document cataloger. Read the document and decide which topics it genuinely "
    "discusses, using ONLY the ontology below. Describe, do not judge sensitivity.\n\n"
    "Reason first inside <think>...</think>: name the candidate subjects, cite the words/context that "
    "confirm or reject each. Then, AFTER </think>, output STRICT JSON on its own line: "
    "{\"topics\":[...],\"tags\":[...]} — topics from the topic list, tags = every descriptor from the "
    "facet vocabulary that characterizes how those topics are discussed (empty lists if none).\n\n"
    f"Topics: {TOPICS}\nTag vocabulary: {FACET_SPEC}")

SYS_DIRECT_FLAT = (
    "You are COLLIE, a document cataloger. Read the document and decide which topics it genuinely "
    "discusses, using ONLY the ontology below. Describe, do not judge sensitivity.\n\n"
    "Output STRICT JSON on its own line: {\"topics\":[...],\"tags\":[...]} — topics from the topic "
    "list, tags = every descriptor from the facet vocabulary that characterizes how those topics are "
    "discussed (empty lists if none).\n\n"
    f"Topics: {TOPICS}\nTag vocabulary: {FACET_SPEC}")

def flat_view(labels):
    """Collapse per-topic facet dicts into {topics:[...], tags:[...]} (order-stable dedup)."""
    topics, tags = [], []
    for l in labels:
        if l["topic"] not in topics:
            topics.append(l["topic"])
        for k in FACET_KEYS:
            v = l.get(k)
            if v and v not in tags:
                tags.append(v)
    return topics, tags

def user_turn(text):
    return "Document:\n" + text

def labels_json(labels):
    return json.dumps({"labels": labels}, ensure_ascii=False)

def flat_json(labels):
    topics, tags = flat_view(labels)
    return json.dumps({"topics": topics, "tags": tags}, ensure_ascii=False)

def truncate(think, max_words):
    if not max_words:
        return think
    w = think.split()
    return think if len(w) <= max_words else " ".join(w[:max_words]) + " …"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-per-source", type=int, default=35)
    ap.add_argument("--max-think-words", type=int, default=0)
    ap.add_argument("--flat", action="store_true",
                    help="emit {topics,tags} targets instead of per-topic facet dicts")
    ap.add_argument("--none-boost", type=int, default=1,
                    help="duplicate empty-label TRAIN docs this many times (abstention signal)")
    a = ap.parse_args()

    opener = (lambda f: gzip.open(f + ".gz", "rt", encoding="utf-8")) \
        if not os.path.exists(SRC) else (lambda f: open(f, encoding="utf-8"))
    rows = [json.loads(l) for l in opener(SRC)]
    # deterministic stratified hold-out by source (hash order, take first N/source)
    import hashlib
    rows.sort(key=lambda r: hashlib.sha256(str(r["i"]).encode()).hexdigest())
    by_src, eval_ids = defaultdict(list), set()
    for r in rows:
        by_src[r["src"]].append(r)
    for src, rs in by_src.items():
        for r in rs[:a.eval_per_source]:
            eval_ids.add(r["i"])

    sys_r = SYS_REASON_FLAT if a.flat else SYS_REASON
    sys_d = SYS_DIRECT_FLAT if a.flat else SYS_DIRECT
    to_json = flat_json if a.flat else labels_json
    prefix = "collie_flat" if a.flat else "collie"
    if a.none_boost > 1:
        prefix += f"_n{a.none_boost}"

    reason, direct, gold = [], [], []
    for r in rows:
        lj = to_json(r["labels"])
        th = truncate(r["think"], a.max_think_words)
        u = user_turn(r["text"])
        split = "eval" if r["i"] in eval_ids else "train"
        reason.append({"messages": [
            {"role": "system", "content": sys_r},
            {"role": "user", "content": u},
            {"role": "assistant", "content": f"<think>{th}</think>\n{lj}"}],
            "src": r["src"], "split": split, "i": r["i"], "teacher": r["model"]})
        direct.append({"messages": [
            {"role": "system", "content": sys_d},
            {"role": "user", "content": u},
            {"role": "assistant", "content": lj}],
            "src": r["src"], "split": split, "i": r["i"], "teacher": r["model"]})
        if split == "eval":
            g = {"i": r["i"], "src": r["src"], "text": r["text"], "labels": r["labels"]}
            if a.flat:
                g["topics"], g["tags"] = flat_view(r["labels"])
            gold.append(g)

    def dump(name, data, split):
        p = f"{HERE}/{name}_{split}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for ex in data:
                if ex["split"] == split:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        return sum(1 for ex in data if ex["split"] == split)

    if a.none_boost > 1:  # duplicate none-docs in train only
        extra_r = [ex for ex in reason if ex["split"] == "train"
                   and '"topics": []' in ex["messages"][2]["content"]]
        extra_d = [ex for ex in direct if ex["split"] == "train"
                   and '"topics": []' in ex["messages"][2]["content"]]
        for _ in range(a.none_boost - 1):
            reason.extend(json.loads(json.dumps(e)) for e in extra_r)
            direct.extend(json.loads(json.dumps(e)) for e in extra_d)
        print(f"none-boost x{a.none_boost}: +{len(extra_r)*(a.none_boost-1)} none-docs per variant")

    for split in ("train", "eval"):
        nr = dump(f"{prefix}_reason", reason, split)
        nd = dump(f"{prefix}_direct", direct, split)
        print(f"{split}: reason={nr} direct={nd}")
    with open(f"{HERE}/{prefix}_eval_gold.jsonl", "w", encoding="utf-8") as f:
        for g in gold:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    print(f"eval gold: {len(gold)}  (flat={a.flat}, max_think_words={a.max_think_words or 'full'})")

if __name__ == "__main__":
    main()
