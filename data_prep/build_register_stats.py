#!/usr/bin/env python3
"""Derive the 'enterprise register' statistics from the teacher labels.

The GRPO reward should not encode my taste in phrasing — it should encode the
register the teacher corpus actually uses. From data/ent_clean.jsonl we take:
  subject_tokens  token -> document frequency over teacher subject terms
  generic_tokens  the most common (least discriminative) subject tokens; a term
                  built only from these is boilerplate, not a filing term
  facet_vocab     the values the teacher used for type / audience / purpose
Written to data/register_stats.json.
"""
import json, os
from collections import Counter

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
STOP = {"the", "a", "an", "of", "and", "or", "for", "to", "in", "on", "with", "by"}
GENERIC_TOP_N = 25


def toks(term):
    return [t for t in str(term).split("_") if t and t not in STOP]


def main():
    rows = [json.loads(l) for l in open(f"{HERE}/ent_clean.jsonl", encoding="utf-8")]
    tokfreq, facet = Counter(), {"type": Counter(), "audience": Counter(), "purpose": Counter()}
    n_terms = 0
    for r in rows:
        e = r["entry"]
        for term in e["subject"]:
            n_terms += 1
            for t in set(toks(term)):
                tokfreq[t] += 1
        for k in facet:
            if e.get(k):
                facet[k][e[k]] += 1

    generic = [t for t, _ in tokfreq.most_common(GENERIC_TOP_N)]
    # keep facet values the teacher used more than once (drops one-off noise)
    facet_vocab = {k: sorted(v for v, c in cnt.items() if c >= 2) for k, cnt in facet.items()}
    out = {"subject_tokens": dict(tokfreq),
           "generic_tokens": generic,
           "facet_vocab": facet_vocab,
           "n_subject_terms": n_terms, "n_docs": len(rows)}
    json.dump(out, open(f"{HERE}/register_stats.json", "w"), indent=1)
    print(f"docs={len(rows)} subject_terms={n_terms} distinct_tokens={len(tokfreq)}")
    print("generic (boilerplate) tokens:", generic)
    for k, v in facet_vocab.items():
        print(f"{k}: {len(v)} values kept, e.g. {v[:6]}")


if __name__ == "__main__":
    main()
