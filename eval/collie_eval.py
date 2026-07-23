#!/usr/bin/env python3
"""COLLIE scorer: multi-label topic F1 + per-facet accuracy.

Gold and pred are each a list of {topic, <facets>} dicts per document.
- Topic layer: multi-label micro P/R/F1 over the set of topics per doc.
- Facet layer: accuracy, computed only over topics that both gold and pred
  agree are present (a facet on a hallucinated/missed topic is meaningless).
Also reports the 'none' confusion (docs where gold=empty vs pred=empty).
"""
import json, sys
from collections import Counter

FACETS = ["scope","publicity","temporality","specificity","register_facet"]

def topic_set(labels): return {l["topic"] for l in labels}
def by_topic(labels): return {l["topic"]: l for l in labels}

def score(gold_rows, pred_rows):
    tp=fp=fn=0
    facet_correct=Counter(); facet_total=Counter()
    none_gold=none_pred=none_both=0
    for g, p in zip(gold_rows, pred_rows):
        gt, pt = topic_set(g), topic_set(p)
        tp += len(gt & pt); fp += len(pt - gt); fn += len(gt - pt)
        gb, pb = by_topic(g), by_topic(p)
        for t in (gt & pt):
            for f in FACETS:
                if f in gb[t] and f in pb[t]:
                    facet_total[f]+=1
                    facet_correct[f]+= (gb[t][f]==pb[t][f])
        if not gt: none_gold+=1
        if not pt: none_pred+=1
        if not gt and not pt: none_both+=1
    P=tp/(tp+fp) if tp+fp else 0.0
    R=tp/(tp+fn) if tp+fn else 0.0
    F=2*P*R/(P+R) if P+R else 0.0
    print(f"TOPIC (multi-label micro):  P {P:.3f}  R {R:.3f}  F1 {F:.3f}   (tp={tp} fp={fp} fn={fn})")
    print(f"NONE handling: gold-none {none_gold}, pred-none {none_pred}, correct-none {none_both}")
    print("FACET accuracy (on agreed topics):")
    for f in FACETS:
        acc = facet_correct[f]/facet_total[f] if facet_total[f] else 0.0
        print(f"  {f:<16} {acc:.3f}  (n={facet_total[f]})")

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--gold",required=True); ap.add_argument("--pred",required=True)
    ap.add_argument("--n",type=int)
    a=ap.parse_args()
    g=[json.loads(l)["labels"] for l in open(a.gold,encoding="utf-8")]
    p=[json.loads(l).get("labels",[]) for l in open(a.pred,encoding="utf-8")]
    if a.n: g,p=g[:a.n],p[:a.n]
    assert len(g)==len(p), f"{len(g)} gold != {len(p)} pred"
    score(g,p)
