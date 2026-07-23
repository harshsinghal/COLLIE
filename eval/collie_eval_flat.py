#!/usr/bin/env python3
"""Scorer for the flat COLLIE output shape: {"topics":[...], "tags":[...]}.

Multi-label micro P/R/F1 for topics and for tags (set semantics), plus the
'none' confusion on topics. Gold rows carry topics/tags (from --flat build);
pred rows are the model's parsed JSON. Rows align positionally.
"""
import json

def sets(row, key):
    return set(row.get(key) or [])

def prf(tp, fp, fn):
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return P, R, F

def score(gold_rows, pred_rows):
    t_tp = t_fp = t_fn = 0
    g_tp = g_fp = g_fn = 0
    none_gold = none_pred = none_both = 0
    for g, p in zip(gold_rows, pred_rows):
        gt, pt = sets(g, "topics"), sets(p, "topics")
        t_tp += len(gt & pt); t_fp += len(pt - gt); t_fn += len(gt - pt)
        gg, pg = sets(g, "tags"), sets(p, "tags")
        g_tp += len(gg & pg); g_fp += len(pg - gg); g_fn += len(gg - pg)
        if not gt: none_gold += 1
        if not pt: none_pred += 1
        if not gt and not pt: none_both += 1
    P, R, F = prf(t_tp, t_fp, t_fn)
    print(f"TOPICS (micro):  P {P:.3f}  R {R:.3f}  F1 {F:.3f}   (tp={t_tp} fp={t_fp} fn={t_fn})")
    P, R, F = prf(g_tp, g_fp, g_fn)
    print(f"TAGS   (micro):  P {P:.3f}  R {R:.3f}  F1 {F:.3f}   (tp={g_tp} fp={g_fp} fn={g_fn})")
    print(f"NONE handling: gold-none {none_gold}, pred-none {none_pred}, correct-none {none_both}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    a = ap.parse_args()
    g = [json.loads(l) for l in open(a.gold, encoding="utf-8")]
    p = [json.loads(l) for l in open(a.pred, encoding="utf-8")]
    assert len(g) == len(p), f"{len(g)} gold != {len(p)} pred"
    score(g, p)
