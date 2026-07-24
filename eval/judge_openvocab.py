#!/usr/bin/env python3
"""LLM-judge scorer for open-vocabulary COLLIE predictions.

Exact string match is meaningless with free labels ("comp_policy" ==
"compensation_guidelines"). For each eval doc the judge (gpt-5.4-mini) sees
the document snippet, gold topics/tags, and predicted topics/tags, and counts
semantic matches (same subject, wording irrelevant). Micro P/R/F1 over the
matched counts; topics and tags scored separately.

Usage: judge_openvocab.py --gold data/ov_gold_eval_ood.jsonl --pred preds.jsonl
Pred rows: {"i", "topics", "tags"} aligned by doc id.
"""
import argparse, json, os, re, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

H = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
     "Content-Type": "application/json"}
MODEL = "gpt-5.4-mini"

PROMPT = """You are scoring a document-cataloging model. Compare PREDICTED topics/tags
against GOLD topics/tags for the same document. Two labels match if they name the same
subject or descriptor — wording, synonyms, and granularity differences do not matter
(e.g. "pay_package" matches "compensation"; "sre_incident" matches "incident_response").
A prediction may match at most one gold label and vice versa.

Document (excerpt):
{doc}

GOLD topics: {gt}   GOLD tags: {gg}
PREDICTED topics: {pt}   PREDICTED tags: {pg}

Output STRICT JSON only:
{{"topic_matches": <int>, "tag_matches": <int>}}"""

def call(g, p):
    body = {"model": MODEL, "max_completion_tokens": 300,
            "messages": [{"role": "user", "content": PROMPT.format(
                doc=g["text"][:900], gt=g["topics"], gg=g["tags"],
                pt=p.get("topics", []), pg=p.get("tags", []))}]}
    for attempt in range(4):
        try:
            r = requests.post("https://api.openai.com/v1/chat/completions",
                              headers=H, json=body, timeout=90)
            if r.status_code >= 429:
                time.sleep(2 * (attempt + 1)); continue
            content = r.json()["choices"][0]["message"]["content"]
            m = re.search(r'\{.*\}', content, re.S)
            d = json.loads(m.group(0))
            tm = min(int(d["topic_matches"]), len(g["topics"]), len(p.get("topics", [])))
            gm = min(int(d["tag_matches"]), len(g["tags"]), len(p.get("tags", [])))
            return {"i": g["i"], "tm": tm, "gm": gm,
                    "gt": len(g["topics"]), "pt": len(p.get("topics", [])),
                    "gg": len(g["tags"]), "pg": len(p.get("tags", []))}
        except Exception:
            time.sleep(2 * (attempt + 1))
    return {"i": g["i"], "tm": 0, "gm": 0, "gt": len(g["topics"]),
            "pt": len(p.get("topics", [])), "gg": len(g["tags"]),
            "pg": len(p.get("tags", []))}

def prf(tp, fp, fn):
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--conc", type=int, default=12)
    a = ap.parse_args()
    gold = [json.loads(l) for l in open(a.gold, encoding="utf-8")]
    preds = {json.loads(l)["i"]: json.loads(l) for l in open(a.pred, encoding="utf-8")}
    pairs = [(g, preds.get(g["i"], {})) for g in gold]
    results = []
    with ThreadPoolExecutor(max_workers=a.conc) as ex:
        futs = [ex.submit(call, g, p) for g, p in pairs]
        for f in as_completed(futs):
            results.append(f.result())
    t_tp = sum(r["tm"] for r in results)
    t_fp = sum(r["pt"] - r["tm"] for r in results)
    t_fn = sum(r["gt"] - r["tm"] for r in results)
    g_tp = sum(r["gm"] for r in results)
    g_fp = sum(r["pg"] - r["gm"] for r in results)
    g_fn = sum(r["gg"] - r["gm"] for r in results)
    P, R, F = prf(t_tp, t_fp, t_fn)
    print(f"TOPICS (judged): P {P:.3f}  R {R:.3f}  F1 {F:.3f}  (tp={t_tp} fp={t_fp} fn={t_fn}, n={len(results)})")
    P, R, F = prf(g_tp, g_fp, g_fn)
    print(f"TAGS   (judged): P {P:.3f}  R {R:.3f}  F1 {F:.3f}  (tp={g_tp} fp={g_fp} fn={g_fn})")

if __name__ == "__main__":
    main()
