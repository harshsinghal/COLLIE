#!/usr/bin/env python3
"""Reference-free faithfulness eval, part 3: harvest judge batch + report.

Metrics (per conditioning mode and overall):
  precise / vague / wrong  — share of emitted topics in each grade
  faithful_rate            — docs where NO topic is wrong AND missed == 0
  avg_missed               — mean count of clearly-discussed-but-omitted subjects
  tag_apt_rate             — apt tags / emitted tags

Usage: score_faith.py --run 06 [--pred data/preds_faith_06.jsonl]
"""
import argparse, json, os, re, requests
from collections import Counter, defaultdict

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
H = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--pred", default=None)
    a = ap.parse_args()
    st = json.load(open(f"{HERE}/state_ov.json"))
    b = requests.get(f"https://api.openai.com/v1/batches/{st[f'faith_judge_{a.run}']['batch_id']}",
                     headers=H, timeout=30).json()
    if b.get("status") != "completed":
        c = b.get("request_counts", {})
        print(f"judge batch: {b.get('status')} {c.get('completed', 0)}/{c.get('total', 0)}")
        return
    raw = requests.get(f"https://api.openai.com/v1/files/{b['output_file_id']}/content",
                       headers=H, timeout=300).text
    man = {json.loads(l)["i"]: json.loads(l)
           for l in open(f"{HERE}/faith_manifest.jsonl", encoding="utf-8")}
    preds = {}
    if a.pred:
        preds = {json.loads(l)["i"]: json.loads(l) for l in open(a.pred, encoding="utf-8")}

    agg = defaultdict(lambda: {"grades": Counter(), "missed": 0, "docs": 0,
                               "faithful": 0, "tags_apt": 0, "tags_total": 0,
                               "parse_fail": 0})
    for line in raw.splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        i = int(r["custom_id"].split("-")[1])
        mode = man[i]["mode"].split(":")[0]
        buckets = [agg[mode], agg["ALL"]]
        try:
            content = r["response"]["body"]["choices"][0]["message"]["content"]
            d = json.loads(re.search(r'\{.*\}', content, re.S).group(0))
            grades = [g for g in d["topic_grades"] if g in ("precise", "vague", "wrong")]
            missed = max(0, min(3, int(d["missed"])))
            tapt = max(0, int(d["tags_apt"]))
        except Exception:
            for bk in buckets:
                bk["parse_fail"] += 1
            continue
        ntags = len(preds.get(i, {}).get("tags", [])) if preds else None
        for bk in buckets:
            bk["docs"] += 1
            bk["grades"].update(grades)
            bk["missed"] += missed
            bk["faithful"] += (missed == 0 and "wrong" not in grades)
            bk["tags_apt"] += tapt
            if ntags is not None:
                bk["tags_total"] += ntags

    order = ["ALL", "canonical", "none", "unseen"]
    print(f"{'mode':<10} {'docs':>5} {'precise':>8} {'vague':>7} {'wrong':>7} "
          f"{'faithful':>9} {'avg_miss':>9} {'tag_apt':>8}")
    for mode in order:
        if mode not in agg:
            continue
        bk = agg[mode]
        tot = sum(bk["grades"].values()) or 1
        tag_rate = (bk["tags_apt"] / bk["tags_total"]) if bk["tags_total"] else float("nan")
        print(f"{mode:<10} {bk['docs']:>5} "
              f"{bk['grades']['precise']/tot:>8.1%} {bk['grades']['vague']/tot:>7.1%} "
              f"{bk['grades']['wrong']/tot:>7.1%} {bk['faithful']/max(1,bk['docs']):>9.1%} "
              f"{bk['missed']/max(1,bk['docs']):>9.2f} {tag_rate:>8.1%}")
    if agg["ALL"]["parse_fail"]:
        print(f"(judge parse failures: {agg['ALL']['parse_fail']})")

if __name__ == "__main__":
    main()
