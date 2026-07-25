#!/usr/bin/env python3
"""Aggregate the faceted faithfulness judge results.

Reports, per conditioning mode and overall:
  subject: precise / vague / wrong shares, avg missed
  facet accuracy: type / audience / time / purpose (judge's facet_ok)
  flags: precision (flags_ok true / emitted)
  redundancy: avg redundant_pairs per entry
  entry_faithful: no wrong subject, missed==0, all facet_ok, all flags ok

Usage: score_facets.py --run facet [--pred data/preds_facet_faith.jsonl]
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
        print(f"judge batch: {b.get('status')} {c.get('completed',0)}/{c.get('total',0)}")
        return
    raw = requests.get(f"https://api.openai.com/v1/files/{b['output_file_id']}/content",
                       headers=H, timeout=600).text
    man = {json.loads(l)["i"]: json.loads(l)
           for l in open(f"{HERE}/faith_manifest.jsonl", encoding="utf-8")}

    agg = defaultdict(lambda: {"sub": Counter(), "missed": 0, "docs": 0, "faithful": 0,
                               "fok": Counter(), "ftot": Counter(),
                               "flags_ok": 0, "flags_tot": 0, "redund": 0, "fail": 0})
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
            grades = [g for g in d["subject_grades"] if g in ("precise", "vague", "wrong")]
            missed = max(0, min(3, int(d["missed"])))
            fok = d["facet_ok"]
            flags = [bool(x) for x in d.get("flags_ok", [])]
            red = max(0, int(d.get("redundant_pairs", 0)))
        except Exception:
            for bk in buckets:
                bk["fail"] += 1
            continue
        ok_all = all(bool(fok.get(k, False)) for k in ("type", "audience", "time", "purpose"))
        faithful = ("wrong" not in grades and missed == 0 and ok_all and all(flags))
        for bk in buckets:
            bk["docs"] += 1
            bk["sub"].update(grades)
            bk["missed"] += missed
            for k in ("type", "audience", "time", "purpose"):
                bk["ftot"][k] += 1
                bk["fok"][k] += bool(fok.get(k, False))
            bk["flags_ok"] += sum(flags)
            bk["flags_tot"] += len(flags)
            bk["redund"] += red
            bk["faithful"] += faithful

    order = ["ALL", "canonical", "none", "unseen"]
    print(f"{'mode':<10} {'docs':>5} {'sub_prec':>9} {'sub_vague':>9} {'sub_wrong':>9} "
          f"{'miss':>5} {'type':>6} {'aud':>6} {'time':>6} {'purp':>6} {'flags':>6} "
          f"{'redund':>7} {'faithful':>9}")
    for mode in order:
        if mode not in agg:
            continue
        bk = agg[mode]
        n = max(1, bk["docs"])
        stot = sum(bk["sub"].values()) or 1
        print(f"{mode:<10} {bk['docs']:>5} "
              f"{bk['sub']['precise']/stot:>9.1%} {bk['sub']['vague']/stot:>9.1%} "
              f"{bk['sub']['wrong']/stot:>9.1%} {bk['missed']/n:>5.2f} "
              + " ".join(f"{bk['fok'][k]/max(1,bk['ftot'][k]):>6.1%}"
                         for k in ("type", "audience", "time", "purpose"))
              + f" {bk['flags_ok']/max(1,bk['flags_tot']):>6.1%} "
              f"{bk['redund']/n:>7.2f} {bk['faithful']/n:>9.1%}")
    if agg["ALL"]["fail"]:
        print(f"(judge parse failures: {agg['ALL']['fail']})")

if __name__ == "__main__":
    main()
