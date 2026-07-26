#!/usr/bin/env python3
"""Teacher-vs-student DPO pairs.

Self-play DPO hit a ceiling: every candidate came from the student, so the
'chosen' side was capped by what the student already knows. Here the chosen
side is the TEACHER's card for the same document — knowledge from outside the
student's distribution — and the rejected side is the student's own greedy
answer. DPO's contrast then actively pushes down on the specific wrong answer
the student would otherwise give, which plain SFT on the same cards cannot do.

Pairs are dropped when they would teach nothing or teach noise:
  identical      student already matches the teacher
  malformed      student card doesn't parse (trivial contrast, wastes capacity)
  not_worse      student is at least as grounded as the teacher (validated
                 proxy) — no evidence the teacher card is actually better

Usage: build_ts_pairs.py --teacher ent_clean.jsonl --student student_cards.jsonl
                         --out ts_pairs.jsonl [--cap 3000]
"""
import argparse, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "train"))
from collie_reward import parse_card, grounding  # noqa: E402

KEYS = ("subject", "type", "audience", "time", "purpose", "content_flags")


def card_json(d):
    return json.dumps({k: d.get(k) for k in KEYS}, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--system", required=True)
    ap.add_argument("--cap", type=int, default=3000)
    a = ap.parse_args()

    system = open(a.system, encoding="utf-8").read()
    teacher = {}
    for l in open(a.teacher, encoding="utf-8"):
        r = json.loads(l)
        teacher[r["i"]] = r
    student = {json.loads(l)["i"]: json.loads(l) for l in open(a.student, encoding="utf-8")}

    pairs, skip = [], {"identical": 0, "malformed": 0, "not_worse": 0, "missing": 0}
    for i, t in teacher.items():
        s = student.get(i)
        if s is None:
            skip["missing"] += 1; continue
        t_json = card_json(t["entry"])
        s_json = card_json(s)
        if t_json == s_json:
            skip["identical"] += 1; continue
        if parse_card(s_json) is None:
            skip["malformed"] += 1; continue
        doc = t["text"]
        t_card, s_card = parse_card(t_json), parse_card(s_json)
        if t_card is None:
            skip["malformed"] += 1; continue
        if grounding(s_card, doc) > grounding(t_card, doc) + 0.05:
            skip["not_worse"] += 1; continue
        pairs.append({"prompt": [{"role": "system", "content": system},
                                 {"role": "user", "content": "Document:\n" + doc}],
                      "chosen": [{"role": "assistant", "content": t_json}],
                      "rejected": [{"role": "assistant", "content": s_json}],
                      "i": i})
    pairs = pairs[:a.cap]
    with open(a.out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"teacher-vs-student pairs: {len(pairs)}   skipped: {skip}", flush=True)


if __name__ == "__main__":
    main()
