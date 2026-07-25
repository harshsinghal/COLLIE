#!/usr/bin/env python3
"""Does the programmatic GRPO reward agree with the LLM judge?

Training on a proxy is only honest if the proxy tracks the thing we care
about. This scores every existing prediction with collie_reward and compares
against the judge's verdicts for the same documents (Spearman + group means).
If the correlation is weak, the reward is not a usable training signal and we
should not spend GPU on it.

Usage: validate_reward.py --run ent --pred data/preds_ent_faith.jsonl
"""
import argparse, json, os, re, sys
import requests

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "train"))
from collie_reward import score_one  # noqa: E402

H = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--pred", required=True)
    a = ap.parse_args()

    data_dir = os.path.join(HERE, "data")
    st = json.load(open(f"{data_dir}/state_ov.json"))
    b = requests.get(f"https://api.openai.com/v1/batches/{st[f'faith_judge_{a.run}']['batch_id']}",
                     headers=H, timeout=30).json()
    raw = requests.get(f"https://api.openai.com/v1/files/{b['output_file_id']}/content",
                       headers=H, timeout=600).text
    man = {json.loads(l)["i"]: json.loads(l) for l in
           open(f"{data_dir}/faith_manifest.jsonl", encoding="utf-8")}
    preds = {json.loads(l)["i"]: json.loads(l) for l in open(a.pred, encoding="utf-8")}

    rows = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        i = int(r["custom_id"].split("-")[1])
        try:
            content = r["response"]["body"]["choices"][0]["message"]["content"]
            d = json.loads(re.search(r'\{.*\}', content, re.S).group(0))
            grades = [g for g in d["subject_grades"] if g in ("precise", "vague", "wrong")]
            missed = max(0, min(3, int(d["missed"])))
            fok = d["facet_ok"]
        except Exception:
            continue
        if not grades or i not in preds:
            continue
        card_json = json.dumps({k: preds[i].get(k) for k in
                                ("subject", "type", "audience", "time", "purpose",
                                 "content_flags")})
        rew = score_one(card_json, man[i]["text"], detail=True)
        judge_quality = (sum(g == "precise" for g in grades) / len(grades)
                         - sum(g == "wrong" for g in grades) / len(grades))
        faithful = ("wrong" not in grades and missed == 0
                    and all(bool(fok.get(k, False)) for k in
                            ("type", "audience", "time", "purpose")))
        rows.append({"reward": rew.get("total", 0.0),
                     "grounding": rew.get("grounding", 0.0),
                     "nonredundant": rew.get("nonredundant", 0.0),
                     "register": rew.get("register", 0.0),
                     "facet_prior": rew.get("facet_prior", 0.0),
                     "judge_quality": judge_quality, "faithful": float(faithful),
                     "wrong_rate": sum(g == "wrong" for g in grades) / len(grades)})

    n = len(rows)
    print(f"paired documents: {n}\n")
    print("Spearman correlation with judge:")
    for comp in ("reward", "grounding", "nonredundant", "register", "facet_prior"):
        rq = spearman([r[comp] for r in rows], [r["judge_quality"] for r in rows])
        rf = spearman([r[comp] for r in rows], [r["faithful"] for r in rows])
        print(f"  {comp:<13} vs judge_quality {rq:+.3f}   vs faithful {rf:+.3f}")

    rows.sort(key=lambda r: r["reward"])
    q = max(1, n // 4)
    print("\nreward quartile -> what the judge said:")
    for name, sl in (("bottom 25%", rows[:q]), ("2nd", rows[q:2 * q]),
                     ("3rd", rows[2 * q:3 * q]), ("top 25%", rows[3 * q:])):
        m = len(sl) or 1
        print(f"  {name:<10} reward {sum(r['reward'] for r in sl)/m:.3f}  "
              f"judge_quality {sum(r['judge_quality'] for r in sl)/m:+.3f}  "
              f"wrong {sum(r['wrong_rate'] for r in sl)/m:.1%}  "
              f"faithful {sum(r['faithful'] for r in sl)/m:.1%}")


if __name__ == "__main__":
    main()
