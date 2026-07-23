#!/usr/bin/env python3
"""Download COLLIE GPT batch results, apply the ontology filter, merge.

Same hard gate as the OpenRouter path: a survivor needs a <think> block with
>=MIN_THINK_WORDS, parseable JSON labels, and only in-ontology topics/facets.
Idempotent: rebuilds reason_clean_gpt.jsonl from whatever batches are complete.
"""
import json, os, re, sys, requests
from collections import Counter

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MANIFEST = f"{HERE}/manifest_2k.jsonl"
OUT = f"{HERE}/reason_clean_gpt.jsonl"
H = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
MIN_THINK_WORDS = 25

ONT = {"compensation","workforce","mergers_acquisitions","financials","pricing","legal",
       "security","credentials","product","strategy","competition","personnel",
       "regulatory","customer_data"}
FVALS = {"scope":{"individual","group","aggregate","org_wide"},
         "publicity":{"public","internal","restricted"},
         "temporality":{"historical","current","forward_looking"},
         "specificity":{"named","figures","both","general"},
         "register_facet":{"report","negotiation","decision","directive",
                            "request","mention","speculation"}}

def extract(content):
    inline = re.search(r"<think>(.*?)</think>", content, re.S)
    think = inline.group(1).strip() if inline else ""
    if len(think.split()) < MIN_THINK_WORDS:
        return None, "no_think"
    body = re.sub(r"<think>.*?</think>", "", content, flags=re.S)
    body = re.sub(r"```(?:json)?|```", "", body).strip()
    m = re.search(r'\{\s*"labels"\s*:.*\}', body, re.S)
    if not m:
        return None, "no_json"
    try:
        labels = json.loads(m.group(0))["labels"]
    except Exception:
        return None, "bad_json"
    for l in labels:
        if l.get("topic") not in ONT:
            return None, f"off_topic:{l.get('topic')}"
        for fk, vals in FVALS.items():
            if l.get(fk) not in vals:
                return None, f"off_facet:{fk}={l.get(fk)}"
    return think, labels

def main():
    man = {json.loads(l)["i"]: json.loads(l) for l in open(MANIFEST, encoding="utf-8")}
    st = json.load(open(f"{HERE}/state.json"))
    survivors, rej = [], Counter()
    persrc_model = Counter()
    for tier in ("base", "boost"):
        key = f"gpt_{tier}"
        if key not in st:
            continue
        b = requests.get(f"https://api.openai.com/v1/batches/{st[key]['batch_id']}",
                         headers=H, timeout=30).json()
        if b.get("status") != "completed" or not b.get("output_file_id"):
            print(f"{tier}: {b.get('status')} — skipping", flush=True)
            continue
        model = st[key]["model"]
        raw = requests.get(f"https://api.openai.com/v1/files/{b['output_file_id']}/content",
                           headers=H, timeout=180).text
        for line in raw.splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            i = int(r["custom_id"].split("-")[1])
            try:
                content = r["response"]["body"]["choices"][0]["message"]["content"]
            except Exception:
                rej["no_response"] += 1; continue
            think, labels = extract(content or "")
            if think is None:
                rej[labels] += 1; continue
            d = man[i]
            survivors.append({"i": i, "src": d["src"], "tier": tier, "id": d["id"],
                              "text": d["text"], "model": model,
                              "think": think, "labels": labels})
            persrc_model[model] += 1
    survivors.sort(key=lambda x: x["i"])
    with open(OUT, "w", encoding="utf-8") as f:
        for s in survivors:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"survivors={len(survivors)} -> {OUT}")
    print("by teacher:", dict(persrc_model))
    print("rejects:", dict(rej.most_common(12)))

if __name__ == "__main__":
    main()
