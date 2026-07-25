#!/usr/bin/env python3
"""Harvest + gate the faceted relabel batch.

Survivor requirements:
  - <think> with >= 25 words
  - keyed JSON parses: subject (1-3 strings), type/audience/time/purpose
    (string or null), content_flags (list, 0-4 strings)
  - all values normalized to snake_case
Writes data/facet_clean.jsonl with doc text, anchor, think, and the entry.
"""
import gzip, json, os, re, requests
from collections import Counter

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
H = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
SINGLE = ("type", "audience", "time", "purpose")

def norm(s):
    s = re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")
    return s[:40]

def extract(content):
    m = re.search(r"<think>(.*?)</think>", content, re.S)
    think = m.group(1).strip() if m else ""
    if len(think.split()) < 25:
        return None, "no_think"
    body = re.sub(r"<think>.*?</think>", "", content, flags=re.S)
    body = re.sub(r"```(?:json)?|```", "", body).strip()
    m = re.search(r'\{\s*"subject"\s*:.*\}', body, re.S)
    if not m:
        return None, "no_json"
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None, "bad_json"
    subj = [norm(t) for t in (d.get("subject") or []) if str(t).strip()]
    subj = [t for t in dict.fromkeys(subj) if t]
    if not 1 <= len(subj) <= 3:
        return None, f"subject_count:{len(subj)}"
    entry = {"subject": subj}
    for k in SINGLE:
        v = d.get(k)
        if v is None or (isinstance(v, str) and not v.strip()):
            entry[k] = None
        elif isinstance(v, str):
            entry[k] = norm(v)
        else:
            return None, f"bad_{k}"
    flags = d.get("content_flags") or []
    if not isinstance(flags, list) or len(flags) > 4:
        return None, "bad_flags"
    entry["content_flags"] = [norm(t) for t in flags if str(t).strip()][:4]
    return think, entry

def main():
    docs = {}
    for l in gzip.open(f"{HERE}/manifest_2k.jsonl.gz", "rt", encoding="utf-8"):
        d = json.loads(l); docs[d["i"]] = d
    for l in open(f"{HERE}/manifest_more.jsonl", encoding="utf-8"):
        d = json.loads(l); docs[d["i"]] = d
    anch = {}
    for fn in ("anchors_2k.jsonl", "anchors_more.jsonl"):
        for l in open(f"{HERE}/{fn}", encoding="utf-8"):
            a = json.loads(l); anch[a["i"]] = a
    st = json.load(open(f"{HERE}/state_ov.json"))
    b = requests.get(f"https://api.openai.com/v1/batches/{st['facet']['batch_id']}",
                     headers=H, timeout=30).json()
    if b.get("status") != "completed":
        print(f"batch: {b.get('status')}"); return
    raw = requests.get(f"https://api.openai.com/v1/files/{b['output_file_id']}/content",
                       headers=H, timeout=600).text
    surv, rej = [], Counter()
    for line in raw.splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        i = int(r["custom_id"].split("-")[1])
        try:
            content = r["response"]["body"]["choices"][0]["message"]["content"] or ""
        except Exception:
            rej["no_response"] += 1; continue
        think, entry = extract(content)
        if think is None:
            rej[entry] += 1; continue
        d = docs[i]; a = anch.get(i, {"regime": "none", "anchor": None})
        surv.append({"i": i, "src": d["src"], "text": d["text"],
                     "regime": a["regime"], "anchor": a["anchor"],
                     "think": think, "entry": entry})
    surv.sort(key=lambda x: x["i"])
    with open(f"{HERE}/facet_clean.jsonl", "w", encoding="utf-8") as f:
        for s in surv:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"survivors={len(surv)}/{st['facet']['n']}")
    print("rejects:", dict(rej.most_common(8)))
    print("by src:", dict(Counter(s["src"] for s in surv)))
    nulls = Counter()
    for s in surv:
        for k in SINGLE:
            if s["entry"][k] is None:
                nulls[k] += 1
    print("null facets:", dict(nulls))
    flags = Counter(fl for s in surv for fl in s["entry"]["content_flags"])
    print("top content_flags:", flags.most_common(10))

if __name__ == "__main__":
    main()
