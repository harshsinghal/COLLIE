#!/usr/bin/env python3
"""Download + filter the open-vocab relabel batches.

Gate (relaxed vs the ontology round — there is no fixed vocabulary now):
  - <think> present with >= MIN_THINK_WORDS words
  - strict-JSON {"topics":[...], "tags":[...]} parses
  - 0 < len(topics) <= 6, all short strings; 0 < len(tags) <= 12
Topics/tags are normalized to lowercase snake_case. Survivors merge with the
per-doc anchor (regime + list) so SFT assembly can reproduce the conditioning.
"""
import json, os, re, requests
from collections import Counter

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
H = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
MIN_THINK_WORDS = 25

def norm(s):
    s = re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")
    return s[:40]

def extract(content):
    m = re.search(r"<think>(.*?)</think>", content, re.S)
    think = m.group(1).strip() if m else ""
    if len(think.split()) < MIN_THINK_WORDS:
        return None, "no_think"
    body = re.sub(r"<think>.*?</think>", "", content, flags=re.S)
    body = re.sub(r"```(?:json)?|```", "", body).strip()
    m = re.search(r'\{\s*"topics"\s*:.*\}', body, re.S)
    if not m:
        return None, "no_json"
    try:
        d = json.loads(m.group(0))
        topics = [norm(t) for t in d["topics"] if str(t).strip()]
        tags = [norm(t) for t in d.get("tags", []) if str(t).strip()]
    except Exception:
        return None, "bad_json"
    topics = [t for t in dict.fromkeys(topics) if t]
    tags = [t for t in dict.fromkeys(tags) if t]
    if not 0 < len(topics) <= 6:
        return None, f"topics_count:{len(topics)}"
    if not 0 < len(tags) <= 12:
        return None, f"tags_count:{len(tags)}"
    return think, {"topics": topics, "tags": tags}

def main():
    import gzip
    man = {json.loads(l)["i"]: json.loads(l)
           for l in gzip.open(f"{HERE}/manifest_2k.jsonl.gz", "rt", encoding="utf-8")}
    anch = {json.loads(l)["i"]: json.loads(l) for l in open(f"{HERE}/anchors_2k.jsonl", encoding="utf-8")}
    st = json.load(open(f"{HERE}/state_ov.json"))
    survivors, rej = [], Counter()
    for tier in ("base", "boost"):
        b = requests.get(f"https://api.openai.com/v1/batches/{st[tier]['batch_id']}",
                         headers=H, timeout=30).json()
        if b.get("status") != "completed" or not b.get("output_file_id"):
            print(f"{tier}: {b.get('status')} — skipping")
            continue
        raw = requests.get(f"https://api.openai.com/v1/files/{b['output_file_id']}/content",
                           headers=H, timeout=180).text
        for line in raw.splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            i = int(r["custom_id"].split("-")[1])
            try:
                content = r["response"]["body"]["choices"][0]["message"]["content"] or ""
            except Exception:
                rej["no_response"] += 1; continue
            think, parsed = extract(content)
            if think is None:
                rej[parsed] += 1; continue
            d, a = man[i], anch[i]
            survivors.append({"i": i, "src": d["src"], "tier": tier, "id": d["id"],
                              "text": d["text"], "model": st[tier]["model"],
                              "regime": a["regime"], "anchor": a["anchor"],
                              "think": think, **parsed})
    survivors.sort(key=lambda x: x["i"])
    out = f"{HERE}/ov_clean.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for s in survivors:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"survivors={len(survivors)} -> {out}")
    print("by regime:", dict(Counter(s["regime"] for s in survivors)))
    print("by src:", dict(Counter(s["src"] for s in survivors)))
    print("rejects:", dict(rej.most_common(10)))

if __name__ == "__main__":
    main()
