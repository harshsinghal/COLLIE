#!/usr/bin/env python3
"""Reference-free faithfulness eval, part 1: sample fresh docs + build prompts.

Samples ~4,000 documents from all six registers that were NEVER used in any
prior round (train or eval), assigns each a conditioning mode —
  canonical (50%): the 14-topic catalog
  none      (25%): no catalog
  unseen    (25%): a holdout-domain vocabulary (education/gov/energy/media/biotech)
— and writes model-input prompt files for the trained COLLIE checkpoints.
No teacher labels are involved: the judge will later score outputs directly
against the documents.
"""
import gzip, hashlib, json, os

DATA = os.path.expanduser("~/workspace/ai_soc/dlp_bench/sources/data")
HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_openvocab_sft import sys_prompt
from sample_more import get_text, clean_ok, sha  # same extraction rules

FILES = {
    "enron.jsonl.gz": (1200, "enron"),
    "finepdfs_english_diverse_10k.jsonl.gz": (1200, "finepdfs"),
    "apache_public_raw/apache_enterprise_like_9k.jsonl.gz": (600, "apache"),
    "chat_like_public_raw/chat_like_public_9k.jsonl.gz": (500, "chat"),
    "github_code_clean_code.jsonl.gz": (300, "ghcode"),
    "loghub.jsonl.gz": (200, "loghub"),
}
CANON = ["compensation","workforce","mergers_acquisitions","financials","pricing","legal",
         "security","credentials","product","strategy","competition","personnel",
         "regulatory","customer_data"]

def _apache_text(r):
    import ast
    def _as(v):
        if isinstance(v,(dict,list)): return v
        try: return ast.literal_eval(v)
        except Exception: return None
    parts=[]
    iss=_as(r.get("raw_issue")) or {}
    f=iss.get("fields",{}) if isinstance(iss,dict) else {}
    if f.get("summary"): parts.append(f"Issue: {f['summary']}")
    d=f.get("description")
    if isinstance(d,str) and d.strip(): parts.append(d.strip())
    cm=_as(r.get("raw_comments")) or []
    for c in (cm[:6] if isinstance(cm,list) else []):
        b=c.get("body") if isinstance(c,dict) else None
        if isinstance(b,str) and b.strip(): parts.append(f"Comment: {b.strip()}")
    return "\n\n".join(parts)

def text_of(r, src):
    if src == "apache":
        return _apache_text(r)
    return get_text(r, src)

def hkey(s):
    return int(hashlib.sha256(s.encode()).hexdigest(), 16)

def main():
    used = set()
    for mf in ("manifest_2k.jsonl.gz", None):
        for l in gzip.open(f"{HERE}/manifest_2k.jsonl.gz", "rt", encoding="utf-8"):
            used.add(json.loads(l)["id"])
        break
    for l in open(f"{HERE}/manifest_more.jsonl", encoding="utf-8"):
        used.add(json.loads(l)["id"])
    hold = json.load(open(f"{HERE}/anchor_pool_holdout.json"))["vocabularies"]

    picked = []
    for fn, (quota, src) in FILES.items():
        rows = []
        for line in gzip.open(os.path.join(DATA, fn), "rt", encoding="utf-8"):
            r = json.loads(line)
            t = text_of(r, src)
            if not clean_ok(t) or r.get("id") in used:
                continue
            rows.append((r, t))
        rows.sort(key=lambda rt: hkey("faith-" + sha(rt[0])))
        for r, t in rows[:quota]:
            picked.append((src, r.get("id", ""), t))

    with open(f"{HERE}/faith_manifest.jsonl", "w", encoding="utf-8") as fm, \
         open(f"{HERE}/faith_infer.jsonl", "w", encoding="utf-8") as fi:
        from collections import Counter
        modes = Counter()
        for k, (src, rid, t) in enumerate(picked):
            h = hkey(f"mode-{k}") % 100
            if h < 50:
                mode, anchor = "canonical", CANON
            elif h < 75:
                mode, anchor = "none", None
            else:
                v = hold[hkey(f"fv-{k}") % len(hold)]
                mode, anchor = f"unseen:{v['domain']}", v["topics"]
            modes[mode.split(":")[0]] += 1
            fm.write(json.dumps({"i": k, "src": src, "id": rid, "mode": mode,
                                 "anchor": anchor, "text": t[:4000]},
                                ensure_ascii=False) + "\n")
            fi.write(json.dumps({"messages": [
                {"role": "system", "content": sys_prompt(anchor, False)},
                {"role": "user", "content": "Document:\n" + t[:4000]},
                {"role": "assistant", "content": ""}],
                "i": k}, ensure_ascii=False) + "\n")
        print(f"docs: {len(picked)}  modes: {dict(modes)}")
        print("by src:", dict(Counter(s for s, _, _ in picked)))

if __name__ == "__main__":
    main()
