#!/usr/bin/env python3
"""Reference-free faithfulness eval, part 2: build + submit the judge batch.

For each doc, the judge (gpt-5.4-mini, Batch API) reads the DOCUMENT and the
model's predicted topics/tags — no gold labels anywhere — and grades three
failure modes:
  - per emitted topic: precise / vague / wrong   (vague = true but so generic
    it barely narrows the doc down, e.g. "business_communication" on everything)
  - missed: how many major subjects the doc clearly discusses that the
    prediction omits (0-3)
  - tags: how many emitted tags aptly characterize the discussion

Usage: build_faith_judge_batch.py --pred data/preds_faith_06.jsonl --run 06
"""
import argparse, json, os, requests

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
H = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}

PROMPT = """You are auditing a document-cataloging model. Read the document and judge whether
the model's catalog entry describes it faithfully. No reference answer exists — judge against
the document alone.

Document:
{doc}

Model's catalog entry:
topics: {topics}
tags: {tags}

Grade:
1. For EACH topic, classify: "precise" (names a real subject of this document specifically),
   "vague" (true but so generic it barely narrows the document down), or "wrong" (not a
   subject of this document).
2. missed: count (0-3) of major subjects the document clearly discusses that the topics
   list omits. Cap at 3.
3. tags_apt: how many of the emitted tags aptly characterize how the content is discussed.

Output STRICT JSON only:
{{"topic_grades": ["precise"|"vague"|"wrong", ...], "missed": <int>, "tags_apt": <int>}}"""

PROMPT_TAGS = """You are auditing a document-tagging model. Read the document and judge whether
the model's tag set describes it faithfully. No reference answer exists — judge against the
document alone.

Document:
{doc}

Model's tags: {tags}

Grade:
1. For EACH tag, classify: "precise" (specifically true of this document — a real subject it
   discusses or an accurate descriptor of how it is discussed), "vague" (true but so generic it
   barely narrows the document down), or "wrong" (not true of this document).
2. missed: count (0-3) of major subjects the document clearly discusses that NO tag reflects.
   Cap at 3.

Output STRICT JSON only:
{{"topic_grades": ["precise"|"vague"|"wrong", ...], "missed": <int>, "tags_apt": 0}}"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--run", required=True, help="short tag, e.g. 06")
    ap.add_argument("--tags-only", action="store_true",
                    help="grade a flat tag set (no topic/tag split)")
    a = ap.parse_args()
    man = {json.loads(l)["i"]: json.loads(l)
           for l in open(f"{HERE}/faith_manifest.jsonl", encoding="utf-8")}
    lines = []
    for l in open(a.pred, encoding="utf-8"):
        p = json.loads(l)
        m = man[p["i"]]
        if a.tags_only:
            content = PROMPT_TAGS.format(doc=m["text"][:3500],
                                         tags=json.dumps(p.get("tags", [])))
        else:
            content = PROMPT.format(doc=m["text"][:3500],
                                    topics=json.dumps(p.get("topics", [])),
                                    tags=json.dumps(p.get("tags", [])))
        lines.append(json.dumps({
            "custom_id": f"FJ{a.run}-{p['i']:05d}",
            "method": "POST", "url": "/v1/chat/completions",
            "body": {"model": "gpt-5.4-mini",
                     "messages": [{"role": "user", "content": content}],
                     "max_completion_tokens": 300}}))
    path = f"{HERE}/faith_judge_batch_{a.run}.jsonl"
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    fid = requests.post("https://api.openai.com/v1/files", headers=H,
                        files={"file": (os.path.basename(path), open(path, "rb"))},
                        data={"purpose": "batch"}).json()["id"]
    b = requests.post("https://api.openai.com/v1/batches", headers=H,
                      json={"input_file_id": fid, "endpoint": "/v1/chat/completions",
                            "completion_window": "24h"}).json()
    print(f"judge batch ({len(lines)} reqs): {b.get('id')} {b.get('status')}")
    st = json.load(open(f"{HERE}/state_ov.json"))
    st[f"faith_judge_{a.run}"] = {"batch_id": b.get("id"), "n": len(lines)}
    json.dump(st, open(f"{HERE}/state_ov.json", "w"), indent=2)

if __name__ == "__main__":
    main()
