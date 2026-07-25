#!/usr/bin/env python3
"""Ask the judge to RANK the K candidate cards for each document — one call
per document, not one per candidate (4x cheaper, and direct comparison gives
more consistent preferences than independent scoring).

The judge ranks on exactly the qualities the faithfulness eval measures:
subjects true and specific to the document, nothing important missed, the
single-valued facets correct, flags actually present, no repetition.
"""
import argparse, json, os, requests

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
H = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}

PROMPT = """You are choosing the best catalog entry for an enterprise document.
Below are {k} candidate entries produced for the SAME document. Judge them against the
document only — there is no reference answer.

A better entry:
- has subjects that are TRUE of this document and SPECIFIC to it (not generic filler),
  phrased the way an enterprise would file it
- misses no major subject the document clearly discusses
- gets type / audience / time / purpose right for this document
- lists content_flags that are actually present
- does not say the same thing twice in different words

Document:
{doc}

Candidates:
{cands}

Rank ALL candidate indices from best to worst, and give each a quality score 0-10.
Output STRICT JSON only:
{{"ranking": [best_index, ..., worst_index], "scores": [score_for_index_0, ...]}}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True)
    ap.add_argument("--run", default="dpo")
    a = ap.parse_args()
    man = {json.loads(l)["i"]: json.loads(l)
           for l in open(f"{HERE}/dpo_manifest.jsonl", encoding="utf-8")}
    lines = []
    for l in open(a.cands, encoding="utf-8"):
        r = json.loads(l)
        cands = r["candidates"]
        if len(set(cands)) < 2:          # identical samples carry no preference
            continue
        listing = "\n".join(f"[{k}] {c}" for k, c in enumerate(cands))
        lines.append(json.dumps({
            "custom_id": f"RK-{r['i']:05d}", "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": "gpt-5.4-mini",
                     "messages": [{"role": "user", "content": PROMPT.format(
                         k=len(cands), doc=man[r["i"]]["text"][:3000], cands=listing)}],
                     "max_completion_tokens": 300}}) + "\n")
    path = f"{HERE}/rank_batch_{a.run}.jsonl"
    open(path, "w", encoding="utf-8").writelines(lines)
    print(f"ranking requests: {len(lines)}")
    fid = requests.post("https://api.openai.com/v1/files", headers=H,
                        files={"file": (os.path.basename(path), open(path, "rb"))},
                        data={"purpose": "batch"}).json()["id"]
    b = requests.post("https://api.openai.com/v1/batches", headers=H,
                      json={"input_file_id": fid, "endpoint": "/v1/chat/completions",
                            "completion_window": "24h"}).json()
    print(f"rank batch: {b.get('id')} {b.get('status')}")
    st = json.load(open(f"{HERE}/state_ov.json"))
    st[f"rank_{a.run}"] = {"batch_id": b.get("id"), "n": len(lines)}
    json.dump(st, open(f"{HERE}/state_ov.json", "w"), indent=2)


if __name__ == "__main__":
    main()
