#!/usr/bin/env python3
"""Build + submit two OpenAI batches for COLLIE v2 labeling.

Single-teacher-per-batch (Batch API requires it): gpt-5.5 labels the rare-topic
tail (tier=boost), gpt-5.4-mini labels the bulk (tier=base). Concise
mandatory-<think> prompt (40-120 words) so traces are distillable into 0.6B
with no truncation. Survivors are filtered downstream by the same ontology gate.
"""
import json, os, requests

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MANIFEST = f"{HERE}/manifest_2k.jsonl"
H = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}

TOPICS = ("compensation, workforce, mergers_acquisitions, financials, pricing, legal, security, "
          "credentials, product, strategy, competition, personnel, regulatory, customer_data")
FACET_SPEC = ("scope=individual|group|aggregate|org_wide; publicity=public|internal|restricted; "
              "temporality=historical|current|forward_looking; specificity=named|figures|both|general; "
              "register_facet=report|negotiation|decision|directive|request|mention|speculation")

SYS = (
    "You are COLLIE, a document cataloger. Read the document and decide which topics it genuinely "
    "discusses and how, using ONLY the ontology below. Describe, do not judge sensitivity.\n\n"
    "You MUST reason first inside <think>...</think> — REQUIRED for every document, even short ones. "
    "Keep it CONCISE: 40-120 words. Inside <think>: name the candidate subjects, cite the specific "
    "words/context that confirm or reject each, and resolve each facet from evidence. Then, AFTER "
    "</think>, output STRICT JSON on its own line: {\"labels\":[{\"topic\":\"<t>\",\"scope\":\"<v>\","
    "\"publicity\":\"<v>\",\"temporality\":\"<v>\",\"specificity\":\"<v>\",\"register_facet\":\"<v>\"}]} "
    "(empty list if no topic applies).\n\n"
    f"Topics: {TOPICS}\nFacets (per topic): {FACET_SPEC}")

TEACHER = {"base": "gpt-5.4-mini", "boost": "gpt-5.5"}

def build(tier):
    model = TEACHER[tier]
    lines = []
    for l in open(MANIFEST, encoding="utf-8"):
        d = json.loads(l)
        if d["tier"] != tier:
            continue
        lines.append(json.dumps({
            "custom_id": f"C-{d['i']:04d}",
            "method": "POST", "url": "/v1/chat/completions",
            "body": {"model": model,
                     "messages": [{"role": "system", "content": SYS},
                                  {"role": "user", "content": "Document:\n" + d["text"]}],
                     "max_completion_tokens": 700}}))
    path = f"{HERE}/gpt_{tier}_batch.jsonl"
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    return path, model, len(lines)

def submit(path):
    fid = requests.post("https://api.openai.com/v1/files", headers=H,
                        files={"file": (os.path.basename(path), open(path, "rb"))},
                        data={"purpose": "batch"}).json()["id"]
    b = requests.post("https://api.openai.com/v1/batches", headers=H,
                      json={"input_file_id": fid, "endpoint": "/v1/chat/completions",
                            "completion_window": "24h"}).json()
    return b["id"], b["status"]

def main():
    state_path = f"{HERE}/state.json"
    st = json.load(open(state_path)) if os.path.exists(state_path) else {}
    for tier in ("base", "boost"):
        path, model, n = build(tier)
        bid, status = submit(path)
        print(f"{tier} ({model}, {n} docs): {bid} {status}")
        st[f"gpt_{tier}"] = {"batch_id": bid, "model": model, "n": n}
    json.dump(st, open(state_path, "w"), indent=2)

if __name__ == "__main__":
    main()
