#!/usr/bin/env python3
"""Open-vocabulary relabel: teacher thinks, then emits free {"topics","tags"}.

Anchor-conditioning: each doc gets one of five anchor regimes (deterministic by
doc-id hash) — canonical 14 / random subset / paraphrase variant / alternative
domain vocabulary / no anchor. The anchor is a SOFT preference ("use when they
fit, otherwise coin your own"), never a constraint. The per-doc anchor is saved
to data/anchors_2k.jsonl so SFT assembly puts the SAME anchor in the student
prompt — the model learns to condition on whatever catalog it is handed.

Two OpenAI batches (API allows one model per batch): gpt-5.4-mini for
tier=base, gpt-5.5 for tier=boost (rare-topic tail).
"""
import gzip, hashlib, json, os, requests

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
H = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
POOL = json.load(open(f"{HERE}/anchor_pool.json"))
TEACHER = {"base": "gpt-5.4-mini", "boost": "gpt-5.5"}

TAG_HINT = ("individual, group, aggregate, org_wide, public, internal, restricted, historical, "
            "current, forward_looking, named, figures, general, report, negotiation, decision, "
            "directive, request, mention, speculation")

def hkey(s):
    return int(hashlib.sha256(s.encode()).hexdigest(), 16)

def anchor_for(i):
    """Deterministic regime + anchor list for doc i. Returns (regime, list|None)."""
    h = hkey(f"anchor-{i}")
    r = h % 100
    if r < 25:
        return "canonical", POOL["canonical"]
    if r < 45:  # subset of 6-9 canonical topics
        k = 6 + (h // 100) % 4
        idx = sorted(range(14), key=lambda j: hkey(f"sub-{i}-{j}"))[:k]
        return "subset", [POOL["canonical"][j] for j in sorted(idx)]
    if r < 65:
        v = POOL["paraphrases"][(h // 100) % len(POOL["paraphrases"])]
        return "paraphrase", v
    if r < 85:
        a = POOL["alternatives"][(h // 100) % len(POOL["alternatives"])]
        return "alternative", a["topics"]
    return "none", None

def sys_prompt(anchor):
    base = (
        "You are COLLIE, a librarian for enterprise documents. Read the document and catalog it: "
        "identify the topic(s) it genuinely discusses and assign descriptive tags. Describe, do not "
        "judge sensitivity.\n\n"
        "You MUST reason first inside <think>...</think> — REQUIRED for every document, even short "
        "ones. Keep it CONCISE: 40-120 words. Inside <think>: name candidate topics, cite the specific "
        "words/context that confirm or reject each, and choose tags from the evidence. Then, AFTER "
        "</think>, output STRICT JSON on its own line: {\"topics\":[...],\"tags\":[...]}.\n\n"
        "topics: 1-4 short snake_case noun phrases naming the subjects discussed. ")
    if anchor:
        base += ("Prefer these catalog topics when they genuinely fit: " + ", ".join(anchor) +
                 ". If the content is not covered by the catalog, coin your own coherent topic "
                 "instead of forcing a bad fit.\n")
    else:
        base += "Coin coherent topics yourself; there is no fixed catalog.\n"
    base += ("tags: 2-8 short descriptors characterizing how the topics are discussed (who/scope, "
             "audience/publicity, time orientation, specificity, speech act). Prefer these when they "
             f"fit: {TAG_HINT}. Coin others when needed.")
    return base

def main():
    docs = [json.loads(l) for l in gzip.open(f"{HERE}/manifest_2k.jsonl.gz", "rt", encoding="utf-8")]
    anchors_out = open(f"{HERE}/anchors_2k.jsonl", "w", encoding="utf-8")
    lines = {"base": [], "boost": []}
    from collections import Counter
    regimes = Counter()
    for d in docs:
        regime, anchor = anchor_for(d["i"])
        regimes[regime] += 1
        anchors_out.write(json.dumps({"i": d["i"], "regime": regime, "anchor": anchor}) + "\n")
        lines[d["tier"]].append(json.dumps({
            "custom_id": f"OV-{d['i']:04d}",
            "method": "POST", "url": "/v1/chat/completions",
            "body": {"model": TEACHER[d["tier"]],
                     "messages": [{"role": "system", "content": sys_prompt(anchor)},
                                  {"role": "user", "content": "Document:\n" + d["text"]}],
                     "max_completion_tokens": 700}}))
    anchors_out.close()
    print("regimes:", dict(regimes))
    st_path = f"{HERE}/state_ov.json"
    st = {}
    for tier, ls in lines.items():
        path = f"{HERE}/ov_{tier}_batch.jsonl"
        open(path, "w", encoding="utf-8").write("\n".join(ls) + "\n")
        fid = requests.post("https://api.openai.com/v1/files", headers=H,
                            files={"file": (os.path.basename(path), open(path, "rb"))},
                            data={"purpose": "batch"}).json()["id"]
        b = requests.post("https://api.openai.com/v1/batches", headers=H,
                          json={"input_file_id": fid, "endpoint": "/v1/chat/completions",
                                "completion_window": "24h"}).json()
        print(f"{tier} ({TEACHER[tier]}, {len(ls)} docs): {b.get('id')} {b.get('status')}")
        st[tier] = {"batch_id": b.get("id"), "model": TEACHER[tier], "n": len(ls)}
    json.dump(st, open(st_path, "w"), indent=2)

if __name__ == "__main__":
    main()
