#!/usr/bin/env python3
"""Faceted-librarian relabel: keyed catalog entry, open vocabulary per facet.

Each document gets ONE answer per facet (NIST/ISO-style faceted classification):
  subject (1-3), type (1), audience (1), time (1), purpose (1),
  content_flags (0-3, descriptive presence only).
All values open-vocabulary. Anchor conditioning (5 regimes, reused from the
saved per-doc assignments) applies to the SUBJECT facet as suggested
vocabulary. Mandatory concise <think>. Single teacher: gpt-5.4-mini.

Covers both manifests (2,480 + 3,000 docs). Submits one batch.
"""
import gzip, json, os, requests

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
H = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"}

def sys_prompt_facets(anchor):
    base = (
        "You are COLLIE, an enterprise librarian. Read the document and produce a faceted catalog "
        "entry — one answer per facet, in the spirit of NIST/ISO data-classification schemes. "
        "Describe; never judge sensitivity or importance.\n\n"
        "You MUST reason first inside <think>...</think> — REQUIRED for every document, even short "
        "ones. Keep it CONCISE: 40-120 words, citing the words/context that resolve each facet. "
        "Then, AFTER </think>, output STRICT JSON on its own line:\n"
        '{"subject":[...],"type":"...","audience":"...","time":"...","purpose":"...","content_flags":[...]}\n\n'
        "Facet rules — every value is open vocabulary (short snake_case), and each facet answers a "
        "DIFFERENT question; never repeat the same information in two facets:\n"
        "- subject: 1-3 terms naming what the document is about. ")
    if anchor:
        base += ("Prefer these catalog subjects when they genuinely fit, coin your own when they "
                 "don't: " + ", ".join(anchor) + "\n")
    else:
        base += "Coin the subjects yourself; there is no fixed catalog.\n"
    base += (
        "- type: what kind of artifact this is (e.g. email_thread, incident_log, source_code, "
        "contract_draft, chat_message, technical_report — coin as needed).\n"
        "- audience: who it is for (e.g. internal_team, public, external_counsel, customers).\n"
        "- time: orientation (e.g. historical, current, forward_looking).\n"
        "- purpose: the act it performs (e.g. request, report, negotiation, decision, instruction, "
        "announcement, speculation).\n"
        "- content_flags: 0-3 DESCRIPTIVE presence flags only (e.g. contains_credentials, "
        "personal_pii, financial_figures, legal_terms, code_snippets). State what is present; "
        "make no sensitivity judgment. Empty list if none apply.\n"
        "Use null for a single-value facet that is genuinely undeterminable from the text.")
    return base

def load_docs():
    docs = []
    for l in gzip.open(f"{HERE}/manifest_2k.jsonl.gz", "rt", encoding="utf-8"):
        docs.append(json.loads(l))
    for l in open(f"{HERE}/manifest_more.jsonl", encoding="utf-8"):
        docs.append(json.loads(l))
    anch = {}
    for fn in ("anchors_2k.jsonl", "anchors_more.jsonl"):
        for l in open(f"{HERE}/{fn}", encoding="utf-8"):
            a = json.loads(l)
            anch[a["i"]] = a
    return docs, anch

def main():
    docs, anch = load_docs()
    path = f"{HERE}/facet_batch.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            a = anch.get(d["i"], {"anchor": None})
            f.write(json.dumps({
                "custom_id": f"FC-{d['i']:05d}", "method": "POST", "url": "/v1/chat/completions",
                "body": {"model": "gpt-5.4-mini",
                         "messages": [{"role": "system", "content": sys_prompt_facets(a["anchor"])},
                                      {"role": "user", "content": "Document:\n" + d["text"]}],
                         "max_completion_tokens": 700}}) + "\n")
    n = sum(1 for _ in open(path))
    fid = requests.post("https://api.openai.com/v1/files", headers=H,
                        files={"file": ("facet_batch.jsonl", open(path, "rb"))},
                        data={"purpose": "batch"}).json()["id"]
    b = requests.post("https://api.openai.com/v1/batches", headers=H,
                      json={"input_file_id": fid, "endpoint": "/v1/chat/completions",
                            "completion_window": "24h"}).json()
    print(f"facet batch ({n} reqs): {b.get('id')} {b.get('status')}")
    st = json.load(open(f"{HERE}/state_ov.json"))
    st["facet"] = {"batch_id": b.get("id"), "n": n}
    json.dump(st, open(f"{HERE}/state_ov.json", "w"), indent=2)

if __name__ == "__main__":
    main()
