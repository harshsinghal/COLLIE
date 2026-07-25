#!/usr/bin/env python3
"""Enterprise-librarian relabel: faceted card, NO catalog, NO anchors.

The subject facet is always free-coined, but steered toward the register an
enterprise would actually file documents under — business functions,
processes, artifacts — specific to the document. This is the SFT baseline;
a later GRPO stage sharpens 'enterprise-friendly' phrasing as a reward.

One prompt for all 5,480 docs (both manifests). Teacher: gpt-5.4-mini.
"""
import gzip, json, os, requests

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
H = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"}

SYS_ENTERPRISE = (
    "You are COLLIE, an enterprise librarian. Read the document and produce a faceted catalog "
    "entry — one answer per facet. Describe; never judge sensitivity or importance.\n\n"
    "You MUST reason first inside <think>...</think> — REQUIRED for every document, even short "
    "ones. Keep it CONCISE: 40-120 words, citing the words/context that resolve each facet. "
    "Then, AFTER </think>, output STRICT JSON on its own line:\n"
    '{"subject":[...],"type":"...","audience":"...","time":"...","purpose":"...","content_flags":[...]}\n\n'
    "Facet rules — each facet answers a DIFFERENT question; never repeat the same information in "
    "two facets:\n"
    "- subject: 1-3 short snake_case terms naming what the document is about, phrased in the "
    "vocabulary an enterprise would use to organize its documents — business functions, "
    "processes, and artifacts (e.g. vendor_contract_negotiation, incident_response, "
    "quarterly_revenue_forecast, employee_onboarding, database_performance_tuning). Be specific "
    "to THIS document; never use a generic term where a specific one fits.\n"
    "- type: what kind of artifact this is (e.g. email_thread, incident_log, source_code, "
    "contract_draft, chat_message, technical_report — coin as needed).\n"
    "- audience: who it is for (e.g. internal_team, public, external_counsel, customers).\n"
    "- time: exactly one of historical, current, forward_looking — pick the dominant orientation.\n"
    "- purpose: the single act it primarily performs (e.g. request, report, negotiation, "
    "decision, instruction, announcement, speculation) — one act, never two glued together.\n"
    "- content_flags: 0-3 DESCRIPTIVE presence flags only (e.g. contains_credentials, "
    "personal_pii, financial_figures, legal_terms, code_snippets). State what is present; make "
    "no sensitivity judgment. Empty list if none apply.\n"
    "Use null for a single-value facet that is genuinely undeterminable from the text.")

def main():
    docs = []
    for l in gzip.open(f"{HERE}/manifest_2k.jsonl.gz", "rt", encoding="utf-8"):
        docs.append(json.loads(l))
    for l in open(f"{HERE}/manifest_more.jsonl", encoding="utf-8"):
        docs.append(json.loads(l))
    path = f"{HERE}/ent_batch.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({
                "custom_id": f"EN-{d['i']:05d}", "method": "POST", "url": "/v1/chat/completions",
                "body": {"model": "gpt-5.4-mini",
                         "messages": [{"role": "system", "content": SYS_ENTERPRISE},
                                      {"role": "user", "content": "Document:\n" + d["text"]}],
                         "max_completion_tokens": 700}}) + "\n")
    fid = requests.post("https://api.openai.com/v1/files", headers=H,
                        files={"file": ("ent_batch.jsonl", open(path, "rb"))},
                        data={"purpose": "batch"}).json()["id"]
    b = requests.post("https://api.openai.com/v1/batches", headers=H,
                      json={"input_file_id": fid, "endpoint": "/v1/chat/completions",
                            "completion_window": "24h"}).json()
    print(f"enterprise batch ({len(docs)} reqs): {b.get('id')} {b.get('status')}")
    st = json.load(open(f"{HERE}/state_ov.json"))
    st["enterprise"] = {"batch_id": b.get("id"), "n": len(docs)}
    json.dump(st, open(f"{HERE}/state_ov.json", "w"), indent=2)

if __name__ == "__main__":
    main()
