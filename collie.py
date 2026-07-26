#!/usr/bin/env python3
"""COLLIE — catalog enterprise documents into faceted cards.

Examples
--------
  # a single file
  python collie.py --file report.txt

  # a folder of text files
  python collie.py --glob 'docs/*.txt'

  # piped text
  cat email.eml | python collie.py

  # straight from a Hugging Face dataset
  python collie.py --hf-dataset HuggingFaceFW/finepdfs --hf-config eng_Latn \
                   --hf-split train --hf-field text --limit 20

Output is one JSON card per document on stdout (JSON Lines), so it pipes
into jq, DuckDB, or a dataframe directly.
"""
import argparse, glob as globmod, json, re, sys

MODEL = "Harsh/collie-ent-direct-0.6b"

SYSTEM = (
    "You are COLLIE, an enterprise librarian. Read the document and produce a faceted catalog "
    "entry — one answer per facet. Describe; never judge sensitivity or importance.\n\n"
    "Output STRICT JSON on its own line:\n"
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

KEYS = ("subject", "type", "audience", "time", "purpose", "content_flags")


def parse(text):
    body = re.sub(r"```(?:json)?|```", "", text)
    m = (re.search(r'\{\s*"subject"\s*:.*?\}', body, re.S)
         or re.search(r'\{\s*"subject"\s*:.*\}', body, re.S))
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None
    return {k: d.get(k) for k in KEYS}


def load_documents(a):
    if a.file:
        yield a.file, open(a.file, encoding="utf-8", errors="ignore").read()
    elif a.glob:
        for p in sorted(globmod.glob(a.glob))[:a.limit or None]:
            yield p, open(p, encoding="utf-8", errors="ignore").read()
    elif a.hf_dataset:
        from datasets import load_dataset
        kw = {"split": a.hf_split, "streaming": True}
        if a.hf_config:
            kw["name"] = a.hf_config
        ds = load_dataset(a.hf_dataset, **kw)
        for k, row in enumerate(ds):
            if a.limit and k >= a.limit:
                break
            txt = row.get(a.hf_field)
            if txt and len(txt.strip()) > 200:
                yield f"{a.hf_dataset}#{k}", txt
    else:
        data = sys.stdin.read()
        if data.strip():
            yield "<stdin>", data


def main():
    ap = argparse.ArgumentParser(description="Catalog documents with COLLIE.")
    src = ap.add_argument_group("input (pick one; default is stdin)")
    src.add_argument("--file")
    src.add_argument("--glob")
    src.add_argument("--hf-dataset", help="e.g. HuggingFaceFW/finepdfs")
    src.add_argument("--hf-config", help="dataset config, e.g. eng_Latn")
    src.add_argument("--hf-split", default="train")
    src.add_argument("--hf-field", default="text")
    ap.add_argument("--limit", type=int, default=0, help="max documents")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-chars", type=int, default=4000,
                    help="documents are truncated to this length (training used 4000)")
    ap.add_argument("--pretty", action="store_true", help="human-readable instead of JSONL")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()

    batch = []
    def flush():
        if not batch:
            return
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": "Document:\n" + t[:a.max_chars]}],
            tokenize=False, add_generation_prompt=True) for _, t in batch]
        enc = tok(prompts, return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=160, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for j, (name, _) in enumerate(batch):
            gen = tok.decode(out[j][enc.input_ids.shape[1]:], skip_special_tokens=True)
            card = parse(gen) or {"error": "unparsed", "raw": gen[:200]}
            if a.pretty:
                print(f"\n=== {name} ===")
                for k in KEYS:
                    print(f"  {k:<15} {card.get(k)}")
            else:
                print(json.dumps({"document": name, **card}, ensure_ascii=False), flush=True)
        batch.clear()

    for name, text in load_documents(a):
        batch.append((name, text))
        if len(batch) >= a.batch_size:
            flush()
    flush()


if __name__ == "__main__":
    main()
