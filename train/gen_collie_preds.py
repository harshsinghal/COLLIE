#!/usr/bin/env python3
"""Generate COLLIE predictions on the eval set and parse labels for scoring.

Loads a trained COLLIE model, rebuilds each eval prompt (system+user), greedily
generates the completion, strips any <think> block, parses the labels JSON, and
writes {"i", "labels"} in the SAME order as the eval file (collie_eval.py aligns
positionally). Malformed output -> empty labels (an honest miss, not a crash).

Env: COLLIE_MODEL_DIR, COLLIE_EVAL (jsonl messages), COLLIE_PRED_OUT, COLLIE_MAXNEW.
"""
import json, os, re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = os.environ["COLLIE_MODEL_DIR"]
EVAL = os.environ["COLLIE_EVAL"]
OUT = os.environ["COLLIE_PRED_OUT"]
MAXNEW = int(os.environ.get("COLLIE_MAXNEW", 400))

def parse_labels(text):
    """Parse either output shape: {"labels":[...]} or flat {"topics":[...],"tags":[...]}.
    Returns a dict of whatever keys parsed; malformed -> empty dict (honest miss)."""
    body = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    body = re.sub(r"```(?:json)?|```", "", body)
    m = re.search(r'\{\s*"(?:labels|topics)"\s*:.*\}', body, re.S)
    if not m:
        return {}
    try:
        d = json.loads(m.group(0))
        return {k: d[k] for k in ("labels", "topics", "tags") if k in d}
    except Exception:
        return {}

def main():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype="bfloat16",
                                                 attn_implementation="sdpa").cuda().eval()
    rows = [json.loads(l) for l in open(EVAL, encoding="utf-8")]
    fout = open(OUT, "w", encoding="utf-8")
    for k, r in enumerate(rows):
        m = r["messages"]
        prompt = tok.apply_chat_template(m[:-1], tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=MAXNEW, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
        parsed = parse_labels(gen)
        row = {"i": r["i"], "labels": parsed.get("labels", []),
               "topics": parsed.get("topics", []), "tags": parsed.get("tags", [])}
        fout.write(json.dumps(row) + "\n")
        if k % 50 == 0:
            print(f"{k}/{len(rows)}", flush=True)
    fout.close()
    print("GEN_COMPLETE", flush=True)

if __name__ == "__main__":
    main()
