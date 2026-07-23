#!/usr/bin/env python3
"""Batched COLLIE eval generation — for hosts where per-token CPU overhead
dominates single-doc generation. Left-padded batches, greedy decode, same
parsing and output contract as gen_collie_preds.py (rows in eval order).

Env: COLLIE_MODEL_DIR, COLLIE_EVAL, COLLIE_PRED_OUT, COLLIE_MAXNEW, COLLIE_GBS.
"""
import json, os, re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = os.environ["COLLIE_MODEL_DIR"]
EVAL = os.environ["COLLIE_EVAL"]
OUT = os.environ["COLLIE_PRED_OUT"]
MAXNEW = int(os.environ.get("COLLIE_MAXNEW", 400))
GBS = int(os.environ.get("COLLIE_GBS", 16))

def parse_labels(text):
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
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype="bfloat16",
                                                 attn_implementation="sdpa").cuda().eval()
    rows = [json.loads(l) for l in open(EVAL, encoding="utf-8")]
    fout = open(OUT, "w", encoding="utf-8")
    done = 0
    for b0 in range(0, len(rows), GBS):
        chunk = rows[b0:b0 + GBS]
        prompts = [tok.apply_chat_template(r["messages"][:-1], tokenize=False,
                                           add_generation_prompt=True) for r in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAXNEW, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for j, r in enumerate(chunk):
            gen = tok.decode(out[j][enc.input_ids.shape[1]:], skip_special_tokens=True)
            parsed = parse_labels(gen)
            fout.write(json.dumps({"i": r["i"], "labels": parsed.get("labels", []),
                                   "topics": parsed.get("topics", []),
                                   "tags": parsed.get("tags", [])}) + "\n")
        fout.flush()
        done += len(chunk)
        print(f"{done}/{len(rows)}", flush=True)
    fout.close()
    print("GEN_COMPLETE", flush=True)

if __name__ == "__main__":
    main()
