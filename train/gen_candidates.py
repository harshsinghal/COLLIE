#!/usr/bin/env python3
"""Sample K candidate catalog cards per document from the SFT checkpoint.

DPO needs preferences over the model's OWN distribution, so we sample with
temperature rather than greedy-decode. Writes one row per document holding all
K candidates, ready for the ranking judge.

Env: COLLIE_MODEL_DIR, COLLIE_DOCS (jsonl with i/text), COLLIE_CAND_OUT,
     COLLIE_K (default 4), COLLIE_TEMP (default 1.0), COLLIE_GBS (docs/batch).
"""
import json, os, re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = os.environ["COLLIE_MODEL_DIR"]
DOCS = os.environ["COLLIE_DOCS"]
OUT = os.environ["COLLIE_CAND_OUT"]
K = int(os.environ.get("COLLIE_K", 4))
TEMP = float(os.environ.get("COLLIE_TEMP", 1.0))
GBS = int(os.environ.get("COLLIE_GBS", 8))
SYS_PATH = os.environ["COLLIE_SYS"]      # file holding the student system prompt


def clean(text):
    body = re.sub(r"```(?:json)?|```", "", text)
    m = re.search(r'\{\s*"subject"\s*:.*?\}', body, re.S) or \
        re.search(r'\{\s*"subject"\s*:.*\}', body, re.S)
    return m.group(0).strip() if m else text.strip()[:400]


def main():
    system = open(SYS_PATH, encoding="utf-8").read()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype="bfloat16", attn_implementation="sdpa").cuda().eval()
    docs = [json.loads(l) for l in open(DOCS, encoding="utf-8")]
    fout = open(OUT, "w", encoding="utf-8")
    done = 0
    for b0 in range(0, len(docs), GBS):
        chunk = docs[b0:b0 + GBS]
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user", "content": "Document:\n" + d["text"]}],
            tokenize=False, add_generation_prompt=True) for d in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=160, do_sample=True,
                                 temperature=TEMP, top_p=0.95,
                                 num_return_sequences=K,
                                 pad_token_id=tok.pad_token_id)
        plen = enc.input_ids.shape[1]
        for j, d in enumerate(chunk):
            cands = []
            for k in range(K):
                seq = out[j * K + k]
                cands.append(clean(tok.decode(seq[plen:], skip_special_tokens=True)))
            fout.write(json.dumps({"i": d["i"], "candidates": cands},
                                  ensure_ascii=False) + "\n")
        fout.flush()
        done += len(chunk)
        print(f"{done}/{len(docs)}", flush=True)
    fout.close()
    print("CANDIDATES_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
