#!/usr/bin/env python3
"""SFT a 0.6B COLLIE model on the messages-format labeling data.

Converts each {messages:[sys,user,assistant]} row to prompt/completion so TRL
masks the prompt and trains only on the assistant turn (the <think>+labels for
the reason variant, labels-only for direct). Same recipe for both variants so
the ONLY difference is the target text.

Env: COLLIE_TRAIN (jsonl), COLLIE_RUN_DIR, COLLIE_HUB_ID (optional).
"""
import json, os
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL = os.environ.get("COLLIE_BASE", "Qwen/Qwen3-0.6B")
TRAIN = os.environ["COLLIE_TRAIN"]
RUN = os.environ.get("COLLIE_RUN_DIR", "/workspace/collie-run")
HUB = os.environ.get("COLLIE_HUB_ID")

def load(tok):
    rows = []
    for l in open(TRAIN, encoding="utf-8"):
        m = json.loads(l)["messages"]
        prompt = tok.apply_chat_template(m[:-1], tokenize=False, add_generation_prompt=True)
        rows.append({"prompt": prompt, "completion": m[-1]["content"]})
    return Dataset.from_list(rows)

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    ds = load(tok)
    print(f"train rows: {len(ds)}  base: {MODEL}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="bfloat16",
                                                 attn_implementation="sdpa")
    cfg = SFTConfig(
        output_dir=RUN,
        max_length=2048,
        packing=False,
        num_train_epochs=float(os.environ.get("COLLIE_EPOCHS", 3)),
        per_device_train_batch_size=int(os.environ.get("COLLIE_BS", 8)),
        gradient_accumulation_steps=int(os.environ.get("COLLIE_ACCUM", 4)),
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        push_to_hub=bool(HUB),
        hub_model_id=HUB,
        hub_strategy="end",
        hub_private_repo=True,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
    trainer.train()
    trainer.save_model(f"{RUN}/final")
    tok.save_pretrained(f"{RUN}/final")
    print("COLLIE_TRAIN_COMPLETE", flush=True)

if __name__ == "__main__":
    main()
