#!/usr/bin/env python3
"""DPO on judge preferences, starting from the COLLIE SFT checkpoint.

Conservative settings: low LR and a firm beta, because the SFT model is
already good and the pairs are a polish signal, not a rewrite. The reference
model is the SFT checkpoint itself, so KL keeps us near known-good behaviour.

Env: DPO_MODEL, DPO_PAIRS, DPO_RUN_DIR, DPO_HUB_ID, DPO_EPOCHS, DPO_BETA.
"""
import json, os
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

MODEL = os.environ.get("DPO_MODEL", "Harsh/collie-ent-direct-0.6b")
PAIRS = os.environ["DPO_PAIRS"]
RUN = os.environ.get("DPO_RUN_DIR", "/workspace/collie-dpo")
HUB = os.environ.get("DPO_HUB_ID")


def main():
    rows = []
    for l in open(PAIRS, encoding="utf-8"):
        r = json.loads(l)
        rows.append({"prompt": r["prompt"], "chosen": r["chosen"],
                     "rejected": r["rejected"]})
    ds = Dataset.from_list(rows)
    print(f"preference pairs: {len(ds)}  base: {MODEL}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype="bfloat16",
                                                 attn_implementation="sdpa")
    cfg = DPOConfig(
        output_dir=RUN,
        beta=float(os.environ.get("DPO_BETA", 0.1)),
        num_train_epochs=float(os.environ.get("DPO_EPOCHS", 1)),
        per_device_train_batch_size=int(os.environ.get("DPO_BS", 4)),
        gradient_accumulation_steps=int(os.environ.get("DPO_ACCUM", 4)),
        learning_rate=5e-7,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_length=2048,          # TRL 1.9 dropped max_prompt_length/max_completion_length
        truncation_mode="keep_end",   # documents are long; keep the end nearest the answer
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        push_to_hub=bool(HUB),
        hub_model_id=HUB,
        hub_strategy="end",
        hub_private_repo=True,
    )
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
    trainer.train()
    trainer.save_model(f"{RUN}/final")
    tok.save_pretrained(f"{RUN}/final")
    print("DPO_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
