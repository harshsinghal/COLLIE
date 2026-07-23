# COLLIE

**Constrained Ontology Labeling for Long-form Information in the Enterprise**

COLLIE is a small (0.6B) document-topic classifier — the first stage of a
three-model DLP cascade:

```
COLLIE  →  what does this document discuss?   (topics + descriptive tags)
SPANIEL →  where is the evidence?             (span extraction)   [github.com/harshsinghal/SPANIEL]
MASTIFF →  how severe is it?                  (rubric-conditioned adjudication)
```

COLLIE is a **librarian, not a judge**: it catalogs what a document discusses
and makes no sensitivity judgment. A BLS salary survey and an individual's
comp negotiation are both `compensation` — the descriptive tags
(`aggregate/public` vs `individual/internal`) are what downstream severity
adjudication consumes.

See [taxonomy.md](taxonomy.md) for the full ontology: 14 topics, 5 facet axes
(scope, publicity, temporality, specificity, register).

## How it's built

1. **Real enterprise-register corpora, all from public sources** — 2,480
   documents sampled across six registers (tail-boosted for rare topics),
   drawn from:
   - **Enron email corpus** — the public archive of real corporate email
     released during the FERC investigation; the classic enterprise-email
     research dataset.
   - **FinePDFs** — diverse real-world PDF documents from the
     [HuggingFaceFW/finepdfs](https://huggingface.co/datasets/HuggingFaceFW/finepdfs)
     dataset on Hugging Face.
   - **Apache Foundation JIRA tickets** — public support/issue tickets
     (summary, description, comment threads) from Apache project trackers.
   - **Public Slack-style chat archives** — messages from openly published
     community Slack archives (e.g. ops/infra communities).
   - **Public GitHub code** and **LogHub system logs** — for the code and
     machine-log registers.

   No synthetic documents were used for training in the current rounds.
2. **Teacher labeling with reasoning traces** — gpt-5.4-mini (bulk) +
   gpt-5.5 (rare-topic tail) label each doc with a concise mandatory
   `<think>` trace (median 78 words) plus strict-JSON labels. A hard filter
   drops any response without genuine reasoning, unparseable JSON, or
   off-ontology labels → 2,096 clean examples.
3. **Controlled distillation experiments** — Qwen3-0.6B fine-tuned in
   matched pairs (reason-first vs direct, structured vs flat output) on
   identical data and eval splits, so every score gap is attributable.

## Why the datasets are not in this repo

The source documents come from third-party corpora and public archives whose
licenses and terms vary — **I don't have the right to redistribute them, so
neither the sampled documents nor the teacher-labeled training sets are
included here.** What ships is everything needed to rebuild them:

- the ontology ([taxonomy.md](taxonomy.md)) — the labeling contract,
- the sampling, teacher-labeling, filtering, and SFT-assembly code
  (`data_prep/`), and
- the exact teacher prompts inside those scripts.

To build your own catalog: point `data_prep/sample_2k.py` at corpora you have
rights to — Hugging Face hosts many suitable public datasets (email, PDFs,
tickets, chat, code, logs) — or use an LLM to generate augmented or fully
synthetic enterprise-register documents and label them with the same pipeline.
The pipeline is corpus-agnostic: anything that yields `{id, text}` records
works.

## Results so far (210-doc held-out eval)

| variant | topic F1 | topic P | topic R | correct-abstain /25 |
|:--|:--:|:--:|:--:|:--:|
| structured reason | 0.618 | 0.620 | 0.616 | 7 |
| structured direct | 0.608 | 0.673 | 0.554 | 0 |
| flat reason | 0.612 | 0.589 | 0.638 | **9** |
| **flat direct** | **0.655** | 0.651 | 0.659 | 0 |

Two findings:
- **Output shape matters more than reasoning.** Dropping per-topic facet
  binding (flat `{"topics":[...],"tags":[...]}`) bought the direct model
  ~5 F1 points — the nested format was taxing topic identification itself.
- **No trace, no abstention.** Across all four runs, only the reasoning
  variants ever correctly return "no topics" on empty docs. Direct models
  tag something on every document.

## Layout

```
taxonomy.md       the ontology (topics + facets) — the contract
data_prep/        corpus sampling, teacher labeling (OpenAI batch + OpenRouter), SFT assembly
train/            0.6B SFT + eval-generation scripts (run on a rented GPU)
eval/             scorers: structured (topic F1 + per-facet acc) and flat (topics/tags F1)
data/             NOT distributed (see above) — rebuilt locally by data_prep/
results/          model predictions per experiment (label ids only, no document text)
journal/          findings, written as they happened
```

## Status

Active development. Current experiment: teaching the flat-direct model to
abstain (`none`-oversampling) without a reasoning trace.
