# COLLIE

<p align="center">
  <img src="assets/collie.png" alt="COLLIE — a clever herder that sorts documents into the categories your organization understands" width="640">
</p>

**Constrained Ontology Labeling for Long-form Information in the Enterprise**

COLLIE is a small (0.6B) model that catalogs enterprise documents. Given any
text an organization produces — email, PDFs, tickets, chat, code, logs — it
fills out a faceted catalog card, one answer per question:

```json
{"subject": ["filesystem_uri_parsing", "s3_path_encoding"],
 "type": "issue_report",
 "audience": "internal_team",
 "time": "current",
 "purpose": "bug_report",
 "content_flags": ["code_snippets", "error_traceback"]}
```

Every value is **open vocabulary** — there is no fixed label set anywhere.
Subjects are written in the register an enterprise would actually file
under (`vendor_contract_negotiation`, `incident_response`,
`windows_cbs_servicing_error`), and the card's structure guarantees the
answers are complementary rather than redundant: each facet answers a
different question, in the spirit of NIST/ISO-style faceted classification.

COLLIE is a **librarian, not a judge**: it describes content and makes no
judgment about sensitivity or importance. `content_flags` state what is
present (`personal_pii`, `financial_figures`) — what you do with that
description is your system's business.

Notably, there is no topic catalog in the prompt — by design, and by
evidence. We tried catalog-conditioning in several forms and it reliably
made a small model force-fit catalog terms onto documents they didn't
describe (see the journal). The bare model describes documents better than
any anchored variant ever did. See [taxonomy.md](taxonomy.md) for the
historical reference ontology the project started from.

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

## Results so far

**Rounds 1–3 — closed-ontology phase** (210-doc held-out eval, exact match):

| variant | topic F1 | topic P | topic R | correct-abstain /25 |
|:--|:--:|:--:|:--:|:--:|
| structured reason | 0.618 | 0.620 | 0.616 | 7 |
| structured direct | 0.608 | 0.673 | 0.554 | 0 |
| flat reason | 0.612 | 0.589 | 0.638 | 9 |
| **flat direct** | **0.655** | 0.651 | 0.659 | 0 |
| flat direct + 3× none-boost | 0.615 | 0.680 | 0.560 | 12 |

- **Output shape matters more than reasoning.** Dropping per-topic facet
  binding (flat `{"topics":[...],"tags":[...]}`) bought the direct model
  ~5 F1 points — the nested format was taxing topic identification itself.
- **Abstention is a data prior, not a capability.** Direct models never
  abstained (0/25) until the training share of empty-label docs was boosted
  8.5%→22%, after which they abstained fine (12/25) — but over-eagerly,
  costing recall. With an open vocabulary the "none" class dissolves anyway
  (log files become `system_error_logging`, not "no topic").

**Rounds 4–5 — open-vocabulary, anchor-conditioned** (LLM-judge semantic
scoring; two OOD axes: registers never trained on — JIRA tickets, system
logs — and anchor vocabularies never trained on — education, government,
energy, media, biotech catalogs). Round 5: 4,414 training docs, two model
sizes, prediction pre-registered before the run:

| topic F1 (judged) | in-dist | register-OOD | anchor-OOD |
|:--|:--:|:--:|:--:|
| reason 0.6B | 0.498 | 0.590 | 0.541 |
| direct 0.6B | **0.548** | 0.587 | **0.559** |
| reason 1.7B | 0.585 | 0.647 | 0.593 |
| **direct 1.7B** | **0.593** | **0.670** | **0.630** |

- The catalog became a prompt-time input: five anchor regimes in training
  (canonical / subset / paraphrase / alternative-domain / none) teach the
  model to prefer whatever catalog it is handed and coin coherent topics
  when the catalog doesn't fit. **Handed never-seen catalogs, every model
  still functions** — anchor-conditioning generalizes.
- **The pre-registered prediction failed.** Round 4 had hinted reasoning
  transfers better out-of-distribution; at 3× data the effect vanished, and
  at 1.7B direct wins every cell — by the most on the hardest axis. A
  distilled trace appears to act as a regularizer for an undertrained
  model, not as a transferable procedure.
- **The shipping recipe after these rounds: direct fine-tune, open
  vocabulary.** 0.6B for throughput, 1.7B for quality.

The full arc — including the failed hypothesis — is written up in
[journal/2026-07-24-five-rounds.md](journal/2026-07-24-five-rounds.md).

**Faithfulness phase — judging output against the document itself**
(3,705 fresh never-trained docs; a grader reads the document and the
model's card and checks each claim; no reference labels):

| design | wrong subjects (worst mode) | precise subjects |
|:--|:--:|:--:|
| topics + catalog anchor | 28% | 51% |
| flat tags + catalog anchor | 19% | 67% |
| faceted card + catalog anchor | 56% | 30% |
| **faceted card, no catalog (final)** | **9%** | **79%** |

Final model (`collie-ent-direct-0.6b`), uniform across all document types:
**79% precise subjects, 9% wrong, 0.39 missed subjects/doc, 92% correct
artifact-type**, low redundancy, 100% parseable cards.

The catalog experiments failed the same way every time: telling a small
model "fill N slots, prefer these words" makes it force-fit the words
whether they describe the document or not. Constrain the **shape** (one
answer per facet), free the **words** — that's the recipe. Written up
plainly in
[journal/2026-07-25-the-catalog-was-the-problem.md](journal/2026-07-25-the-catalog-was-the-problem.md).

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

The v1 recipe is settled: **anchor-free faceted catalog cards,
enterprise-register subjects, direct fine-tune**. Current model:
`collie-ent-direct-0.6b` (earlier round models kept for the record).
Next: a GRPO pass rewarding specific, non-redundant, enterprise-sounding
phrasing — the remaining gaps are style, not correctness.
