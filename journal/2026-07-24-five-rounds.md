# Five rounds with a librarian — and a prediction that failed on schedule

*2026-07-24*

COLLIE started as a constrained classifier: 14 enterprise topics, 5 facet
axes, a decoder that couldn't emit anything else. Five experimental rounds
later it's an open-vocabulary librarian that conditions on whatever catalog
you hand it — and along the way it falsified the most interesting hypothesis
we had. This is the story in order, because the order is the lesson.

## Round 1–2: the cage was the bottleneck, not the brain

We trained matched pairs of Qwen3-0.6B models — one that reasons before
answering (`<think>` trace distilled from a teacher), one that answers
directly — first with structured per-topic facets, then with a flat
`{"topics":[...],"tags":[...]}` output.

The surprise wasn't reasoning. It was **format**. Dropping per-topic facet
binding bought the direct model ~5 F1 points (0.608 → 0.655): forcing a
0.6B model to nest five facet values under each topic inside strict JSON was
taxing the part that actually identifies topics. Reasoning, in-distribution,
was a wash — precision traded for recall.

## Round 3: abstention is a data prior wearing a capability costume

Across the first rounds, only reasoning models ever correctly said "no
topics here" (7–9/25 vs literally 0/25 for direct). It looked like a deep
finding: *no trace, no abstention*. It wasn't. Oversampling empty-label docs
from 8.5% to 22% of training taught the direct model to abstain fine
(12/25) — overshooting, in fact, into false abstentions. The "capability
gap" was a class-imbalance artifact. Cheap experiment, big correction.

Then the design question dissolved entirely: with an open vocabulary, a log
file isn't "no topic" — it's `system_error_logging`. The abstention class
was an artifact of the cage.

## Round 4: the pivot — catalog as input, not constraint

The redesign: COLLIE sees a topic vocabulary *in the prompt* (ours, a
paraphrased one, a partial one, someone else's domain catalog, or none) and
prefers it when it fits, coining coherent topics when it doesn't. Trained
across five anchor regimes, evaluated by an LLM judge (exact match dies
with free labels), with two registers — JIRA tickets and system logs —
held out of training entirely.

Round 4's result teased the hypothesis we wanted true: in-distribution,
direct won as usual, but on the held-out registers the order *flipped*
(reason 0.567 vs direct 0.553). Maybe the trace is the transferable
procedure — evidence-citing travels, memorized mappings don't?

## Round 5: pre-register, scale, find out

We wrote the prediction down before running: *if the trace is the
transferable procedure, reasoning's OOD advantage should widen at 1.7B*
(the size the SPANIEL thinking-arc showed can actually hold a trace). Then:
3× the training data (4,414 docs), a second model size, and a harder OOD
axis — five anchor vocabularies never seen in training (education,
government, energy, media, biotech).

| topic F1 (judged) | in-dist | register-OOD | anchor-OOD |
|:--|:--:|:--:|:--:|
| reason 0.6B | 0.498 | 0.590 | 0.541 |
| direct 0.6B | **0.548** | 0.587 | **0.559** |
| reason 1.7B | 0.585 | 0.647 | 0.593 |
| **direct 1.7B** | **0.593** | **0.670** | **0.630** |

The prediction failed symmetrically. At 1.7B, direct wins everywhere — by
the most on the hardest axis. And round 4's OOD edge for reasoning
evaporated at 3× data (0.590 vs 0.587, a wash). The likeliest reading: with
only 1.5k training docs, the trace was acting as a *regularizer* for an
undertrained model. Give the direct model enough data and it generalizes
fine — to unseen registers and unseen catalogs — without spending a single
inference token on thinking.

## What survived five rounds

1. **Output shape > reasoning.** The single biggest quality lever was
   simplifying what the model must emit.
2. **Anchor-conditioning works.** Handed never-seen catalogs, every model
   still functions (0.54–0.63). The catalog is genuinely a runtime input —
   train-time variety is what bought that.
3. **Scale is boring and undefeated.** 1.7B beats 0.6B in all 12 cells.
   More data helped the direct models most.
4. **Distilled reasoning is not a generalization vehicle.** Same conclusion
   the SPANIEL thinking arc reached on span extraction, now replicated on
   cataloging with proper OOD tests at two model sizes. A 68-word distilled
   trace buys recall early and nothing once data is adequate. (Caveat kept
   honest: this indicts trace *distillation*, not reasoning in the large.)

The recipe that ships: **direct fine-tune, open vocabulary, varied
anchors.** 0.6B when catalog throughput is the point, 1.7B when quality is.

The meta-lesson is the one worth keeping: write the prediction down before
the run. A hypothesis that fails *on schedule* teaches more than a vibe
that survives by never being pinned.
