# The catalog was the problem

*2026-07-25*

This entry covers four quick experiments that ended with a surprise: the
feature we thought was COLLIE's core idea — handing the model a topic
catalog to work from — turned out to be the thing holding it back. We
removed it, and the model got dramatically better.

## What we were trying to do

COLLIE reads an enterprise document and files it: what is this about, what
kind of document is it, who is it for. Our plan from the start was
"catalog-conditioned" filing — you give the model your organization's list
of topics in the prompt, and it prefers your words when they fit.

To check quality, we stopped comparing the model's answers to a teacher's
answers and started asking a simpler, harder question: **read the document,
read the model's output — is the output actually true of this document?**
A grader model does this check on thousands of fresh documents. We call it
the faithfulness eval, and it's what caught everything that follows.

## What kept going wrong

Every time we put a catalog in the prompt, quality fell off a cliff — and
the *shape* of the failure was always the same.

- With the model's original output format (pick 1-4 topics, prefer the
  catalog), **28% of its topics were wrong** on catalog-prompted documents.
  With no catalog: 6%.
- We loosened the format to a free tag list. Better — wrong dropped to
  19% — but the gap between "catalog" and "no catalog" remained.
- We then moved to a structured catalog card (subject / type / audience /
  time / purpose / content flags — one answer per box, which fixed
  redundancy nicely). But the subject box said "1-3 subjects, prefer the
  catalog," and wrongness **exploded to 56%**.

The pattern, in plain terms: tell a small model *"fill this many slots,
and here are the words the teacher likes"* and it fills the slots with
teacher-pleasing words whether they fit the document or not. It behaves
like a student gaming a rubric. Meanwhile, whenever we gave it **no list
at all**, the same model quietly did excellent work: 82% of its
self-written subjects were precise, only 8% wrong.

The model was never the problem. The homework instructions were.

## The fix: let the librarian speak

We dropped the catalog completely. The model now fills the same card, but
the subject line is always in its own words — steered only by an
instruction to *sound like an enterprise filing system* (things like
`vendor_contract_negotiation`, `incident_response`,
`windows_cbs_servicing_error`), not like an academic or a chatbot.

Results on 3,705 fresh documents, judged against the documents themselves:

| what we measure | with catalog (worst case) | final model (no catalog) |
|:--|:--:|:--:|
| subjects that are wrong | 56% | **9%** |
| subjects that are precise | 30% | **79%** |
| important subjects missed per doc | 1.7 | **0.4** |
| "what kind of document is this" correct | — | **92%** |

And it's uniform — every document type, every slice of the test set,
within a point of the same score. The card format survived (one answer per
box, redundancy stays low, every single output parsed cleanly), only the
catalog died.

## What we learned

1. **Judge outputs against the input, not against a teacher.** Matching
   the teacher rewarded the exact force-fitting that was breaking the
   model. Checking against the document caught it immediately.
2. **A suggestion list plus a quota is a trap for small models.** Any
   phrasing of "pick N, prefer these" produced the same failure, in every
   format we tried. Small models read preferences as requirements.
3. **Structure helps; vocabulary constraints hurt.** Fixed *boxes* (one
   answer per question) made outputs cleaner and non-redundant. A fixed
   *word list* made them wrong. Constrain the shape, free the words.
4. **Trust the bare model more.** Its unprompted descriptions were the
   best output all along. Most of our engineering effort went into adding
   things that made it worse.

## What's next

The remaining soft spots are style, not correctness: `purpose` and
`audience` are right ~70-74% of the time, and subject lists sometimes say
the same thing twice in different words. The plan is a GRPO pass — reward
the model for phrasing that is specific, non-redundant, and
enterprise-sounding — on top of this checkpoint
(`collie-ent-direct-0.6b` on Hugging Face).
