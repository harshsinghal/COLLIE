# COLLIE ontology v1
**Constrained Ontology Labeling for Long-form Information in the Enterprise**

*COLLIE is a librarian, not a judge. It catalogs what a document discusses —
topic(s) plus descriptive facets — and makes NO judgment about sensitivity or
severity. Severity is MASTIFF's job (rubric-conditioned), and it operates on
COLLIE's description. "Constrained" = the decoder emits only valid ontology
labels; it cannot invent a topic or facet value outside this file.*

Two layers, both emitted per document (or per segment):
- **Topics** — the subject(s) discussed. Multi-label; a document may carry
  several, or `none`.
- **Facets** — cross-cutting descriptors that characterize *how* each topic is
  discussed. These are the raw material MASTIFF turns into severity.

## Layer 1 — Topic ontology (the subjects)

Purely descriptive. "Is this subject discussed?" — not "is it sensitive?"

| Label | The subject being discussed |
|:--|:--|
| `compensation` | Pay, equity, bonuses, offers, raises |
| `workforce` | Hiring, layoffs, RIF, reorg, headcount |
| `mergers_acquisitions` | Acquisitions, mergers, divestitures, deals |
| `financials` | Revenue, forecasts, results, budgets, financial performance |
| `pricing` | Prices, discounts, margins, contract terms |
| `legal` | Litigation, investigations, settlements, liability, contracts |
| `security` | Breaches, vulnerabilities, incidents, security posture |
| `credentials` | Passwords, API keys, tokens, or other secrets present in the text |
| `product` | Roadmap, features, launches, technical design |
| `strategy` | Business strategy, market moves, partnerships, restructuring |
| `competition` | Competitors, win/loss, market positioning |
| `personnel` | Individual performance, conduct, HR matters, complaints |
| `regulatory` | Regulations, audits, compliance, certifications |
| `customer_data` | Customer PII/PHI, data handling, privacy |

## Layer 2 — Facet ontology (the descriptors)

Each present topic gets characterized along these axes. Values are a fixed set
(constrained). Not every facet applies to every topic; emit what's determinable.

| Facet | Values | What it captures |
|:--|:--|:--|
| `scope` | `individual` · `group` · `aggregate` · `org_wide` | Who/how many the discussion is about |
| `publicity` | `public` · `internal` · `restricted` | Is this already public, or internal-only? |
| `temporality` | `historical` · `current` · `forward_looking` | Past fact, present state, or pre-decision/pre-announcement |
| `specificity` | `named` · `figures` · `both` · `general` | Named individuals? specific numbers? or general talk? |
| `register` | `report` · `negotiation` · `decision` · `directive` · `request` · `mention` · `speculation` | The speech act — what kind of discussion it is |

*`register` value notes: `directive` = an order/instruction to do something; `request` = an ask for action or information (distinct from an order — "could you send…", "please review"). `mention` = passing reference; `report` = describing state; `speculation` = hypothesizing.*

## Output shape (per segment)

```json
{"topics": ["compensation", "workforce"],
 "facets": {"scope": "individual", "publicity": "internal",
            "temporality": "forward_looking", "specificity": "both",
            "register": "decision"}}
```

Worked contrast — same topic, opposite facets:

| Text | topics | facets |
|:--|:--|:--|
| "BLS survey: median engineer pay rose 4% last year" | `compensation` | scope=aggregate, publicity=public, temporality=historical, specificity=figures, register=report |
| "We can offer Jane 180 base, equity refresh pending VP sign-off" | `compensation` | scope=individual, publicity=internal, temporality=forward_looking, specificity=both, register=negotiation |

MASTIFF then maps (topics + facets + SPANIEL's evidence spans) → severity via a
rubric. COLLIE never decides low vs. high; it only describes.

## Open design questions (your calls)

1. **Topic list** — 14 subjects above. Missing anything your target enterprises
   care about (export_control? board_governance? source_code / IP?), or is
   anything here noise?
2. **Facet set** — 5 facets. `scope`/`publicity`/`temporality` are the strongest
   severity drivers; `specificity`/`register` are softer. Keep all five, or trim
   to the load-bearing three for the MVP?
3. **Facet granularity** — values are coarse on purpose. Do any need more values
   (e.g. `publicity` → add `regulated` for PII/PHI)?
4. **Per-segment vs. per-document** — emit one label set for the whole document,
   or per segment (turn/paragraph)? Per-segment is more useful (localizes) and
   composes with SPANIEL, but is a harder labeling task. MVP could be
   per-document; SPANIEL adds the localization.
5. **Data implication** — because COLLIE describes rather than judges, training
   data no longer needs a "benign vs sensitive" split. It needs broad coverage
   of each topic across *all facet combinations* — the aggregate/public case and
   the individual/internal case are both `compensation`, and COLLIE must tag both
   identically at the topic layer while distinguishing them at the facet layer.
