# Two ways of teaching COLLIE that didn't work

*2026-07-25*

After the catalog experiments settled, COLLIE had a solid but imperfect
scorecard: it names what a document is about correctly ~79% of the time,
identifies what kind of document it is 92% of the time, but only gets
`purpose` right 69% of the time and `audience` 74%.

Those last two looked like *taste* problems rather than knowledge problems
— the sort of thing reinforcement learning is supposed to fix. So we tried
twice. Neither worked, and understanding **why** turned out to be more
useful than a win would have been.

## Attempt zero: the reward function that we never used

The plan was GRPO, where the model generates answers and a reward function
scores them. A scoring function has to be fast and free, so we wrote one
that reads the document and the card and checks measurable things: are the
subject words actually present in the document? Do the facets repeat each
other? Do the terms look like enterprise filing vocabulary?

Before spending anything on GPUs, we checked something simple: **does our
cheap scorer agree with the expensive LLM judge?** We had 3,705 already-judged
cards sitting around, so this cost nothing.

It mostly didn't agree. Correlation with the judge:

| reward component | agreement with judge |
|:--|:--:|
| grounding (words appear in the document) | +0.16 |
| non-redundancy | −0.01 |
| enterprise-register phrasing | +0.05 |
| facet plausibility | +0.02 |

Only grounding carried any signal. And the reason is structural, not a
tuning problem: you cannot tell whether `purpose: bug_report` is *correct
for this document* using a statistic that never reads the meaning of the
document. Three of our four components were, in effect, measuring nothing.

Training against that would have optimized noise. We threw the plan away
before it cost a dollar. **Validating a proxy metric against ground truth,
before training on it, is a ten-minute job that saved a wasted day.**

## Attempt one: let the model rank its own work (self-play DPO)

New plan. Ask COLLIE to catalog the same document four times, have a good
judge rank the four attempts, then teach it: *be more like your best
attempt, less like your worst*. This is DPO, and the judge only runs once,
offline, so it's cheap.

The preference data looked great — the judge saw a clear winner in 1,199 of
1,200 documents, with a median quality gap of 5 points out of 10.

The result: **nothing moved.** 37.8% of the cards changed, but every score
stayed within one point.

| | before | after |
|:--|--:|--:|
| subjects precise | 79.3% | 78.7% |
| subjects wrong | 9.1% | 9.4% |
| purpose correct | 69.4% | 68.4% |
| fully faithful cards | 20.6% | 20.9% |

The diagnostic explains it. The judge scored COLLIE's *best of four*
attempts at a median **8 out of 10**. The model was already producing good
cards; picking its best one and reinforcing it has nowhere to go.

More importantly: if `purpose` were wrong 31% of the time because of bad
luck in sampling, then at least one of four attempts would usually get it
right, and this method would have caught it. It didn't — which means when
COLLIE gets `purpose` wrong, it gets it wrong **all four times**. That's not
a preference problem. That's not knowing the answer.

## Attempt two: show it the teacher's answer (teacher-vs-student DPO)

If comparing the model to itself hits a ceiling, compare it to something
better. We already had GPT's card for thousands of documents. So: the
teacher's card is the "good" example, COLLIE's own card is the "bad" one,
and DPO pushes it from one toward the other.

Building those pairs produced the most interesting number of the whole
exercise. Out of 3,200 documents:

- **1,894 (59%) were thrown out because COLLIE's card was already as good
  as the teacher's.**

The student has caught up with the teacher on what we can measure. There
was never much teacher advantage left to transfer.

The result, predictably by now: another wash — plus a regression.

| | SFT | self-play DPO | teacher-vs-student DPO |
|:--|--:|--:|--:|
| subjects precise | 79.3% | 78.7% | 79.3% |
| audience correct | 74.1% | 74.3% | 74.9% |
| **time correct** | **76.3%** | 76.4% | **71.9%** |
| content flags correct | 70.2% | 70.5% | 72.3% |
| fully faithful cards | 20.6% | 20.9% | 21.4% |

Flags and audience nudged up. `time` fell more than four points. Net zero,
paid for with a regression.

That regression is our own fault in an instructive way. To decide which
pairs were worth keeping, we filtered on *grounding* — the one signal we'd
validated. But grounding says nothing about whether `time` is right, so the
filter happily kept pairs that were worse on `time` and the model dutifully
learned from them. **If you filter training data on one quality, you are
silently ignoring every other quality.**

## What we actually learned

1. **Check your proxy before you train on it.** Ours agreed with reality on
   one of four axes. Ten minutes of correlation checking against data we
   already had prevented a pointless training run.
2. **Self-play can only reach the best version of what you already are.**
   When the best of four attempts is already an 8/10, preference learning
   has nothing left to teach. Consistent errors across all attempts are
   knowledge gaps, not sampling noise.
3. **A small model can catch up to its teacher.** On 59% of documents our
   0.6B model matched GPT's card quality. Distillation had already given
   nearly everything it had to give.
4. **Filters have blind spots that become regressions.** Selecting pairs on
   one metric let another metric quietly degrade.
5. **Two different methods failing the same way is a real answer.** The
   remaining errors aren't style or preference. They are capacity: at 0.6B,
   this is roughly the ceiling. If we want a better COLLIE, the lever is a
   bigger model — which earlier experiments already showed beats every other
   knob we've turned.

So COLLIE v1 ships as the plain fine-tuned model. Both RL checkpoints stay
published for the record, and neither is recommended.

Total cost of finding this out: about $6 and an afternoon. Negative results
are cheap when you check your assumptions early — and this one closes a door
we'd otherwise have kept pushing on.
