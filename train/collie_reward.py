#!/usr/bin/env python3
"""Verifiable reward for GRPO on COLLIE catalog cards — no LLM judge in the loop.

Every component is computed from the document and the card alone, so a GRPO
step costs nothing but compute. The LLM judge stays OUT of training and is used
only as the held-out evaluator afterwards (don't train on your eval metric).

Components (each in [0,1], combined by WEIGHTS; schema validity is a hard gate):

  valid        schema gate — parses, six facets, subject 1-3, single-valued
               facets single, time in the closed set, <=3 flags. Fail -> 0.
  grounding    are the subject terms lexically supported by the document?
               (proxy for "is this actually what the document is about")
  nonredundant 1 - overlap among subject terms and across facets
               ("useful together, not saying the same thing twice")
  register     do the terms look like enterprise filing vocabulary? Derived
               from the teacher corpus: compound shape + at least one
               distinctive (non-boilerplate) token.
  facet_prior  do type/audience/purpose values match the teacher's vocabulary
               for that facet (the weakest facets in the SFT model).

Stats come from data/register_stats.json (built by build_register_stats.py from
the teacher labels) so "enterprise-sounding" is learned from data, not my taste.
"""
import json, math, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
STATS_PATH = os.environ.get(
    "COLLIE_REGISTER_STATS",
    os.path.join(os.path.dirname(HERE), "data", "register_stats.json"))

TIME_OK = {"historical", "current", "forward_looking"}
FACETS = ("subject", "type", "audience", "time", "purpose", "content_flags")
STOP = {"the", "a", "an", "of", "and", "or", "for", "to", "in", "on", "with", "by"}

WEIGHTS = {"grounding": 0.45, "nonredundant": 0.25, "register": 0.20, "facet_prior": 0.10}

_stats = None


def stats():
    global _stats
    if _stats is None:
        if os.path.exists(STATS_PATH):
            _stats = json.load(open(STATS_PATH))
        else:                                   # degrade gracefully, not silently
            _stats = {"subject_tokens": {}, "generic_tokens": [], "facet_vocab": {}}
    return _stats


def toks(term):
    return [t for t in str(term).split("_") if t and t not in STOP]


# ---------------------------------------------------------------- schema gate

def parse_card(text):
    """Return the card dict, or None if it is not a well-formed COLLIE card."""
    body = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    body = re.sub(r"```(?:json)?|```", "", body)
    m = re.search(r'\{\s*"subject"\s*:.*?\}', body, re.S)
    if not m:
        m = re.search(r'\{\s*"subject"\s*:.*\}', body, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None
    if not all(k in d for k in FACETS):
        return None
    subj = d.get("subject")
    if not isinstance(subj, list) or not 1 <= len(subj) <= 3:
        return None
    if not all(isinstance(s, str) and s.strip() for s in subj):
        return None
    for k in ("type", "audience", "purpose"):
        v = d.get(k)
        if v is not None and (not isinstance(v, str) or "_and_" in v or "," in v):
            return None
    t = d.get("time")
    if t is not None and t not in TIME_OK:
        return None
    fl = d.get("content_flags")
    if not isinstance(fl, list) or len(fl) > 3:
        return None
    if not all(isinstance(f, str) for f in fl):
        return None
    return d


# ------------------------------------------------------------------ grounding

def _grounded_token(tok, doc_low):
    """Token supported by the document, tolerating inflection (parsing/parse)."""
    if len(tok) < 3:
        return False
    if tok in doc_low:
        return True
    if len(tok) >= 5:                       # prefix match: encoding -> encoded
        return tok[:max(4, len(tok) - 3)] in doc_low
    return False


def grounding(card, doc):
    doc_low = doc.lower()
    scores = []
    for term in card["subject"]:
        tt = toks(term)
        if not tt:
            scores.append(0.0); continue
        hit = sum(_grounded_token(t, doc_low) for t in tt)
        scores.append(hit / len(tt))
    return sum(scores) / len(scores) if scores else 0.0


# --------------------------------------------------------------- redundancy

def _jaccard(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


def nonredundant(card):
    subj = card["subject"]
    pen = 0.0
    if len(subj) > 1:
        pairs = [(i, j) for i in range(len(subj)) for j in range(i + 1, len(subj))]
        pen += sum(_jaccard(toks(subj[i]), toks(subj[j])) for i, j in pairs) / len(pairs)
    singles = [card.get(k) for k in ("type", "audience", "purpose") if card.get(k)]
    values = [s.lower() for s in subj] + [str(v).lower() for v in singles] \
        + [str(f).lower() for f in card["content_flags"]]
    dupes = len(values) - len(set(values))
    pen += 0.25 * dupes
    return max(0.0, 1.0 - min(1.0, pen))


# ----------------------------------------------------------------- register

def register(card):
    st = stats()
    vocab = st.get("subject_tokens", {})
    generic = set(st.get("generic_tokens", []))
    scores = []
    for term in card["subject"]:
        tt = toks(term)
        if not tt:
            scores.append(0.0); continue
        # compound shape: enterprise filing terms are typically 2-4 tokens
        shape = 1.0 if 2 <= len(tt) <= 4 else (0.4 if len(tt) == 1 else 0.6)
        # at least one distinctive (non-boilerplate) token
        distinctive = 1.0 if any(t not in generic for t in tt) else 0.0
        # tokens the teacher corpus actually uses for subjects
        known = sum(1 for t in tt if t in vocab) / len(tt)
        scores.append(0.4 * shape + 0.35 * distinctive + 0.25 * known)
    return sum(scores) / len(scores) if scores else 0.0


# --------------------------------------------------------------- facet prior

def facet_prior(card):
    fv = stats().get("facet_vocab", {})
    got = []
    for k in ("type", "audience", "purpose"):
        v = card.get(k)
        known = set(fv.get(k, []))
        if not v:
            got.append(0.3)                       # null: allowed but unrewarded
        elif v in known:
            got.append(1.0)
        else:
            tt = toks(v)
            overlap = set()
            for term in known:
                overlap |= set(toks(term))
            got.append(0.5 * (sum(t in overlap for t in tt) / len(tt)) if tt else 0.0)
    return sum(got) / len(got)


# -------------------------------------------------------------------- reward

def score_one(completion, document, detail=False):
    card = parse_card(completion)
    if card is None:
        return ({"total": 0.0, "valid": 0.0} if detail else 0.0)
    parts = {"grounding": grounding(card, document),
             "nonredundant": nonredundant(card),
             "register": register(card),
             "facet_prior": facet_prior(card)}
    total = sum(WEIGHTS[k] * v for k, v in parts.items())
    if detail:
        parts.update({"total": total, "valid": 1.0})
        return parts
    return total


def make_reward_fn():
    """TRL GRPO reward: (completions, document, **cols) -> list[float]."""
    def reward(completions, document, **kwargs):
        out = []
        for c, doc in zip(completions, document):
            text = c[0]["content"] if isinstance(c, list) else c
            out.append(score_one(text, doc))
        return out
    return reward


if __name__ == "__main__":
    doc = ("Issue: [Python] FileSystem.from_uri doesn't decode %-encoded characters "
           "in path. Traceback shows ArrowInvalid: Cannot parse URI for the S3 bucket.")
    good = json.dumps({"subject": ["filesystem_uri_parsing", "s3_path_encoding"],
                       "type": "issue_report", "audience": "internal_team",
                       "time": "current", "purpose": "bug_report",
                       "content_flags": ["code_snippets"]})
    redundant = json.dumps({"subject": ["uri_parsing_issue", "uri_parsing_problem",
                                        "parsing_uri"],
                            "type": "issue_report", "audience": "internal_team",
                            "time": "current", "purpose": "bug_report",
                            "content_flags": []})
    ungrounded = json.dumps({"subject": ["quarterly_revenue_forecast"],
                             "type": "issue_report", "audience": "internal_team",
                             "time": "current", "purpose": "report",
                             "content_flags": []})
    generic = json.dumps({"subject": ["report", "information"], "type": "document",
                          "audience": "internal", "time": "current",
                          "purpose": "report", "content_flags": []})
    broken = '{"subject": ["a","b","c","d"], "type": "x"}'
    for name, c in (("good", good), ("redundant", redundant),
                    ("ungrounded", ungrounded), ("generic", generic),
                    ("broken", broken)):
        d = score_one(c, doc, detail=True)
        print(f"{name:>11}: total={d['total']:.3f}  " +
              " ".join(f"{k}={d[k]:.2f}" for k in
                       ("grounding", "nonredundant", "register", "facet_prior")
                       if k in d))
