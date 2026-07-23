#!/usr/bin/env python3
"""Sample ~2400 docs from dlp_bench for COLLIE v2 labeling.

Broad random draw across corpora for base coverage, PLUS a keyword-targeted
boost that oversamples docs likely to carry the rare tail topics
(compensation, M&A, credentials, customer_data, regulatory) — because those
labels are unknown pre-labeling, we bias the draw with signal words. Sample
2400 so ~2000 survive the no-think / parse / ontology filter downstream.
"""
import gzip, json, hashlib, os, re, ast

DATA = os.path.expanduser("~/workspace/ai_soc/dlp_bench/sources/data")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "data", "manifest_2k.jsonl")

# deterministic PRNG (Math.random-free equivalent): hash-based shuffle key
def hkey(s):
    return int(hashlib.sha256(s.encode()).hexdigest(), 16)

CORPORA = {  # file -> (base_quota, short_src)
    "enron.jsonl.gz": (760, "enron"),
    "finepdfs_english_diverse_10k.jsonl.gz": (560, "finepdfs"),
    "apache_public_raw/apache_enterprise_like_9k.jsonl.gz": (340, "apache"),
    "chat_like_public_raw/chat_like_public_9k.jsonl.gz": (240, "chat"),
    "github_code_clean_code.jsonl.gz": (170, "ghcode"),
    "loghub.jsonl.gz": (110, "loghub"),
}

# rare-topic signal words -> boost pool (searched in enron + finepdfs + apache)
RARE = re.compile(r"\b("
    r"salary|salaries|compensation|equity|bonus|stock option|severance|"          # compensation
    r"acquisition|acquire|merger|divestiture|due diligence|term sheet|"            # M&A
    r"password|passwd|api[_ ]?key|secret key|access token|private key|credential|" # credentials
    r"ssn|social security|customer data|personal data|gdpr|hipaa|phi\b|pii\b|"     # customer_data
    r"audit|compliance|regulation|regulatory|sec filing|10-k|certification"        # regulatory
    r")\b", re.I)

BOOST_FILES = ["enron.jsonl.gz",
               "finepdfs_english_diverse_10k.jsonl.gz",
               "apache_public_raw/apache_enterprise_like_9k.jsonl.gz"]
BOOST_TARGET = 300

def _asdict(v):
    """raw_issue / raw_comments may be dicts/lists already, or repr-strings."""
    if isinstance(v, (dict, list)):
        return v
    if isinstance(v, str):
        try:
            return ast.literal_eval(v)
        except Exception:
            return None
    return None

def _apache_text(r):
    """Extract summary + description + comment bodies from a JIRA issue record."""
    parts = []
    iss = _asdict(r.get("raw_issue")) or {}
    fields = iss.get("fields", {}) if isinstance(iss, dict) else {}
    if fields.get("summary"):
        parts.append(f"Issue: {fields['summary']}")
    desc = fields.get("description")
    if isinstance(desc, str) and desc.strip():
        parts.append(desc.strip())
    comments = _asdict(r.get("raw_comments")) or []
    if isinstance(comments, list):
        for c in comments[:6]:
            b = c.get("body") if isinstance(c, dict) else None
            if isinstance(b, str) and b.strip():
                parts.append(f"Comment: {b.strip()}")
    return "\n\n".join(parts)

def get_text(r, src):
    if src == "chat":
        return r.get("body_text", "")
    if src == "apache":
        return _apache_text(r)
    return r.get("text", "")

def sha(r):
    return r.get("text_sha256") or r.get("id") or \
        hashlib.sha256(json.dumps(r, sort_keys=True)[:2000].encode()).hexdigest()

def clean_ok(t):
    return t and 200 <= len(t) <= 20000

def read(fn):
    for line in gzip.open(os.path.join(DATA, fn), "rt", encoding="utf-8"):
        yield json.loads(line)

def main():
    picked, seen = [], set()
    # base broad draw: hash-shuffle each corpus, take quota
    for fn, (quota, src) in CORPORA.items():
        rows = [(r, t) for r in read(fn) if clean_ok(t := get_text(r, src))]
        rows.sort(key=lambda rt: hkey(sha(rt[0])))
        for r, t in rows[:quota]:
            if sha(r) in seen:
                continue
            seen.add(sha(r))
            picked.append((src, r, t))
    base_n = len(picked)
    # rare-topic boost: keyword hits not already taken
    boost = []
    for fn in BOOST_FILES:
        src = CORPORA[fn][1]
        for r in read(fn):
            t = get_text(r, src)
            if not clean_ok(t) or sha(r) in seen:
                continue
            if RARE.search(t[:6000]):
                boost.append((src, r, t))
    boost.sort(key=lambda x: hkey(sha(x[1])))
    for src, r, t in boost[:BOOST_TARGET]:
        if sha(r) in seen:
            continue
        seen.add(sha(r))
        picked.append((src, r, t))
    # write manifest (tier: base=broad draw, boost=rare-topic keyword hits)
    with open(OUT, "w", encoding="utf-8") as f:
        for i, (src, r, t) in enumerate(picked):
            f.write(json.dumps({
                "i": i, "src": src, "id": r.get("id", f"{src}-{i}"),
                "tier": "base" if i < base_n else "boost",
                "text": t[:4000],
            }) + "\n")
    from collections import Counter
    print(f"base={base_n} boost={len(picked)-base_n} total={len(picked)}")
    print("by source:", dict(Counter(s for s, _, _ in picked)))

if __name__ == "__main__":
    main()
