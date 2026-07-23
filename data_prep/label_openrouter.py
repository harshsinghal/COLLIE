#!/usr/bin/env python3
"""COLLIE v2 labeler: multi-teacher reasoning traces via OpenRouter.

Round-robins each doc across a diverse reasoning-model panel, asks for a
mandatory <think> block then strict-JSON labels, captures the trace from
either the native `reasoning` field or inline <think>, and HARD-FILTERS:
a survivor must have >=MIN_THINK_WORDS of reasoning, parseable labels, and
every topic/facet inside the ontology. Non-survivors are logged and dropped.

Runs locally (network-bound, not GPU). Resumable: appends survivors and
records every attempted doc-id so reruns skip finished work.

Env: OR_KEY (required). Optional: OR_CONC, OR_LIMIT.
"""
import json, os, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

KEY = os.environ["OR_KEY"]
HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MANIFEST = f"{HERE}/manifest_2k.jsonl"
OUT = f"{HERE}/reason_clean_v2.jsonl"        # survivors
ATTEMPTED = f"{HERE}/attempted_v2.jsonl"     # every doc-id we've tried (resume)
REJECTS = f"{HERE}/rejects_v2.jsonl"         # dropped, with reason
URL = "https://openrouter.ai/api/v1/chat/completions"
CONC = int(os.environ.get("OR_CONC", "10"))
LIMIT = int(os.environ.get("OR_LIMIT", "0")) or None
MIN_THINK_WORDS = 30

# weighted panel: (model, weight). kimi-k3 is pricey -> minority slice.
PANEL = [
    ("deepseek/deepseek-r1-0528", 25),
    ("moonshotai/kimi-k2-thinking", 20),
    ("z-ai/glm-5.2", 20),
    ("qwen/qwen3-235b-a22b-thinking-2507", 20),
    ("minimax/minimax-m2.5", 10),
    ("moonshotai/kimi-k3", 5),
]
# expand to a weighted schedule, then deterministically interleave so that
# adjacent doc indices route to different teachers (not blocked by model).
import hashlib as _hl
_raw = [m for m, w in PANEL for _ in range(w)]
SCHED = [_raw[k] for k in sorted(range(len(_raw)),
         key=lambda k: _hl.sha256(str(k).encode()).hexdigest())]

ONT = {"compensation","workforce","mergers_acquisitions","financials","pricing","legal",
       "security","credentials","product","strategy","competition","personnel",
       "regulatory","customer_data"}
FVALS = {"scope":{"individual","group","aggregate","org_wide"},
         "publicity":{"public","internal","restricted"},
         "temporality":{"historical","current","forward_looking"},
         "specificity":{"named","figures","both","general"},
         "register_facet":{"report","negotiation","decision","directive",
                            "request","mention","speculation"}}

SYS = ("You are COLLIE, a document cataloger. Read the document and decide which topics it "
"genuinely discusses and how, using ONLY the ontology below. Describe, do not judge sensitivity.\n\n"
"You MUST reason first inside <think>...</think> — REQUIRED for every document. Inside <think>: "
"name the candidate subjects, cite the specific words/context that confirm or reject each, and "
"resolve each facet from evidence (40-120 words). Then, AFTER </think>, output the final answer as "
"STRICT JSON on its own line: {\"labels\":[{\"topic\":\"<t>\",\"scope\":\"<v>\",\"publicity\":\"<v>\","
"\"temporality\":\"<v>\",\"specificity\":\"<v>\",\"register_facet\":\"<v>\"}]}  (empty list if no topic).\n\n"
"Topics: compensation, workforce, mergers_acquisitions, financials, pricing, legal, security, "
"credentials, product, strategy, competition, personnel, regulatory, customer_data\n"
"Facets (per topic): scope=individual|group|aggregate|org_wide; publicity=public|internal|restricted; "
"temporality=historical|current|forward_looking; specificity=named|figures|both|general; "
"register_facet=report|negotiation|decision|directive|request|mention|speculation")

_lock = threading.Lock()

def extract(msg):
    """Return (think_text, labels_list) or (None, reason_str) on failure."""
    content = msg.get("content") or ""
    reasoning = (msg.get("reasoning") or "").strip()
    inline = re.search(r"<think>(.*?)</think>", content, re.S)
    think = reasoning or (inline.group(1).strip() if inline else "")
    if len(think.split()) < MIN_THINK_WORDS:
        return None, "no_think"
    # strip inline think + code fences, then find the labels JSON (allow pretty-print)
    body = re.sub(r"<think>.*?</think>", "", content, flags=re.S)
    body = re.sub(r"```(?:json)?|```", "", body).strip()
    m = re.search(r'\{\s*"labels"\s*:.*\}', body, re.S)
    if not m:
        return None, "no_json"
    try:
        labels = json.loads(m.group(0))["labels"]
    except Exception:
        return None, "bad_json"
    for l in labels:
        if l.get("topic") not in ONT:
            return None, f"off_topic:{l.get('topic')}"
        for fk, vals in FVALS.items():
            if l.get(fk) not in vals:
                return None, f"off_facet:{fk}={l.get(fk)}"
    return think, labels

def call(doc):
    model = SCHED[doc["i"] % len(SCHED)]
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYS},
                     {"role": "user", "content": "Document:\n" + doc["text"]}],
        "temperature": 0.4,
        "max_tokens": 3000,
        "include_reasoning": True,
        "reasoning": {"max_tokens": 1200},
    }
    for attempt in range(4):
        try:
            r = requests.post(URL, headers={"Authorization": f"Bearer {KEY}"},
                              json=payload, timeout=180)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 * (attempt + 1)); continue
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            think, labels = extract(msg)
            if think is None:
                return {"ok": False, "i": doc["i"], "model": model, "reason": labels}
            return {"ok": True, "i": doc["i"], "src": doc["src"], "id": doc["id"],
                    "text": doc["text"], "model": model, "think": think, "labels": labels}
        except Exception as e:
            if attempt == 3:
                return {"ok": False, "i": doc["i"], "model": model,
                        "reason": f"err:{type(e).__name__}"}
            time.sleep(2 * (attempt + 1))

def main():
    docs = [json.loads(l) for l in open(MANIFEST, encoding="utf-8")]
    done = set()
    if os.path.exists(ATTEMPTED):
        done = {json.loads(l)["i"] for l in open(ATTEMPTED, encoding="utf-8")}
    todo = [d for d in docs if d["i"] not in done]
    if LIMIT:
        todo = todo[:LIMIT]
    print(f"docs={len(docs)} already_attempted={len(done)} todo={len(todo)} panel={len(PANEL)}",
          flush=True)
    surv = 0; rej = 0; t0 = time.time()
    fout = open(OUT, "a", encoding="utf-8")
    fatt = open(ATTEMPTED, "a", encoding="utf-8")
    frej = open(REJECTS, "a", encoding="utf-8")
    with ThreadPoolExecutor(max_workers=CONC) as ex:
        futs = [ex.submit(call, d) for d in todo]
        for k, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            with _lock:
                fatt.write(json.dumps({"i": res["i"], "model": res["model"]}) + "\n")
                if res["ok"]:
                    fout.write(json.dumps(res) + "\n"); surv += 1
                else:
                    frej.write(json.dumps(res) + "\n"); rej += 1
                if k % 50 == 0 or k == len(todo):
                    for f in (fout, fatt, frej): f.flush()
                    rate = k / (time.time() - t0)
                    print(f"{k}/{len(todo)} surv={surv} rej={rej} "
                          f"({rate:.1f}/s, eta {(len(todo)-k)/rate/60:.1f}m)", flush=True)
    for f in (fout, fatt, frej): f.close()
    print(f"DONE surv={surv} rej={rej} yield={surv/max(1,surv+rej):.1%}", flush=True)

if __name__ == "__main__":
    main()
