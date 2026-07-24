#!/usr/bin/env python3
"""Tags-only COLLIE: transform existing open-vocab data — no new labeling.

Output shape becomes a single flat set: {"tags":[...]} = topics ∪ tags from
the round-4/5 teacher labels (subject tags + descriptor tags, deduped, order
preserved: subjects first). No slot quota, no privileged topic layer — the
anchor becomes suggested tag vocabulary only.

Writes: tags_direct_train.jsonl (from ov_clean + scale survivors already
merged in ov_*_train) and a tags-only inference file for the 3,705-doc
faithfulness corpus (same docs, same modes -> before/after comparison).
"""
import json, os

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

def sys_prompt_tags(anchor):
    base = ("You are COLLIE, a librarian for enterprise documents. Read the document and catalog "
            "it with descriptive tags: short snake_case terms that together say what the document "
            "is about and how it discusses it. Include tags for the subjects discussed and tags "
            "for the manner of discussion (who/scope, audience, time orientation, specificity, "
            "speech act). Describe, do not judge sensitivity.\n\n"
            "Output STRICT JSON on its own line: {\"tags\":[...]} — 3 to 10 tags.\n")
    if anchor:
        base += ("Suggested vocabulary — use these terms as tags when they genuinely apply, and "
                 "coin your own when they don't: " + ", ".join(anchor))
    else:
        base += "Coin the tags yourself; there is no fixed vocabulary."
    return base

def merge(topics, tags):
    out = []
    for t in list(topics) + list(tags):
        if t and t not in out:
            out.append(t)
    return out[:12]

def main():
    # training set: rebuild from the direct train file (carries topics+tags in
    # the assistant JSON) — parse each row, transform target + system prompt.
    n = 0
    with open(f"{HERE}/tags_direct_train.jsonl", "w", encoding="utf-8") as out:
        for l in open(f"{HERE}/ov_direct_train.jsonl", encoding="utf-8"):
            r = json.loads(l)
            sysmsg, usr, asst = (m["content"] for m in r["messages"])
            d = json.loads(asst)
            merged = merge(d.get("topics", []), d.get("tags", []))
            if not merged:
                continue
            # recover the anchor from the old system prompt
            anchor = None
            key = "Prefer these catalog topics when they genuinely fit: "
            if key in sysmsg:
                seg = sysmsg.split(key, 1)[1].split(". If the content", 1)[0]
                anchor = [t.strip() for t in seg.split(",") if t.strip()]
            out.write(json.dumps({"messages": [
                {"role": "system", "content": sys_prompt_tags(anchor)},
                {"role": "user", "content": usr},
                {"role": "assistant", "content": json.dumps({"tags": merged}, ensure_ascii=False)}],
                "src": r.get("src"), "i": r.get("i")}, ensure_ascii=False) + "\n")
            n += 1
    print(f"tags_direct_train.jsonl: {n} rows")

    # faithfulness corpus inference file (same 3,705 docs + modes as run 06)
    m = 0
    with open(f"{HERE}/tags_faith_infer.jsonl", "w", encoding="utf-8") as out:
        for l in open(f"{HERE}/faith_manifest.jsonl", encoding="utf-8"):
            d = json.loads(l)
            out.write(json.dumps({"messages": [
                {"role": "system", "content": sys_prompt_tags(d["anchor"])},
                {"role": "user", "content": "Document:\n" + d["text"]},
                {"role": "assistant", "content": ""}],
                "i": d["i"]}, ensure_ascii=False) + "\n")
            m += 1
    print(f"tags_faith_infer.jsonl: {m} rows")

if __name__ == "__main__":
    main()
