#!/usr/bin/env python3
"""Render the COLLIE demo GIF: documents typing in, catalog cards coming out.

Every example is a REAL document from the evaluation corpus paired with the
REAL card collie-ent-direct-0.6b produced for it — nothing is mocked up.
Writes assets/collie_demo.gif.
"""
import json, os
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "collie_demo.gif")
EXAMPLES = os.path.join(HERE, "demo_examples.json")

W, H = 1000, 620
PAD = 26
LINE = 21
FONT_PATH = "/System/Library/Fonts/Menlo.ttc"

BG = (17, 20, 28)
CHROME = (30, 34, 45)
DIM = (108, 118, 138)
DOC = (176, 186, 205)
KEY = (110, 170, 255)
VAL = (126, 214, 158)
FLAG = (240, 176, 96)
PROMPT = (96, 220, 180)
WHITE = (226, 232, 240)
RED, YEL, GRN = (255, 95, 86), (255, 189, 46), (39, 201, 63)


def fonts():
    return (ImageFont.truetype(FONT_PATH, 15), ImageFont.truetype(FONT_PATH, 15, index=1),
            ImageFont.truetype(FONT_PATH, 13))


def base_frame(title):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 34], fill=CHROME)
    for k, c in enumerate((RED, YEL, GRN)):
        d.ellipse([16 + k * 20, 12, 26 + k * 20, 22], fill=c)
    f, fb, fs = fonts()
    d.text((W // 2 - 90, 9), title, font=fs, fill=DIM)
    return img, d


def wrap(text, width):
    out = []
    for raw in text.split("\n"):
        raw = raw.rstrip()
        while len(raw) > width:
            cut = raw.rfind(" ", 0, width)
            cut = cut if cut > width * 0.6 else width
            out.append(raw[:cut])
            raw = raw[cut:].lstrip()
        out.append(raw)
    return out


def draw_doc(d, lines, y, f, reveal=None):
    shown = lines if reveal is None else lines[:reveal]
    for ln in shown:
        d.text((PAD, y), ln, font=f, fill=DOC)
        y += LINE
    return y


def draw_card(d, card, y, f, fb, reveal_keys):
    d.text((PAD, y), "{", font=f, fill=WHITE); y += LINE
    order = ["subject", "type", "audience", "time", "purpose", "content_flags"]
    for k in order[:reveal_keys]:
        v = card.get(k)
        d.text((PAD + 22, y), f'"{k}"', font=fb, fill=KEY)
        kw = d.textlength(f'"{k}"', font=fb)
        d.text((PAD + 22 + kw, y), ": ", font=f, fill=WHITE)
        vx = PAD + 22 + kw + d.textlength(": ", font=f)
        colour = FLAG if k == "content_flags" else VAL
        d.text((vx, y), json.dumps(v, ensure_ascii=False), font=fb, fill=colour)
        y += LINE
    if reveal_keys >= len(order):
        d.text((PAD, y), "}", font=f, fill=WHITE); y += LINE
    return y


def build():
    examples = json.load(open(EXAMPLES, encoding="utf-8"))
    f, fb, fs = fonts()
    frames, durations = [], []
    charw = f.getlength("M")
    cols = int((W - 2 * PAD) / charw)

    for ex in examples:
        doc_lines = wrap(ex["text"].strip(), cols)[:9]
        cmd = f'$ python collie.py --file {ex["name"]}'
        # 1. command types in
        for k in range(0, len(cmd) + 1, 4):
            img, d = base_frame("COLLIE — enterprise document librarian")
            d.text((PAD, 52), cmd[:k] + ("█" if k < len(cmd) else ""), font=fb, fill=PROMPT)
            frames.append(img); durations.append(35)
        # 2. document reveals
        for r in range(1, len(doc_lines) + 1):
            img, d = base_frame("COLLIE — enterprise document librarian")
            d.text((PAD, 52), cmd, font=fb, fill=PROMPT)
            d.text((PAD, 84), f'[{ex["source"]}]', font=fs, fill=DIM)
            draw_doc(d, doc_lines, 106, f, reveal=r)
            frames.append(img); durations.append(70)
        y_card = 106 + len(doc_lines) * LINE + 26
        # 3. card fills in, facet by facet
        for r in range(0, 7):
            img, d = base_frame("COLLIE — enterprise document librarian")
            d.text((PAD, 52), cmd, font=fb, fill=PROMPT)
            d.text((PAD, 84), f'[{ex["source"]}]', font=fs, fill=DIM)
            draw_doc(d, doc_lines, 106, f)
            d.text((PAD, y_card - 24), "catalog card", font=fs, fill=DIM)
            draw_card(d, ex["card"], y_card, f, fb, r)
            frames.append(img); durations.append(150 if r else 260)
        durations[-1] = 1900          # hold the finished card
    frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=True)
    mb = os.path.getsize(OUT) / 1e6
    print(f"wrote {OUT}  frames={len(frames)}  {mb:.1f} MB")


if __name__ == "__main__":
    build()
