"""Rewrites the method letters on figures/geometry_illustrated.png after the flat plane stopped being Method B.

The four labels sit on white, so each is painted over and redrawn in Segoe UI at the size that reproduces the
original label's width. Run once; it edits the PNG in place. ponytail: label boxes are hard-coded for the 1400 px
illustration; re-measure them if the figure is ever regenerated.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
FONT = "C:/Windows/Fonts/segoeui.ttf"
# (search box, old text, new text)
LABELS = [((1074, 592, 1398, 642), "Depth Z at torso (A, D)", "Depth Z at torso (A, C)"),
          ((1074, 760, 1398, 808), "Box-bottom ray (B, C)", "Box-bottom ray (B, D)"),
          ((195, 870, 395, 912), "Flat plane (B)", "Flat plane (D)"),
          ((165, 1050, 335, 1092), "Terrain (C)", "Terrain (B)")]

im = Image.open(HERE / "figures/geometry_illustrated.png").convert("RGB")
px = im.load()
draw = ImageDraw.Draw(im)
for (x0, y0, x1, y1), old, new in LABELS:
    ink = [(x, y) for x in range(x0, x1) for y in range(y0, y1) if sum(px[x, y]) < 600]
    xs, ys = [p[0] for p in ink], [p[1] for p in ink]
    bx0, by0, bx1, by1 = min(xs), min(ys), max(xs) + 1, max(ys) + 1
    colour = sorted(px[x, y] for x, y in ink)[len(ink) // 2]
    size = 10
    while ImageFont.truetype(FONT, size + 1).getlength(old) <= bx1 - bx0:
        size += 1
    font = ImageFont.truetype(FONT, size)
    draw.rectangle((bx0 - 3, by0 - 3, bx1 + 3, by1 + 3), fill="white")
    l, t, _, _ = font.getbbox(new)
    draw.text((bx0 - l, by0 - t), new, fill=colour, font=font)
    print(f"{old!r} -> {new!r}: box {bx0},{by0},{bx1},{by1} size {size} colour {colour}")
im.save(HERE / "figures/geometry_illustrated.png")
