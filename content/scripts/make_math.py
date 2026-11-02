#!/usr/bin/env python3
"""Typesets each card's `math_tex` lines into one PNG per theme.

Run: python content/scripts/make_math.py [slug ...]
Each `math_tex` entry is one line: prose with formulas inside $...$ (matplotlib mathtext).
Output: app/src/main/assets/math/<slug>_light.png and <slug>_dark.png (committed; CI never
runs matplotlib). Exits non-zero if a line fails to parse or is wider than the canvas.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "DejaVu Serif"

ROOT = Path(__file__).resolve().parent.parent
PROBLEMS = ROOT / "problems"
OUT = ROOT.parent / "app/src/main/assets/math"
COLORS = {"light": ("#1b2430", "#5b6675"), "dark": ("#e8edf3", "#9aa6b5")}
WIDTH_IN = 4.4
FS = 12.5
TALL = ("\\frac", "\\sum", "\\prod", "\\int", "\\binom", "\\underset", "\\overset")


def line_height(s):
    return 0.62 if any(t in s for t in TALL) else 0.36


def render(slug, lines, theme):
    main, muted = COLORS[theme]
    heights = [line_height(s) for s in lines]
    total = sum(heights) + 0.2
    fig = plt.figure(figsize=(WIDTH_IN, total))
    y = total - 0.1
    texts = []
    for s, h in zip(lines, heights):
        prose = "$" not in s
        t = fig.text(0.5, (y - h / 2) / total, s, ha="center", va="center",
                     fontsize=FS - 1.5 if prose else FS,
                     color=muted if prose else main)
        texts.append(t)
        y -= h
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    limit = WIDTH_IN * fig.dpi * 0.97
    for s, t in zip(lines, texts):
        if t.get_window_extent(r).width > limit:
            raise ValueError(f"{slug}: line too wide, split it: {s}")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{slug}_{theme}.png", dpi=300, transparent=True)
    plt.close(fig)


def main():
    want = set(sys.argv[1:])
    done, bad = 0, 0
    for f in sorted(PROBLEMS.glob("*.yaml")):
        if want and f.stem not in want:
            continue
        d = yaml.safe_load(f.read_text(encoding="utf-8"))
        lines = d.get("math_tex")
        if not lines:
            continue
        try:
            for theme in COLORS:
                render(f.stem, lines, theme)
            done += 1
        except Exception as e:  # mathtext ParseException, too wide, ...
            bad += 1
            print("FAIL", f.stem, "-", str(e).splitlines()[0][:160])
    print("rendered", done, "cards,", bad, "failed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
