"""Flash cards are derived, not written: check the derivation and the generated asset."""
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import flashcards  # noqa: E402

asset = ROOT.parent / "app" / "src" / "main" / "assets" / "flashcards.json"
doc = json.loads(asset.read_text(encoding="utf-8"))
cards = doc["cards"]
version = int((ROOT / "version.txt").read_text().strip())
assert doc["contentVersion"] == version, "flashcards.json is stale: run build.py"

problems = {p.stem: yaml.safe_load(p.read_text(encoding="utf-8"))
            for p in (ROOT / "problems").glob("*.yaml")}
sheets = {s.stem: yaml.safe_load(s.read_text(encoding="utf-8"))
          for s in (ROOT / "sheets").glob("*.yaml")}
full = {s: p for s, p in problems.items() if p.get("summary")}

# the deck is internally sound
assert flashcards.check(cards, set(problems), set(sheets)) == []
ids = [c["id"] for c in cards]
assert len(ids) == len(set(ids))
assert [c["order"] for c in cards] == list(range(1, len(cards) + 1))

# every full card yields define, idea, use, pitfalls and one flash card per interview question
by_slug = {}
for c in cards:
    by_slug.setdefault(c["slug"], []).append(c)
for slug, p in full.items():
    mine = {c["id"] for c in by_slug[slug]}
    for kind in ("define", "idea", "use", "pitfalls"):
        assert f"{slug}/{kind}" in mine, (slug, kind)
    for i in range(1, len(p["interview_qa"]) + 1):
        assert f"{slug}/qa{i}" in mine, (slug, i)
    has_math = bool(p.get("math_tex") or p.get("math"))
    assert (f"{slug}/math" in mine) == has_math, slug

# link-only entries have nothing to quiz
for slug, p in problems.items():
    if not p.get("summary"):
        assert slug not in by_slug, slug

# decks follow the sheets (a card in several sheets is in several decks)
member = {}
for sid, sheet in sheets.items():
    for sec in sheet["sections"]:
        for slug in sec["problems"]:
            member.setdefault(slug, set()).add(sid)
for slug, mine in by_slug.items():
    expected = member.get(slug, set())
    for c in mine:
        if c["kind"] in ("compare", "family"):
            continue
        assert set(c["decks"]) == expected, (c["id"], c["decks"], expected)
assert {"dl-core", "llm-engineer"} <= set(next(c for c in cards
                                                if c["id"] == "self-attention/idea")["decks"])

# families and compare cards carry the shared idea / when-to-pick text
fam = next(c for c in cards if c["id"] == "family/time-series")
assert fam["kind"] == "family" and fam["slug"] in problems and len(fam["back"]) > 40
cmp_ = next(c for c in cards if c["id"] == "compare/compare-ridge-vs-lasso")
assert cmp_["front"].endswith("when to pick which?") and cmp_["back"].startswith("• ")

# formulas stay as $...$ so the app can typeset them
math_card = next(c for c in cards if c["id"] == "adam-optimizer/math")
assert "$" in math_card["back"] and math_card["back"].count("\n") >= 2

# derive() on a tiny synthetic input
p = {"slug": "x", "title": "X", "summary": "S.", "intuition": "I.", "approach": ["a"],
     "solution": "c", "examples": [{"input": "q?", "output": "a."}], "description": "d",
     "bruteForce": "b", "edgeCases": ["e"], "dryRun": []}
out = flashcards.derive([p], [{"id": "s", "entries": [{"slug": "x"}]}], [])
assert [c["id"] for c in out] == ["x/define", "x/idea", "x/qa1", "x/use", "x/pitfalls"]
assert out[3]["back"] == "Use it when: d\n\nAvoid it when: b"
assert all(c["decks"] == ["s"] for c in out)

print("flashcards OK:", len(cards), "cards")
