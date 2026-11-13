#!/usr/bin/env python3
"""Derives the flash cards from the converted content (no hand-written cards).

Used by build.py, which writes app/src/main/assets/flashcards.json:
    {"contentVersion": N, "cards": [{id, slug, kind, front, back, decks, order}, ...]}

Kinds (one flash card each, so every question is one tap to flip):
  define    "What is <title>?"                        -> summary
  idea      "<title>: what is the key insight?"       -> intuition
  qa        each interview question                   -> its answer
  use       "When should you use <title>, and when not?" -> use / avoid
  pitfalls  "Common pitfalls of <title>?"             -> bullet list
  math      "Key formulas: <title>"                   -> LaTeX lines (one per line)
  compare   "<Compare card>: when to pick which?"     -> when_to_use bullets
  family    "<Family>: what is the shared idea?"      -> key_idea

Text keeps its $...$ formulas; the app typesets them. `decks` lists the sheet ids that
contain the card, `slug` points at the concept card ("Open concept"), `order` follows the
sheet order so new cards come in curriculum order.
"""

KINDS = ("define", "idea", "qa", "use", "pitfalls", "math", "compare", "family")


def _bullets(items):
    return "\n".join("• " + s for s in items if s)


def derive(problems, sheets, patterns):
    """problems/sheets/patterns are the already converted JSON structures from build.py."""
    by_slug = {p["slug"]: p for p in problems}
    decks = {}
    seq = []
    for sheet in sheets:
        for entry in sheet["entries"]:
            slug = entry["slug"]
            decks.setdefault(slug, [])
            if sheet["id"] not in decks[slug]:
                decks[slug].append(sheet["id"])
            if slug not in seq:
                seq.append(slug)
    for slug in by_slug:  # cards that are in no sheet still get flash cards
        if slug not in seq:
            seq.append(slug)

    cards = []

    def add(card_id, slug, kind, front, back, deck_ids):
        back = (back or "").strip()
        if front and back:
            cards.append({"id": card_id, "slug": slug, "kind": kind, "front": front.strip(),
                          "back": back, "decks": list(deck_ids), "order": len(cards) + 1})

    for slug in seq:
        p = by_slug.get(slug)
        if p is None or not (p.get("intuition") and p.get("approach") and p.get("solution")):
            continue  # link-only entries have no notes to quiz
        d = decks.get(slug, [])
        title = p["title"]
        add(f"{slug}/define", slug, "define", f"What is {title}?", p.get("summary"), d)
        add(f"{slug}/idea", slug, "idea", f"{title}: what is the key insight?",
            p.get("intuition"), d)
        for i, qa in enumerate(p.get("examples") or [], 1):
            add(f"{slug}/qa{i}", slug, "qa", qa["input"], qa["output"], d)
        use = []
        if p.get("description"):
            use.append("Use it when: " + p["description"])
        if p.get("bruteForce"):
            use.append("Avoid it when: " + p["bruteForce"])
        add(f"{slug}/use", slug, "use", f"When should you use {title}, and when not?",
            "\n\n".join(use), d)
        add(f"{slug}/pitfalls", slug, "pitfalls", f"Common pitfalls of {title}?",
            _bullets(p.get("edgeCases") or []), d)
        formulas = [s for s in (p.get("dryRun") or []) if s.strip()]
        if formulas:
            add(f"{slug}/math", slug, "math", f"Key formulas: {title}", "\n".join(formulas), d)

    for pat in patterns:
        members = [s for s in pat["problems"] if s in by_slug]
        if not members:
            continue
        deck_ids = []
        for s in members:
            for x in decks.get(s, []):
                if x not in deck_ids:
                    deck_ids.append(x)
        if pat["id"].startswith("compare-"):
            add(f"compare/{pat['id']}", members[0], "compare",
                f"{pat['name']}: when to pick which?", _bullets(pat["whenToUse"]), deck_ids)
        else:
            add(f"family/{pat['id']}", members[0], "family",
                f"{pat['name']}: what is the shared idea?", pat["keyIdea"], deck_ids)
    return cards


def check(cards, problem_slugs, sheet_ids):
    """Returns a list of error strings (empty when the deck is sound)."""
    errors = []
    seen = set()
    for c in cards:
        cid = c.get("id")
        if not cid or cid in seen:
            errors.append(f"flash card id missing or duplicated: {cid!r}")
        seen.add(cid)
        if c.get("kind") not in KINDS:
            errors.append(f"{cid}: unknown kind {c.get('kind')!r}")
        if not str(c.get("front", "")).strip() or not str(c.get("back", "")).strip():
            errors.append(f"{cid}: empty front or back")
        if c.get("slug") not in problem_slugs:
            errors.append(f"{cid}: slug {c.get('slug')!r} has no concept card")
        for d in c.get("decks") or []:
            if d not in sheet_ids:
                errors.append(f"{cid}: unknown deck {d!r}")
        for field in ("front", "back"):
            if str(c.get(field, "")).count("$") % 2:
                errors.append(f"{cid}: unbalanced $ in {field}")
    return errors
