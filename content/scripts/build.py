#!/usr/bin/env python3
"""Build app/src/main/assets/problems.json from the YAML content (ML cards).

Usage:  python content/scripts/build.py

Runs validation first (aborts with a non-zero exit code on any error),
then writes the JSON consumed by the Android app. Works from any cwd.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate  # noqa: E402

OUT_FILE = (validate.CONTENT_DIR.parent / "app" / "src" / "main"
            / "assets" / "problems.json")


def convert_problem(p):
    """Maps ML card fields onto the JSON keys the (ex-Revise) app reads.

    area->platform, tags(topics)->topics, code->solution, time/space (compute/
    memory)->time/space, when_to_use->description, when_not_to_use->bruteForce,
    math->dryRun, interview_qa->examples(input=q, output=a),
    hyperparameters->constraints, pitfalls->edgeCases.
    """
    code = p.get("code")
    return {
        "slug": p["slug"],
        "title": p["title"],
        "platform": p["area"],
        "url": p["url"],
        "difficulty": p["difficulty"],
        "topics": list(p["topics"]),
        "patterns": list(p.get("patterns") or []),
        "summary": _prose(p.get("summary")),
        "bruteForce": _prose(p.get("when_not_to_use")),
        "intuition": _prose(p.get("intuition")),
        "whyItWorks": _prose(p.get("why_it_works")),
        "approach": ([_prose(s) for s in p["approach"]]
                     if p.get("approach") else None),
        "time": _text(p.get("time")),
        "space": _text(p.get("space")),
        "solution": code.rstrip("\n") if code else None,
        "keyLines": list(p.get("key_lines") or []),
        # formula lines keep their layout (monospace block in the app)
        "dryRun": [s.rstrip() for s in p.get("math") or []],
        "description": _prose(p.get("when_to_use")),
        "examples": [{"input": _prose(e["q"]), "output": _prose(e["a"]),
                      "explanation": ""}
                     for e in p.get("interview_qa") or []],
        "constraints": [_prose(c) for c in p.get("hyperparameters") or []],
        "edgeCases": [_prose(c) for c in p.get("pitfalls") or []],
        "related": list(p.get("related") or []),
        # box-art lines keep their spacing
        "diagram": [p["diagram"]] if p.get("diagram") else [],
    }


def _text(v):
    return v.strip() if isinstance(v, str) and v.strip() else None


def _prose(v):
    """Join hard-wrapped lines so text fills the phone's width.

    YAML `|` blocks keep the editor's ~70-char line breaks; the app would show
    them as short lines. Blank lines stay as paragraph breaks.
    """
    text = _text(v)
    if text is None:
        return None
    paragraphs = []
    for para in text.split("\n\n"):
        joined = ""
        for line in (ln.strip() for ln in para.splitlines()):
            if not line:
                continue
            # "non-" + "empty" -> "non-empty" (a word hyphenated across lines)
            if len(joined) >= 2 and joined[-1] == "-" and joined[-2].isalnum():
                joined += line
            else:
                joined = f"{joined} {line}" if joined else line
        if joined:
            paragraphs.append(joined)
    return "\n\n".join(paragraphs)


def convert_sheet(sheet):
    entries = []
    order = 0
    for sec in sheet["sections"]:
        for slug in sec["problems"]:
            order += 1
            entries.append({"slug": slug, "section": sec["name"],
                            "order": order})
    return {
        "id": sheet["id"],
        "name": sheet["name"],
        "description": sheet["description"].strip(),
        "entries": entries,
    }


def convert_pattern(d):
    """Prose is unwrapped like problem prose; the template keeps its lines.
    `order` only sets the position in the list and is not emitted."""
    return {
        "id": d["id"],
        "name": d["name"].strip(),
        "summary": _prose(d["summary"]),
        "whenToUse": [_prose(s) for s in d["when_to_use"]],
        "keyIdea": _prose(d["key_idea"]),
        "template": d["template"].rstrip("\n"),
        "complexity": _prose(d["complexity"]),
        "pitfalls": [_prose(s) for s in d["pitfalls"]],
        "problems": list(d["problems"]),
        "aliases": list(d["aliases"]),
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    (errors, warnings, problems, sheets, version,
     patterns) = validate.validate()
    for w in warnings:
        print(f"warning: {w}")
    if errors:
        for e in errors:
            print(f"error: {e}", file=sys.stderr)
        print(f"\nbuild aborted: {len(errors)} validation error(s)",
              file=sys.stderr)
        return 1

    out_problems = [convert_problem(problems[s]) for s in sorted(problems)]
    out_sheets = [convert_sheet(s) for s in sheets]
    out_patterns = [convert_pattern(d) for d in patterns]  # by `order`
    doc = {"contentVersion": version, "problems": out_problems,
           "sheets": out_sheets, "patterns": out_patterns}

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
        f.write("\n")

    full = sum(1 for p in out_problems
               if p["intuition"] and p["approach"] and p["solution"])
    detailed = sum(1 for p in out_problems
                   if p["description"] and p["examples"]
                   and p["constraints"])
    entries = sum(len(s["entries"]) for s in out_sheets)
    keyed = sum(1 for p in out_problems if p["keyLines"])
    print(f"wrote {OUT_FILE}")
    print(f"contentVersion={version} problems={len(out_problems)} "
          f"full={full} detailed={detailed} key_lines={keyed} "
          f"sheets={len(out_sheets)} sheet_entries={entries} "
          f"patterns={len(out_patterns)}")
    for s in out_sheets:
        print(f"  {s['id']}: {len(s['entries'])} entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
