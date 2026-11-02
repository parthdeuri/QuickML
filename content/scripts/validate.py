#!/usr/bin/env python3
"""Validate the QuickML content files.

Usage:  python content/scripts/validate.py

Checks every problem, sheet and pattern YAML and exits with a non-zero
status (printing one line per card found) if anything is wrong.
Works from any working directory. Needs only the stdlib and pyyaml.
"""
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("pyyaml is required: pip install -r content/requirements.txt")

CONTENT_DIR = Path(__file__).resolve().parent.parent
PROBLEMS_DIR = CONTENT_DIR / "problems"
SHEETS_DIR = CONTENT_DIR / "sheets"
PATTERNS_DIR = CONTENT_DIR / "patterns"
VERSION_FILE = CONTENT_DIR / "version.txt"

DIFFICULTIES = {"Easy", "Medium", "Hard"}  # shown as Beginner/Intermediate/Advanced
AREAS = {"Classical ML", "Deep Learning", "NLP", "Vision", "RL", "MLOps",
         "Math"}

REQUIRED_KEYS = ["slug", "title", "url", "difficulty", "topics", "area"]
# Fields that make up a "full" card. All or none of them must be present.
CONTENT_KEYS = ["summary", "intuition", "approach", "time", "space", "code"]
OPTIONAL_KEYS = ["patterns", "pitfalls", "related", "diagram", "math_tex"]
# Optional monospace box-art picture (architecture / data flow).
MATH_DIR = Path(__file__).resolve().parent.parent.parent / "app/src/main/assets/math"
DIAGRAM_DIR = Path(__file__).resolve().parent.parent.parent / "app/src/main/assets/diagrams"
# 1-based line numbers of `code` holding the core idea. Required for full
# cards, forbidden for metadata-only ones.
KEY_LINES_KEY = "key_lines"
KEY_LINES_COUNT = (1, 3)
# Also all-or-nothing together with CONTENT_KEYS (full cards only).
DETAIL_KEYS = ["when_to_use", "interview_qa", "hyperparameters"]
# Explanation fields, also all-or-nothing with CONTENT_KEYS.
EXPLAIN_KEYS = ["when_not_to_use", "why_it_works", "math"]
QA_KEYS = ["q", "a"]
ALL_KEYS = set(REQUIRED_KEYS + CONTENT_KEYS + DETAIL_KEYS + EXPLAIN_KEYS
               + OPTIONAL_KEYS + [KEY_LINES_KEY])

# Sheets whose size is fixed; the build fails if the count drifts.
EXPECTED_SHEET_COUNTS = {"ml-core": 47, "dl-core": 39, "nlp-vision-rl": 38}
SLUG_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")

MAX_CODE_LINE = 70  # only a warning: phone screens are narrow
MATH_LINES = (2, 14)
QA_COUNT = (3, 5)
HYPER_COUNT = (2, 5)

# Pattern library (patterns/<id>.yaml). `order` sets the position in
# problems.json and is not emitted itself.
PATTERN_KEYS = ["id", "name", "order", "summary", "when_to_use", "key_idea",
                "template", "complexity", "pitfalls", "problems", "aliases"]
PATTERN_WHEN_TO_USE = (3, 5)
PATTERN_PITFALLS = (2, 4)
PATTERN_TEMPLATE_LINES = 25


def _is_text(v):
    return isinstance(v, str) and v.strip() != ""



def _strings(v):
    if isinstance(v, str):
        yield v
    elif isinstance(v, list):
        for x in v:
            yield from _strings(x)
    elif isinstance(v, dict):
        for x in v.values():
            yield from _strings(x)


try:  # optional: CI may not have matplotlib; the local run does
    from matplotlib import mathtext as _mathtext
except Exception:  # pragma: no cover
    _mathtext = None


def _check_inline_math(text, where, err):
    """Formulas are written $...$; they must pair up and parse (matplotlib mathtext subset)."""
    if text.count("$") % 2:
        err(f"{where}: unbalanced $ in: {text[:60]}")
        return
    if _mathtext is None:
        return
    for seg in re.findall(r"\$([^$]+)\$", text):
        try:
            _mathtext.MathTextParser("path").parse("$" + seg + "$")
        except Exception as e:
            err(f"{where}: formula does not parse ({str(e).splitlines()[0][:80]}): {seg[:60]}")

def _is_text_list(v):
    return isinstance(v, list) and all(_is_text(x) for x in v)


def read_version(errors):
    try:
        text = VERSION_FILE.read_text(encoding="utf-8").strip()
        version = int(text)
        if version < 1:
            raise ValueError
        return version
    except (OSError, ValueError):
        errors.append(
            f"{VERSION_FILE.name}: must contain a positive integer")
        return 1


def _load_yaml(path, errors):
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        errors.append(f"{path.name}: cannot parse YAML ({e})")
        return None
    if not isinstance(data, dict):
        errors.append(f"{path.name}: top level must be a mapping")
        return None
    return data


def check_details(p, err):
    """when_to_use / interview_qa / hyperparameters of a full card."""
    if "when_to_use" in p and not _is_text(p["when_to_use"]):
        err("'when_to_use' must be a non-empty string")
    qa = p.get("interview_qa")
    if qa is not None:
        lo, hi = QA_COUNT
        if not isinstance(qa, list) or not lo <= len(qa) <= hi:
            err(f"'interview_qa' must be a list of {lo}-{hi} items")
        else:
            for i, e in enumerate(qa, 1):
                if not isinstance(e, dict):
                    err(f"interview_qa #{i} must be a mapping")
                    continue
                extra = set(e) - set(QA_KEYS)
                if extra:
                    err(f"interview_qa #{i}: unknown key(s) "
                        f"{', '.join(sorted(extra))}")
                for k in QA_KEYS:
                    if not _is_text(e.get(k)):
                        err(f"interview_qa #{i}: '{k}' must be a "
                            "non-empty string")
    hp = p.get("hyperparameters")
    if hp is not None:
        lo, hi = HYPER_COUNT
        if not (_is_text_list(hp) and lo <= len(hp) <= hi):
            err(f"'hyperparameters' must be a list of {lo}-{hi} "
                "non-empty strings")


def check_explanations(p, err):
    """when_not_to_use / why_it_works / math of a full card."""
    for k in ("when_not_to_use", "why_it_works"):
        if k in p and not _is_text(p[k]):
            err(f"'{k}' must be a non-empty string")
    m = p.get("math")
    if m is not None:
        lo, hi = MATH_LINES
        if not (_is_text_list(m) and lo <= len(m) <= hi):
            err(f"'math' must be a list of {lo}-{hi} non-empty strings")
        else:
            for i, line in enumerate(m, 1):
                if len(line) > MAX_CODE_LINE:
                    err(f"math line {i} is {len(line)} chars "
                        f"(> {MAX_CODE_LINE})")


def check_key_lines(p, err):
    """key_lines of a full card: 1-3 distinct line numbers of `solution`,
    each on a meaningful line (not blank, not class/def, not bare return)."""
    kl = p.get(KEY_LINES_KEY)
    lo, hi = KEY_LINES_COUNT
    if kl is None:
        err(f"full card is missing '{KEY_LINES_KEY}'")
        return
    if (not isinstance(kl, list) or not lo <= len(kl) <= hi
            or not all(isinstance(i, int) and not isinstance(i, bool)
                       for i in kl)):
        err(f"'{KEY_LINES_KEY}' must be a list of {lo}-{hi} integers")
        return
    if len(set(kl)) != len(kl):
        err(f"'{KEY_LINES_KEY}' has duplicate line numbers")
    src = p.get("code")
    if not _is_text(src):
        return
    lines = src.splitlines()
    for i in kl:
        if not 1 <= i <= len(lines):
            err(f"'{KEY_LINES_KEY}' line {i} is outside the code "
                f"(1-{len(lines)})")
            continue
        code = lines[i - 1].strip()
        if not code:
            err(f"'{KEY_LINES_KEY}' line {i} is blank")
        elif re.match(r"(class|def|async\s+def)\b", code) or code == "return":
            err(f"'{KEY_LINES_KEY}' line {i} is a class/def line or a bare "
                f"return: {code}")


def _sentences(text):
    return len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s])


def check_problem(path, p, errors, warnings):
    """Validate one card mapping; returns normalised dict or None."""
    name = path.name
    start = len(errors)

    def err(msg):
        errors.append(f"{name}: {msg}")

    unknown = set(p) - ALL_KEYS
    if unknown:
        err(f"unknown key(s): {', '.join(sorted(unknown))}")
    for k in REQUIRED_KEYS:
        if k not in p or p[k] in (None, ""):
            err(f"missing required key '{k}'")

    slug = p.get("slug")
    if _is_text(slug) and slug != path.stem:
        err(f"slug '{slug}' does not match filename '{path.stem}'")

    for k in ("title",):
        if k in p and not _is_text(p[k]):
            err(f"'{k}' must be a non-empty string")

    if slug is not None and not (isinstance(slug, str)
                                 and SLUG_RE.match(slug)):
        err("slug must be lowercase-kebab-case")

    if p.get("area") not in AREAS and "area" in p:
        err(f"area '{p.get('area')}' not in {sorted(AREAS)}")

    url = p.get("url")
    if url is not None:
        if not (isinstance(url, str) and url.startswith("https://")):
            err("url must start with https://")

    if p.get("difficulty") not in DIFFICULTIES and "difficulty" in p:
        err(f"difficulty '{p.get('difficulty')}' not in "
            f"{sorted(DIFFICULTIES)}")

    if "topics" in p and not (_is_text_list(p["topics"])
                              and p["topics"]):
        err("'topics' must be a non-empty list of strings")

    for k in ("patterns", "pitfalls", "related"):
        if k in p and p[k] is not None and not _is_text_list(p[k]):
            err(f"'{k}' must be a list of non-empty strings")

    dg = p.get("diagram")
    if dg is not None:
        if not (isinstance(dg, str) and re.fullmatch(r"[a-z0-9-]+", dg)):
            err("'diagram' must be an image name such as 'k-means'")
        else:
            for theme in ("light", "dark"):
                if not (DIAGRAM_DIR / f"{dg}_{theme}.png").exists():
                    err(f"diagram image missing: {dg}_{theme}.png "
                        "(run make_diagrams.py)")

    mt = p.get("math_tex")
    if mt is not None:
        if not _is_text_list(mt):
            err("'math_tex' must be a list of non-empty strings")
        elif len(mt) != len(p.get("math") or []):
            err(f"'math_tex' has {len(mt)} lines but 'math' has {len(p.get('math') or [])}")
        else:
            for theme in ("light", "dark"):
                if not (MATH_DIR / f"{slug}_{theme}.png").exists():
                    err(f"math image missing: {slug}_{theme}.png (run make_math.py)")

    for key, val in p.items():
        if key in ("code", "math", "math_tex", "key_lines", "diagram", "template"):
            continue
        for text in _strings(val):
            if "$" in text:
                _check_inline_math(text, f"'{key}'", err)

    # Full vs metadata-only: all or nothing.
    full_keys = CONTENT_KEYS + DETAIL_KEYS + EXPLAIN_KEYS
    present = [k for k in full_keys if p.get(k) not in (None, "", [])]
    if present and len(present) != len(full_keys):
        missing = [k for k in full_keys if k not in present]
        err("partial content: has "
            f"{', '.join(present)} but is missing {', '.join(missing)}")
    if not present and p.get(KEY_LINES_KEY) is not None:
        err(f"'{KEY_LINES_KEY}' is only allowed on full problems")
    if present:
        check_details(p, err)
        check_explanations(p, err)
        check_key_lines(p, err)
        if _is_text(p.get("intuition")) and _sentences(p["intuition"]) < 2:
            warnings.append(f"{name}: 'intuition' is under 2 sentences")
        for k in ("summary", "intuition", "time", "space"):
            if k in p and not _is_text(p[k]):
                err(f"'{k}' must be a non-empty string")
        if "approach" in p and not (_is_text_list(p["approach"])
                                    and p["approach"]):
            err("'approach' must be a non-empty list of strings")
        if "code" in p:
            src = p["code"]
            if not _is_text(src):
                err("'code' must be a non-empty string")
            else:
                try:
                    compile(src, str(slug or path.stem), "exec")
                except SyntaxError as e:
                    err(f"code does not compile: {e}")
                for i, line in enumerate(src.splitlines(), 1):
                    if len(line) > MAX_CODE_LINE:
                        warnings.append(
                            f"{name}: code line {i} is "
                            f"{len(line)} chars (> {MAX_CODE_LINE})")
        for k, lo, hi in (("pitfalls", 2, 4), ("related", 1, 3)):
            n = len(p.get(k) or [])
            if not lo <= n <= hi:
                warnings.append(
                    f"{name}: '{k}' has {n} item(s), expected {lo}-{hi}")

    return None if len(errors) > start else p


def load_problems(errors, warnings):
    """Returns {slug: problem dict} for all valid problem files."""
    problems = {}
    files = sorted(PROBLEMS_DIR.glob("*.yaml"))
    if not files:
        errors.append(f"no problem files found in {PROBLEMS_DIR}")
    seen = {}
    for path in files:
        data = _load_yaml(path, errors)
        if data is None:
            continue
        slug = data.get("slug")
        if isinstance(slug, str):
            if slug in seen:
                errors.append(
                    f"{path.name}: duplicate slug '{slug}' "
                    f"(also in {seen[slug]})")
            seen[slug] = path.name
        ok = check_problem(path, data, errors, warnings)
        if ok is not None:
            problems[ok["slug"]] = ok
    # related slugs must exist (checked against every file's slug)
    for slug, p in problems.items():
        for r in p.get("related") or []:
            if r not in seen:
                errors.append(f"{slug}.yaml: related slug '{r}' "
                              "has no problem file")
            elif r == slug:
                errors.append(f"{slug}.yaml: related lists itself")
    return problems


def load_sheets(problems, errors):
    """Returns a list of validated sheet dicts."""
    sheets = []
    ids = set()
    files = sorted(SHEETS_DIR.glob("*.yaml"))
    if not files:
        errors.append(f"no sheet files found in {SHEETS_DIR}")
    for path in files:
        data = _load_yaml(path, errors)
        if data is None:
            continue
        name = path.name
        start = len(errors)
        for k in ("id", "name", "description", "sections"):
            if k not in data or data[k] in (None, ""):
                errors.append(f"{name}: missing required key '{k}'")
        sid = data.get("id")
        if sid is not None:
            if sid in ids:
                errors.append(f"{name}: duplicate sheet id '{sid}'")
            ids.add(sid)
            if sid != path.stem:
                errors.append(
                    f"{name}: sheet id '{sid}' does not match filename")
        sections = data.get("sections")
        if not isinstance(sections, list) or not sections:
            errors.append(f"{name}: 'sections' must be a non-empty list")
            continue
        in_sheet = set()
        total = 0
        for i, sec in enumerate(sections, 1):
            if (not isinstance(sec, dict) or not _is_text(sec.get("name"))
                    or not _is_text_list(sec.get("problems"))
                    or not sec.get("problems")):
                errors.append(
                    f"{name}: section #{i} needs a 'name' and a "
                    "non-empty 'problems' list of slugs")
                continue
            for slug in sec["problems"]:
                total += 1
                if slug not in problems:
                    errors.append(
                        f"{name}: section '{sec['name']}' references "
                        f"'{slug}' which has no (valid) problem file")
                if slug in in_sheet:
                    errors.append(f"{name}: slug '{slug}' appears more "
                                  "than once in the sheet")
                in_sheet.add(slug)
        expected = EXPECTED_SHEET_COUNTS.get(sid)
        if expected is not None and total != expected:
            errors.append(
                f"{name}: expected {expected} problems, found {total}")
        if len(errors) == start:
            sheets.append(data)
    return sheets


def is_full(p):
    return all(p.get(k) for k in CONTENT_KEYS + DETAIL_KEYS + EXPLAIN_KEYS)


def check_pattern(path, d, problems, errors, warnings):
    """Validate one pattern mapping; returns it or None."""
    name = path.name
    start = len(errors)

    def err(msg):
        errors.append(f"{name}: {msg}")

    unknown = set(d) - set(PATTERN_KEYS)
    if unknown:
        err(f"unknown key(s): {', '.join(sorted(unknown))}")
    missing = [k for k in PATTERN_KEYS if d.get(k) in (None, "", [])]
    if missing:
        err(f"missing required key(s): {', '.join(missing)}")
        return None
    if d["id"] != path.stem:
        err(f"id '{d['id']}' does not match filename '{path.stem}'")
    if not isinstance(d["order"], int) or isinstance(d["order"], bool):
        err("'order' must be an integer")
    for k in ("id", "name", "summary", "key_idea", "template",
              "complexity"):
        if not _is_text(d[k]):
            err(f"'{k}' must be a non-empty string")
    for k, (lo, hi) in (("when_to_use", PATTERN_WHEN_TO_USE),
                        ("pitfalls", PATTERN_PITFALLS)):
        if not (_is_text_list(d[k]) and lo <= len(d[k]) <= hi):
            err(f"'{k}' must be a list of {lo}-{hi} non-empty strings")
    for k in ("problems", "aliases"):
        if not _is_text_list(d[k]):
            err(f"'{k}' must be a non-empty list of strings")
        elif len(set(d[k])) != len(d[k]):
            err(f"'{k}' has duplicates")
    if _is_text_list(d["problems"]):
        for slug in d["problems"]:
            if slug not in problems:
                err(f"problem '{slug}' has no (valid) problem file")
    if _is_text(d["key_idea"]) and not 2 <= _sentences(d["key_idea"]) <= 4:
        warnings.append(f"{name}: 'key_idea' should be 2-4 sentences")
    src = d["template"]
    if _is_text(src):
        lines = src.rstrip("\n").splitlines()
        if len(lines) > PATTERN_TEMPLATE_LINES:
            err(f"template has {len(lines)} lines "
                f"(max {PATTERN_TEMPLATE_LINES})")
        try:
            compile(src, f"{path.stem} template", "exec")
        except SyntaxError as e:
            err(f"template does not compile: {e}")
        for i, line in enumerate(lines, 1):
            if len(line) > MAX_CODE_LINE:
                warnings.append(f"{name}: template line {i} is "
                                f"{len(line)} chars (> {MAX_CODE_LINE})")
    return None if len(errors) > start else d


def load_patterns(problems, errors, warnings):
    """Returns validated patterns sorted by `order`."""
    parsed, valid = [], []  # parsed: has every key; valid: no errors
    files = sorted(PATTERNS_DIR.glob("*.yaml"))
    if not files:
        warnings.append(f"no pattern files found in {PATTERNS_DIR}")
    for path in files:
        data = _load_yaml(path, errors)
        if data is None:
            continue
        ok = check_pattern(path, data, problems, errors, warnings)
        if all(data.get(k) not in (None, "", []) for k in PATTERN_KEYS):
            parsed.append(data)
        if ok is not None:
            valid.append(data)
    # ids, names, orders unique; every alias maps to exactly one pattern
    for key in ("id", "name", "order"):
        seen = {}
        for d in parsed:
            v = str(d[key])
            if v in seen:
                errors.append(f"{d['id']}.yaml: duplicate {key} "
                              f"'{v}' (also in {seen[v]}.yaml)")
            seen[v] = d["id"]
    owner = {}
    for d in parsed:
        if not _is_text_list(d["aliases"]):
            continue
        for a in d["aliases"]:
            if a in owner and owner[a] != d["id"]:
                errors.append(f"{d['id']}.yaml: alias '{a}' is already "
                              f"used by {owner[a]}.yaml")
            owner.setdefault(a, d["id"])
    # Coverage is only a warning: problems are added in parallel.
    uncovered = {}
    for slug, p in sorted(problems.items()):
        if is_full(p):
            for s in p.get("patterns") or []:
                if s not in owner:
                    uncovered.setdefault(s, []).append(slug)
    if parsed:
        for s, slugs in sorted(uncovered.items()):
            warnings.append(f"pattern string '{s}' (used by "
                            f"{', '.join(slugs)}) is not an alias of any "
                            "pattern")
    valid.sort(key=lambda d: (d["order"], d["id"]))
    return valid


def validate():
    """Returns (errors, warnings, problems, sheets, version, patterns)."""
    errors, warnings = [], []
    version = read_version(errors)
    problems = load_problems(errors, warnings)
    sheets = load_sheets(problems, errors)
    patterns = load_patterns(problems, errors, warnings)
    return errors, warnings, problems, sheets, version, patterns


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    errors, warnings, problems, sheets, _, patterns = validate()
    for w in warnings:
        print(f"warning: {w}")
    if errors:
        for e in errors:
            print(f"error: {e}", file=sys.stderr)
        print(f"\nvalidation FAILED: {len(errors)} error(s)",
              file=sys.stderr)
        return 1
    full = sum(1 for p in problems.values() if is_full(p))
    sizes = ", ".join(
        f"{s['id']}={sum(len(x['problems']) for x in s['sections'])}"
        for s in sheets)
    print(f"validation OK: {len(problems)} problems ({full} full), "
          f"{len(sheets)} sheet(s) [{sizes}], {len(patterns)} "
          f"pattern(s), {len(warnings)} warning(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
