# QuickML content pipeline

YAML in this folder is the source of truth. `scripts/build.py` validates it and writes
`app/src/main/assets/problems.json`, which the Android app imports. Bump `version.txt` whenever
content changes so installed apps re-import (user progress is never touched).

```
pip install -r content/requirements.txt
python content/scripts/validate.py
python content/scripts/build.py
python content/tests/test_<slug>.py     # per-card code checks (numpy/sklearn/torch)
```

## Layout
- `problems/<slug>.yaml` — one concept card per file, filename == slug (kebab-case).
- `sheets/<id>.yaml` — ordered lists of slugs in named sections. Pin sizes in `EXPECTED_SHEET_COUNTS`.
- `patterns/<id>.yaml` — optional "Compare/technique" cards (none yet).
- `tests/` — scripts that execute each card's `code` and compare with a reference library.

## Card fields (all-or-nothing for a full card)
Required: `slug, title, url (https reference), difficulty (Easy|Medium|Hard = Beginner/Intermediate/Advanced),
area (Classical ML|Deep Learning|NLP|Vision|RL|MLOps|Math), topics[]`.
Full card adds all of: `summary, intuition, approach[] ("step — why"), time, space (compute/memory),
code (Python, lines <= 70 chars), key_lines[1-3], when_to_use, when_not_to_use, why_it_works,
math[2-14 formula lines, <= 70 chars], interview_qa[3-5 of {q, a}], hyperparameters[2-5]`.
Optional: `pitfalls[2-4], related[1-3 slugs], patterns[]`.

The app still uses the old DSA column names; `build.py` maps ML fields onto them:
`area->platform, code->solution, when_to_use->description, when_not_to_use->bruteForce,
math->dryRun, interview_qa->examples (q->input, a->output), hyperparameters->constraints,
pitfalls->edgeCases, time/space = compute/memory`.

## Writing rules
Formulas are plain Unicode (ŷ = σ(wᵀx + b)). Intuition is the key insight, not a restatement.
Every `code` block must run; outputs/shapes are produced by executing, never by hand, and from-scratch
NumPy code is asserted against sklearn/PyTorch in `tests/`.
