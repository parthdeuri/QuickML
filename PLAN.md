# QuickML plan and status

Personal Android app for revising ML and deep learning in a reels-style feed with spaced repetition.
Reuses the Revise (DSA) app's architecture. No local Android SDK: every build runs in GitHub
Actions and the user reports pass or fail.

## Decisions (locked)
| Topic | Decision |
|---|---|
| Stack | Kotlin 2.1 + Jetpack Compose, Room, WorkManager, manual DI, no nav library |
| Package | `com.parthdeuri.quickml`, signed with a stable key from GitHub secrets |
| Math | LaTeX between `$...$` in card text, typeset in-app by `MathText` (jlatexmath-android); `math_tex` feeds the Math box, Unicode `math` kept for speech |
| Diagrams | Pre-drawn PNGs (light and dark) from `content/scripts/make_diagrams.py`, shown in the Approach tab; card field `diagram: <name>` |
| Content | One YAML per concept card, validated by `content/scripts/validate.py` |
| Verification | Every card's code is run in `content/tests/` against sklearn / PyTorch / SciPy |
| AI | Optional "Ask" tab (Gemini, own key, off by default) |

## Where the content lives
- `content/problems/*.yaml` concept cards (the folder keeps its old name; the app calls them cards).
- `content/sheets/*.yaml` three sheets: ML Core, Deep Learning Core, NLP / Vision / RL.
- `content/patterns/*.yaml` technique families (quiz) and "Compare: A vs B" cards.
- Scripts: `assemble.py` (sheets and family tags), `make_families.py`, `make_compare.py`,
  `make_diagrams.py` (needs matplotlib, output committed); `tools/check_links.py` checks every reference URL.

## Field mapping (YAML to the older app columns)
`area -> platform`, `code -> solution`, `when_to_use -> description`, `when_not_to_use -> bruteForce`,
`math -> dryRun`, `interview_qa -> examples`, `hyperparameters -> constraints`,
`pitfalls -> edgeCases`, `time/space -> compute/memory`. The database keeps these column names on
purpose: renaming them would need a migration and gives users nothing. Kotlin class and function
names were renamed from Problem to Concept; table and column names were not.

## Done
- Skeleton, CI, signing, stable-key builds; logo (flash cards with a bolt).
- 178 concept cards with all code tests passing; 22 families, 12 compare cards, 4 sheets.
- Card wording (card / learned / implementation), Ask prompt includes math.
- Reference links all return HTTP 200 (`tools/check_links.py`).
- 17 diagrams as pre-drawn light/dark images (replaced the text art that broke on phones).
- Typeset maths: LaTeX inline in all card text and in the Math box, rendered live in the app.
  The APK stays small (assets about 1.1 MB).

## Done: LLM Engineer roadmap, concepts and techniques (sheet `llm-engineer`, 92 entries)
Roadmap sections 5-10, 12, 13, 16-18 are in (62 new cards, area `LLM`; fairness-metrics is `MLOps`).
New cards have no image diagram yet (an arrow-flow line ends the Intuition text); real diagrams come in a
later run. Not covered yet: sections 14-15 (agents, multimodal), 11 and 19 (tools, LLMOps); foundations,
projects and career are not revision material. Several cards cite abstract pages only; facts beyond the
abstracts were checked by the card's own tests, not by the cited page.

## Next: flash cards (planned, nothing started)
Auto-generated from the existing cards, spaced repetition, new bottom tab.
1. Content: `build.py` derives about 800-900 cards (interview Q&A, key insight, when to use,
   pitfalls, formulas, compare cards) into `flashcards.json`; validator and `test_flashcards.py`.
2. Data: DB migration 7 to 8 (`flashcards` content table, `flash_review` user table, kept apart from
   `review_state` so Revise counts are unaffected), `FlashDao`/`FlashRepository` on `Sm2Scheduler`,
   importer keeps reviews, backup includes them.
3. UI: Flash tab with deck picker and type chips, due then new cards up to a daily limit, tap to
   flip, Forgot/Hard/Good/Easy with interval preview, session summary, "Open concept" link.
   Move `RatingRow` to a shared file. Use `MathText` for formulas.
4. Wrap-up: dashboard due count, daily-limit setting, README/PLAN update, version bump.

## Not done
- Scenario quiz ("which loss or metric fits this?", "what does this curve show?") needs a new quiz
  screen and question content.
- Practice mode masks code lines only, not formula lines.
- Play Store prep: privacy policy (Ask sends text to Google when on), store listing, AAB build,
  licence and attribution check.
- Later ideas: cloud sync, FSRS scheduler, other AI providers.
- Phone check still owed: diagrams, inline formulas, Math box; progress kept after the DB v7 update.

## Known card limits
- Decision tree: matches sklearn exactly up to depth 4; deeper trees differ only through sklearn's
  random tie-breaking between equally good splits.
- Several cards are deliberate simplifications; each says so in its pitfalls or text.
- CS229's syllabus page could not be fetched (certificate error), so coverage was checked against
  d2l.ai, the scikit-learn user guide, Made With ML (section level), CS224N and the Hugging Face
  course instead.

## Workflow rules
1. Edit YAML, run `validate.py`, `build.py`, and the card's test.
2. Bump `content/version.txt` so installed apps re-import (user progress is never touched).
3. Commit and push; check the Actions tab; install the `quickml-apk` artifact over the old one.
