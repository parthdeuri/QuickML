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
- 116 concept cards with all code tests passing; 22 families, 12 compare cards, 17 diagrams.
- Card wording (card / learned / implementation), Ask prompt includes math.
- Reference links all return HTTP 200 (`tools/check_links.py`).

## Not done
- Scenario quiz ("which loss or metric fits this?", "what does this curve show?") needs a new quiz
  screen and question content.
- Practice mode masks code lines only, not formula lines.
- Play Store prep: privacy policy (Ask sends text to Google when on), store listing, AAB build,
  licence and attribution check.
- Later ideas: cloud sync, FSRS scheduler, other AI providers.

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
