# QuickML

Android app for revising machine learning and deep learning in a swipeable, reels-style feed with
spaced repetition. Built from the Revise (DSA) app's architecture.

- **Stack:** Kotlin + Jetpack Compose, Room, WorkManager. Built only in GitHub Actions
  (`.github/workflows/build.yml`); the APK is the `quickml-apk` artifact.
- **Content:** YAML in `content/` (see `content/README.md`): 116 concept cards, 3 sheets,
  33 technique families and 23 "Compare: A vs B" cards. Every card's code is checked against
  sklearn or PyTorch by `content/tests/`.
- **Features:** feed modes (Learn, Shuffle, Revise, Quiz, Families, Bookmarks), notes, own
  implementations, dashboard, backup/restore, read-aloud, reminders, widget, optional AI "Ask" tab
  (Gemini, your own key, off by default).

## Content workflow
```
pip install -r content/requirements.txt
python content/scripts/validate.py
python content/scripts/build.py          # writes app/src/main/assets/problems.json
for t in content/tests/test_*.py; do python $t; done
```
Bump `content/version.txt` after any content change so installed apps re-import.
`content/scripts/assemble.py` rebuilds the sheets and family tags, `make_families.py` and
`make_compare.py` regenerate the family and compare cards.

## Signing
`python tools/make_signing_key.py` creates the permanent key outside the repo; put it in the
`QUICKML_KEYSTORE_BASE64` / `QUICKML_KEYSTORE_PASSWORD` GitHub secrets. Never commit keys.

## Flash cards
The Flash tab quizzes recall with spaced repetition. The 2157 cards are derived from the concept cards by
`content/scripts/flashcards.py` (written to `app/src/main/assets/flashcards.json` by `build.py`); nothing
is hand-written. Run `python content/tests/test_flashcards.py` after `build.py`.
