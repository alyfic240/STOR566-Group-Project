# Final presentation (Dec 1: 12-minute talk, 3 minutes of questions)

The deck can be exported to PowerPoint and PDF. This folder holds what the deck is built from, so the numbers
on the slides cannot drift from the paper:

| File | Content |
|---|---|
| `speaker_notes.md` | Slide titles and speaker notes with a suggested time per slide (the 12 talk slides add up to 11:50) |
| `QA_PREP.md` | Prepared answers to likely questions, with the backup slide to show for each |
| `figures/` | The figures on the slides (`python3 src/slide_figs.py results slides/figures`), plus two shared with the paper |
| `build_deck.py` | Writes the slide files from `paper/generated/numbers.json` and `results/` |
| `deck_root/project/` | The deck's source files as published: `deck.json` (slide order), `deck.css`, one HTML file per slide in `slides/` (speaker notes are in each file's `<aside>`) |
| `assets.json` | Maps the image ids used in the slide files (`/_blob/<id>`) to the files in `figures/` |

Slides 1–12 are the talk; slides 13–20 are backup slides for questions.
