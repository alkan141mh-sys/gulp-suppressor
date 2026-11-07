# data/

The repository has the **labels** (timestamps and label names only) and `playlist_ids.txt`, which lists the lecture number and YouTube video ID for each lecture. Lectures 037–040 are the held-out test set.

The tools expect this layout locally:

- `raw/` holds the lecture audio as `NNN - <title>.wav`. It is not in the repository; download it from the video IDs.
- `cache/` holds the frame features that `features.py` computes (not in the repository).
- `labels/` holds the original Audacity label tracks (start, end, label).
- `labels_review/` and `review/` hold the review-tool queues, the decisions (`decisions.json`) and the blind-audit keys.

Label names:

- `gulp`, `gulp_high`, `gulp_low` and `gulp_overlap` are gulps (a swallow, loud or quiet, or overlapping speech).
- Everything else, such as `breath` or `none`, counts as not a gulp.
