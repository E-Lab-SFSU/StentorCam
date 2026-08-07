# StentorCam

Downstream analysis pipeline for *Stentor* (single-cell ciliate) tracking and
behavior quantification, consuming video recorded by
[RoboCam 3.1](https://github.com/dairyking98/RoboCam3.1) (and its
predecessors). This document describes the repository **as currently
uploaded** — it is a snapshot for orientation, not a design spec. Nothing
here has been reorganized or fixed yet; see the "Known gaps" section below
for what's rough.

There is no `requirements.txt`, `pyproject.toml`, environment file, or test
suite in the repo yet — dependencies below are inferred from each script's
imports.

---

## Two independent workflows

The repo currently contains **two separate, non-interoperating** tracking
pipelines. Nothing here converts one workflow's output into the other's
input format.

### 1. Fiji / TrackMate workflow (semi-manual)

```
video frames (PNG sequence)
  → stentor_preprocess.ijm   (Fiji macro: median-projection background
                               subtraction, Otsu binarization, manual
                               TrackMate detection/tracking pause)
  → per-track CSV export from TrackMate (one or more CSV files)
  → csv_compiler.py           (merge many CSVs into one, reassigning
                               track ids sequentially)
  → full_data_plot.py / track_plot.py   (population or per-track speed
                               plots)
```

- **`stentor_preprocess.ijm`** — Fiji/ImageJ macro. Loads a PNG frame
  sequence (hardcodes the first frame's name as `frame0001.png`), converts
  to 8-bit, computes a median-intensity projection as background, subtracts
  it, and launches TrackMate. The macro **pauses twice** for a human: once
  to manually tune TrackMate's detector and run tracking (`waitForUser`),
  and once after a *second* background subtraction + Otsu threshold to save
  the resulting binary mask by hand. Not automatable in its current form.
- **`csv_compiler.py`** — merges every CSV in a folder into one, matching
  columns via a small alias table (`track id`/`trackid`/`id`,
  `x`/`position_x`, `y`/`position_y`, `t`/`frame`/`time` — i.e. built for
  TrackMate/Fiji's export headers). Renumbers to a `new_track_id` every time
  the source `track id` changes, to keep tracks from different files or
  different segments distinct. Rows with a non-numeric track id are
  silently dropped.
- **`full_data_plot.py`** — reads a single compiled CSV (`-i`), assuming the
  Fiji export's layout: skips the first **4 lines** unconditionally, then
  reads columns by **fixed position** (`x`=index 1, `y`=index 2, `frame`=
  index 3, `track_id`=index 4), not by header name. Computes per-frame
  speed for every track, plots the population mean ± 1 SD and a
  frame-window sliding average, with a light/laser-color-coded shaded
  region assumed at frames 900–1800 (i.e. a fixed 30/30/30 s pre/on/post
  laser structure at 30 fps → 2700 total frames = 90 s).
- **`track_plot.py`** — same CSV format assumption as above, but plots up to
  5 individually-named tracks (`-s1`..`-s5`, each a comma-separated list of
  track ids to stitch together) side-by-side: raw trail, inverted-axis
  "expanded" trail, and per-track speed, plus the same population-average
  panels as `full_data_plot.py`. Includes a hardcoded reference circle
  (assumed well boundary) and a hardcoded 1 mm scale bar position, both
  tuned to one specific camera/zoom setup.

Both plotting scripts hardcode the same speed conversion:
`dist_px / dt_frames * 30 / 56.25` → mm/s, i.e. **30 fps and 56.25 px/mm
are assumed constants**, not derived from the input data or CLI arguments.
Both also hardcode the 2700-frame (90 s) timeline in axis limits and the
laser on/off shading. Any recording at a different frame rate, duration, or
camera zoom needs these constants hand-edited before the plots are
meaningful.

### 2. Standalone Python/OpenCV workflow (automatic)

> **Note:** the descriptions below (`stentTrack.py`, `multiTest.py`) reflect
> an earlier snapshot of this workflow's logic and are kept here for
> historical context. The files currently uploaded to the repo implement
> the same ideas as Colab notebooks — see
> [§3 Colab notebook pipeline](#3-colab-notebook-pipeline-currently-uploaded)
> below for what's actually in the repo today, including the CSV export
> that used to be missing.

```
video (mp4)
  → stentTrack.py   (single cell)      → CSV + ffmpeg overlay video
  → multiTest.py    (multiple cells)   → ffmpeg overlay video only (no CSV yet)
```

- **`stentTrack.py`** — single-stentor tracker. Per video: computes a
  median-frame background, then per frame does background subtraction →
  percentile threshold → largest contour → ellipse-fit to classify pose as
  `CONTRACTED` or `ELONGATED` (contour-to-ellipse fit error above/below a
  threshold) → for elongated cells, fits a minimum enclosing triangle and
  scores its three edges (by length dominance vs. alignment with recent
  motion direction) to pick a head edge / tail point, with a
  low-confidence fallback that re-scores assuming backward motion. Writes
  one CSV row per frame (`frame, x, y, pose, movement, direction_deg,
  contour`) and a debug overlay video (contour, pose box/triangle,
  head/tail markers, motion arrow) composited over the source video via a
  `subprocess` call to `ffmpeg`. Several `#debugging` comments mark
  in-progress tuning code the file's own docstring says not to treat as
  load-bearing.
- **`multiTest.py`** — despite the filename, its own header docstring names
  it `contour_video.py`; this mismatch is real and worth fixing/renaming
  once the file's identity settles. The most substantial file in the repo:
  a multi-cell tracker with a real correction pipeline —
  1. **Pass 1 (detection)**: background subtraction (with an optional
     circular ROI mask), Otsu threshold, distance-transform watershed to
     split touching cells, contour + centroid extraction per frame.
  2. **Pass 1b (identity)**: frame-to-frame track assignment via the
     Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) on
     centroid distance.
  3. **Pass 2 (correction sweep)**: drops tracks shorter than
     `min_track_len` as noise; linearly interpolates (centroid *and*
     resampled contour shape) across gaps up to `max_gap` frames; detects
     suspiciously large blobs (likely two cells merged by the mask) and
     re-splits them with a *seeded* watershed using neighboring tracks'
     predicted centroids as seeds.
  4. **Pass 2b (optional `--n_cells` enforcement)**: if the true cell count
     per frame is known ahead of time, per-frame reconciles detected count
     against it — under-detection tries a finer watershed re-seed first,
     then falls back to splitting a merge partner's blob or, last resort,
     pure interpolation; over-detection scores and drops the least
     track-consistent excess detections.
  5. **Pass 3**: renders an overlay (dashed = gap-filled, thick = merge-
     split, dotted = soft-recovered, cross = fully synthesized/forced,
     solid = real detection) and composites it over the source video via
     `ffmpeg`.
  This script **does not currently write a CSV** — only the overlay video
  — so none of its tracking data (positions, corrections applied, cell
  count) is persisted for the plotting scripts or anything else to consume.

`stentTrack.py`'s own CSV schema (`frame, x, y, pose, movement,
direction_deg, contour`) does not match what `full_data_plot.py` /
`track_plot.py` expect (the Fiji/TrackMate fixed-column layout above), so
today there is no script that takes either Python tracker's output straight
into the plotting scripts without a manual reformatting step.

### 3. Colab notebook pipeline (currently uploaded)

This is the workflow actually present in the repo today, as five Colab
notebooks meant to be run top-to-bottom in order. Each notebook follows the
same `#@title`-form-cell convention (install cell → helper cells → a final
hardcoded-path run cell), so cells can be collapsed and run in sequence
without editing code, only the constants at the top of the last cell.

```
video (mp4)
  → ROI_Selector.ipynb      (click 3 points on frame 1 → circular ROI:
                                center + radius, saved to roi_config.json)
  → stentorDetect2.ipynb      (multi-cell background-subtraction detector +
                                Hungarian-algorithm tracker + gap-fill/merge-
                                split correction sweep + optional n_cells
                                enforcement → tracks CSV + ffmpeg debug
                                overlay video)
  → trackID_Overlay.ipynb   (re-renders a clean fading-trail overlay for a
                                chosen subset of TRACK_IDs from that CSV,
                                with an optional on-screen laser-on/off
                                indicator)
  → multiParams.ipynb         (per-track pose/head-tail/motion-direction
                                analysis on the tracks CSV → analyzed CSV
                                with pose, movement, direction_deg columns
                                appended)
  → overlayMultiParams.ipynb  (renders the analyzed CSV's pose box, long-
                                axis line, and head/tail markers back onto
                                the source video)
```

- **`ROI_Selector.ipynb`** — loads the first frame of a video, then uses
  an HTML5 canvas + `google.colab.output.register_callback` (in place of
  Plotly's `FigureWidget.on_click`, which doesn't reliably fire in Colab's
  widget manager) to let you click 3 points on the well boundary. Fits a
  circle through those points and writes `roi_config.json`
  (`roi_cx`, `roi_cy`, `roi_r`) for the next notebook to consume.
- **`stentorDetect2.ipynb`** — the multi-cell detector/tracker/corrector
  (Pass 1 detection → Pass 1b Hungarian-algorithm identity assignment →
  Pass 2 correction sweep → optional Pass 2b `n_cells` enforcement →
  Pass 3 overlay render), matching the pipeline described for `multiTest.py`
  above, **plus a CSV export** (`export_tracks_csv`, columns `TRACK_ID`,
  `FRAME`, `POSITION_X`, `POSITION_Y`, `contour_points`) that the earlier
  script didn't have. This closes the "no CSV" gap noted below/in the
  earlier snapshot. Supports both Otsu and Bernsen local thresholding.
- **`trackID_Overlay.ipynb`** — loads that tracks CSV back in and renders
  a second overlay video restricted to chosen `TRACK_IDS`, adding a fading
  motion trail per track and a "LIGHT ON" indicator over a configurable
  laser-on/off time window. Independent of `stentorDetect2.ipynb`'s own
  debug overlay — this one is meant as a cleaner, presentation-ready render.
- **`multiParams.ipynb`** *(new — converted from `multiParams.py` in this
  update)* — walks the tracks CSV one `track_id` at a time and replays
  `stentTrack.py`'s single-cell classification logic (ellipse/aspect-ratio
  pose call, contour-width head/tail split, rolling 6-frame motion
  direction) independently per track, so a multi-cell video gets the same
  per-cell morphology/motion labels a single-cell video would. Appends
  `pose`, `movement`, and `direction_deg` columns and writes an analyzed
  CSV. Previously a CLI script (`argparse`, `--input`/`--output`); the
  notebook version replaces the CLI with an install cell and a hardcoded
  `INPUT_CSV`/`OUTPUT_CSV` run cell, matching the other notebooks' style.
- **`overlayMultiParams.ipynb`** *(new — converted from
  `overlayMultiParams.py` in this update)* — reads the analyzed CSV and
  draws each detection's oriented bounding box, long axis, head/tail-split
  line, and (for `ELONGATED` poses) head/tail centroid markers onto the
  source video via plain `cv2.VideoWriter` (no `ffmpeg` compositing step,
  unlike the other three notebooks). Also burns in the `TRACK_ID`, pose,
  movement, and direction-in-degrees as text per detection. Same CLI→
  hardcoded-cell conversion as `multiParams.ipynb`.

Unlike workflow 1, this pipeline **is** self-consistent end to end: each
notebook's output format is exactly what the next one expects, with no
manual reformatting step. It still doesn't interoperate with workflow 1's
CSV shape or plotting scripts (`full_data_plot.py`/`track_plot.py`) — see
Known gaps.

---

## Relationship to RoboCam 3.1

Both workflows above take an already-encoded **mp4 video** as their entry
point (`stentTrack.py --video`, `multiTest.py --video`, or a PNG sequence
for the Fiji macro). RoboCam 3.1's raw-burst capture mode writes each
well's frames directly as one memory-mapped `.npy` stack (see that repo's
`PROJECT_STATE.md` §§ 3–4, 8) and only produces an mp4/MKV as a
*post-processing* output, so video is no longer the only frame source
available upstream — whether tracking here should eventually read
`.npy`/PNG output directly instead of round-tripping through an encoded
video is an open question for later, not addressed by anything in this
repo yet.

---

## Dependencies (inferred from imports; no lockfile exists)

- `opencv-python` (`cv2`) — `stentTrack.py`, `multiTest.py`
- `numpy` — `stentTrack.py`, `multiTest.py`, `full_data_plot.py`,
  `track_plot.py`
- `pandas` — `stentTrack.py` (CSV output via `DataFrame.to_csv`)
- `tqdm` — `stentTrack.py`, `multiTest.py`
- `scipy` — `multiTest.py` only (`scipy.optimize.linear_sum_assignment`)
- `matplotlib` — `full_data_plot.py`, `track_plot.py` (both also attempt a
  custom style, `plt.style.use('BME163')`, that isn't bundled here — falls
  back to matplotlib's default style with a printed warning if missing)
- `ffmpeg` — external binary, invoked via `subprocess.run` by
  `stentTrack.py` and `multiTest.py` to composite the PNG overlay sequence
  back onto the source video. Not a Python dependency; must be on `PATH`.
- Fiji/ImageJ with the TrackMate plugin — external GUI application,
  required to run `stentor_preprocess.ijm` and produce its CSV exports.
  Entirely manual, not scriptable from this repo.

### Colab notebook pipeline (§3)

Each notebook installs its own dependencies in its first cell via `!pip`/
`!apt-get`, so nothing extra needs to be pre-installed to run them in
Google Colab:

- `opencv-python-headless` — all five notebooks
- `numpy` — all five notebooks
- `pandas` — `multiParams.ipynb`, `overlayMultiParams.ipynb`
- `tqdm` — `stentorDetect2.ipynb`, `multiParams.ipynb`
- `scipy` — `stentorDetect2.ipynb` only (`linear_sum_assignment`)
- `matplotlib` — `ROI_Selector.ipynb` (frame preview only)
- `ffmpeg` — external binary, `apt-get`-installed by `stentorDetect2.ipynb`
  and `trackID_Overlay.ipynb`; invoked via `subprocess.run` to composite
  the PNG overlay sequence back onto the source video. `overlayMultiParams.
  ipynb` writes video directly with `cv2.VideoWriter` instead, so it does
  not need `ffmpeg`.
- `google.colab` (`drive`, `output`, `files`) — Colab-only; used for Drive
  mounting (`stentorDetect2.ipynb`, `trackID_Overlay.ipynb`) and the
  click-to-select-ROI callback (`ROI_Selector.ipynb`). These notebooks
  will not run as-is outside Colab without stubbing these out.

## Known gaps

- No automated tests.
- No dependency manifest (`requirements.txt` etc.) — the Colab notebooks
  sidestep this by installing their own dependencies in-cell, but that
  only works inside Colab.
- ~~`multiTest.py` produces no CSV, only a rendered overlay video~~ —
  resolved in `stentorDetect2.ipynb`, which now calls `export_tracks_csv`
  after the correction sweep, so `TRACK_ID`/`FRAME`/`POSITION_X`/
  `POSITION_Y`/`contour_points` are persisted for downstream use (feeding
  `trackID_Overlay-2.ipynb`, `multiParams.ipynb`, and
  `overlayMultiParams.ipynb`). It still doesn't record *which* frames were
  gap-filled, merge-split, soft-recovered, or forced — the `corrected`
  label used to pick the overlay's drawing style in
  `stentorDetect2.ipynb`'s `draw_overlay` is not one of the exported
  columns.
- The Fiji-workflow plotting scripts (`full_data_plot.py`, `track_plot.py`)
  still hardcode fps (30), a px/mm calibration (56.25), and a fixed
  2700-frame/90 s recording length with laser-on window at frames
  900–1800 — none of this is derived from the input CSV or exposed as a
  CLI argument, so a different frame rate, recording duration, or camera
  zoom silently produces a mislabeled/wrong-scale plot rather than an
  error. The Colab notebook pipeline (§3) sidesteps this for the laser
  window specifically — `trackID_Overlay.ipynb` takes
  `LASER_ON_TIME`/`LASER_OFF_TIME` in seconds and converts using the
  video's own fps — but has no equivalent px/mm calibration or plotting
  step of its own yet.
- The two workflows still don't interoperate: nothing converts the Colab
  pipeline's tracks CSV (§3) into the Fiji/TrackMate fixed-column layout
  `full_data_plot.py`/`track_plot.py` expect, or vice versa.
- `ROI_Selector.ipynb`'s ROI selection is still a fully manual
  click-3-points step per video; nothing detects the well boundary
  automatically.
- `multiParams.ipynb` and `overlayMultiParams.ipynb` were converted from
  CLI scripts (`argparse`, `--input`/`--output`/`--video`/`--csv`) to
  Colab notebooks with hardcoded path variables in this update — like the
  rest of the pipeline, running them on a different file means editing the
  constants in the last cell rather than passing flags.
- No existing path consumes RoboCam 3.1's raw `.npy` well-stacks directly —
  everything here starts from an already-encoded video or PNG sequence.
