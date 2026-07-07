# EVALUATION.md — honest numbers, caveats first

Running log of evaluation results, methodology decisions, and failure modes.
The rule: every number ships with its caveat.

## Data notes (learned the hard way, 2026-07-07)

- **FLOGA access**: yearly HDF5 files on Dropbox (12–82 GB each; 240 GB total
  @10m, 140 GB @20m, 20 GB @60m). We read them **remotely via HTTP range
  requests** (`scripts/floga_remote.py`) — single events extract to ~0.5 GB
  local files; no bulk download.
- **Band order** in `sen2_*` arrays is *alphabetical*:
  `B02 B03 B04 B05 B06 B07 B11 B12 B8A` at 20 m (9 bands). So NIR = B8A =
  index 8, SWIR2 = B12 = index 7. Verified against FLOGA's config/notebook
  (60 m has 11 bands, NIR at index 10 — same alphabetical convention).
- **`sen2_*_cloud_*` is not a binary cloud mask** — it's the Sentinel-2 Scene
  Classification Layer (SCL): 0 nodata, 1 saturated, 2 dark, 3 shadow,
  4 vegetation, 5 bare, 6 water, 7 unclassified, 8/9 cloud, 10 cirrus.
- **`sea_mask`**: 0 = land, 255 = sea, 253 = (unresolved — investigate),
  small values 1–4 rare.
- **`label`**: 0 unburned, 1 burned, **2 = burned by a different event**
  (seen in event 59) — must be excluded from both pred and truth, not
  treated as unburned.
- Event bounding boxes are whole Sentinel-2 scenes (~110×127 km), mostly sea
  for island fires — always report metrics on valid (land, clear-sky) pixels
  and say so.

## Ground-truth caveats

- Labels are Hellenic Fire Service **perimeter polygons rasterized**, not
  pixel truth. Unburned islands inside a perimeter count as "burned" in the
  label; dNBR sees them as unburned → shows up as FN fringe that may actually
  be *label* error. Document, don't hide.
- Evia 2021 (event 61): label says **27,394 ha** on valid pixels; media/EFFIS
  figures cite ~50 k ha. RESOLVED (see finding below): event 61 is a *partial
  view* — the fire straddles the T34SFJ/T34SGJ tile boundary and appears in
  both events 61 and 62. Deduplicated size 49,435 ha ≈ the EFFIS figure.

## ★ Finding 1 (2026-07-07): duplicate events leak across FLOGA's official splits

The Evia 2021 megafire appears as TWO events — 61 (T34SGJ, 27,554 ha) and
62 (T34SFJ, 45,393 ha) — from the *identical* Sentinel-2 acquisitions
(datatakes 20210803T090601 / 20210818T090559, orbit R050), because the fire
sits on the tile boundary and adjacent S2 tiles overlap ~10 km.

Evidence (`scripts/check_duplicate_event.py`; no georef in the HDFs, so
alignment via phase correlation of the land masks): recovered shift 111.3 km
east / 1.5 km north (exactly the tile offset), land-mask agreement 77.9 %,
**burned-label intersection 23,512 ha = 85.3 % of event 61's entire burn**.
Union 49,435 ha ≈ EFFIS ~50 k ha.

**FLOGA's published `data_split.csv` assigns 62 → train, 61 → val.** So the
largest Greek fire of the period sits on both sides of the benchmark split
with pixel-identical imagery and labels — a textbook spatial leak. Implications:
- Published val numbers are likely optimistic (megafire seen in training).
- Our own splits must group events by *spatial overlap*, not trust event ids
  as independent units. Scan all years for further duplicate pairs (same
  datatake, adjacent tiles) before designing spatial-block CV.
- Numbers like "total 175,561 ha burned in 2021" double-count overlaps —
  never sum raw event labels.
- Caveat before publicizing: verify 61/62 aren't intentionally distinct
  (e.g. two administrative fire records); check the FLOGA paper's dedup
  discussion, and raise as a question/issue with the authors first.

## Results log

### 2026-07-07 — dNBR physics baseline, Evia 2021 (event 61), 20 m

- Setup: NBR = (B8A − B12)/(B8A + B12); dNBR = NBR_pre − NBR_post;
  burned := dNBR > 0.27 (USGS moderate-low+). Valid px only (land, clear in
  both SCLs). Pre 2021-08-03, post 2021-08-18.
- **IoU 0.861, F1 0.925.** Baseline hectares 24,039 vs label 27,394.
- Caveat: **this is dNBR's best case** — an enormous, mostly high-severity
  fire with clean pre/post scenes. Expect the baseline to degrade on small,
  low-severity, agricultural, or cloud-affected events. The baseline bar must
  be computed over a *distribution* of fires (size-stratified), not one hero
  event. Label-size scan of all 2021 events in progress for exactly this.
- Failure texture: thin FN fringe at perimeter edges + FN patches inside the
  perimeter (surviving vegetation → see ground-truth caveats). FP negligible.

### 2026-07-07 — dNBR baseline, event 62 (Evia, full view, T34SFJ), 20 m

- Same recipe. **IoU 0.858, F1 0.924** (45,307 ha valid-label vs 40,093 ha
  predicted). Near-identical to event 61 — consistent with it being the same
  fire (→ Finding 1).

### 2021 fire-size census (labels of all 105 events, `2021_label_stats.csv`)

- Quartiles: 49 / 110 / 338 ha; max 45,393 ha (Evia). The median FLOGA fire
  is ~100 ha ≈ 2.5k pixels at 20 m — megafires are the exception. Pooled
  metrics will be dominated by Evia-class events; all reporting must be
  size-stratified. Note: totals across events double-count tile overlaps
  (Finding 1).

### 2026-07-07 — dNBR baseline across the fire-size ladder (2021, 20 m)

| event | burned (label) | pre→post gap | F1 | IoU | failure mode |
|---|---|---|---|---|---|
| 62 (Evia) | 45,393 ha | 15 d | **0.924** | 0.858 | — (dNBR's best case) |
| 61 (Evia, dup view) | 27,554 ha | 15 d | 0.925 | 0.861 | — |
| 40 | 184 ha | 5 d | 0.874 | 0.777 | — |
| 54 | 389 ha | 5 d | 0.819 | 0.693 | mild over-prediction |
| 80 | 111 ha | 5 d | **0.275** | 0.159 | agricultural mosaic: harvest/tillage between acquisitions reads as burn (3.5× over-prediction) |
| 141 | 874 ha | **55 d** | **0.047** | 0.024 | long revisit gap: 55 days of summer senescence → 12,162 ha predicted vs 874 labeled |

Takeaways:
- The physics bar is **not one number**: F1 spans 0.05–0.93 driven by fire
  size, land cover, and revisit gap. Pooled "baseline F1" would be a lie;
  report per-stratum.
- The two dominant failure modes (agricultural false alarms; long-gap
  senescence) are *context* problems, not spectral ones — exactly what a
  U-Net with spatial context should fix. This is the ML value proposition,
  now measured rather than assumed.
- Possible baseline improvements to keep the fight fair (before crowning the
  U-Net): RBR instead of dNBR, CLC mask to exclude agriculture, per-event
  dNBR offset (dNBR of unburned surroundings subtracted). Try before step 3.

## Methodology decisions

- **Splits**: spatial-block CV, never random pixel/patch splits (adjacent
  patches leak). FLOGA's own event-level splits (`data_split.csv`) are the
  published-comparison reference; our stricter splits are the headline.
- **Metrics**: IoU + F1 on burned class, per-event; report size-stratified
  aggregates (small/medium/large fires), not just a single pooled number
  (pooled numbers are dominated by megafires like Evia).
- **Severity**: labels are binary → ML learns *extent*; severity stays a
  physics (dNBR class) layer. Two honest layers, no fake severity truth.
