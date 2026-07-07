# pyroPythia — Project Spec

**The fire oracle** (Pythia of fire) — burned-area mapping for Greek wildfires
from Sentinel-2: dNBR/RBR physics baseline → U-Net segmentation → interactive
web map → honest evaluation.

**Naming:** pyroPythia = pyro (fire) + Pythia (the Delphic oracle) — camelCase
like iPhone. The oracle grows toward *prediction* (fire-risk forecasting).

## What we're building

Input: a Greek wildfire (location + rough date, or a FLOGA event id).
Output: pixel-precise burn-scar + severity map on an interactive web map,
with honest accuracy numbers.

1. **Data pipeline** — FLOGA events (pre/post Sentinel-2 pairs + Fire Service
   labels); later CDSE STAC (`stac.dataspace.copernicus.eu/v1`) for fresh
   fires outside FLOGA.
2. **Physics baseline** — NBR/dNBR/RBR thresholding (USGS severity classes).
   A real map with zero ML; the honest bar the U-Net must beat.
3. **U-Net segmentation** — burned/unburned + severity. Start from FLOGA's
   BAM-CD baseline as the published reference; beat or honestly match it.
4. **Evaluation layer (the signature)** — IoU/F1 vs Fire Service ground truth,
   **spatial-block CV** (adjacent pixels/events leak — never random splits),
   per-fire breakdown (big vs small fires), validation against EFFIS
   perimeters, failure modes documented in EVALUATION.md.
5. **Web map** — interactive burn perimeter + severity overlay (maplibre/
   leaflet), hectares burned, severity breakdown. Time-wipe pre↔post imagery
   slider.
6. **Deploy on AWS** — static-artifact pattern: inference is offline/
   scheduled, the map is static files + tiles.

**Growth path (parking lot, not scope):** fresh-fire mode via CDSE STAC within
days of a new fire; Sentinel-1 SAR fusion for cloud/smoke robustness (CaBuAr,
EO4WildFires); operational monitor validated against EFFIS.

## Data sources (all free)

- **FLOGA:** https://github.com/Orion-AI-Lab/FLOGA (paper: arXiv 2311.03339) —
  326 Greek fires 2017–2021, Sentinel-2 pre/post pairs, Hellenic Fire Service
  ground truth, code MIT / data CC BY 4.0, PyTorch baseline (BAM-CD).
  Yearly HDF5s on Dropbox (12–82 GB each); we read them remotely via HTTP
  range requests (`scripts/floga_remote.py`) — no bulk download.
- **EFFIS** burnt-area perimeters — independent validation layer.
- **CDSE STAC v1** for fresh Sentinel-2 (needs free account); **Microsoft
  Planetary Computer** for georeferenced scene copies (no account).

## Build spine (ship-not-sprawl, each step a working artifact)

1. **One fire end-to-end, no ML:** one FLOGA event (Evia 2021), dNBR,
   severity classes, scored vs the Fire Service label. ✅
2. **Baseline over the dataset:** dNBR/RBR thresholds tuned on a train split,
   evaluated properly — the honest bar.
3. **U-Net:** train on FLOGA (subset first), beat the baseline; compare
   against published BAM-CD numbers (replicating their patch protocol).
4. **Evaluation layer:** spatial-block CV, per-fire analysis, EFFIS
   cross-check, EVALUATION.md with failure notes.
5. **Web map + deploy:** interactive viewer, AWS, scheduled/offline inference.

## Hard-won data notes

See EVALUATION.md ("Data notes" + "Georeferencing") before touching FLOGA
data: alphabetical band order, SCL-not-binary cloud masks, label==2 semantics,
geographic (lat/lon) pixel grid — NOT the S2 UTM grid, no georeferencing in
the HDFs (we recover it per event: `scripts/georef_affine.py`).

## Known risks

- **Label edge cases** — Fire Service perimeters are polygons, not pixel
  truth; partially-burned agricultural land and clouds in post-fire images
  are the classic failure modes. Document, don't hide.
- **Geospatial CRS hell** — reprojection bugs are silent; sanity-check every
  raster/vector overlay visually early.

## Working conventions (house rules)

- **Ask before deploying** anything with a slow feedback loop.
- Disable, don't delete.
- Batch edits, then one commit; don't panic on CDN/cache lag.
- New ideas → parking lot, not this repo's scope. Risk is breadth without
  shipping.

## Related repos

- `o-ilios` — sibling flagship; reuse: baseline-first discipline,
  EVALUATION.md style, worker→static-artifact serving, React patterns.
- `archeologic` — sibling flagship #2 (LLM-agent lane).
- `CaptainJimbo.github.io` — portfolio site; gets a card at launch.
