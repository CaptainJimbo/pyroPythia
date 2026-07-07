# pyroPythia 🔥🔮

**The fire oracle.** Maps burned land from satellite imagery: give it a Greek
wildfire — a place and a date — and it returns a pixel-precise, severity-graded
burn map, rendered on an interactive web map, with accuracy numbers you can trust.

> The Pythia of Delphi read the future in vapors rising from the earth.
> This one reads the aftermath of fire in light reflected from it — and one day,
> the danger before it starts.

## What it does

1. **Fetches two satellite snapshots** — pre-fire and post-fire Sentinel-2
   imagery (~10 m resolution, free, ESA Copernicus).
2. **Computes burn indices** — NBR / dNBR / RBR: charred ground and healthy
   vegetation reflect near-infrared light very differently, so the difference
   image makes the burn scar light up. A decent map with zero ML — the baseline.
3. **Segments the burn scar with a U-Net** — pixel-by-pixel: burned or not,
   and how severely (low / moderate / high). The ML that beats the physics.
4. **Renders an interactive web map** — burn perimeter over terrain, hectares
   burned, severity breakdown.
5. **Reports honest accuracy** — IoU against Hellenic Fire Service ground
   truth, **spatial-block cross-validation** (no leakage from neighboring
   pixels), failure modes documented.

## Data

- **[FLOGA](https://github.com/Orion-AI-Lab/FLOGA)** — ML-ready dataset from
  NOA's Orion-AI-Lab: 326 Greek wildfire events (2017–2021), paired pre/post
  Sentinel-2 + MODIS imagery, burnt-area ground truth annotated by the
  **Hellenic Fire Service**. MIT-licensed, with a PyTorch baseline (BAM-CD).
- **EFFIS** fire perimeters — independent validation.
- **Copernicus Data Space Ecosystem (CDSE)** — fresh Sentinel-2 for fires
  outside FLOGA.

## Status

🚧 Early scaffold — private while under construction.

## License & attribution

Code is [MIT](LICENSE) © Dimitris Kogias.

This project builds on open data and gratefully credits:

- **FLOGA dataset** — © [Orion-AI-Lab, National Observatory of Athens](https://orionlab.space.noa.gr/),
  licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (data) and
  MIT (code). We use the imagery, masks, and Hellenic Fire Service labels in
  modified form (single-event extracts, derived statistics, reprojected map
  layers). Please cite their paper:
  > Sdraka et al., *"FLOGA: A machine learning ready dataset, a benchmark and
  > a novel deep learning model for burnt area mapping with Sentinel-2"*,
  > IEEE JSTARS 2024. [arXiv:2311.03339](https://arxiv.org/abs/2311.03339)
- **Sentinel-2 imagery** — contains modified Copernicus Sentinel data
  (2017–2021), ESA. Scene copies for georeferencing via
  [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/).
- **Burnt-area ground truth** — Hellenic Fire Service, via FLOGA.
- Web map basemap imagery © Esri; coastlines from
  [GSHHG](https://www.soest.hawaii.edu/pwessel/gshhg/).

## Stack (planned)

- **ML:** Python, PyTorch, semantic segmentation (U-Net / BAM-CD)
- **Geospatial:** rasterio, rioxarray, geopandas, xarray; STAC (pystac-client)
- **Web map:** interactive burn-scar viewer (leaflet/maplibre family)
- **Deploy:** AWS

---

*Built by [Dimitris Kogias](https://captainjimbo.github.io) — physicist & AI/ML
systems engineer. Sibling projects:
[Ο Ήλιος — The Living Sun](https://github.com/CaptainJimbo/o-ilios) (solar CV) ·
[ArcheoLogic](https://github.com/CaptainJimbo/archeologic) (AI archaeologist).
Fires on Earth, storms on the Sun.*
