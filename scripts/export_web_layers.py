"""Export a georeferenced event's burn layers for the web map.

Produces, in web/data/:
  <name>_severity.png   RGBA dNBR severity overlay (EPSG:4326, north-up)
  <name>_severity.json  overlay bounds [w, s, e, n]
  <name>_perimeter.geojson  burn perimeter (Fire Service label), simplified
  <name>_meta.json      stats for the UI

Usage: python scripts/export_web_layers.py data/floga/FLOGA_2021_event62_sen2_20.h5 evia
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import rasterio
from matplotlib.colors import to_rgba
from pyproj import Transformer
from rasterio import features
from rasterio.transform import Affine, from_bounds
from rasterio.warp import Resampling, reproject
from shapely.geometry import mapping, shape
from shapely.ops import transform as shp_transform, unary_union

from dnbr_evia import BURN_THRESHOLD, SEV_BOUNDS, SEV_COLORS, SEV_LABELS, nbr

OUT = Path("web/data")


def main() -> None:
    path = Path(sys.argv[1])
    name = sys.argv[2]
    OUT.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "r") as hdf:
        year = list(hdf.keys())[0]
        event_id = list(hdf[year].keys())[0]
        ev = hdf[year][event_id]
        pre, post = ev["sen2_20_pre"][()], ev["sen2_20_post"][()]
        label = ev["label"][0]
        scl_pre, scl_post = ev["sen2_20_cloud_pre"][0], ev["sen2_20_cloud_post"][0]
        sea = ev["sea_mask"][0]
        attrs = dict(ev.attrs)

    geo = json.loads(path.with_suffix(".georef.json").read_text())
    a, b, c, d, e, f_ = geo["affine"]
    src_transform = Affine(a, b, e, c, d, f_)
    src_crs = f"EPSG:{geo['epsg']}"

    dnbr = nbr(pre) - nbr(post)
    bad_scl = [0, 1, 3, 8, 9, 10]
    valid = ((sea == 0) & ~np.isin(scl_pre, bad_scl) & ~np.isin(scl_post, bad_scl)
             & ~np.isnan(dnbr) & (label != 2))

    # severity class index: 0 = transparent, 1..len(SEV_LABELS) = classes;
    # only show burn classes (low and up) so the basemap stays visible
    sev = np.zeros(label.shape, dtype=np.uint8)
    for i, (lo, hi) in enumerate(zip(SEV_BOUNDS[:-1], SEV_BOUNDS[1:])):
        if SEV_LABELS[i] in ("regrowth", "unburned"):
            continue
        sev[valid & (dnbr > lo) & (dnbr <= hi)] = i + 1

    # crop to the labeled fire + margin (dNBR positives elsewhere in the
    # scene — other fires, ag fields — would inflate the box)
    rows, cols = np.where(label == 1)
    m = 100
    r0, r1 = max(rows.min() - m, 0), min(rows.max() + m, label.shape[0])
    c0, c1 = max(cols.min() - m, 0), min(cols.max() + m, label.shape[1])
    sev_c = sev[r0:r1, c0:c1]
    crop_transform = src_transform * Affine.translation(c0, r0)

    # destination: north-up EPSG:4326 grid at ~0.0002 deg
    tr = Transformer.from_crs(geo["epsg"], 4326, always_xy=True)
    corners = [(cc, rr) for rr in (0, sev_c.shape[0]) for cc in (0, sev_c.shape[1])]
    lonlat = [tr.transform(*(crop_transform * p)) for p in corners]
    w, s = min(p[0] for p in lonlat), min(p[1] for p in lonlat)
    e_, n = max(p[0] for p in lonlat), max(p[1] for p in lonlat)
    res = 0.0002
    dw, dh = int((e_ - w) / res), int((n - s) / res)
    dst = np.zeros((dh, dw), dtype=np.uint8)
    reproject(sev_c, dst, src_transform=crop_transform, src_crs=src_crs,
              dst_transform=from_bounds(w, s, e_, n, dw, dh), dst_crs="EPSG:4326",
              resampling=Resampling.nearest)

    lut = np.zeros((len(SEV_LABELS) + 1, 4), dtype=np.uint8)
    for i, color in enumerate(SEV_COLORS):
        r_, g_, b_, _ = to_rgba(color)
        lut[i + 1] = [int(r_ * 255), int(g_ * 255), int(b_ * 255), 200]
    rgba = lut[dst]

    import matplotlib.image
    matplotlib.image.imsave(OUT / f"{name}_severity.png", rgba)
    (OUT / f"{name}_severity.json").write_text(json.dumps({"bounds": [w, s, e_, n]}))

    # burn perimeter from the label, in lon/lat, simplified
    shapes = [shape(g) for g, v in features.shapes(
        (label == 1).astype(np.uint8), transform=src_transform) if v == 1]
    poly = unary_union(shapes)
    poly4326 = shp_transform(lambda x, y: tr.transform(x, y), poly).simplify(0.0005)
    (OUT / f"{name}_perimeter.geojson").write_text(json.dumps({
        "type": "Feature", "geometry": mapping(poly4326),
        "properties": {"event": event_id, "year": year},
    }))

    px_ha = geo["px_area_m2"] / 1e4
    meta = {
        "name": name, "year": year, "event": event_id,
        "pre_date": attrs["pre_image_date"], "post_date": attrs["post_image_date"],
        "label_ha": round(float((label == 1).sum() * px_ha)),
        "dnbr_ha": round(float(((dnbr > BURN_THRESHOLD) & valid).sum() * px_ha)),
        "severity_ha": {SEV_LABELS[i]: round(float((sev == i + 1).sum() * px_ha))
                        for i in range(len(SEV_LABELS))
                        if SEV_LABELS[i] not in ("regrowth", "unburned")},
        "center": [round((w + e_) / 2, 5), round((s + n) / 2, 5)],
    }
    (OUT / f"{name}_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"wrote {OUT}/{name}_*.{{png,json,geojson}}")


if __name__ == "__main__":
    main()
