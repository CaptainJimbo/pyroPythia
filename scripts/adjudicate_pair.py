"""Adjudicate a candidate duplicate pair using recovered georeferencing.

Unlike blob phase-correlation (which can lock onto a spurious offset), this
maps event B's burned pixels through both geotransforms into event A's grid
— including across UTM zones — and computes the true geographic IoU.

Usage: python scripts/adjudicate_pair.py <event_a.h5> <event_b.h5>
Requires .georef.json sidecars (run scripts/georef_event.py on both first).
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np
from pyproj import Transformer


def load(path: Path):
    with h5py.File(path, "r") as hdf:
        year = list(hdf.keys())[0]
        event_id = list(hdf[year].keys())[0]
        label = hdf[year][event_id]["label"][0] == 1
    geo = json.loads(path.with_suffix(".georef.json").read_text())
    return event_id, label, geo


def main() -> None:
    id_a, burn_a, geo_a = load(Path(sys.argv[1]))
    id_b, burn_b, geo_b = load(Path(sys.argv[2]))
    aa, ab, ac, ad, ae, af = geo_a["affine"]
    ba, bb, bc, bd, be, bf = geo_b["affine"]

    rows, cols = np.where(burn_b)
    east_b = ba * cols + bb * rows + be
    north_b = bc * cols + bd * rows + bf
    if geo_a["epsg"] != geo_b["epsg"]:
        tr = Transformer.from_crs(geo_b["epsg"], geo_a["epsg"], always_xy=True)
        east_b, north_b = tr.transform(east_b, north_b)

    # invert a's affine to map UTM -> a's pixel indices
    inv = np.linalg.inv(np.array([[aa, ab], [ac, ad]]))
    ca = np.round(inv[0, 0] * (east_b - ae) + inv[0, 1] * (north_b - af)).astype(int)
    ra = np.round(inv[1, 0] * (east_b - ae) + inv[1, 1] * (north_b - af)).astype(int)
    inside = (ra >= 0) & (ra < burn_a.shape[0]) & (ca >= 0) & (ca < burn_a.shape[1])
    inter = int(burn_a[ra[inside], ca[inside]].sum())

    na, nb = int(burn_a.sum()), int(burn_b.sum())
    iou = inter / (na + nb - inter)
    ha = lambda n: n * geo_a["px_area_m2"] / 10_000
    frac_a, frac_b = inter / na, inter / nb
    verdict = "DUPLICATE" if iou > 0.5 else ("PARTIAL" if max(frac_a, frac_b) > 0.5 else "distinct")

    print(f"pair {id_a}|{id_b}: geo-IoU {iou:.3f}  "
          f"inter {ha(inter):,.0f} ha = {frac_a:.1%} of {id_a} / {frac_b:.1%} of {id_b}  "
          f"[fit inliers {geo_a['inliers']}/{geo_b['inliers']}]  {verdict}")


if __name__ == "__main__":
    main()
