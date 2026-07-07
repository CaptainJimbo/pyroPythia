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
    px = 20.0

    rows, cols = np.where(burn_b)
    east_b = geo_b["ul_east"] + (cols + 0.5) * px
    north_b = geo_b["ul_north"] - (rows + 0.5) * px
    if geo_a["epsg"] != geo_b["epsg"]:
        tr = Transformer.from_crs(geo_b["epsg"], geo_a["epsg"], always_xy=True)
        east_b, north_b = tr.transform(east_b, north_b)

    ca = np.round((east_b - geo_a["ul_east"]) / px - 0.5).astype(int)
    ra = np.round((geo_a["ul_north"] - north_b) / px - 0.5).astype(int)
    inside = (ra >= 0) & (ra < burn_a.shape[0]) & (ca >= 0) & (ca < burn_a.shape[1])
    inter = int(burn_a[ra[inside], ca[inside]].sum())

    na, nb = int(burn_a.sum()), int(burn_b.sum())
    iou = inter / (na + nb - inter)
    ha = lambda n: n * 400 / 10_000
    frac_a, frac_b = inter / na, inter / nb
    verdict = "DUPLICATE" if iou > 0.5 else ("PARTIAL" if max(frac_a, frac_b) > 0.5 else "distinct")

    print(f"pair {id_a}|{id_b}: geo-IoU {iou:.3f}  "
          f"inter {ha(inter):,.0f} ha = {frac_a:.1%} of {id_a} / {frac_b:.1%} of {id_b}  "
          f"[land agreement {geo_a['land_agreement']:.0%}/{geo_b['land_agreement']:.0%}]"
          f"  {verdict}")


if __name__ == "__main__":
    main()
