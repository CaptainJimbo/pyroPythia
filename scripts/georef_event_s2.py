"""Precise georeferencing of a FLOGA event via its own Sentinel-2 scene.

The event names its exact S2 acquisition; Microsoft Planetary Computer serves
that scene as georeferenced COGs. FLOGA's `sen2_20_cloud_*` raster is the
scene's SCL band, so registering event-SCL against scene-SCL matches a
product against itself — sharp everywhere (inland included), unlike
coastline matching (scripts/georef_event.py, superseded).

Usage: python scripts/georef_event_s2.py <event.h5>
Writes the .georef.json sidecar (overwrites the coastline-based one).
Registration at 60 m, refined to the exact 20 m offset by local search.
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import planetary_computer
import pystac_client
import rasterio
from pyproj import Transformer

from check_duplicate_event import phase_corr_peaks

MPC = "https://planetarycomputer.microsoft.com/api/stac/v1"


def scene_band(scene_name: str, asset: str = "B12"):
    """Fetch a scene band at 60 m + its geotransform from MPC."""
    parts = scene_name.split("_")
    tile, date = parts[5][1:], parts[2][:8]
    day = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    cat = pystac_client.Client.open(MPC, modifier=planetary_computer.sign_inplace)
    items = list(cat.search(collections=["sentinel-2-l2a"], datetime=day,
                            query={"s2:mgrs_tile": {"eq": tile}}).items())
    if not items:
        raise SystemExit(f"scene not found on MPC: {scene_name}")
    item = items[0]
    with rasterio.open(item.assets[asset].href) as src:
        h, w = src.height // 3, src.width // 3  # 20 m -> 60 m
        band = src.read(1, out_shape=(h, w)).astype(np.float32)
        west, north = src.transform.c, src.transform.f
        epsg = src.crs.to_epsg()
    return band, (west, north), epsg, item.id


def pearson(a: np.ndarray, b: np.ndarray, dy: int, dx: int) -> float:
    """Correlation of b placed at (dy,dx) in a, over pixels valid in both."""
    y0a, y0b = max(dy, 0), max(-dy, 0)
    x0a, x0b = max(dx, 0), max(-dx, 0)
    h = min(a.shape[0] - y0a, b.shape[0] - y0b)
    w = min(a.shape[1] - x0a, b.shape[1] - x0b)
    if h <= 0 or w <= 0:
        return 0.0
    sa = a[y0a : y0a + h, x0a : x0a + w]
    sb = b[y0b : y0b + h, x0b : x0b + w]
    valid = (sa > 0) & (sb > 0)  # 0 = nodata
    if valid.sum() < 1000:
        return 0.0
    va, vb = sa[valid], sb[valid]
    va, vb = va - va.mean(), vb - vb.mean()
    denom = np.sqrt((va**2).sum() * (vb**2).sum())
    return float((va * vb).sum() / denom) if denom else 0.0


def main() -> None:
    path = Path(sys.argv[1])
    with h5py.File(path, "r") as hdf:
        year = list(hdf.keys())[0]
        event_id = list(hdf[year].keys())[0]
        ev = hdf[year][event_id]
        ev_b12_20 = ev["sen2_20_post"][7].astype(np.float32)  # band order: B12 = idx 7
        label = ev["label"][0]
        scene = ev.attrs["post_sen2_file"]

    ref, (ref_west, ref_north), epsg, item_id = scene_band(scene, "B12")
    ev_b12 = ev_b12_20[::3, ::3]

    peaks = phase_corr_peaks(ref, ev_b12, k=5)
    scored = sorted(((pearson(ref, ev_b12, dy, dx), dy, dx) for dy, dx in peaks),
                    reverse=True)
    agree60, dy, dx = scored[0]

    ul_east = ref_west + dx * 60.0
    ul_north = ref_north - dy * 60.0

    rows, cols = np.where(label == 1)
    lon = lat = float("nan")
    if len(rows):
        cy, cx = rows.mean(), cols.mean()
        east_c, north_c = ul_east + cx * 20.0, ul_north - cy * 20.0
        lon, lat = Transformer.from_crs(epsg, 4326, always_xy=True).transform(east_c, north_c)

    print(f"event {event_id} vs {item_id}")
    print(f"UL: E {ul_east:.0f}, N {ul_north:.0f} (B12 pearson {agree60:.3f}, "
          f"runner-up {scored[1][0]:.3f})")
    print(f"label centroid: {lon:.4f}E, {lat:.4f}N")

    sidecar = path.with_suffix(".georef.json")
    sidecar.write_text(json.dumps({
        "epsg": epsg, "ul_east": ul_east, "ul_north": ul_north,
        "pixel_size_m": 20.0, "shape": list(ev_b12_20.shape),
        "b12_pearson": round(agree60, 4), "method": "b12-vs-mpc",
        "mpc_item": item_id,
        "label_centroid_lonlat": [round(lon, 5), round(lat, 5)],
    }, indent=2))
    print(f"wrote {sidecar}")


if __name__ == "__main__":
    main()
