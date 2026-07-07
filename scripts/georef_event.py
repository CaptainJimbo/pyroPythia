"""Recover the geotransform of a FLOGA event by coastline registration.

FLOGA HDF5s carry no georeferencing. But each event names its Sentinel-2
scene (-> MGRS tile -> UTM search window), and its sea_mask draws coastlines.
Phase-correlating that against a land raster built from GSHHG full-resolution
coastlines pins the array to UTM coordinates.

Usage: python scripts/georef_event.py data/floga/FLOGA_2021_event61_sen2_20.h5
Writes a JSON sidecar (crs, UL corner, pixel size) next to the input file and
prints the label centroid in lon/lat for sanity checking.

Registration runs at 40 m (2x downsample) — +/-1 ref pixel => ~2 px at 20 m,
plenty for pair adjudication and web mapping. Inland scenes without coastline
in view will fail loudly (low agreement) rather than silently.
"""

import json
import sys
from pathlib import Path

import geopandas as gpd
import h5py
import mgrs
import numpy as np
from pyproj import Transformer
from rasterio import features
from rasterio.transform import from_origin
from shapely.geometry import box

from check_duplicate_event import phase_corr_peaks

GSHHS = "data/ref/gshhg/GSHHS_shp/f/GSHHS_f_L1.shp"
REF_RES = 40.0  # m
MARGIN = 100_000  # m of search window around the tile


def tile_from_scene(scene: str) -> str:
    return scene.split("_")[5][1:]  # 'T34SGJ' -> '34SGJ'


def build_ref_land(tile: str, span: float):
    """Rasterized GSHHG land over tile SW corner +/- margin, in tile's UTM zone."""
    zone, _, east0, north0 = mgrs.MGRS().MGRSToUTM(tile)
    epsg = 32600 + zone
    west, south = east0 - MARGIN, north0 - MARGIN
    east, north = east0 + span + MARGIN, north0 + span + MARGIN

    # window bbox in lon/lat (GSHHG is EPSG:4326), padded a bit
    tr = Transformer.from_crs(epsg, 4326, always_xy=True)
    xs, ys = [west, east, west, east], [south, south, north, north]
    lons, lats = tr.transform(xs, ys)
    bbox = box(min(lons) - 0.2, min(lats) - 0.2, max(lons) + 0.2, max(lats) + 0.2)

    land = gpd.read_file(GSHHS, bbox=bbox).to_crs(epsg)
    w = int((east - west) / REF_RES)
    h = int((north - south) / REF_RES)
    transform = from_origin(west, north, REF_RES, REF_RES)
    raster = features.rasterize(
        land.geometry, out_shape=(h, w), transform=transform, fill=0, default_value=1
    ).astype(np.float32)
    return raster, (west, north), epsg


def main() -> None:
    path = Path(sys.argv[1])
    with h5py.File(path, "r") as hdf:
        year = list(hdf.keys())[0]
        event_id = list(hdf[year].keys())[0]
        ev = hdf[year][event_id]
        sea = ev["sea_mask"][0]
        label = ev["label"][0]
        scene = ev.attrs["post_sen2_file"]

    tile = tile_from_scene(scene)
    ev_land = (sea == 0).astype(np.float32)[::2, ::2]  # 20 m -> 40 m
    span = max(sea.shape) * 20.0
    ref, (ref_west, ref_north), epsg = build_ref_land(tile, span)

    def land_iou(dy: int, dx: int) -> float:
        # IoU of LAND, not raw agreement: mostly-sea scenes agree ~85% under
        # arbitrary shifts, so raw agreement can't detect a mis-lock
        h = min(ev_land.shape[0], ref.shape[0] - dy) if dy >= 0 else ev_land.shape[0] + dy
        w = min(ev_land.shape[1], ref.shape[1] - dx) if dx >= 0 else ev_land.shape[1] + dx
        if h <= 0 or w <= 0:
            return 0.0
        sub_ref = ref[max(dy, 0) : max(dy, 0) + h, max(dx, 0) : max(dx, 0) + w]
        sub_ev = ev_land[max(-dy, 0) : max(-dy, 0) + h, max(-dx, 0) : max(-dx, 0) + w]
        both = ((sub_ref > 0) & (sub_ev > 0)).sum()
        either = ((sub_ref > 0) | (sub_ev > 0)).sum()
        return both / max(either, 1)

    # the global correlation peak can be a spurious lock — verify top peaks
    # by land-IoU and keep the best-fitting one
    peaks = phase_corr_peaks(ref, ev_land, k=8)
    scored = sorted(((land_iou(dy, dx), dy, dx) for dy, dx in peaks), reverse=True)
    agree, dy, dx = scored[0]
    runner_up = scored[1][0]
    # event pixel (0,0) sits at ref pixel (dy, dx)
    ul_east = ref_west + dx * REF_RES
    ul_north = ref_north - dy * REF_RES

    rows, cols = np.where(label == 1)
    cy, cx = rows.mean(), cols.mean()
    east_c, north_c = ul_east + cx * 20.0, ul_north - cy * 20.0
    lon, lat = Transformer.from_crs(epsg, 4326, always_xy=True).transform(east_c, north_c)

    print(f"event {event_id} tile {tile} -> EPSG:{epsg}")
    print(f"UL corner: E {ul_east:.0f}, N {ul_north:.0f} "
          f"(land-IoU {agree:.1%}, runner-up peak {runner_up:.1%})")
    print(f"label centroid: {lon:.4f}E, {lat:.4f}N")

    sidecar = path.with_suffix(".georef.json")
    sidecar.write_text(json.dumps({
        "epsg": epsg, "ul_east": ul_east, "ul_north": ul_north,
        "pixel_size_m": 20.0, "shape": list(sea.shape),
        "land_agreement": round(float(agree), 4),
        "label_centroid_lonlat": [round(lon, 5), round(lat, 5)],
    }, indent=2))
    print(f"wrote {sidecar}")


if __name__ == "__main__":
    main()
