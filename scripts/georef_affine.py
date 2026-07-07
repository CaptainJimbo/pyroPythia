"""Fit a full affine geotransform for a FLOGA event against its S2 scene.

FLOGA arrays live on a geographic grid (not the scene's UTM grid), so a
translation-only fit cannot lock. Instead: phase-correlate a grid of local
patches of the event's post-fire B12 against the MPC copy of the same
acquisition, then least-squares an affine map (event pixel -> scene UTM).
The determinant gives exact per-pixel area, fixing the ~4 % hectare bias of
the 400 m^2 assumption.

Usage: python scripts/georef_affine.py <event.h5>
Sidecar written: <event>.georef.json with affine [a,b,c,d,e,f]:
  east  = a*col + b*row + e
  north = c*col + d*row + f      (col,row in 20 m event pixels)
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np
from pyproj import Transformer

from check_duplicate_event import phase_corr_peaks
from georef_event_s2 import pearson, scene_band

DS = 3  # event 20 m -> 60 m downsample factor
PATCH = 96  # patch size in 60 m px (~5.8 km) — small enough that the
# event-vs-UTM scale mismatch (~10 %) drifts <1 px inside a patch... (10% of
# 96 is ~10 px, but the iterative affine absorbs it after the first pass)
MARGIN = 80  # local search window half-extra around the predicted position
STRIDE = 120
MIN_R = 0.5
# NB: lat/lon -> UTM is not exactly affine over 110 km; residuals of a few
# ref px at scene edges are projection curvature, not matching noise. 8 px
# (160 m) is fine for pair adjudication; don't chase tighter with an affine.
FINAL_THRESH = 8.0


def global_estimate(ref: np.ndarray, ev: np.ndarray):
    """Rough translation to seed the iteration (scale error smears this)."""
    best = (0.0, 0, 0)
    for dy, dx in phase_corr_peaks(ref, ev, k=5):
        r = pearson(ref, ev, dy, dx)
        if r > best[0]:
            best = (r, dy, dx)
    _, dy, dx = best
    return np.array([1.0, 0.0, dx]), np.array([0.0, 1.0, dy])


def patch_matches(ref: np.ndarray, ev: np.ndarray, cx: np.ndarray, cy: np.ndarray):
    """Control points by matching small patches inside locally-predicted
    windows of the reference (prediction via current affine cx, cy)."""
    pts = []
    for py in range(0, ev.shape[0] - PATCH, STRIDE):
        for px in range(0, ev.shape[1] - PATCH, STRIDE):
            sub = ev[py : py + PATCH, px : px + PATCH]
            if (sub > 0).mean() < 0.8 or sub.std() < 50:
                continue  # nodata or featureless (open sea)
            ex, ey = px + PATCH / 2, py + PATCH / 2
            rx = cx[0] * ex + cx[1] * ey + cx[2]
            ry = cy[0] * ex + cy[1] * ey + cy[2]
            x0 = int(rx - PATCH / 2 - MARGIN)
            y0 = int(ry - PATCH / 2 - MARGIN)
            x1, y1 = x0 + PATCH + 2 * MARGIN, y0 + PATCH + 2 * MARGIN
            if x0 < 0 or y0 < 0 or y1 > ref.shape[0] or x1 > ref.shape[1]:
                continue
            win = ref[y0:y1, x0:x1]
            best_r, best = 0.0, None
            for dy, dx in phase_corr_peaks(win, sub, k=3):
                r = pearson(win, sub, dy, dx)
                if r > best_r:
                    best_r, best = r, (dy, dx)
            if best and best_r >= MIN_R:
                dy, dx = best
                c = PATCH / 2
                pts.append((ex, ey, x0 + dx + c, y0 + dy + c, best_r))
    return pts


def fit_affine(pts):
    """Least squares [ref_x, ref_y] = A @ [ev_x, ev_y, 1], with staged
    outlier rejection (coarse first — initial fits are polluted by bad
    matches, so a tight threshold rejects everything)."""
    arr = np.array(pts, dtype=np.float64)
    total = len(arr)
    resid = np.zeros(len(arr))
    for thresh in [50.0, 20.0, FINAL_THRESH]:
        X = np.c_[arr[:, 0], arr[:, 1], np.ones(len(arr))]
        cx = np.linalg.lstsq(X, arr[:, 2], rcond=None)[0]
        cy = np.linalg.lstsq(X, arr[:, 3], rcond=None)[0]
        resid = np.hypot(X @ cx - arr[:, 2], X @ cy - arr[:, 3])
        keep = resid < thresh
        if keep.sum() < 4:
            break
        arr = arr[keep]
        resid = resid[keep]
    return cx, cy, float(resid.max()), len(arr), total


def main() -> None:
    path = Path(sys.argv[1])
    with h5py.File(path, "r") as hdf:
        year = list(hdf.keys())[0]
        event_id = list(hdf[year].keys())[0]
        ev = hdf[year][event_id]
        if "sen2_20_post" in ev:
            b12 = ev["sen2_20_post"][7].astype(np.float32)
        else:
            b12 = ev["b12_post"][0].astype(np.float32)  # masks-only extracts
        label = ev["label"][0]
        scene = ev.attrs["post_sen2_file"]

    ref, (ref_west, ref_north), epsg, item_id = scene_band(scene, "B12")
    ev60 = b12[::DS, ::DS]
    cx, cy = global_estimate(ref, ev60)
    pts = []
    for it in range(3):
        pts = patch_matches(ref, ev60, cx, cy)
        if len(pts) < 4:
            raise SystemExit(f"iter {it}: only {len(pts)} control points — cannot fit affine")
        cx, cy, max_resid, inliers, total = fit_affine(pts)
        print(f"iter {it}: {inliers}/{total} inliers, max residual {max_resid:.2f} ref px")

    # compose: event 20 m px -> 60 m ds px -> ref 60 m px -> UTM meters
    # ref_px_x = cx0*(col/DS) + cx1*(row/DS) + cx2 ; east = ref_west + 60*ref_px_x
    a = 60.0 * cx[0] / DS
    b = 60.0 * cx[1] / DS
    e = ref_west + 60.0 * cx[2]
    c = -60.0 * cy[0] / DS
    d = -60.0 * cy[1] / DS
    f_ = ref_north - 60.0 * cy[2]
    px_area = abs(a * d - b * c)

    rows, cols = np.where(label == 1)
    lon = lat = float("nan")
    if len(rows):
        east_c = a * cols.mean() + b * rows.mean() + e
        north_c = c * cols.mean() + d * rows.mean() + f_
        lon, lat = Transformer.from_crs(epsg, 4326, always_xy=True).transform(east_c, north_c)

    print(f"event {event_id} vs {item_id}")
    print(f"control points: {inliers}/{total} inliers, max residual {max_resid:.2f} ref px")
    print(f"pixel size: {np.hypot(a, c):.2f} x {np.hypot(b, d):.2f} m, area {px_area:.1f} m2/px "
          f"(vs 400 assumed)")
    print(f"label centroid: {lon:.4f}E, {lat:.4f}N | burned {len(rows) * px_area / 1e4:,.0f} ha "
          f"(was {len(rows) * 400 / 1e4:,.0f})")

    sidecar = path.with_suffix(".georef.json")
    sidecar.write_text(json.dumps({
        "epsg": epsg, "affine": [a, b, c, d, e, f_], "method": "affine-b12-mpc",
        "px_area_m2": round(px_area, 2), "inliers": inliers, "control_points": total,
        "max_residual_refpx": round(float(max_resid), 2), "mpc_item": item_id,
        "label_centroid_lonlat": [round(lon, 5), round(lat, 5)],
    }, indent=2))
    print(f"wrote {sidecar}")


if __name__ == "__main__":
    main()
