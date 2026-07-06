"""Step 1: dNBR burn-severity map for one FLOGA event vs the Fire Service label.

Physics baseline, zero ML: NBR = (B8A - B12) / (B8A + B12), dNBR = pre - post,
USGS severity classes, IoU/F1 of `dNBR > threshold` against the label.

Usage: python scripts/dnbr_evia.py [path/to/event.h5]
"""

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

DEFAULT = "data/floga/FLOGA_2021_event61_sen2_20.h5"

# band axis is alphabetical: B02 B03 B04 B05 B06 B07 B11 B12 B8A
B04, B03, B02 = 2, 1, 0  # RGB
B12, B8A = 7, 8  # SWIR2, narrow NIR

# USGS dNBR severity thresholds (unscaled)
SEV_BOUNDS = [-0.5, -0.1, 0.1, 0.27, 0.44, 0.66, 1.3]
SEV_LABELS = ["regrowth", "unburned", "low", "moderate-low", "moderate-high", "high"]
SEV_COLORS = ["#1a9850", "#f7f7f7", "#fee08b", "#fdae61", "#f46d43", "#a50026"]
BURN_THRESHOLD = 0.27  # moderate-low and up counts as burned for IoU/F1


def nbr(img: np.ndarray) -> np.ndarray:
    nir = img[B8A].astype(np.float32)
    swir2 = img[B12].astype(np.float32)
    nodata = (nir <= 0) & (swir2 <= 0)
    denom = nir + swir2
    denom[denom == 0] = 1
    out = (nir - swir2) / denom
    out[nodata] = np.nan
    return out


def rgb(img: np.ndarray, gain: float = 3.0) -> np.ndarray:
    x = img[[B04, B03, B02]].astype(np.float32) / 10000.0
    return np.clip(np.moveaxis(x, 0, -1) * gain, 0, 1)


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
    with h5py.File(path, "r") as hdf:
        year = list(hdf.keys())[0]
        event_id = list(hdf[year].keys())[0]
        ev = hdf[year][event_id]
        pre = ev["sen2_20_pre"][()]
        post = ev["sen2_20_post"][()]
        label = ev["label"][0]
        cloud_pre = ev["sen2_20_cloud_pre"][0]
        cloud_post = ev["sen2_20_cloud_post"][0]
        sea = ev["sea_mask"][0]
        attrs = dict(ev.attrs)

    dnbr = nbr(pre) - nbr(post)

    # sea_mask: 0 = land, 255 = sea; "cloud" rasters are the S2 Scene
    # Classification Layer: 0 nodata, 1 saturated, 3 shadow, 8/9 cloud, 10 cirrus
    bad_scl = [0, 1, 3, 8, 9, 10]
    valid = (
        (sea == 0)
        & ~np.isin(cloud_pre, bad_scl)
        & ~np.isin(cloud_post, bad_scl)
        & ~np.isnan(dnbr)
    )
    pred = (dnbr > BURN_THRESHOLD) & valid
    truth = (label == 1) & valid

    # metrics on the full scene, plots cropped to the fire + margin
    tp = np.sum(pred & truth)
    fp = np.sum(pred & ~truth)
    fn = np.sum(~pred & truth)
    iou = tp / (tp + fp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn)
    ha = lambda px: px * 400 / 10_000

    print(f"event {event_id} ({attrs['pre_image_date']} -> {attrs['post_image_date']})")
    print(f"valid pixels: {valid.mean():.1%} (rest: sea/cloud/nodata)")
    print(f"label burned:  {ha(truth.sum()):>9,.0f} ha")
    print(f"dNBR>{BURN_THRESHOLD} burned: {ha(pred.sum()):>9,.0f} ha")
    print(f"IoU = {iou:.3f}   F1 = {f1:.3f}   (physics baseline, zero ML)")

    rows, cols = np.where(label == 1)
    m = 200  # margin in pixels
    r0, r1 = max(rows.min() - m, 0), min(rows.max() + m, label.shape[0])
    c0, c1 = max(cols.min() - m, 0), min(cols.max() + m, label.shape[1])
    win = np.s_[r0:r1, c0:c1]

    # figure: pre / post / dNBR severity / label vs prediction agreement
    fig, ax = plt.subplots(2, 2, figsize=(16, 12))
    ax[0, 0].imshow(rgb(pre)[win])
    ax[0, 0].set_title(f"pre-fire {attrs['pre_image_date']}")
    ax[0, 1].imshow(rgb(post)[win])
    ax[0, 1].set_title(f"post-fire {attrs['post_image_date']}")

    sev = np.ma.masked_where(~valid, dnbr)[win]
    cmap = ListedColormap(SEV_COLORS)
    im = ax[1, 0].imshow(sev, cmap=cmap, norm=BoundaryNorm(SEV_BOUNDS, cmap.N))
    cbar = fig.colorbar(im, ax=ax[1, 0], shrink=0.8, ticks=[])
    for y, s in zip(np.linspace(0.08, 0.92, len(SEV_LABELS)), SEV_LABELS):
        cbar.ax.text(1.3, y, s, transform=cbar.ax.transAxes, fontsize=8, va="center")
    ax[1, 0].set_title("dNBR severity (USGS classes)")

    agree = np.zeros(label.shape, dtype=np.uint8)  # 0 bg, 1 TP, 2 FP, 3 FN
    agree[pred & truth] = 1
    agree[pred & ~truth] = 2
    agree[~pred & truth] = 3
    cmap_a = ListedColormap(["#f0f0f0", "#2b8cbe", "#e34a33", "#fdbb84"])
    ax[1, 1].imshow(agree[win], cmap=cmap_a, vmin=0, vmax=3)
    ax[1, 1].set_title(f"blue=TP red=FP orange=FN — IoU {iou:.3f}, F1 {f1:.3f}")

    for a in ax.flat:
        a.set_axis_off()
    fig.suptitle(f"FLOGA {year} event {event_id} — dNBR physics baseline", fontsize=14)
    fig.tight_layout()
    out_png = Path("artifacts") / f"dnbr_{year}_event{event_id}.png"
    out_png.parent.mkdir(exist_ok=True)
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
