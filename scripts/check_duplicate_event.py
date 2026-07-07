"""Test whether two FLOGA events are overlapping views of the same fire.

No georeferencing exists in the HDF files, so alignment is estimated by phase
correlation of the land masks (coastlines are unambiguous registration
targets), then the burned labels are compared in the shared frame.

Usage: python scripts/check_duplicate_event.py <event_a.h5> <event_b.h5>
"""

import sys

import h5py
import numpy as np


def load(path: str):
    with h5py.File(path, "r") as hdf:
        year = list(hdf.keys())[0]
        event_id = list(hdf[year].keys())[0]
        ev = hdf[year][event_id]
        land = (ev["sea_mask"][0] == 0).astype(np.float32)
        burned = ev["label"][0] == 1
    return event_id, land, burned


def phase_corr_peaks(a: np.ndarray, b: np.ndarray, k: int = 1) -> list[tuple[int, int]]:
    """Top-k candidate shifts (dy, dx) mapping b's frame onto a's, by phase
    correlation. The global peak can be a spurious lock (repetitive coastline,
    nodata edges), so callers should verify candidates against a fit metric."""
    h = max(a.shape[0], b.shape[0])
    w = max(a.shape[1], b.shape[1])
    pa = np.zeros((2 * h, 2 * w), np.float32)
    pb = np.zeros((2 * h, 2 * w), np.float32)
    pa[: a.shape[0], : a.shape[1]] = a - a.mean()
    pb[: b.shape[0], : b.shape[1]] = b - b.mean()
    fa, fb = np.fft.rfft2(pa), np.fft.rfft2(pb)
    cross = fa * np.conj(fb)
    cross /= np.abs(cross) + 1e-12
    corr = np.fft.irfft2(cross, s=pa.shape)

    peaks = []
    suppress = 25  # px radius around an accepted peak
    for _ in range(k):
        dy, dx = np.unravel_index(np.argmax(corr), corr.shape)
        corr[max(dy - suppress, 0) : dy + suppress, max(dx - suppress, 0) : dx + suppress] = -np.inf
        dy, dx = int(dy), int(dx)
        if dy > h:
            dy -= 2 * h
        if dx > w:
            dx -= 2 * w
        peaks.append((dy, dx))
    return peaks


def phase_corr_shift(a: np.ndarray, b: np.ndarray) -> tuple[int, int]:
    """Single best shift (dy, dx) that maps b's frame onto a's."""
    return phase_corr_peaks(a, b, k=1)[0]


def shift_into(a_shape, b: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Place b into a's frame given b->a shift (dy, dx)."""
    out = np.zeros(a_shape, dtype=bool)
    ys = slice(max(dy, 0), min(dy + b.shape[0], a_shape[0]))
    xs = slice(max(dx, 0), min(dx + b.shape[1], a_shape[1]))
    if ys.start >= ys.stop or xs.start >= xs.stop:
        return out
    out[ys, xs] = b[ys.start - dy : ys.stop - dy, xs.start - dx : xs.stop - dx]
    return out


def main() -> None:
    id_a, land_a, burn_a = load(sys.argv[1])
    id_b, land_b, burn_b = load(sys.argv[2])

    dy, dx = phase_corr_shift(land_a, land_b)
    km = 20 / 1000
    print(f"alignment shift (event {id_b} -> event {id_a} frame): "
          f"dy={dy}px ({dy * km:.1f} km), dx={dx}px ({dx * km:.1f} km)")

    land_b_in_a = shift_into(land_a.shape, land_b > 0, dy, dx)
    coast_agree = (land_b_in_a & (land_a > 0)).sum() / max(land_b_in_a.sum(), 1)
    print(f"land-mask agreement in overlap: {coast_agree:.1%} "
          f"(sanity check on the alignment)")

    burn_b_in_a = shift_into(land_a.shape, burn_b, dy, dx)
    inter = (burn_a & burn_b_in_a).sum()
    ha = lambda px: px * 400 / 10_000
    print(f"burned in {id_a}: {ha(burn_a.sum()):,.0f} ha, "
          f"in {id_b}: {ha(burn_b.sum()):,.0f} ha")
    print(f"intersection: {ha(inter):,.0f} ha "
          f"= {inter / burn_a.sum():.1%} of {id_a}, "
          f"{inter / burn_b.sum():.1%} of {id_b}")
    # b extends beyond a's frame, so compute the union arithmetically
    union = ha(burn_a.sum() + burn_b.sum() - inter)
    print(f"union (deduplicated fire size): {union:,.0f} ha")


if __name__ == "__main__":
    main()
