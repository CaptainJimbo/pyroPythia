"""Confirm candidate duplicate FLOGA events by remote label alignment.

For each pair, reads ONLY the two label rasters over HTTP ranges, finds the
shift that best aligns them via phase correlation of the labels themselves,
and reports the aligned IoU. Two distinct fires can share a datatake; they
cannot share a rasterized burn-scar *shape*. IoU>0.5 => duplicate views,
0.1-0.5 => partial overlap (Evia-style), <0.1 => not a duplicate.

Pairs are hardcoded from find_duplicate_candidates.py triage (overlapping
tiles + similar size), plus the proven Evia pair as a positive control.
"""

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

from check_duplicate_event import phase_corr_shift, shift_into
from floga_remote import HttpRangeFile, floga_url

PAIRS = [
    # (year, event_a, event_b, official splits a|b)
    (2017, "45", "83", "test|val"),
    (2017, "53", "86", "train|train"),
    (2017, "43", "85", "test|train"),
    (2018, "28", "39", "val|test"),
    (2018, "7", "25", "train|train"),
    (2018, "19", "40", "train|val"),
    (2019, "39", "40", "val|train"),
    (2019, "50", "74", "test|train"),
    (2020, "92", "93", "train|train"),
    (2020, "63", "69", "test|train"),
    (2021, "61", "62", "val|train"),  # positive control (proven: 85% of 61 in 62)
]


def main() -> None:
    handles: dict[int, h5py.File] = {}

    def labels(year: int, event_id: str) -> np.ndarray:
        if year not in handles:
            handles[year] = h5py.File(HttpRangeFile(floga_url(year, 20)), "r")
        return handles[year][str(year)][event_id]["label"][0] == 1

    print(f"{'year':>5} {'pair':>9} {'splits':>12} {'ha_a':>7} {'ha_b':>7} "
          f"{'shift_km':>12} {'IoU':>6}  verdict")
    for year, a, b, splits in PAIRS:
        la, lb = labels(year, a), labels(year, b)
        dy, dx = phase_corr_shift(la.astype(np.float32), lb.astype(np.float32))
        lb_in_a = shift_into(la.shape, lb, dy, dx)
        inter = (la & lb_in_a).sum()
        union = la.sum() + lb.sum() - inter
        iou = inter / union if union else 0.0
        ha = lambda x: x.sum() * 400 / 10_000
        verdict = "DUPLICATE" if iou > 0.5 else ("PARTIAL" if iou > 0.1 else "distinct")
        print(f"{year:>5} {a + '|' + b:>9} {splits:>12} {ha(la):>7,.0f} {ha(lb):>7,.0f} "
              f"{f'{dy * 0.02:.1f},{dx * 0.02:.1f}':>12} {iou:>6.3f}  {verdict}",
              flush=True)


if __name__ == "__main__":
    main()
