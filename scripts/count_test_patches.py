"""Count positive 256x256 patches per test event — remote labels only.

Estimates the share of FLOGA's benchmark test patches that come from
duplicate-contaminated events. Patch grid mirrors create_dataset.py:
non-overlapping 256x256 tiles from (0,0); a patch is positive if it
contains any burned (label==1) pixel.

Writes stats/floga/test_patch_counts.csv.
"""

import csv

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

from floga_remote import HttpRangeFile, floga_url

PATCH = 256
CONTAMINATED = {(2017, "43"), (2017, "45"), (2018, "39"), (2019, "50"), (2021, "15")}


def main() -> None:
    split = {}
    with open("stats/floga/data_split.csv") as fh:
        for r in csv.DictReader(fh):
            split[(int(r["year"]), r["event_id"])] = r["set"]
    test = sorted([k for k, s in split.items() if s == "test"])

    handles: dict[int, h5py.File] = {}
    rows = []
    missing = []
    for i, (year, event_id) in enumerate(test):
        if year not in handles:
            handles[year] = h5py.File(HttpRangeFile(floga_url(year, 20)), "r")
        if event_id not in handles[year][str(year)]:
            # data_split.csv references events absent from the HDF files
            missing.append((year, event_id))
            print(f"[{i + 1}/{len(test)}] {year} ev {event_id}: MISSING from HDF", flush=True)
            continue
        label = handles[year][str(year)][event_id]["label"][0]
        h, w = label.shape
        gh, gw = h // PATCH, w // PATCH
        crop = label[: gh * PATCH, : gw * PATCH] == 1
        patches = crop.reshape(gh, PATCH, gw, PATCH).any(axis=(1, 3))
        n_pos = int(patches.sum())
        rows.append({
            "year": year, "event_id": event_id, "positive_patches": n_pos,
            "burned_px": int((label == 1).sum()),
            "contaminated": (year, event_id) in CONTAMINATED,
        })
        print(f"[{i + 1}/{len(test)}] {year} ev {event_id}: {n_pos} positive patches"
              f"{'  <-- CONTAMINATED' if (year, event_id) in CONTAMINATED else ''}",
              flush=True)

    with open("stats/floga/test_patch_counts.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    total = sum(r["positive_patches"] for r in rows)
    bad = sum(r["positive_patches"] for r in rows if r["contaminated"])
    print(f"\npositive test patches: {total}; from contaminated events: {bad} "
          f"({bad / total:.1%})")
    if missing:
        print(f"split-file events MISSING from HDFs: {missing}")


if __name__ == "__main__":
    main()
