"""Scan burned-area sizes for every event in a FLOGA year — labels only, remote.

Writes data/floga/<year>_label_stats.csv with per-event burned hectares and
metadata. ~20-40 MB fetched per event; run in the background.

Usage: python scripts/scan_labels.py [year] [--gsd 20]
"""

import argparse
import csv
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

from floga_remote import HttpRangeFile, floga_url


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("year", type=int, nargs="?", default=2021)
    p.add_argument("--gsd", type=int, default=20)
    args = p.parse_args()

    out = Path("data/floga") / f"{args.year}_label_stats.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    f = HttpRangeFile(floga_url(args.year, args.gsd))
    rows = []
    with h5py.File(f, "r") as hdf:
        events = sorted(hdf[str(args.year)], key=int)
        for i, event_id in enumerate(events):
            ev = hdf[str(args.year)][event_id]
            label = ev["label"][0]
            px_per_ha = (args.gsd**2) / 10_000
            burned_ha = float((label == 1).sum() * px_per_ha)
            other_fire_ha = float((label == 2).sum() * px_per_ha)
            attrs = dict(ev.attrs)
            tile = attrs.get("post_sen2_file", "").split("_")[5]
            rows.append(
                {
                    "event_id": event_id,
                    "pre_date": attrs.get("pre_image_date", ""),
                    "post_date": attrs.get("post_image_date", ""),
                    "tile": tile,
                    "burned_ha": round(burned_ha),
                    "other_fire_ha": round(other_fire_ha),
                }
            )
            print(
                f"[{i + 1}/{len(events)}] event {event_id}: {burned_ha:,.0f} ha "
                f"(fetched {f.bytes_fetched / 2**20:.0f} MiB)",
                flush=True,
            )

    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    sizes = np.array([r["burned_ha"] for r in rows])
    print(f"\nwrote {out}")
    print(f"events: {len(sizes)}, total burned: {sizes.sum():,.0f} ha")
    print(f"size quartiles (ha): {np.percentile(sizes, [0, 25, 50, 75, 100]).round(0)}")


if __name__ == "__main__":
    main()
