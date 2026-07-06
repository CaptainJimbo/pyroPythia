"""Confirm which FLOGA 2021 event is the Evia fire by burned-area size.

Reads only the label array (int8, bzip2-compressed) for the candidate events.
Evia 2021 burned ~50k ha — an order of magnitude above the same-day Attica fire.
"""

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

from floga_remote import HttpRangeFile, floga_url

CANDIDATES = ["61", "59"]  # T34SGJ (N. Evia?) and T34SGH (Attica?), both pre=Aug 3


def main() -> None:
    f = HttpRangeFile(floga_url(2021, sen2_gsd=20))
    with h5py.File(f, "r") as hdf:
        for event_id in CANDIDATES:
            label = hdf["2021"][event_id]["label"][0]
            values, counts = np.unique(label, return_counts=True)
            burned_px = int(counts[values == 1].sum())
            ha = burned_px * 400 / 10_000  # 20 m px -> ha
            print(f"event {event_id}: label values {dict(zip(values.tolist(), counts.tolist()))}")
            print(f"event {event_id}: burned = {burned_px} px = {ha:,.0f} ha")
    print(f"bytes fetched: {f.bytes_fetched / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
