"""Extract a single FLOGA event from the remote yearly HDF5 to a local file.

Usage: python scripts/extract_event.py <year> <event_id> [--gsd 20] [--out data/floga]

Pulls only that event's arrays over HTTP ranges and writes a small local .h5
with the same dataset names, downcast where lossless, gzip-compressed.
"""

import argparse
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

from floga_remote import HttpRangeFile, floga_url


def downcast(name: str, arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.int32 and arr.min() >= np.iinfo(np.int16).min and arr.max() <= np.iinfo(np.int16).max:
        return arr.astype(np.int16)
    return arr


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("year", type=int)
    p.add_argument("event_id")
    p.add_argument("--gsd", type=int, default=20)
    p.add_argument("--out", default="data/floga")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"FLOGA_{args.year}_event{args.event_id}_sen2_{args.gsd}.h5"

    f = HttpRangeFile(floga_url(args.year, args.gsd))
    with h5py.File(f, "r") as hdf, h5py.File(out_file, "w") as out:
        ev = hdf[str(args.year)][args.event_id]
        grp = out.create_group(f"{args.year}/{args.event_id}")
        for k, v in ev.attrs.items():
            grp.attrs[k] = v
        for name in ev:
            arr = ev[name][()]
            arr = downcast(name, arr)
            grp.create_dataset(name, data=arr, compression="gzip", compression_opts=4)
            print(f"{name}: {arr.shape} {arr.dtype} "
                  f"(fetched so far: {f.bytes_fetched / 2**20:.0f} MiB)", flush=True)

    size = out_file.stat().st_size / 2**20
    print(f"\nwrote {out_file} ({size:.0f} MiB); "
          f"total fetched: {f.bytes_fetched / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
