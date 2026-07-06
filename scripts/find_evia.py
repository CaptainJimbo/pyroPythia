"""Scan the FLOGA 2021 file's event metadata to locate the Evia (Euboea) fire.

Reads only attributes and dataset shapes — no pixel data. The Evia fire
(started 2021-08-03, northern Evia, ~50k ha) should stand out as an early-
August event on a zone-34S tile with a very large bounding box.
"""

import h5py
import hdf5plugin  # noqa: F401

from floga_remote import HttpRangeFile, floga_url


def main() -> None:
    f = HttpRangeFile(floga_url(2021, sen2_gsd=20), chunk=2 * 2**20)
    rows = []
    with h5py.File(f, "r") as hdf:
        year = "2021"
        for event_id in hdf[year]:
            ev = hdf[year][event_id]
            attrs = dict(ev.attrs)
            shape = ev["label"].shape  # (1, H, W) at 20 m GSD
            h, w = shape[1], shape[2]
            tile = attrs.get("post_sen2_file", "").split("_")[5]  # e.g. T35SNV
            rows.append(
                (
                    int(event_id),
                    attrs.get("pre_image_date", "?"),
                    attrs.get("post_image_date", "?"),
                    tile,
                    h,
                    w,
                    h * w,
                )
            )

    rows.sort(key=lambda r: -r[6])
    print(f"{'id':>4} {'pre':>10} {'post':>10} {'tile':>7} {'HxW':>12} {'km2 bbox':>9}")
    for r in rows:
        km2 = r[6] * (20 * 20) / 1e6  # 20 m pixels -> km^2
        print(f"{r[0]:>4} {r[1]:>10} {r[2]:>10} {r[3]:>7} {r[4]:>5}x{r[5]:<6} {km2:>8.0f}")
    print(f"\nbytes fetched: {f.bytes_fetched / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
