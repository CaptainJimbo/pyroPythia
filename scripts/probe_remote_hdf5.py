"""Probe a FLOGA HDF5 file on Dropbox via HTTP range requests — no full download.

Proves we can list events and read metadata (and later, single events) out of
the tens-of-GB yearly files while keeping the local footprint near zero.
"""

import io
import sys
import time

import h5py
import hdf5plugin  # noqa: F401  (registers the BZip2 filter for h5py)
import requests

FLOGA_2021_20M = (
    "https://www.dropbox.com/scl/fo/3sqbs3tioox7s5vb4jmwl/h/"
    "S2%2020m%20-%20MODIS%20500m/FLOGA_dataset_2021_sen2_20_mod_500.h5"
    "?rlkey=5p3e7wa5al4cy9x34pmtp9g6d&dl=1"
)


class HttpRangeFile(io.RawIOBase):
    """Read-only file over HTTP range requests, sharing Dropbox session cookies."""

    def __init__(self, share_url: str, chunk: int = 4 * 2**20):
        self.session = requests.Session()
        self.share_url = share_url
        self._refresh_url()
        # HEAD on the content URL omits content-length; probe with a 1-byte range
        r = self._ranged_get("bytes=0-0")
        self.size = int(r.headers["content-range"].split("/")[-1])
        self.pos = 0
        self.chunk = chunk
        self.cache: dict[int, bytes] = {}  # block index -> bytes
        self.bytes_fetched = 0

    def _refresh_url(self) -> None:
        r = self.session.head(self.share_url, allow_redirects=True, timeout=30)
        r.raise_for_status()
        self.url = r.url

    def _ranged_get(self, range_header: str) -> requests.Response:
        # Dropbox content tokens are short-lived and 403 doubles as throttling;
        # re-resolve the URL and back off before retrying
        for attempt in range(6):
            r = self.session.get(
                self.url, headers={"Range": range_header}, timeout=60
            )
            if r.status_code in (403, 429) and attempt < 5:
                time.sleep(2**attempt)
                self._refresh_url()
                continue
            r.raise_for_status()
            return r
        raise RuntimeError("unreachable")

    def _block(self, idx: int) -> bytes:
        if idx not in self.cache:
            start = idx * self.chunk
            end = min(start + self.chunk, self.size) - 1
            r = self._ranged_get(f"bytes={start}-{end}")
            self.cache[idx] = r.content
            self.bytes_fetched += len(r.content)
        return self.cache[idx]

    def readinto(self, b) -> int:
        n = min(len(b), self.size - self.pos)
        out, pos = b"", self.pos
        while len(out) < n:
            idx, off = divmod(pos + len(out), self.chunk)
            blk = self._block(idx)
            out += blk[off : off + n - len(out)]
        b[:n] = out
        self.pos += n
        return n

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def tell(self) -> int:
        return self.pos

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True


def main() -> None:
    f = HttpRangeFile(FLOGA_2021_20M)
    print(f"remote file size: {f.size / 2**30:.2f} GiB")

    with h5py.File(f, "r") as hdf:
        print("root keys:", list(hdf.keys()))
        for year in hdf.keys():
            events = list(hdf[year].keys())
            print(f"{year}: {len(events)} events; first 10: {events[:10]}")
            ev = hdf[year][events[0]]
            print(f"event {events[0]} members: {list(ev.keys())}")
            for name, item in ev.items():
                if isinstance(item, h5py.Dataset):
                    print(f"  {name}: shape={item.shape} dtype={item.dtype}")
                if dict(item.attrs):
                    print(f"  {name} attrs: {dict(item.attrs)}")
            if dict(ev.attrs):
                print("event attrs:", dict(ev.attrs))
    print(f"bytes fetched: {f.bytes_fetched / 2**20:.1f} MiB (of {f.size / 2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
    sys.exit(0)
