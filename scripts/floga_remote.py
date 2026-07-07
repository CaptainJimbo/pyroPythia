"""Remote access to FLOGA HDF5 files on Dropbox via HTTP range requests.

The yearly files are 12-82 GB; this reader lets h5py pull only the bytes it
needs (metadata, single events) so the local footprint stays near zero.
"""

import io
import time

import requests

FOLDER = "https://www.dropbox.com/scl/fo/3sqbs3tioox7s5vb4jmwl/h"
RLKEY = "rlkey=5p3e7wa5al4cy9x34pmtp9g6d"


def floga_url(year: int, sen2_gsd: int = 20) -> str:
    """Dropbox share URL for a yearly FLOGA HDF5 file (sen2_gsd: 10, 20 or 60)."""
    sub = f"S2%20{sen2_gsd}m%20-%20MODIS%20500m"
    return (
        f"{FOLDER}/{sub}/FLOGA_dataset_{year}_sen2_{sen2_gsd}_mod_500.h5"
        f"?{RLKEY}&dl=1"
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
        # Dropbox content tokens are short-lived, 403 doubles as throttling,
        # and long sessions get transient disconnects — back off, re-resolve
        # the URL, and retry on all of it
        for attempt in range(8):
            try:
                r = self.session.get(
                    self.url, headers={"Range": range_header}, timeout=60
                )
                if r.status_code in (403, 429):
                    raise requests.HTTPError(f"{r.status_code} (throttle)")
                r.raise_for_status()
                return r
            except requests.RequestException as e:
                if attempt == 7:
                    raise
                time.sleep(2**attempt)
                try:
                    self._refresh_url()
                except requests.RequestException:
                    pass  # next loop iteration retries with the old URL
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
