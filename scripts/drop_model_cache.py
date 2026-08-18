"""Drop the page cache held by model files, without root.

    python scripts/drop_model_cache.py <path> [<path> ...]

On an integrated GPU the "device memory" is system RAM, and the driver reports
free memory close to MemFree rather than MemAvailable. Streaming a multi-GB
checkpoint therefore leaves the cache holding memory the driver will not hand
out, and the next engine start fails sizing even though the RAM is reclaimable.
posix_fadvise(DONTNEED) evicts just those pages; drop_caches would need root
and would throw away everyone else's cache too.
"""
from __future__ import annotations

import os
import sys


def _meminfo():
    out = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            k, _, v = line.partition(":")
            out[k] = int(v.split()[0]) // 1024
    return out


def drop(path):
    dropped = 0
    for root, _, files in os.walk(path) if os.path.isdir(path) else [
            (os.path.dirname(path), None, [os.path.basename(path)])]:
        for name in files:
            full = os.path.join(root, name)
            try:
                fd = os.open(full, os.O_RDONLY)
            except OSError:
                continue
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                dropped += os.fstat(fd).st_size
            finally:
                os.close(fd)
    return dropped


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    before = _meminfo()
    total = sum(drop(p) for p in sys.argv[1:])
    after = _meminfo()
    print(f"advised away {total / 2**30:.1f} GiB of file pages")
    print(f"MemFree   {before['MemFree']:6d} -> {after['MemFree']:6d} MB")
    print(f"Cached    {before['Cached']:6d} -> {after['Cached']:6d} MB")


if __name__ == "__main__":
    main()
