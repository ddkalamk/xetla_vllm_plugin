#!/usr/bin/env python3
"""Drop the page cache for model weight files.

torch.xpu.mem_get_info() reports MemFree, not MemAvailable, so on the integrated
Xe2 GPUs the page cache of the weights is counted as unavailable and comes out of
the KV budget. vLLM prefetches every checkpoint into cache on load, so without
this a sweep loses several GiB per model until the engine refuses to start.

Globbing *.safetensors misses what matters: the HF cache keeps weights in
hub/**/blobs/<sha256> with no extension, and the snapshot entries are symlinks.
"""
import os
import sys

MIN_SIZE = 32 * 1024 * 1024

n = 0
freed = 0
for d in sys.argv[1:]:
    for root, _, files in os.walk(d):
        for fn in files:
            p = os.path.join(root, fn)
            try:
                size = os.path.getsize(p)
                if size < MIN_SIZE:
                    continue
                fd = os.open(p, os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    n += 1
                    freed += size
                finally:
                    os.close(fd)
            except OSError:
                pass
print(f"[evict] {n} files, {freed / 2**30:.1f} GiB")
