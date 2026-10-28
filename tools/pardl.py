#!/usr/bin/env python3
"""Parallel, resumable download of one URL into a file: N range workers, 64 MiB chunks, completed chunks recorded
in <file>.chunks so a restart continues. Usage: pardl.py URL FILE [N]. One connection to huggingface.co gives
about 6 MB/s here; 24 give over 200 MB/s."""
import json
import os
import sys
import threading
import time
import urllib.request

url, path = sys.argv[1], sys.argv[2]
n_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 24
CHUNK = 64 << 20
with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=120) as r:
    size = int(r.headers["Content-Length"])
    final_url = r.url
done_path = path + ".chunks"
done = set(json.load(open(done_path))) if os.path.exists(done_path) else set()
n_chunks = (size + CHUNK - 1) // CHUNK
fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
if os.fstat(fd).st_size != size:
    os.ftruncate(fd, size)
lock = threading.Lock()
todo = [c for c in range(n_chunks) if c not in done]
got = [0]
t0 = time.time()


def save_done():
    json.dump(sorted(done), open(done_path + ".tmp", "w"))
    os.replace(done_path + ".tmp", done_path)


def worker():
    while True:
        with lock:
            if not todo:
                return
            c = todo.pop(0)
        lo, hi = c * CHUNK, min(size, (c + 1) * CHUNK) - 1
        for attempt in range(30):
            try:
                rq = urllib.request.Request(final_url, headers={"Range": f"bytes={lo}-{hi}"})
                with urllib.request.urlopen(rq, timeout=300) as r:
                    buf = bytearray()
                    while len(buf) < hi - lo + 1:
                        b = r.read(min(4 << 20, hi - lo + 1 - len(buf)))
                        if not b:
                            raise IOError("short read")
                        buf += b
                os.pwrite(fd, buf, lo)
                with lock:
                    done.add(c)
                    got[0] += len(buf)
                    if len(done) % 8 == 0 or len(done) == n_chunks:
                        save_done()
                        el = time.time() - t0
                        print(f"{os.path.basename(path)}: {len(done)}/{n_chunks} chunks, {got[0] / el / 1e6:.0f} MB/s", flush=True)
                break
            except Exception:
                time.sleep(2 + attempt)
        else:
            print(f"chunk {c} failed", flush=True)


threads = [threading.Thread(target=worker) for _ in range(n_workers)]
for t in threads:
    t.start()
for t in threads:
    t.join()
os.fsync(fd)
os.close(fd)
with lock:
    save_done()
if len(done) == n_chunks:
    print(f"complete {path} ({size} bytes) in {time.time() - t0:.0f}s", flush=True)
else:
    print(f"incomplete: {n_chunks - len(done)} chunks missing", flush=True)
    sys.exit(1)
