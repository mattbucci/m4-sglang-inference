#!/usr/bin/env python3
"""Prune orphaned blobs from the Hugging Face hub cache's shared blob store.

On this box the hub keeps file contents in ``hub/blobs`` and every snapshot
file is a symlink into it, so deleting a ``models--*`` directory frees
nothing until the blobs no snapshot references are removed. Files newer
than --min-age-min are kept so an active download is never touched.

    python scripts/common/hf_cache_prune.py            # dry run
    python scripts/common/hf_cache_prune.py --delete
"""
import argparse
import glob
import os
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub", default=os.path.expanduser("~/.cache/huggingface/hub"))
    ap.add_argument("--min-age-min", type=float, default=20)
    ap.add_argument("--delete", action="store_true")
    a = ap.parse_args()
    referenced = {os.path.realpath(l) for l in glob.glob(os.path.join(a.hub, "models--*", "snapshots", "*", "**"), recursive=True)
                  if os.path.islink(l)}
    now = time.time()
    total = n = 0
    for root, _, files in os.walk(os.path.join(a.hub, "blobs")):
        for name in files:
            p = os.path.join(root, name)
            if name.startswith(".") or os.path.realpath(p) in referenced:
                continue
            st = os.stat(p)
            if now - st.st_mtime < a.min_age_min * 60:
                continue
            total += st.st_size
            n += 1
            if a.delete:
                os.remove(p)
    print(f"{'removed' if a.delete else 'orphaned'}: {n} blobs, {total / 1e9:.1f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
