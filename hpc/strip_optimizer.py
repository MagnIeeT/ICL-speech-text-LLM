#!/usr/bin/env python3
"""Strip `optimizer_state` from saved LoRA checkpoints to reclaim disk.

optimizer_state is only ever WRITTEN (never loaded — no resume path) and is
unused at inference, so removing it is lossless for inference/analysis. Each
.pt is re-saved atomically (temp file + os.replace) so an interruption can't
corrupt the original.

Usage:
  python hpc/strip_optimizer.py <dir>            # dry-run, recursive
  python hpc/strip_optimizer.py <dir> --apply    # actually rewrite
"""
import argparse
import os
import sys
from glob import glob

# Repo root on path so the pickled TrainingConfig (config.*) unpickles.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="dir to scan recursively for *.pt")
    ap.add_argument("--apply", action="store_true", help="rewrite files (default: dry-run)")
    args = ap.parse_args()

    files = sorted(glob(os.path.join(os.path.abspath(args.target), "**", "*.pt"), recursive=True))
    if not files:
        print(f"No .pt files under {args.target}")
        return

    reclaim = 0
    changed = 0
    for f in files:
        try:
            ck = torch.load(f, map_location="cpu", weights_only=False)
        except Exception as exc:
            print(f"  SKIP (load failed): {f} ({exc})")
            continue
        if not isinstance(ck, dict) or ck.get("optimizer_state") is None:
            continue
        before = os.path.getsize(f)
        changed += 1
        if args.apply:
            ck["optimizer_state"] = None
            tmp = f + ".tmp"
            torch.save(ck, tmp)
            os.replace(tmp, f)
            after = os.path.getsize(f)
            reclaim += before - after
            print(f"  stripped {(before-after)/1e6:6.0f}MB  {f}")
        else:
            print(f"  would strip  {before/1e6:6.0f}MB file  {f}")

    verb = "Reclaimed" if args.apply else "Would rewrite"
    if args.apply:
        print(f"\n{verb} {reclaim/1e9:.2f} GB across {changed} file(s).")
    else:
        print(f"\n{changed} file(s) have optimizer_state. Re-run with --apply to strip.")


if __name__ == "__main__":
    main()
