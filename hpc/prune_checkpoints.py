#!/usr/bin/env python3
"""Prune non-best LoRA checkpoints for a training run (or a whole date dir).

For each run it reads the matching training log, finds the best epoch per
validation mode (original / fresh / fixed) using the average score over the
TRAINING datasets, and keeps the union of those best epochs. Everything else
is deleted. When a kept epoch has both a `_final` and a `_periodic` file, only
`_final` is kept.

SAFETY:
  * Dry-run by default; pass --apply to actually delete.
  * A run is skipped (nothing deleted) unless its log contains
    "Training completed successfully" -> never touches an in-progress run.
  * Never deletes the last remaining checkpoint of a run.

Usage:
  # one run
  python hpc/prune_checkpoints.py <ckpt>/2026-07-30/201011_af_h-cremad_nosym
  # whole date dir (all runs inside)
  python hpc/prune_checkpoints.py <ckpt>/2026-07-30
  # actually delete
  python hpc/prune_checkpoints.py <run_or_date_dir> --apply

Wire into training .sh AFTER training finishes, e.g.:
  python hpc/prune_checkpoints.py "$CHECKPOINT_DIR/$RUN_NAME" --apply
"""
import argparse
import ast
import os
import re
import sys
from glob import glob

LOGS_ROOT_DEFAULT = os.path.expanduser("~/training/symbol_training/logs")
EPOCH_RE = re.compile(r"Epoch (\d+) validation: (\{.*\})\s*$")
CKPT_RE = re.compile(r"lora_epoch(\d+)_(\w+)\.pt$")


def find_log(run_dir, logs_root):
    """run_dir = .../checkpoints/<date>/<run_name>  ->  <logs_root>/<date>/<run_name>.log"""
    run_name = os.path.basename(run_dir.rstrip("/"))
    date = os.path.basename(os.path.dirname(run_dir.rstrip("/")))
    cand = os.path.join(logs_root, date, run_name + ".log")
    return cand if os.path.exists(cand) else None


def parse_log(log_path):
    """Return (completed, train_datasets, {epoch: {mode: {ds: score}}})."""
    train_ds, epochs, completed = [], {}, False
    with open(log_path) as f:
        for line in f:
            if "Training completed successfully" in line:
                completed = True
            m = re.search(r"Loaded (\w+) TRAIN:", line)
            if m and m.group(1) not in train_ds:
                train_ds.append(m.group(1))
            em = EPOCH_RE.search(line)
            if em:
                try:
                    d = ast.literal_eval(em.group(2))
                except (ValueError, SyntaxError):
                    continue
                epochs[int(em.group(1))] = {
                    mode: {ds: v.get("score", 0.0) for ds, v in dd.items()}
                    for mode, dd in d.get("all_modes", {}).items()
                }
    return completed, train_ds, epochs


def best_epochs(epochs, train_ds):
    """Union of the best epoch per validation mode, scored on train datasets."""
    keep = set()
    modes = {m for ep in epochs.values() for m in ep}
    for mode in modes:
        best_ep, best_avg = None, -1.0
        for ep in sorted(epochs):
            sc = epochs[ep].get(mode)
            if not sc:
                continue
            tasks = train_ds if train_ds else list(sc)
            avg = sum(sc.get(t, 0.0) for t in tasks) / max(len(tasks), 1)
            if avg > best_avg:
                best_avg, best_ep = avg, ep
        if best_ep is not None:
            keep.add(best_ep)
    return keep


def plan_run(run_dir, logs_root):
    """Return (keep_files, del_files, reason) for one run dir; del_files empty on skip."""
    ckpts = sorted(glob(os.path.join(run_dir, "*.pt")))
    if not ckpts:
        return [], [], "no .pt files"
    log_path = find_log(run_dir, logs_root)
    if not log_path:
        return ckpts, [], "SKIP: no matching log found"
    completed, train_ds, epochs = parse_log(log_path)
    if not completed:
        return ckpts, [], "SKIP: training not completed"
    if not epochs:
        return ckpts, [], "SKIP: no validation entries in log"
    keep_eps = best_epochs(epochs, train_ds)
    if not keep_eps:
        return ckpts, [], "SKIP: could not determine best epochs"

    keep, delete = [], []
    for f in ckpts:
        m = CKPT_RE.search(f)
        if not m:
            keep.append(f)  # unknown file -> keep to be safe
            continue
        ep, kind = int(m.group(1)), m.group(2)
        k = ep in keep_eps
        if k and kind == "periodic" and os.path.exists(
            os.path.join(run_dir, f"lora_epoch{ep}_final.pt")
        ):
            k = False  # prefer _final when both exist for a kept epoch
        (keep if k else delete).append(f)

    if not keep:  # never leave a run with zero checkpoints
        return ckpts, [], "SKIP: refusing to delete all checkpoints"
    reason = f"train={train_ds or 'ALL(val)'} keep_epochs={sorted(keep_eps)}"
    return keep, delete, reason


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="a run checkpoint dir, or a date dir containing run dirs")
    ap.add_argument("--logs-root", default=LOGS_ROOT_DEFAULT)
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry-run)")
    args = ap.parse_args()

    target = os.path.abspath(args.target.rstrip("/"))
    if glob(os.path.join(target, "*.pt")):
        run_dirs = [target]                       # target is a single run dir
    else:
        run_dirs = sorted(d for d in glob(os.path.join(target, "*")) if os.path.isdir(d))
    if not run_dirs:
        print(f"No run dirs / checkpoints under {target}")
        return

    total_del = 0
    for rd in run_dirs:
        keep, delete, reason = plan_run(rd, args.logs_root)
        print(f"\n### {os.path.basename(rd)}  [{reason}]")
        for f in keep:
            print(f"  KEEP   {os.path.getsize(f)/1e6:6.0f}MB  {os.path.basename(f)}")
        for f in delete:
            sz = os.path.getsize(f)
            total_del += sz
            print(f"  DELETE {sz/1e6:6.0f}MB  {os.path.basename(f)}")
            if args.apply:
                os.remove(f)

    verb = "Deleted" if args.apply else "Would delete"
    print(f"\n{verb} {total_del/1e9:.2f} GB across {len(run_dirs)} run(s).")
    if not args.apply:
        print("Dry-run only. Re-run with --apply to delete.")


if __name__ == "__main__":
    main()
