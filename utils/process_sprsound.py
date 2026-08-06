#!/usr/bin/env python3
"""Process SPRSound (respiratory sounds) into our {audio, sound_label, text} format.

RECORD-level: each whole recording → its record_annotation (normal / cas / das / cas and das).
"Poor Quality" recordings are dropped. Opaque-label testbed for the definition/legend claim.
test2022 has no annotations, so we stratify-split train2022 into train/val/test.

Usage:
  python utils/process_sprsound.py --raw_dir ~/data/raw/SPRSound/BioCAS2022 --out_base ~/data
Then set SPRSOUND_{TRAIN,VAL,TEST}_PATH in .env to the saved dirs.
"""
import argparse, glob, json, os, random
from collections import defaultdict, Counter
import numpy as np
import soundfile as sf
from datasets import Dataset, Audio

TARGET_SR = 16000
LABEL_MAP = {"normal": "normal", "cas": "cas", "das": "das", "cas & das": "cas and das"}  # drop "poor quality"


def norm_record(t):
    return LABEL_MAP.get(str(t).strip().lower())


def build_rows(wav_dir, json_dir):
    rows = []
    for wav in sorted(glob.glob(os.path.join(wav_dir, "*.wav"))):
        base = os.path.splitext(os.path.basename(wav))[0]
        jf = os.path.join(json_dir, base + ".json")
        if not os.path.exists(jf):
            continue
        lab = norm_record(json.load(open(jf)).get("record_annotation"))
        if lab is None:
            continue
        try:
            data, sr = sf.read(wav)
        except Exception as e:
            print(f"  skip {base}: {e}"); continue
        if data.ndim > 1:
            data = data.mean(axis=1)
        rows.append({
            "id": base,
            "audio": {"array": np.asarray(data, dtype=np.float32), "sampling_rate": sr},
            "sound_label": lab,
            "text": "",
            "few_shot_examples": [],
        })
    return rows


def strat_split(rows, seed=0, frac=(0.7, 0.15, 0.15)):
    by = defaultdict(list)
    for r in rows:
        by[r["sound_label"]].append(r)
    tr, va, te = [], [], []
    rng = random.Random(seed)
    for lab, items in by.items():
        rng.shuffle(items)
        n = len(items); a = int(frac[0] * n); b = a + int(frac[1] * n)
        tr += items[:a]; va += items[a:b]; te += items[b:]
    rng.shuffle(tr); rng.shuffle(va); rng.shuffle(te)
    return {"train": tr, "validation": va, "test": te}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", required=True, help="SPRSound/BioCAS2022 dir")
    p.add_argument("--out_base", default=os.environ.get("BASE_DATA_DIR", os.path.expanduser("~/data")))
    args = p.parse_args()

    rows = build_rows(os.path.join(args.raw_dir, "train2022_wav"), os.path.join(args.raw_dir, "train2022_json"))
    print(f"usable recordings (Poor Quality dropped): {len(rows)}")
    print("label dist:", dict(Counter(r["sound_label"] for r in rows)))
    splits = strat_split(rows)
    for name, srows in splits.items():
        out = os.path.join(args.out_base, f"sprsound_{name}")
        ds = Dataset.from_list(srows).cast_column("audio", Audio(sampling_rate=TARGET_SR))
        ds.save_to_disk(out)
        print(f"saved {name}: {len(srows)} ({dict(Counter(r['sound_label'] for r in srows))}) → {out}")
    print("Done. Set in .env: SPRSOUND_TRAIN_PATH/VAL_PATH/TEST_PATH → these dirs.")


if __name__ == "__main__":
    main()
