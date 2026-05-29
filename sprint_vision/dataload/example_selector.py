"""
ExampleSelector — mirrors ICI's BaseMultiTaskDataset._select_examples().

In ICI, few-shot examples are pre-attached to each test sample as a
"few_shot_examples" column, built by FewShotGenerator using semantic
similarity (BAAI/llm-embedder cosine similarity over text transcripts).

For MedFMC image classification, there is no text transcript to embed,
so semantic similarity selection is not applicable. Instead we use
stratified random sampling: equal examples from each class, selected
once before the inference loop and reused for every test sample.
This matches standard practice in few-shot medical imaging literature.

Design mirrors:
    ICI/dataload/multi_task_dataset.py  :: BaseMultiTaskDataset._select_examples()
    ICI/utils/generate_fewshots.py      :: FewShotGenerator (selection logic only)
"""

import json
import random
from typing import Dict, List, Optional


class ExampleSelector:
    """
    Loads a MedFMC training JSON and provides balanced few-shot examples.

    Args:
        train_json_path: Path to colon_train.json (or chest/endo equivalent).
        seed:            Random seed — guarantees reproducibility across runs.
    """

    def __init__(self, train_json_path: str, seed: int = 42):
        self.seed = seed
        # Independent RNG so nothing else in the process affects selection
        self.rng = random.Random(seed)

        with open(train_json_path, "r") as f:
            all_samples = json.load(f)

        # Group samples by class label.
        # Label lives in conversations[1]["value"] — same field sprint_eval.py reads.
        self.class_pools: Dict[str, List[Dict]] = {}
        for sample in all_samples:
            label = str(sample["conversations"][1]["value"]).strip()
            if label not in self.class_pools:
                self.class_pools[label] = []
            self.class_pools[label].append({
                "id":    sample.get("id", ""),   # needed to exclude self during training
                "image": sample["image"],
                "label": label,
            })

        # Shuffle each pool once at construction so repeated select() calls
        # always draw from the same deterministic ordering.
        for label in self.class_pools:
            self.rng.shuffle(self.class_pools[label])

        all_labels = sorted(self.class_pools.keys())
        sizes = {lbl: len(self.class_pools[lbl]) for lbl in all_labels}
        print(f"[ExampleSelector] Loaded train pool: {sizes} (seed={seed})")

    def select(
        self,
        n_shots: int,
        exclude_id: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> List[Dict]:
        """
        Return n_shots stratified examples.

        Mirrors ICI's _select_examples() but for image classification:
        instead of taking from a pre-ranked similarity list, we take
        floor(n_shots / n_classes) from each class, distributing the
        remainder randomly.

        Args:
            n_shots:    Total examples to return.  0 → empty list.
            exclude_id: Sample ID to exclude (prevents a training sample
                        from selecting itself as its own ICL example).
            seed:       When provided, a fresh random.Random(seed) is used
                        for this call only.  Allows per-item deterministic
                        selection during training (seed = base_seed + item_idx)
                        while keeping inference using the shared self.rng.

        Returns:
            List of dicts with keys "id", "image" (relative path), "label".
        """
        if n_shots == 0:
            return []

        classes = sorted(self.class_pools.keys())
        n_classes = len(classes)
        base = n_shots // n_classes
        remainder = n_shots % n_classes

        # Use a per-call RNG when seed is given (training); shared RNG otherwise (inference).
        rng = random.Random(seed) if seed is not None else self.rng

        remainder_classes = rng.sample(classes, remainder) if remainder > 0 else []

        selected: List[Dict] = []
        for label in classes:
            # Exclude the target sample itself so it doesn't appear as its own example.
            pool = (
                [s for s in self.class_pools[label] if s["id"] != exclude_id]
                if exclude_id
                else self.class_pools[label]
            )
            count = base + (1 if label in remainder_classes else 0)
            count = min(count, len(pool))

            if seed is not None:
                # Per-item training path: sample randomly from the full pool.
                chosen = rng.sample(pool, count) if count <= len(pool) else list(pool)
            else:
                # Inference path: deterministic first-N from pre-shuffled pool.
                chosen = pool[:count]

            selected.extend(chosen)

        rng.shuffle(selected)
        return selected
