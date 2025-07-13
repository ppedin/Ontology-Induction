from __future__ import annotations

"""
Updated question generator with train/test split support.

Usage examples
--------------
>>> from datasets.graphragbench_medical.graphragbench_medical_questions.graphragbench_medical_questions_generator import generate_questions

# Iterate over the training partition (80 % stratified, seed‑42)
for rec in generate_questions(split="train", fraction=0.8, seed=42):
    ...

# Iterate over the test partition
for rec in generate_questions(split="test", fraction=0.8, seed=42):
    ...

If ``split`` is omitted or set to "all", behaviour is identical to the previous
implementation (returns the full dataset).
"""

import json
import random
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Tuple

from datasets.general_questions_generator import make_record

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DATA_PATH = (
    "C:/Users/paolo/Desktop/Ontology-Induction/"
    "datasets/graphragbench_medical/graphragbench_medical_questions/"
    "graphragbench_medical_questions.json"
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _load_json(path: Path) -> List[Dict[str, Any]]:
    """Load the raw JSON list from *path*."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _stratified_split(
    data: Sequence[Dict[str, Any]],
    fraction: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """Return two lists of indices (train, test) with a stratified split.

    The split preserves the distribution of ``question_type`` across the corpus.
    ``fraction`` is the proportion assigned to the *training* partition (0–1).
    ``seed`` guarantees deterministic shuffling.
    """

    if not (0.0 < fraction < 1.0):
        raise ValueError("fraction must be in the open interval (0, 1)")

    rnd = random.Random(seed)
    buckets: Dict[str, List[int]] = {}
    for idx, item in enumerate(data):
        buckets.setdefault(item.get("question_type", "Unknown"), []).append(idx)

    train_idx: List[int] = []
    test_idx: List[int] = []

    for indices in buckets.values():
        rnd.shuffle(indices)
        k = round(len(indices) * fraction)
        train_idx.extend(indices[:k])
        test_idx.extend(indices[k:])

    # Preserve global order (optional – keeps parity with original file)
    train_idx.sort()
    test_idx.sort()
    return train_idx, test_idx

# --------------------------------------------------------------------------- #
# Main public API
# --------------------------------------------------------------------------- #

def generate_questions(
    *,
    split: str = "all",  # "train", "test", or "all"
    fraction: float = 0.8,
    seed: int = 42,
    path: Path | None = None,
) -> Iterator[Dict[str, Any]]:
    """Yield question records from the requested data split.

    Parameters
    ----------
    split : {"train", "test", "all"}, default "all"
        Partition to iterate. "all" reproduces the previous behaviour.
    fraction : float, default 0.8
        Proportion of examples allocated to the *training* split.
    seed : int, default 42
        Seed for the deterministic stratified shuffle.
    path : Path | None
        Optional path to a custom JSON file (mainly for testing).
    """

    # Load dataset
    path = Path(path or DATA_PATH).expanduser().resolve()
    data = _load_json(path)

    # Compute / cache split indices if needed
    if split in {"train", "test"}:
        train_idx, test_idx = _stratified_split(data, fraction, seed)
        selected = train_idx if split == "train" else test_idx
    elif split == "all":
        selected = range(len(data))
    else:
        raise ValueError("split must be one of {'train', 'test', 'all'}")

    for i in selected:
        item = data[i]
        evidence: List[str] = []

        # "evidence" può essere lista o stringa
        ev = item.get("evidence", [])
        if isinstance(ev, list):
            evidence.extend(ev)
        elif isinstance(ev, str):
            evidence.append(ev)

        # "evidence_relations" è singola stringa
        ev_rel = item.get("evidence_relations")
        if ev_rel:
            evidence.append(ev_rel)

        metadata = {
            "Id": item.get("id"),
            "Question_Type": item.get("question_type"),
        }

        yield make_record(
            question=item.get("question", ""),
            answer=item.get("answer", ""),
            evidence=evidence,
            metadata=metadata,
        )

# --------------------------------------------------------------------------- #
# Quick test (CLI)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys

    split = "train"
    fraction = 0.2
    seed = 42


    print(f"Testing generator on: {DATA_PATH} (split={split})")
    itr = generate_questions(split=split, fraction=fraction, seed=seed)
    for rec in itr:
        if rec["Metadata"]["Question_Type"] == "Complex Reasoning":
            print(rec)
