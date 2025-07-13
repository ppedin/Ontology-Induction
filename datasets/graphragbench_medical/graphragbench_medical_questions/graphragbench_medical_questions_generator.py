"""
GraphragBench-Medical - Questions Generator
-------------------------------------------
Converte il JSON del dataset medicale nello schema universale definito in
`datasets/general_questions_generator.py`.

• Question   -> "question"
• Answer     -> "answer"
• Evidence   -> concatenazione di "evidence" + "evidence_relations"
• Metadata   -> {"Id": id, "Question_Type": question_type}
"""

from __future__ import annotations

import json
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterator, List

from datasets.general_questions_generator import make_record


# Percorso predefinito al JSON sorgente
DATA_PATH = "C:/Users/paolo/Desktop/Ontology-Induction/datasets/graphragbench_medical/graphragbench_medical_questions/graphragbench_medical_questions.json"


def _load_json(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def generate_questions(path: Path | None = None) -> Iterator[Dict[str, Any]]:
    """
    Generatore che produce record conformi allo schema universale.

    Parameters
    ----------
    path : Path | None
        Se fornito, usa un file JSON alternativo; altrimenti DATA_PATH.
    """
    path = Path(path or DATA_PATH).expanduser().resolve()
    data = _load_json(path)

    for item in data:
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
# Quick test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print(f"Testing generator on: {DATA_PATH}")
    for rec in islice(generate_questions(), 10):
        print(json.dumps(rec, indent=2, ensure_ascii=False))
