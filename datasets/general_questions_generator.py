"""
General Questions Generator
---------------------------
Definisce un helper per uniformare i record provenienti da diversi dataset.

Schema universale:
{
    "Question": str,
    "Answer": str,
    "Evidence": list[str],
    "Metadata": dict
}
"""

from typing import Any, Dict, List


def make_record(
    question: str,
    answer: str,
    evidence: List[str],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Ritorna un dizionario conforme allo schema universale."""
    return {
        "Question": question,
        "Answer": answer,
        "Evidence": evidence,
        "Metadata": metadata,
    }
