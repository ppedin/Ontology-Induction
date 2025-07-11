"""
LightRAG – Phase 4: summary embedding
------------------------------------

Legge *entities.parquet* e *relationships.parquet* prodotti dallo
stadio di summarization, genera gli embedding del campo ``summary`` con
un modello *sentence embedding* leggero e salva i risultati arricchiti
in una nuova directory.

Requisiti:
* pandas
* pyarrow
* sentence‑transformers
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
from sentence_transformers import SentenceTransformer
import pyarrow  # necessario per IO parquet con pandas

__all__ = ["embed_summaries"]

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def _safe_encode(
    model: SentenceTransformer,
    texts: List[str],
    *,
    batch_size: int,
) -> List[Optional[List[float]]]:
    """
    Converte *texts* in embedding con gestione robusta degli errori.

    Se l'encoding batch va in errore, si passa al fallback riga‑per‑riga.
    Ritorna una lista lunga quanto *texts* che può contenere ``None``
    nei punti in cui l'encoding non è riuscito.
    """
    try:
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=False,
        )
        # Rende ogni embedding una *plain* python list
        return [emb.tolist() if hasattr(emb, "tolist") else list(emb) for emb in embeddings]
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("Batch encoding fallito: %s – fallback riga‑per‑riga.", exc)
        out: List[Optional[List[float]]] = []
        for idx, text in enumerate(texts):
            try:
                emb = model.encode(
                    text,
                    show_progress_bar=False,
                    convert_to_numpy=False,
                )
                out.append(emb.tolist() if hasattr(emb, "tolist") else list(emb))
            except Exception as row_exc:  # pylint: disable=broad-exception-caught
                logger.error("Encoding fallito per la riga %d: %s", idx, row_exc)
                out.append(None)
        return out


def _process_file(
    file_path: Path,
    model: SentenceTransformer,
    *,
    batch_size: int,
    output_dir: Path,
) -> None:
    """
    Aggiunge la colonna ``embedding`` a un singolo file parquet
    e lo scrive in *output_dir*.
    """
    if not file_path.exists():
        logger.warning("File non trovato: %s – ignorato.", file_path)
        return

    df = pd.read_parquet(file_path, engine="pyarrow")

    if "summary" not in df.columns:
        logger.warning("Colonna 'summary' assente in %s – ignorato.", file_path)
        return

    logger.info("Encoding dei summary in %s …", file_path.name)
    df["embedding"] = _safe_encode(
        model,
        df["summary"].astype(str).tolist(),
        batch_size=batch_size,
    )

    out_path = output_dir / file_path.name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, engine="pyarrow", index=False)

    print(f"{file_path.name}: {len(df):,} righe elaborate.")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def embed_summaries(input_dir: Path, output_dir: Path, batch_size: int = 64) -> None:
    """
    Genera gli embedding dei riassunti e li salva in nuovi file parquet.

    Parameters
    ----------
    input_dir : Path
        Directory contenente *entities.parquet* e *relationships.parquet*.
    output_dir : Path
        Directory di destinazione; verrà creata se non esiste.
    batch_size : int, default 64
        Dimensione dei batch per ``SentenceTransformer.encode``.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    for pattern in ("entities*.parquet", "relationships*.parquet"):
        matched_files = sorted(input_dir.glob(pattern))
        if not matched_files:
            logger.warning("Nessun file trovato per il pattern %s", pattern)
            continue
        for file_path in matched_files:
            _process_file(
                file_path,
                model,
                batch_size=batch_size,
                output_dir=output_dir,
            )

# --------------------------------------------------------------------------- #
# CLI helper
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    input_dir = "C:/Users/paolo/Desktop/Ontology-Induction/outputs/graphragbench_medical/postprocessing/summarization"
    output_dir = "C:/Users/paolo/Desktop/Ontology-Induction/outputs/graphragbench_medical/postprocessing/embedding"
    batch_size = 64

    embed_summaries(input_dir, output_dir, batch_size=batch_size)
