"""
indexing.py  –  Vector indexing for entities & relationships embeddings
----------------------------------------------------------------------
Legge i file .parquet prodotti dal processo di embedding e costruisce
due indici FAISS (entities, relationships) insieme ai relativi metadati,
persistendo il tutto su disco per un retrieval semantico efficiente.

Requisiti Python:
    pandas          >= 1.0
    numpy           >= 1.20
    faiss‑cpu       (o faiss‑gpu)
    pathlib, glob, pickle, logging

Usage (modulo):
    from pathlib import Path
    from postprocessing.lightrag.indexing import build_faiss_indices

    build_faiss_indices(
        input_dir=Path("outputs/graphragbench_medical/postprocessing/embedding"),
        output_dir=Path("outputs/graphragbench_medical/postprocessing/faiss_index")
    )
"""

from __future__ import annotations

import glob
import logging
import pickle
from pathlib import Path
from typing import Final, List, Dict, Sequence, Tuple

import faiss  # type: ignore
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Configurazione logging
# --------------------------------------------------------------------------- #
LOGGER_NAME: Final[str] = "lightrag.indexing"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(LOGGER_NAME)


# --------------------------------------------------------------------------- #
# Costanti interne
# --------------------------------------------------------------------------- #
REQUIRED_COLUMNS: Final[Sequence[str]] = (
    "name",
    "type",
    "description",
    "summary",
    "embedding",
)

GROUPS: Final[Tuple[Tuple[str, str, str], ...]] = (
    # (file‑prefix, index‑file, metadata‑file)
    ("entities_", "entities.index", "entity_metadata.pkl"),
    ("relationships_", "relationships.index", "relationship_metadata.pkl"),
)


# --------------------------------------------------------------------------- #
# Funzione pubblica
# --------------------------------------------------------------------------- #
def build_faiss_indices(input_dir: Path, output_dir: Path) -> None:
    """
    Costruisce (o ricostruisce) gli indici FAISS per entità e relazioni.

    Parameters
    ----------
    input_dir : Path
        Cartella contenente i file .parquet con embeddings:
        `entities_*.parquet` e/o `relationships_*.parquet`.

    output_dir : Path
        Destinazione degli indici e dei metadati:
            ├── entities.index
            ├── relationships.index
            ├── entity_metadata.pkl
            └── relationship_metadata.pkl
    """
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Input directory:  %s", input_dir)
    logger.info("Output directory: %s", output_dir)

    for prefix, index_name, meta_name in GROUPS:
        files = _collect_parquet_files(input_dir, prefix)
        if not files:
            logger.warning("Nessun file corrispondente a '%s*.parquet' trovato.", prefix)
            continue

        df = _read_and_validate_parquets(files, prefix)
        embeddings = _build_embedding_matrix(df["embedding"])

        logger.info(
            "Gruppo '%s': %d embeddings, dimensione vettori = %d",
            prefix.rstrip("_"),
            embeddings.shape[0],
            embeddings.shape[1],
        )

        # Normalizzazione per usare similarity coseno con IndexFlatIP
        _l2_normalize_inplace(embeddings)

        index = _build_faiss_index(embeddings.shape[1])
        index.add(embeddings)

        metadata = _extract_metadata(df)

        # Persistenza
        faiss.write_index(index, str(output_dir / index_name))
        with open(output_dir / meta_name, "wb") as fh:
            pickle.dump(metadata, fh, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(
            "Persistenza completata: %s (%s vettori) e %s",
            index_name,
            len(metadata),
            meta_name,
        )

    logger.info("Indicizzazione conclusa con successo.")


# --------------------------------------------------------------------------- #
# Helpers interni
# --------------------------------------------------------------------------- #
def _collect_parquet_files(base_dir: Path, prefix: str) -> List[Path]:
    pattern = str(base_dir / f"{prefix}*.parquet")
    return [Path(p) for p in glob.glob(pattern)]


def _read_and_validate_parquets(files: List[Path], prefix: str) -> pd.DataFrame:
    dfs = [pd.read_parquet(p) for p in files]
    df = pd.concat(dfs, ignore_index=True)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"File '{prefix}*.parquet' mancanti delle colonne obbligatorie: {missing}"
        )

    # Garanzia che 'embedding' sia lista di float
    if not df["embedding"].map(
        lambda x: isinstance(x, (list, tuple, np.ndarray))
    ).all():
        raise TypeError(
            f"Colonna 'embedding' deve contenere sequenze di float (lista o tuple)."
        )

    return df


def _build_embedding_matrix(col: pd.Series) -> np.ndarray:
    """Stack embedding column into shape (N, D) float32 array with checks."""
    arr = np.asarray(col.tolist(), dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(
            f"Embedding matrix attesa di forma (N, D), trovata {arr.shape}"
        )
    return arr


def _l2_normalize_inplace(mat: np.ndarray) -> None:
    """Modifica in‑place normalizzando ogni vettore a norma L2 unitaria."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    # Evita divisione per zero
    norms[norms == 0] = 1.0
    mat /= norms


def _build_faiss_index(d: int) -> faiss.IndexFlatIP:
    """Crea un indice FAISS basato su inner product (compatibile con coseno)."""
    index = faiss.IndexFlatIP(d)
    if not index.is_trained:
        raise RuntimeError("L'indice FAISS non risulta 'trained' (IndexFlatIP dovrebbe).")
    return index


def _extract_metadata(df: pd.DataFrame) -> List[Dict[str, str]]:
    """Ritorna lista di dict con name, type, summary, description."""
    return df[["name", "type", "summary", "description"]].to_dict(orient="records")


# --------------------------------------------------------------------------- #
# Esecuzione da CLI (opzionale)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    input_dir = Path("C:/Users/paolo/Desktop/Ontology-Induction/outputs/graphragbench_medical/postprocessing/embedding")
    output_dir = Path("C:/Users/paolo/Desktop/Ontology-Induction/outputs/graphragbench_medical/postprocessing/indexing")
    build_faiss_indices(input_dir, output_dir)