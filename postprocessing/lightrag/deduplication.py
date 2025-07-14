from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import List

import pandas as pd


# --------------------------------------------------------------------------- #
# Logging configuration                                                       #
# --------------------------------------------------------------------------- #
LOGGER_NAME = __name__
logger = logging.getLogger(LOGGER_NAME)
if not logger.handlers:
    # Configurazione di default (utile quando il modulo è lanciato come script)
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
    )


# --------------------------------------------------------------------------- #
# Funzione pubblica                                                           #
# --------------------------------------------------------------------------- #
def deduplicate_entities(input_dir: Path, output_dir: Path) -> None:
    """
    Deduplica i file `entities.parquet` e copia `relationships.parquet`.

    Parameters
    ----------
    input_dir : Path
        Directory che contiene i file di input.
    output_dir : Path
        Directory in cui verranno salvati i file di output. Viene creata se
        non esiste.

    Note
    ----
    • Se un record ha campi mancanti, la riga viene scartata con warning.
    • In caso di conflitto di `entity_type` all’interno dello stesso
      `entity_name`, viene mantenuto il primo valore (ordine originale) e
      viene emesso un warning.
    """
    input_dir = Path(input_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    entities_path = input_dir / "entities.parquet"
    rels_path = input_dir / "relationships.parquet"

    if not entities_path.is_file():
        raise FileNotFoundError(f"File non trovato: {entities_path}")
    if not rels_path.is_file():
        raise FileNotFoundError(f"File non trovato: {rels_path}")

    # --------------------------------------------------------------------- #
    # 1. Lettura del file entities.parquet                                  #
    # --------------------------------------------------------------------- #
    logger.info("Lettura entità da %s", entities_path)
    df = pd.read_parquet(entities_path, engine="pyarrow")
    initial_count = len(df)

    required_cols: List[str] = [
        "entity_name",
        "entity_type",
        "entity_description",
        "source_chunk",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Il file entities.parquet è privo delle colonne richieste: {missing_cols}"
        )

    # --------------------------------------------------------------------- #
    # 2. Pulizia righe con campi mancanti                                   #
    # --------------------------------------------------------------------- #
    null_mask = df[required_cols].isnull().any(axis=1)
    if null_mask.any():
        n_null = int(null_mask.sum())
        logger.warning(
            "Saranno scartate %d righe con valori mancanti in %s",
            n_null,
            required_cols,
        )
        df = df[~null_mask]

    # --------------------------------------------------------------------- #
    # 3. Deduplicazione                                                     #
    # --------------------------------------------------------------------- #
    def _aggregate(group: pd.DataFrame) -> pd.Series:
        """Aggrega i record omonimi in un’unica riga."""
        # 3a. Gestione entity_type
        first_type = group["entity_type"].iloc[0]
        unique_types = group["entity_type"].unique()
        if len(unique_types) > 1:
            logger.warning(
                "Conflitto di entity_type per '%s': %s. "
                "Verrà mantenuto '%s'.",
                group.name,
                list(unique_types),
                first_type,
            )

        # 3b. Concatenazione descrizioni e source_chunk
        join = " ||| "
        descriptions = join.join(group["entity_description"].astype(str))
        source_chunks = join.join(group["source_chunk"].astype(str))

        return pd.Series(
            {
                "entity_name": group.name,
                "entity_type": first_type,
                "entity_description": descriptions,
                "source_chunk": source_chunks,
            }
        )

    logger.info("Esecuzione deduplicazione…")
    dedup_df = (
        df.groupby("entity_name", sort=False, as_index=False)
        .apply(_aggregate)
        .reset_index(drop=True)
    )

    final_count = len(dedup_df)
    deduplicated = initial_count - final_count
    logger.info(
        "Entità iniziali: %d | Entità finali: %d | Record unificati: %d",
        initial_count,
        final_count,
        deduplicated,
    )

    # --------------------------------------------------------------------- #
    # 4. Scrittura file deduplicato                                         #
    # --------------------------------------------------------------------- #
    entities_out = output_dir / "entities.parquet"
    logger.info("Scrittura entità deduplicate in %s", entities_out)
    dedup_df.to_parquet(entities_out, index=False, engine="pyarrow")

    # --------------------------------------------------------------------- #
    # 5. Deduplicazione e scrittura relationships.parquet                   #
    # --------------------------------------------------------------------- #
    logger.info("Lettura relazioni da %s", rels_path)
    rels_df = pd.read_parquet(rels_path, engine="pyarrow")
    rels_required_cols = [
        "head",
        "tail",
        "relation_description",
        "relation_keywords",
        "source_chunk",
    ]
    missing_rels_cols = [c for c in rels_required_cols if c not in rels_df.columns]
    if missing_rels_cols:
        raise ValueError(
            f"Il file relationships.parquet è privo delle colonne richieste: {missing_rels_cols}"
        )

    null_mask_rels = rels_df[rels_required_cols].isnull().any(axis=1)
    if null_mask_rels.any():
        n_null = int(null_mask_rels.sum())
        logger.warning(
            "Saranno scartate %d righe con valori mancanti in %s (relazioni)",
            n_null,
            rels_required_cols,
        )
        rels_df = rels_df[~null_mask_rels]

    def _aggregate_rels(group: pd.DataFrame) -> pd.Series:
        join = " ||| "
        head = group["head"].iloc[0]
        tail = group["tail"].iloc[0]
        descs = join.join(group["relation_description"].astype(str))
        sources = join.join(group["source_chunk"].astype(str))
        return pd.Series(
            {
                "head": head,
                "tail": tail,
                "relation_description": descs,
                "relation_keywords": group.name,
                "source_chunk": sources,
            }
        )

    logger.info("Deduplicazione relazioni…")
    initial_rels_count = len(rels_df)

    # Evita il FutureWarning: includi esplicitamente solo le colonne necessarie
    grouped_rels = rels_df.groupby("relation_keywords", sort=False)

    dedup_rels_df = (
        grouped_rels[rels_required_cols]
        .apply(_aggregate_rels)
        .reset_index(drop=True)
    )

    final_rels_count = len(dedup_rels_df)
    deduplicated_rels = initial_rels_count - final_rels_count
    logger.info(
        "Relazioni iniziali: %d | Relazioni finali: %d | Record unificati: %d",
        initial_rels_count,
        final_rels_count,
        deduplicated_rels,
    )

    rels_out = output_dir / "relationships.parquet"
    logger.info("Scrittura relazioni deduplicate in %s", rels_out)
    dedup_rels_df.to_parquet(rels_out, index=False, engine="pyarrow")


# --------------------------------------------------------------------------- #
# Esecuzione da linea di comando (opzionale)                                  #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    BASE_PATH = Path("C:/Users/paolo/Desktop/Ontology-Induction/outputs/exp_7_13/graphragbench_medical")
    deduplicate_entities(
        input_dir=BASE_PATH / "pragmarag_prompt" / "postprocessing" / "graph_formatter",
        output_dir=BASE_PATH / "pragmarag_prompt" / "postprocessing" / "deduplication"
    )
    