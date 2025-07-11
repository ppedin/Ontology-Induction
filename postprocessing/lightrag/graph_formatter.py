# graph_formatter.py
"""
Utility for post‑processing entity‑relation extractions produced by the
LightRAG baseline.

It reads the raw *.parquet parts written by the extraction stage,
normalises the rows, and writes two clean datasets:

    • entities.parquet
    • relationships.parquet

Both files are written with a *safe* “write‑to‑tmp then atomic‑rename”
mechanism so that a crash or ^C never leaves a half‑written Parquet file.

---------------------------------------------------------------------------
Public API
---------------------------------------------------------------------------

    format_graph(input_dir: pathlib.Path, output_dir: pathlib.Path) -> None

Example
-------
>>> from pathlib import Path
>>> from graph_formatter import format_graph
>>> format_graph(
...     input_dir=Path("outputs/extraction/graphragbench_medical/"
...                    "graphragbench_medical_extraction/parts"),
...     output_dir=Path("outputs/graphragbench_medical/postprocessing/"
...                     "graph_formatter")
... )
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------#
# Logging configuration
# ---------------------------------------------------------------------------#
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------#
# Constants
# ---------------------------------------------------------------------------#
# Number of (entity or relationship) rows accumulated before the buffer is
# flushed to disk.
BATCH_SIZE = 3


# ---------------------------------------------------------------------------#
# Safe writer
# ---------------------------------------------------------------------------#
class SafeParquetWriter:
    """
    Incremental, failure‑tolerant writer for a single Parquet file.

    *   Accumulates rows in memory.
    *   Flushes every `batch_size` rows to a temporary *.tmp file using an
        internal :class:`pyarrow.parquet.ParquetWriter`.
    *   On :py:meth:`finalize`, closes the underlying writer and
        atomically renames the *.tmp file -> final *.parquet file.
    """

    def __init__(
        self,
        output_path: Path,
        schema: pa.schema,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        self._output_path: Path = output_path.with_suffix(".parquet")
        self._tmp_path: Path = self._output_path.with_suffix(".tmp")
        self._schema: pa.schema = schema
        self._batch_size: int = batch_size

        self._buffer: List[Dict[str, Any]] = []
        self._writer: pq.ParquetWriter | None = None

        # If an old tmp file exists (previous crash), remove it.
        if self._tmp_path.exists():
            logger.warning(
                "Removing stale temporary file: %s", self._tmp_path.as_posix()
            )
            self._tmp_path.unlink()

    # --------------------------------------------------------------------- #
    # Public helpers
    # --------------------------------------------------------------------- #
    def append_rows(self, rows: Iterable[Dict[str, Any]]) -> None:
        """
        Add iterable of *row dicts* to the internal buffer.  Flush to disk
        automatically when the buffer reaches ``batch_size`` elements.
        """
        for row in rows:
            self._buffer.append(row)
            if len(self._buffer) >= self._batch_size:
                self._flush()

    def finalize(self) -> None:
        """
        Write any remaining rows, close the writer, and make the final,
        atomic rename from *.tmp -> *.parquet.
        """
        self._flush(force=True)

        if self._writer is not None:
            self._writer.close()  # close ParquetWriter
            self._writer = None

        # Atomic rename (`replace` is atomic on the same filesystem).
        self._tmp_path.replace(self._output_path)
        logger.info("Wrote %s", self._output_path.as_posix())

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _flush(self, *, force: bool = False) -> None:
        """Write buffered rows if buffer is full or `force` is True."""
        if not self._buffer:
            return
        if not force and len(self._buffer) < self._batch_size:
            return

        # Convert list‑of‑dicts -> Arrow Table.
        batch_table: pa.Table = pa.Table.from_pylist(self._buffer, schema=self._schema)

        # Lazily create the ParquetWriter on first flush.
        if self._writer is None:
            self._writer = pq.ParquetWriter(
                where=str(self._tmp_path),
                schema=self._schema,
                compression="snappy",
            )

        self._writer.write_table(batch_table)
        logger.debug("Flushed %d rows to %s", len(self._buffer), self._tmp_path.name)
        self._buffer.clear()


# ---------------------------------------------------------------------------#
# Core formatting function
# ---------------------------------------------------------------------------#
def format_graph(input_dir: Path, output_dir: Path) -> None:
    """
    Convert raw extraction parts into entity & relationship tables.

    Parameters
    ----------
    input_dir
        Directory containing the `*.parquet` parts produced by the
        extraction step.
    output_dir
        Directory where the two clean Parquet files will be written.
    """
    # ------------------------------------------------------------------ #
    # Validation & I/O preparation
    # ------------------------------------------------------------------ #
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory {input_dir!s} does not exist.")

    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob("*.parquet"))
    if not input_files:
        raise FileNotFoundError(f"No *.parquet parts found in {input_dir!s}")

    logger.info("Found %d input part‑files.", len(input_files))

    # ------------------------------------------------------------------ #
    # Schema definitions
    # ------------------------------------------------------------------ #
    entity_schema = pa.schema(
        [
            pa.field("entity_name", pa.string()),
            pa.field("entity_type", pa.string()),
            pa.field("entity_description", pa.string()),
            pa.field("source_chunk", pa.string()),
        ]
    )

    relation_schema = pa.schema(
        [
            pa.field("head", pa.string()),
            pa.field("tail", pa.string()),
            pa.field("relation_description", pa.string()),
            pa.field("relation_keywords", pa.string()),
            pa.field("source_chunk", pa.string()),
        ]
    )

    # ------------------------------------------------------------------ #
    # Writers
    # ------------------------------------------------------------------ #
    entity_writer = SafeParquetWriter(
        output_path=output_dir / "entities",
        schema=entity_schema,
    )
    relation_writer = SafeParquetWriter(
        output_path=output_dir / "relationships",
        schema=relation_schema,
    )

    # ------------------------------------------------------------------ #
    # Processing loop
    # ------------------------------------------------------------------ #
    total_entities = 0
    total_relations = 0
    corrupt_rows = 0

    for fpath in input_files:
        logger.info("Processing %s …", fpath.name)

        try:
            pf = pq.ParquetFile(fpath)
        except Exception as exc:
            logger.error("Cannot read %s: %s", fpath.name, exc)
            continue

        # Iterate in moderate Arrow record‑batches to keep memory bounded.
        for batch in pf.iter_batches(batch_size=3):  # tweak as needed
            # Arrow → PyObjects for convenience.
            table = pa.Table.from_batches([batch])
            data = table.to_pylist()

            entity_rows: List[Dict[str, Any]] = []
            relation_rows: List[Dict[str, Any]] = []


            for row in data:
                sha1: str | None = row.get("sha1")
                parsed: Any = row.get("payload_json")


                if sha1 is None or parsed is None:
                    corrupt_rows += 1
                    logger.debug(
                        "Skipping row: missing sha1 or parsed in file %s", fpath.name
                    )
                    continue

                # The extraction pipeline *should* have serialised `parsed`
                # as a dict.  Sometimes it comes as a JSON‑encoded string.
                if not isinstance(parsed, dict):
                    try:
                        parsed = json.loads(parsed)
                    except Exception:
                        corrupt_rows += 1
                        logger.debug(
                            "Skipping row: cannot parse JSON in sha1=%s (%s)",
                            sha1,
                            fpath.name,
                        )
                        continue

                # ------------------  entities ------------------ #
                for ent in parsed.get("entities", []):
                    print(ent)
                    try:
                        entity_rows.append(
                            {
                                "entity_name": ent["entity_name"],
                                "entity_type": ent["entity_type"],
                                "entity_description": ent.get("entity_description", ""),
                                "source_chunk": sha1,
                            }
                        )
                    except Exception:
                        corrupt_rows += 1
                        logger.debug(
                            "Malformed entity in sha1=%s (%s)", sha1, fpath.name
                        )

                # ---------------  relationships --------------- #
                for rel in parsed.get("relationships", []):
                    try:
                        relation_rows.append(
                            {
                                "head": rel["head"],
                                "tail": rel["tail"],
                                "relation_description": rel.get(
                                    "relation_description", ""
                                ),
                                "relation_keywords": rel.get("relation_keywords", ""),
                                "source_chunk": sha1,
                            }
                        )
                    except Exception:
                        corrupt_rows += 1
                        logger.debug(
                            "Malformed relationship in sha1=%s (%s)", sha1, fpath.name
                        )

            # Batch append to writers.
            entity_writer.append_rows(entity_rows)
            relation_writer.append_rows(relation_rows)

            total_entities += len(entity_rows)
            total_relations += len(relation_rows)

    # ------------------------------------------------------------------ #
    # Finalise (flush last batches & rename)
    # ------------------------------------------------------------------ #
    entity_writer.finalize()
    relation_writer.finalize()

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    logger.info("------------------------------------------------------------")
    logger.info("Formatting completed.")
    logger.info("Entities written      : %d", total_entities)
    logger.info("Relationships written : %d", total_relations)
    logger.info("Corrupt / skipped rows: %d", corrupt_rows)
    logger.info("------------------------------------------------------------")


# ---------------------------------------------------------------------------#
# Command‑line entry point
# ---------------------------------------------------------------------------#
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Post‑process extracted graph parts into two clean Parquet files "
            "(entities & relationships)."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing raw extraction *.parquet parts.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where the formatted Parquet files will be created.",
    )
    args = parser.parse_args()

    format_graph(args.input_dir, args.output_dir)