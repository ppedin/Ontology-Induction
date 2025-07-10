import pyarrow as pa
import pyarrow.parquet as pq
import pathlib
import pandas as pd
import os

class SafeParquetWriter:
    def __init__(self, final_path: pathlib.Path, schema: pa.Schema, batch_size: int = 500):
        self.final_path = final_path
        self.tmp_path   = final_path.with_suffix(".tmp")
        self.schema = schema
        self.batch_size = batch_size
        self.rows = []

    def append_rows(self, rows: list[dict]):
        self.rows.extend(rows)
        if len(self.rows) >= self.batch_size:
            self._flush()

    def _flush(self):
        if not self.rows:
            return
        table = pa.Table.from_pylist(self.rows, schema=self.schema)
        # 1) scrivi in file *temporaneo*
        with pq.ParquetWriter(self.tmp_path, self.schema, compression="zstd") as w:
            w.write_table(table)
        # 2) fsync per garantire che sia su disco
        with open(self.tmp_path, "rb") as fout:
            fout.flush()
            os.fsync(fout.fileno())
        # 3) move atomico → sostituisce o crea il file finale
        os.replace(self.tmp_path, self.final_path)
        self.rows.clear()

    def close(self):
        self._flush()