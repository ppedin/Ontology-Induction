from __future__ import annotations

import json
import multiprocessing as mp
import pickle
from pathlib import Path
from typing import Iterable, List, Literal, Sequence

import faiss
import igraph as ig  # pip install python-igraph
import numpy as np
from tqdm import tqdm

from llm.call_embedder import get_embeddings_sync

# ---------------------------------------------------------------------------#
# Parametri di default (coerenti con sentence_indexing)                      #
# ---------------------------------------------------------------------------#
DEF_EMBED_MODEL       = "text-embedding-3-small"
DEF_EMBED_DIM         = 1536
DEF_BATCH_SIZE        = 128
DEF_NUM_WORKERS       = max(1, mp.cpu_count() - 1)
DEF_INDEX_FILENAME    = "faiss.index"
DEF_MAPPING_FILENAME  = "elements.jsonl"
DEF_PROGRESS_FILENAME = "progress.txt"

# ---------------------------------------------------------------------------#
# API pubblico                                                                #
# ---------------------------------------------------------------------------#
def build_graph_index(
    graph_source: str | Path,
    index_dir: str | Path,
    *,
    embed_model: str = DEF_EMBED_MODEL,
    embed_dim: int = DEF_EMBED_DIM,
    batch_size: int = DEF_BATCH_SIZE,
    num_workers: int = DEF_NUM_WORKERS,
    index_filename: str = DEF_INDEX_FILENAME,
    mapping_filename: str = DEF_MAPPING_FILENAME,
    progress_filename: str = DEF_PROGRESS_FILENAME,
) -> None:
    """Crea/aggiorna un indice FAISS per nodi + archi di un grafo.

    `graph_source` può essere:
    • path a file .graphml
    • path a pickle contenente oggetto igraph.Graph
    """
    indexer = _GraphIndexer(
        graph_source=graph_source,
        index_dir=Path(index_dir),
        embed_model=embed_model,
        embed_dim=embed_dim,
        batch_size=batch_size,
        num_workers=num_workers,
        index_filename=index_filename,
        mapping_filename=mapping_filename,
        progress_filename=progress_filename,
    )
    indexer.run()


# ---------------------------------------------------------------------------#
# Implementazione                                                             #
# ---------------------------------------------------------------------------#
class _GraphIndexer:
    """Indicizzatore di nodi e archi (unico FAISS Index)."""

    # ------------------------ init -------------------------------------- #
    def __init__(
        self,
        *,
        graph_source: str | Path,
        index_dir: Path,
        embed_model: str,
        embed_dim: int,
        batch_size: int,
        num_workers: int,
        index_filename: str,
        mapping_filename: str,
        progress_filename: str,
    ) -> None:
        self.graph_source   = Path(graph_source)
        self.index_dir      = index_dir
        self.embed_model    = embed_model
        self.embed_dim      = embed_dim
        self.batch_size     = batch_size
        self.num_workers    = max(1, num_workers)
        self.index_path     = self.index_dir / index_filename
        self.mapping_path   = self.index_dir / mapping_filename
        self.progress_path  = self.index_dir / progress_filename

        self.index_dir.mkdir(parents=True, exist_ok=True)
        if not self.mapping_path.exists():
            self.mapping_path.touch()

        # FAISS
        if self.index_path.exists():
            self.index: faiss.Index = faiss.read_index(str(self.index_path))
            assert self.index.d == self.embed_dim, "Index dim ≠ embed dim"
        else:
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.embed_dim))
            faiss.write_index(self.index, str(self.index_path))

        # progress
        self.next_id = self.index.ntotal
        self.last_processed = (
            int(self.progress_path.read_text() or -1)
            if self.progress_path.exists()
            else -1
        )

        # carica grafo in memoria
        self.graph = self._load_graph()

        # costruiamo la lista completa (nodi prima, poi archi)
        self.elements: list[_GraphElement] = self._collect_elements()

    # ------------------------ public run -------------------------------- #
    def run(self) -> None:
        """Esegue embedding e indicizzazione con fault‑tolerance."""
        total = len(self.elements)
        pbar = tqdm(
            enumerate(self.elements), total=total, desc="Graph indexing", unit="el"
        )

        buffer_texts: list[str] = []
        buffer_meta: list[_GraphElement] = []
        B = self.batch_size

        for idx, element in pbar:
            if idx <= self.last_processed:
                continue

            buffer_texts.append(element.text_for_embed)
            buffer_meta.append(element)

            if len(buffer_texts) >= B:
                self._flush_batch(buffer_texts, buffer_meta)
                self._save_progress(idx)
                buffer_texts.clear()
                buffer_meta.clear()

        # flush finale
        if buffer_texts:
            self._flush_batch(buffer_texts, buffer_meta)
            self._save_progress(total - 1)

    # ------------------------ helpers ----------------------------------- #
    def _flush_batch(
        self, texts: list[str], meta: list["_GraphElement"]
    ) -> None:
        vecs = (
            get_embeddings_sync(texts, model=self.embed_model)
            if self.num_workers == 1
            else self._embed_multiproc(texts)
        )
        ids = np.arange(self.next_id, self.next_id + vecs.shape[0])
        self.next_id += vecs.shape[0]
        self.index.add_with_ids(vecs, ids)

        with self.mapping_path.open("a", encoding="utf-8") as f:
            for _id, m in zip(ids, meta):
                f.write(json.dumps(m.to_metadata(int(_id))) + "\n")

        faiss.write_index(self.index, str(self.index_path))

    def _embed_multiproc(self, texts: list[str]) -> np.ndarray:
        """Multiprocessing semplice per evitare bottle‑neck di rete."""
        # dividiamo in sotto‑batch <= self.batch_size
        mini = [
            texts[i : i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]
        with mp.Pool(processes=self.num_workers) as pool:
            arrays = pool.starmap(
                get_embeddings_sync,
                [(mb, self.embed_model) for mb in mini],
            )
        return np.vstack(arrays)

    def _save_progress(self, idx: int) -> None:
        self.progress_path.write_text(str(idx))
        self.last_processed = idx

    # ------------------------ graph utils ------------------------------- #
    def _load_graph(self) -> ig.Graph:
        if self.graph_source.suffix == ".pkl":
            with open(self.graph_source, "rb") as fh:
                return pickle.load(fh)
        if self.graph_source.suffix == ".graphml":
            return ig.Graph.Read_GraphML(str(self.graph_source))
        raise ValueError("Formato grafo non riconosciuto (usa .graphml o .pkl)")

    def _collect_elements(self) -> list["_GraphElement"]:
        """Crea la lista ordinata (nodi seguiti da archi)."""
        elems: list[_GraphElement] = []

        # nodi
        for v in self.graph.vs:
            name = str(v["name"]) if "name" in v.attributes() else ""
            vtype = str(v["type"]) if "type" in v.attributes() else "Unknown"
            elems.append(
                _GraphElement(
                    kind="node",
                    label=name,
                    type=vtype,
                    text_for_embed=name,
                )
            )

        # archi
        for e in self.graph.es:
            kw = str(e["keyword"]) if "keyword" in e.attributes() else ""
            etype = str(e["description"])[:30] + "…" if "description" in e.attributes() else "edge"
            src, dst = e.tuple
            elems.append(
                _GraphElement(
                    kind="edge",
                    label=kw or f"{self.graph.vs[src]['name']}→{self.graph.vs[dst]['name']}",
                    type=etype,
                    text_for_embed=kw if kw else f"{self.graph.vs[src]['name']} {self.graph.vs[dst]['name']}",
                    src=src,
                    dst=dst,
                )
            )
        return elems


# ---------------------------------------------------------------------------#
# Dataclass semplificato                                                     #
# ---------------------------------------------------------------------------#
class _GraphElement:
    """Rappresenta un nodo o un arco con i metadati necessari."""

    def __init__(
        self,
        *,
        kind: Literal["node", "edge"],
        label: str,
        type: str,
        text_for_embed: str,
        src: int | None = None,
        dst: int | None = None,
    ):
        self.kind = kind
        self.label = label
        self.type = type
        self.text_for_embed = text_for_embed
        self.src = src
        self.dst = dst

    def to_metadata(self, idx: int) -> dict:
        base = {
            "id": idx,
            "kind": self.kind,
            "label": self.label,
            "type": self.type,
        }
        if self.kind == "edge":
            base |= {"src": self.src, "dst": self.dst}
        return base


# ---------------------------------------------------------------------------#
# Test rapido                                                                #
# ---------------------------------------------------------------------------#
if __name__ == "__main__":
    GRAPH_PATH = Path(
        "outputs/exp_7_13/graphragbench_medical/lightrag/postprocessing/graph_builder/graphragbench_medical.graphml"
    )
    INDEX_OUT = Path("outputs/graphragbench_medical/graph_index")

    build_graph_index(
        graph_source=GRAPH_PATH,
        index_dir=INDEX_OUT,
        num_workers=4,      # modifica per scalare
    )

    print("Indicizzazione grafo completata.")
