from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Iterable, List, Sequence

import faiss
import numpy as np
import spacy
from tqdm import tqdm

from llm.call_embedder import get_embeddings_sync

# ---------------------------------------------------------------------------#
# Config – valori di default                                                 #
# ---------------------------------------------------------------------------#
DEFAULT_EMBED_MODEL   = "text-embedding-3-small"
DEFAULT_EMBED_DIM     = 1536            # ← text‑embedding‑3‑small
DEFAULT_BATCH_SIZE    = 128
DEFAULT_NUM_WORKERS   = max(1, mp.cpu_count() - 1)
DEF_INDEX_FILENAME    = "faiss.index"
DEF_MAPPING_FILENAME  = "sentences.jsonl"
DEF_PROGRESS_FILENAME = "progress.txt"
DEF_CHUNK_SENTENCES   = 3              # n. frasi per blocco da indicizzare


# ---------------------------------------------------------------------------#
# API di alto livello (funzione “one‑shot”)                                  #
# ---------------------------------------------------------------------------#
def build_sentence_index(
    corpus: Iterable[str] | Sequence[str],
    index_dir: str | Path,
    *,
    sentences_per_chunk: int = DEF_CHUNK_SENTENCES,
    embed_model: str = DEFAULT_EMBED_MODEL,
    embed_dim: int = DEFAULT_EMBED_DIM,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    index_filename: str = DEF_INDEX_FILENAME,
    mapping_filename: str = DEF_MAPPING_FILENAME,
    progress_filename: str = DEF_PROGRESS_FILENAME,
) -> None:
    """Costruisce (o aggiorna) un indice FAISS per un corpus testuale.

    Parametri ricevuti *via codice*, non via CLI – quindi facilmente
    richiamabile da orchestrator o da notebook.
    """
    indexer = _SentenceIndexer(
        index_dir=Path(index_dir),
        sentences_per_chunk=sentences_per_chunk,
        embed_model=embed_model,
        embed_dim=embed_dim,
        batch_size=batch_size,
        num_workers=num_workers,
        index_filename=index_filename,
        mapping_filename=mapping_filename,
        progress_filename=progress_filename,
    )
    indexer.index_corpus(corpus)


# ---------------------------------------------------------------------------#
# Implementazione                                                            #
# ---------------------------------------------------------------------------#
class _SentenceIndexer:
    """Classe interna – non esportata – che incapsula il processo vero e proprio."""

    # ---------------------------- init ---------------------------------- #
    def __init__(
        self,
        *,
        index_dir: Path,
        sentences_per_chunk: int,
        embed_model: str,
        embed_dim: int,
        batch_size: int,
        num_workers: int,
        index_filename: str,
        mapping_filename: str,
        progress_filename: str,
    ) -> None:
        self.index_dir         = index_dir
        self.sentences_per_ck  = sentences_per_chunk
        self.embed_model       = embed_model
        self.embed_dim         = embed_dim
        self.batch_size        = batch_size
        self.num_workers       = max(1, num_workers)
        self.index_path        = self.index_dir / index_filename
        self.mapping_path      = self.index_dir / mapping_filename
        self.progress_path     = self.index_dir / progress_filename

        # -------------------- create paths ------------------------------- #
        self.index_dir.mkdir(parents=True, exist_ok=True)
        if not self.mapping_path.exists():
            self.mapping_path.touch()

        # -------------------- spaCy splitter ----------------------------- #
        self._nlp = spacy.blank("en")
        self._nlp.add_pipe("sentencizer")

        # -------------------- FAISS index -------------------------------- #
        if self.index_path.exists():
            self.index: faiss.Index = faiss.read_index(str(self.index_path))
            assert self.index.d == self.embed_dim, "Index dim ≠ embedding dim"
        else:
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.embed_dim))
            faiss.write_index(self.index, str(self.index_path))

        # -------------------- recovery / progress ------------------------ #
        self.next_id = self.index.ntotal
        self.last_processed_chunk = (
            int(self.progress_path.read_text() or -1)
            if self.progress_path.exists()
            else -1
        )

    # ------------------------- public ----------------------------------- #
    def index_corpus(self, corpus_chunks: Iterable[str] | Sequence[str]) -> None:
        """Loop principale: scorre il corpus e indicizza."""

        enumerated = (
            enumerate(corpus_chunks)
            if not isinstance(corpus_chunks, Sequence)
            else enumerate(list(corpus_chunks))
        )

        # tqdm progress bar
        pbar = tqdm(enumerated, desc="Indexing", unit="chunk", leave=False)

        # -------------- pipeline con (eventuale) multiprocessing ---------- #
        # Ogni chunk viene:
        #   1) suddiviso in frasi
        #   2) re‑aggregato in blocchi da `sentences_per_ck`
        #   3) inviato in batch a get_embeddings_sync
        #
        # Se num_workers==1 ⇒ esecuzione seriale
        # Altrimenti uso mp.Pool (process‑safe; FAISS scrive solo nel proc principale)
        #
        for chunk_id, chunk_text in pbar:
            if chunk_id <= self.last_processed_chunk:        # fault tolerance
                continue

            # 1) pre‑processing & splitting
            blocks = self._prepare_blocks(chunk_text)
            if not blocks:
                self._save_progress(chunk_id)
                continue

            # 2) embed (seriale o parallelo)
            embeds = self._batch_embed(blocks)

            # 3) update index
            self._add_to_index(embeds, blocks)

            # 4) persist on disk
            self._persist(chunk_id)

    # -------------------- helper: prepare blocks ------------------------ #
    def _prepare_blocks(self, text: str) -> List[str]:
        sents = [
            s.text.strip()
            for s in self._nlp(text).sents
            if s.text.strip()
        ]
        if not sents:
            return []
        k = self.sentences_per_ck
        return [" ".join(sents[i : i + k]) for i in range(0, len(sents), k) if len(sents[i : i + k]) == k]

    # -------------------- helper: embedding ----------------------------- #
    def _batch_embed(self, blocks: List[str]) -> np.ndarray:
        """Embedding parallelo a livello di chiamata API.

        ⚠️ OpenAI impone rate‑limit: num_workers > 1 usa multithread
           con cautela; qui utilizzo multiprocessing.pool sui batch per
           ragioni di semplicità. In alternativa puoi:
           • usare asyncio + semaforo
           • sfruttare FAISS GPU e ridurre network bottleneck
        """
        if self.num_workers == 1 or len(blocks) <= self.batch_size:
            return get_embeddings_sync(blocks, model=self.embed_model)

        # Suddivido in mini‑batch per evitare request troppo grandi
        mini_batches = [
            blocks[i : i + self.batch_size] for i in range(0, len(blocks), self.batch_size)
        ]
        with mp.Pool(processes=self.num_workers) as pool:
            arrays = pool.starmap(
                get_embeddings_sync,
                [(mb, self.embed_model) for mb in mini_batches],
            )
        return np.vstack(arrays)

    # -------------------- helper: FAISS & mapping ----------------------- #
    def _add_to_index(self, vecs: np.ndarray, blocks: List[str]) -> None:
        ids = np.arange(self.next_id, self.next_id + vecs.shape[0])
        self.next_id += vecs.shape[0]
        self.index.add_with_ids(vecs, ids)

        # mapping jsonl
        with self.mapping_path.open("a", encoding="utf-8") as f:
            for _id, txt in zip(ids, blocks):
                f.write(json.dumps({"id": int(_id), "sentence": txt}) + "\n")

    # -------------------- persistence & recovery ------------------------ #
    def _persist(self, chunk_id: int) -> None:
        faiss.write_index(self.index, str(self.index_path))
        self._save_progress(chunk_id)

    def _save_progress(self, chunk_id: int) -> None:
        self.progress_path.write_text(str(chunk_id))
        self.last_processed_chunk = chunk_id


# ---------------------------------------------------------------------------#
# Main di test (richiamabile con `python -m postprocessing.sentence_indexing`)#
# ---------------------------------------------------------------------------#
if __name__ == "__main__":
    from datasets.datasets_utils import GraphRAGBenchMedical  # type: ignore

    DATASET_BASE = Path("C:/Users/paolo/Desktop/Ontology-Induction/datasets")
    DATASET_NAME = "graphragbench_medical"
    CORPUS_PATH  = DATASET_BASE / DATASET_NAME / f"{DATASET_NAME}_corpus"

    index_out = Path("outputs") / DATASET_NAME / "sentence_index"

    # generator che restituisce i chunk di testo da indicizzare
    chunk_gen = GraphRAGBenchMedical(str(CORPUS_PATH), chunk_size=512).get_documents()

    # build index
    build_sentence_index(
        corpus=chunk_gen,
        index_dir=index_out,
        sentences_per_chunk=4,
        num_workers=4,                # modifica qui per scalare
    )

    print("Indicizzazione completata.")
