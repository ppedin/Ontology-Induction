import argparse
import os
import json
from pathlib import Path
from typing import List, Iterator, Dict

import faiss
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# SentenceIndexer: builds or updates a FAISS index over sentence embeddings
# ---------------------------------------------------------------------------

class SentenceIndexer:
    """Incremental, fault‑tolerant sentence indexer.

    * Each sentence receives a monotonically‑increasing ID.
    * Embeddings are stored in a FAISS **IndexIDMap** ⇒ supports `add_with_ids`.
    * A companion JSONL file stores metadata (ID, text, chunk, doc).
    * On restart we reload index + mapping, detect the last used ID, and resume.
    """

    def __init__(
        self,
        index_dir: Path,
        embedder_name: str = "all-mpnet-base-v2",
        batch_size: int = 128,
    ) -> None:
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self.embedder = SentenceTransformer(embedder_name)
        self.dimension = self.embedder.get_sentence_embedding_dimension()

        # ---------------- FAISS index -----------------
        self.faiss_path = self.index_dir / "sentences.faiss"
        if self.faiss_path.exists():
            # Carica l'indice di base esistente...
            base_index = faiss.read_index(str(self.faiss_path))
            # ...e avvolgilo SEMPRE in un IndexIDMap.
            self.index = faiss.IndexIDMap(base_index)
            print(f"[SentenceIndexer] Loaded FAISS index ({self.index.ntotal} vectors).")
        else:
            # Crea un nuovo indice da zero
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dimension))
            print("[SentenceIndexer] Created new FAISS IndexIDMap.")

        # ---------------- Mapping file -----------------
        self.mapping_path = self.index_dir / "sentences.jsonl"
        if self.mapping_path.exists():
            with self.mapping_path.open("r", encoding="utf-8") as f:
                self.next_id = sum(1 for _ in f)
            # reopen in append mode
            self.mapping_file = self.mapping_path.open("a", encoding="utf-8")
            print(f"[SentenceIndexer] Mapping file already has {self.next_id} records.")
        else:
            self.mapping_file = self.mapping_path.open("w", encoding="utf-8")
            self.next_id = 0
            print("[SentenceIndexer] Created new mapping file.")

        # ---------------- SpaCy splitter -----------------
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("sentencizer")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_chunks(self, chunk_iter: Iterator[Dict]):
        """Iterate over chunks and add their sentences to the index."""
        buffer_sent: List[str] = []
        buffer_meta: List[dict] = []

        for chunk in tqdm(chunk_iter, desc="Indexing chunks"):
            text = chunk["text"]
            chunk_idx = chunk.get("chunk_idx")
            doc_id = chunk.get("doc_id")

            for sent in self._split_sentences(text):
                sent_id = self.next_id + len(buffer_sent)
                buffer_sent.append(sent)
                buffer_meta.append({
                    "id": sent_id,
                    "sentence": sent,
                    "chunk_idx": chunk_idx,
                    "doc_id": doc_id,
                })

                if len(buffer_sent) >= self.batch_size:
                    self._flush(buffer_sent, buffer_meta)
                    buffer_sent.clear()
                    buffer_meta.clear()

        if buffer_sent:
            self._flush(buffer_sent, buffer_meta)

        self.mapping_file.close()
        print("[SentenceIndexer] Completed. Total sentences:", self.index.ntotal)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_sentences(self, text: str) -> List[str]:
        return [s.text.strip() for s in self.nlp(text).sents if s.text.strip()]

    def _flush(self, sentences: List[str], meta: List[dict]):
        embeddings = self.embedder.encode(
            sentences,
            convert_to_numpy=True,
            batch_size=len(sentences),
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype("float32")

        ids = np.arange(self.next_id, self.next_id + len(sentences)).astype("int64")
        self.index.add_with_ids(embeddings, ids)
        self.next_id += len(sentences)

        # Persist
        faiss.write_index(self.index, str(self.faiss_path))
        for m in meta:
            self.mapping_file.write(json.dumps(m, ensure_ascii=False) + "\n")
        self.mapping_file.flush()
        print(f"[SentenceIndexer] Flushed {len(sentences)} sentences (total {self.index.ntotal}).")

# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

def build_chunk_iterator(dataset_base_path: str) -> Iterator[dict]:
    """Yields dicts with keys: text, chunk_idx, doc_id."""
    from datasets.datasets_utils import GraphRAGBenchMedical  # type: ignore

    corpus_path = os.path.join(dataset_base_path, "graphragbench_medical/graphragbench_medical_corpus")
    graphragbench_medical = GraphRAGBenchMedical(corpus_path, chunk_size=512)

    for i, chunk in enumerate(graphragbench_medical.get_documents()):
        yield {
            "chunk_idx": i,
            "doc_id": chunk.get("doc_id", "unknown") if isinstance(chunk, dict) else "unknown",
            "text": chunk["text"] if isinstance(chunk, dict) else chunk,
        }


def main():
    parser = argparse.ArgumentParser(description="SentenceIndexer for GraphRAGBenchMedical")
    parser.add_argument("--index_dir", type=Path, default="evaluation/retrieval_analysis/graphragbench_medical/sentence_index")
    parser.add_argument("--dataset_base", type=str, default="C:/Users/paolo/Desktop/Ontology-Induction/datasets")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    indexer = SentenceIndexer(args.index_dir, batch_size=args.batch_size)
    indexer.index_chunks(build_chunk_iterator(args.dataset_base))


if __name__ == "__main__":
    main()
