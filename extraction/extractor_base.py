from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pyarrow as pa
from tqdm import tqdm
from pydantic import BaseModel
from google import genai
from google.genai import types
from typing import Union
import pathlib
import threading
import signal

from llm.call_llm import call_gemini
from extraction.safe_writer import SafeParquetWriter
import uuid

stop_event = threading.Event()

def _handle_signal(signum, frame):
    print(f"[i] Caught signal {signum}. Shutting down gracefully…")
    stop_event.set()

for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, _handle_signal)

SCHEMA = pa.schema([
    ("chunk_id", pa.string()),
    ("row_type", pa.string()),
    ("name", pa.string()),
    ("entity_type", pa.string()),
    ("description", pa.string()),
    ("head", pa.string()),
    ("tail", pa.string()),
    ("relation", pa.string())
])

class TripleExtractorBase:
    def __init__(   
        self,
        llm_client: genai.Client,
        model_name: str,
        system_prompt: str,
        response_schema: BaseModel,
        output_path: Union[str, pathlib.Path],
        max_workers: int = 4,
        batch_size: int = 500,
        thinking_budget: int = 512,
        verbose: bool = False
    ):
        self.llm_client = llm_client
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.response_schema = response_schema
        self.max_workers = max_workers
        self.thinking_budget = thinking_budget
        self.verbose = verbose
        self.writer = SafeParquetWriter(output_path, SCHEMA, batch_size)
        self.processed_log = Path(output_path).with_suffix(".processed.txt")
        self.processed_chunks = set(self.processed_log.read_text().splitlines()) if self.processed_log.exists() else set()
        self.lock = threading.Lock()

    def _save_chunk_id(self, chunk_id: str):
        with self.processed_log.open("a") as f:
            f.write(chunk_id + "\n")
        self.processed_chunks.add(chunk_id)

    def _submit(self, executor, chunk_id: str, text: str):
        return executor.submit(self._process_single_chunk, chunk_id, text)
    
    def build_prompt(self, text: str):
        return f"The text is: \n{text}\n" 

    def _process_single_chunk(self, chunk_id: str, text: str):
        if chunk_id in self.processed_chunks:
            return []                       # already done
        user_prompt = self.build_prompt(text)
        result = call_gemini(                    
            gemini_client=self.llm_client,
            model_name=self.model_name,
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            response_schema=self.response_schema,
            thinking_budget=self.thinking_budget,
            verbose=self.verbose
        )
        rows = self._result_to_rows(result, chunk_id)
        return rows
    
    def _result_to_rows(self, result: BaseModel, chunk_id: str) -> list[dict]:
        with self.lock:
            rows = []
            for ent in getattr(result, "entities", []):
                rows.append({
                "chunk_id": chunk_id,
                "row_type": "entity",
                "name": ent.entity_name,
                "entity_type": getattr(ent, "entity_type", None),
                "description": getattr(ent, "entity_description", None),
                "head": None,
                "tail": None,
                "relation": None
            })

            for rel in getattr(result, "relationships", []):
                rows.append({
                    "chunk_id": chunk_id,
                    "row_type": "relationship",
                    "name": None,
                    "entity_type": None,
                    "description": None,
                    "head": rel.source_entity if hasattr(rel, "source_entity") else getattr(rel, "head", None),
                    "tail": rel.target_entity if hasattr(rel, "target_entity") else getattr(rel, "tail", None),
                    "relation": getattr(rel, "relationship_description", None) or getattr(rel, "relation", None),
                })
        return rows

    def extract_from_chunks(self, chunks_iter):
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = []
            for i, chunk in enumerate(chunks_iter):
                if stop_event.is_set():
                    break
                chunk_id = f"Chunk_{str(uuid.uuid4())}"
                futures.append(self._submit(pool, chunk_id, chunk))

            for fut in tqdm(as_completed(futures), total=len(futures), desc="LLM extraction"):
                try:
                    rows = fut.result()
                    self.writer.append_rows(rows)
                    self._save_chunk_id(fut.chunk_id)
                except Exception as e:
                    print("[!] Worker error:", e)

        self.writer.close()
