from __future__ import annotations
import hashlib, json, os, signal, time, typing as t
from dataclasses import dataclass
from pathlib import Path
from queue import Queue, Empty
from threading import Event, Thread
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ValidationError
from tqdm import tqdm

T = t.TypeVar("T", bound=BaseModel)

@dataclass
class _WorkItem:
    idx: int
    text: str
    sha1: str                    # deterministic fingerprint

class TripleExtractorBase(t.Generic[T]):
    """
    Abstract base class – subclass and implement:
        * build_user_prompt(chunk: str) -> str
        * response_schema: ClassVar[type[BaseModel]]
    """
    # ---- MUST override in subclass ----
    response_schema: type[T] = BaseModel  # type: ignore
    system_prompt: str = ""

    # -----------------------------------
    def __init__(
        self,
        llm_client,
        model_name: str,
        output_path: Path,
        max_workers: int = 4,
        batch_size: int = 5,
        thinking_budget: int = 512,
        verbose: bool = False,
    ):
        self.llm_client = llm_client
        self.model_name = model_name
        self.output_path = output_path
        self.parts_dir = output_path.with_suffix("") / "parts"
        self.manifest = output_path.with_suffix("") / "progress.jsonl"
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.thinking_budget = thinking_budget
        self.verbose = verbose

        self.parts_dir.mkdir(parents=True, exist_ok=True)
        self.manifest.touch(exist_ok=True)  # ensures file exists

        # quick lookup of already‑processed fingerprints
        with self.manifest.open() as f:
            self._seen_sha1: set[str] = {line.strip() for line in f if line.strip()}

        # graceful shutdown handling
        self._stop_event = Event()
        signal.signal(signal.SIGINT,  self._signal_handler)  # Ctrl‑C
        signal.signal(signal.SIGTERM, self._signal_handler)

    # -------- public driver --------
    def extract_from_chunks(self, chunks: t.Iterable[str]) -> None:
        """
        Main entry point.  Streams chunks → parallel LLM calls → parquet parts.
        """
        work_queue: Queue[list[pa.RecordBatch]] = Queue(maxsize=self.max_workers * 2)

        # writer thread --------------------------------------------------------
        def writer() -> None:
            while not (self._stop_event.is_set() and work_queue.empty()):
                try:
                    record_batches = work_queue.get(timeout=1)
                except Empty:
                    continue
                self._write_part(record_batches)
                work_queue.task_done()

        writer_thread = Thread(target=writer, daemon=True)
        writer_thread.start()

        # producer / worker pool ----------------------------------------------
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            current_batch: list[_WorkItem] = []
            futures = []
            for idx, chunk in enumerate(chunks):
                sha1 = hashlib.sha1(chunk.encode()).hexdigest()
                if sha1 in self._seen_sha1:
                    continue  # already processed previous run

                current_batch.append(_WorkItem(idx, chunk, sha1))
                if len(current_batch) == self.batch_size:
                    futures.append(pool.submit(self._process_batch, current_batch))
                    current_batch = []

            if current_batch:
                futures.append(pool.submit(self._process_batch, current_batch))

            # collect finished tasks and pass to writer
            for f in as_completed(futures):
                record_batches = f.result()
                work_queue.put(record_batches)

        # wait for remaining writes, then close
        work_queue.join()
        self._stop_event.set()
        writer_thread.join()

    # -------- subclass hooks --------
    def build_user_prompt(self, chunk: str) -> str:  # → override
        raise NotImplementedError

    # -------- internal helpers --------
    def _process_batch(self, batch: list[_WorkItem]) -> list[pa.RecordBatch]:
        record_batches: list[pa.RecordBatch] = []
        for item in batch:
            if self._stop_event.is_set():
                break
            try:
                result = self._call_llm(item.text)
                arrow_rb = self._to_record_batch(item.sha1, result)
                record_batches.append(arrow_rb)
            except Exception as e:   # broad catch is fine here; we log & continue
                if self.verbose:
                    print(f"[WARN] idx={item.idx} sha1={item.sha1[:8]} error: {e}")
        return record_batches

    def _call_llm(self, chunk: str) -> T:
        user_prompt = self.build_user_prompt(chunk)

        # Retry loop with back‑off
        delay = 1
        while True:
            try:
                from llm.call_llm import call_gemini  # local convenience
                response = call_gemini(
                    gemini_client=self.llm_client,
                    model_name=self.model_name,
                    system_prompt=self.system_prompt,
                    user_prompt=user_prompt,
                    response_schema=self.response_schema,
                    thinking_budget=self.thinking_budget,
                    verbose=self.verbose,
                )
                return response
            except (ValidationError, RuntimeError) as e:
                if delay > 64:  # fail after ~2 min
                    raise
                time.sleep(delay)
                delay *= 2

    def _to_record_batch(self, sha1: str, parsed: T) -> pa.RecordBatch:
        """
        Convert the pydantic object graph to a flat Arrow RecordBatch.
        For hierarchical objects you can normalise into two tables; here we keep a single JSON column.
        """
        json_col = json.dumps(parsed.dict())
        array = pa.array([json_col])
        sha1_array = pa.array([sha1])
        return pa.record_batch([sha1_array, array], names=["sha1", "payload_json"])

    def _write_part(self, record_batches: list[pa.RecordBatch]) -> None:
        if not record_batches:
            return
        table = pa.Table.from_batches(record_batches)
        part_name = f"part-{record_batches[0]['sha1'][0].as_py()[:12]}-{int(time.time()*1e6)}.parquet"
        part_path = self.parts_dir / part_name
        pq.write_table(table, part_path, compression="zstd")

        # append to manifest atomically
        with self.manifest.open("a") as m:
            for sha in table.column("sha1"):
                m.write(sha.as_py() + "\n")
                self._seen_sha1.add(sha.as_py())

        if self.verbose:
            print(f"[WRITE] {part_path.name} (rows={table.num_rows})")

    # -------- utilities --------
    def _signal_handler(self, signum, frame):
        print(f"[INFO] Received signal {signum} → stopping gracefully…")
        self._stop_event.set()