# dataset_utils.py
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, List, Union
import json


DATASETS_BASE_PATH = "C:/Users/paolo/Desktop/Ontology-Induction/datasets"


def _chunk_stream(lines: Iterator[str], chunk_size: int) -> Iterator[str]:
    """
    """
    buffer: list[str] = []
    for line in lines:
        buffer.extend(line.split())
        while len(buffer) >= chunk_size:
            yield " ".join(buffer[:chunk_size])
            buffer = buffer[chunk_size:]
    if buffer:
        yield " ".join(buffer)

###############################################################################
# Classe astratta
###############################################################################
class BaseDataset(ABC):
    """
    """

    def __init__(
        self,
        paths: Union[str, Path, List[str], List[Path]],
        chunk_size: int = 512,
        allowed_ext: set[str] | None = None,
    ):
        if isinstance(paths, (str, Path)):
            paths = [paths]

        expanded: list[Path] = []
        for p in paths:
            p = Path(p)
            if p.is_dir():
                globbed = (f for f in p.rglob("*") if f.is_file())
                if allowed_ext:
                    globbed = (f for f in globbed if f.suffix in allowed_ext)
                expanded.extend(sorted(globbed))
            elif p.is_file():
                if (not allowed_ext) or p.suffix in allowed_ext:
                    expanded.append(p)

        if not expanded:
            raise FileNotFoundError("No valid files found in the given paths.")
        self.paths: List[Path] = expanded
        self.chunk_size = chunk_size

    def stream_documents(self) -> Iterator[str]:
        for path in self.paths:
            yield from self._stream_file(path)

    get_documents = stream_documents  # alias retro-compatibilità

    def get_all_text(self) -> str:
        """Concatena l’intero corpus in RAM (uso solo per file piccoli!)."""
        return "\n\n".join(" ".join(self.stream_documents()))


    @abstractmethod
    def _stream_file(self, path: Path) -> Iterator[str]:
        ...


###############################################################################
# Implementazioni concrete
###############################################################################
class GraphRAGBenchMedical(BaseDataset):

    def __init__(self, paths, chunk_size: int = 512):
        super().__init__(paths, chunk_size, allowed_ext={".json"})

    def _stream_file(self, path: Path) -> Iterator[str]:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        context = data[0]["context"]
        yield from _chunk_stream(iter([context]), self.chunk_size)


class GraphRAGBenchCS(BaseDataset):

    def __init__(self, paths, chunk_size: int = 512):
        super().__init__(paths, chunk_size, allowed_ext={".md", ".txt"})

    def _stream_file(self, path: Path) -> Iterator[str]:
        with path.open(encoding="utf-8") as f:
            yield from _chunk_stream(f, self.chunk_size)


class DocFinQA(BaseDataset):

    def __init__(self, paths, chunk_size: int = 512):
        super().__init__(paths, chunk_size, allowed_ext={".json"})

    def _stream_file(self, path: Path) -> Iterator[str]:
        try:
            import ijson 
            with path.open("rb") as f:
                buffer: list[str] = []
                for obj in ijson.items(f, "item"):
                    if "Context" in obj:
                        buffer.extend(str(obj["Context"]).split())
                        while len(buffer) >= self.chunk_size:
                            yield " ".join(buffer[:self.chunk_size])
                            buffer = buffer[self.chunk_size:]
                if buffer:
                    yield " ".join(buffer)
        except ModuleNotFoundError:
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
            contexts = (d["Context"] for d in data if "Context" in d)
            yield from _chunk_stream(contexts, self.chunk_size)


class HousingQATSVStatutes(BaseDataset):

    def __init__(self, paths, chunk_size: int = 512, skip_header: bool = True):
        self.skip_header = skip_header
        super().__init__(paths, chunk_size, allowed_ext={".tsv"})

    def _stream_file(self, path: Path) -> Iterator[str]:
        with path.open(encoding="utf-8") as f:
            if self.skip_header:
                next(f, None)          # salta la prima riga se è l'intestazione
            yield from _chunk_stream(f, self.chunk_size)



if __name__ == "__main__":
    # --- esempio Statutes TSV ---
    graphragbench_medical = GraphRAGBenchMedical(
        DATASETS_BASE_PATH + "/graphragbench_medical/graphragbench_medical_corpus",
        chunk_size=512
    )
    for i, chunk in enumerate(graphragbench_medical.get_documents()):
        print(f"[graphragbench_medical chunk {i}] {chunk} ...")
        if i == 2:
            break

