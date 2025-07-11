from pathlib import Path
from pydantic import BaseModel
from typing import List, Literal

from extraction.extractor_base import TripleExtractorBase


class MedicalEntity(BaseModel):
    entity_name: str
    entity_type: Literal[
        "Body Part", "Clinical Finding", "Event", "Organism",
        "Pharmaceutical/Biological Product", "Procedure", "Specimen",
        "Substance", "Staging", "Physical Force", "Disease", "Symptom",
    ]
    entity_description: str


class Relationship(BaseModel):
    head: str
    tail: str
    relation: str


class TripleExtractionOutput(BaseModel):
    entities: List[MedicalEntity]
    relationships: List[Relationship]


class MedicalExtractor(TripleExtractorBase[TripleExtractionOutput]):
    response_schema = TripleExtractionOutput  # tells the base class what to expect

    def __init__(
        self,
        client,
        model_name: str,
        output_path: Path,
        max_workers: int = 6,
        batch_size: int = 3,
        thinking_budget: int = 512,
        verbose: bool = False,
    ):
        from typing import get_args
        prompt_path = (
            Path("C:/Users/paolo/Desktop/Ontology-Induction/extraction/prompt_template.txt")
        )
        system_prompt_template = prompt_path.read_text(encoding="utf-8")
        entity_types = ", ".join(get_args(MedicalEntity.__annotations__["entity_type"]))
        self.system_prompt = system_prompt_template.replace("{entity_types}", entity_types)
        super().__init__(
            llm_client=client,
            model_name=model_name,
            output_path=output_path,
            max_workers=max_workers,
            batch_size=batch_size,
            thinking_budget=thinking_budget,
            verbose=verbose,
        )

    # ---- domain‑specific prompt -------------------------------------------
    def build_user_prompt(self, chunk: str) -> str:
        return f"""You are an expert medical ontology curator.

Text:
\"\"\"{chunk}\"\"\"

Extract all entities and relations present in the text.  
Answer *only* with a JSON matching the provided schema.
"""