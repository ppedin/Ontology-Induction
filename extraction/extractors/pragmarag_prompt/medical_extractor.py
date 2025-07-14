from pathlib import Path
from pydantic import BaseModel
from typing import List, Literal

from extraction.extractor_base import TripleExtractorBase
import json

class MedicalEntity(BaseModel):
    entity_name: str
    entity_type: str
    entity_description: str


class Relationship(BaseModel):
    head: str
    tail: str
    relation_description: str
    relation_keywords: str


class TripleExtractionOutput(BaseModel):
    entities: List[MedicalEntity]
    relationships: List[Relationship]

def format_graph_for_prompt(graph: dict) -> str:
    """
    Format a single abstract graph dictionary into a readable string
    for use as the user prompt to an LLM.

    The graph must contain:
    - "entity_types": list of {"name": ..., "description": ...}
    - "relationship_types": list of {"name": ..., "description": ...}
    """
    entities = graph.get("entity_types", [])
    relations = graph.get("relationship_types", [])

    entity_block = "\n".join(
        f"- {e['name']}: {e['description']}" for e in entities
    )
    relation_block = "\n".join(
        f"- {r['name']}: {r['description']}" for r in relations
    )

    return f"""Entity Types:
{entity_block}

Relation Types:
{relation_block}"""

GRAPH_PATH = "C:/Users/paolo/Desktop/Ontology-Induction/outputs/exp_7_13/graphragbench_medical/pragmarag_prompt/extraction/abstract_questions_subgraphs_abstraction3.jsonl"


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
            Path("C:/Users/paolo/Desktop/Ontology-Induction/extraction/extractors/pragmarag_prompt/pragmarag_prompt_prompt_template.txt")
        )
        system_prompt_template = prompt_path.read_text(encoding="utf-8")
        with open(GRAPH_PATH, "r") as f:
            graph_schema = json.load(f)
            formatted_graph_schema = format_graph_for_prompt(graph_schema)
        #  entity_types = ", ".join(get_args(MedicalEntity.__annotations__["entity_type"]))
        self.system_prompt = system_prompt_template.replace("{SCHEMA}", formatted_graph_schema)
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
        return f"""

        Text:
        \"\"\"{chunk}\"\"\"

        """
    

if __name__ == "__main__":
    with open(GRAPH_PATH, "r") as f:
        graph = json.load(f)
        print(format_graph_for_prompt(graph))