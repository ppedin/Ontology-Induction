from extraction.extractor_base import TripleExtractorBase
from extraction.medical_schema import TripleExtractionOutput
from pathlib import Path

MEDICAL_ENTITY_TYPES = [
        "Body Part", "Clinical Finding", "Event", "Organism", "Pharmaceutical/Biological Product", 
        "Procedure", "Specimen", "Substance", "Staging", "Physical Force", "Disease", "Symptom" 
    ] # first level concepts in the SNOMED-CT ontology

class MedicalExtractor(TripleExtractorBase):
    def __init__(self, client, model_name, output_path, max_workers, batch_size, thinking_budget, verbose):
        with open(Path("extraction/prompt_template.txt"), "r") as f:
            system_prompt_template = f.read()
        super().__init__(
            llm_client=client,
            model_name=model_name,
            system_prompt=system_prompt_template.replace("{entity_types}", ", ".join(MEDICAL_ENTITY_TYPES)),
            response_schema=TripleExtractionOutput,
            output_path=output_path,
            max_workers=max_workers,
            batch_size=batch_size,
            thinking_budget=512,
            verbose=True
        )