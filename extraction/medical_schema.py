from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class MedicalEntity(BaseModel):
    entity_name: str
    entity_type: Literal[
        "Body Part", "Clinical Finding", "Event", "Organism", "Pharmaceutical/Biological Product", 
        "Procedure", "Specimen", "Substance", "Staging", "Physical Force", "Disease", "Symptom" 
    ]   #  first level concepts in the SNOMED-CT ontology
    entity_description: str

class Relationship(BaseModel):
    head: str
    tail: str
    relation: str

class TripleExtractionOutput(BaseModel):
    entities: List[MedicalEntity]
    relationships: List[Relationship]
