from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class Entity(BaseModel):
    entity_name: str
    entity_type: Literal[
        "Action", "Event", "Intangible", "Organization", "Person", "Place", "Product"
    ]  #  subtypes of Thing in Schema.org
    entity_description: str

class Relationship(BaseModel):
    head: str
    tail: str
    relation: str

class TripleExtractionOutput(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]
