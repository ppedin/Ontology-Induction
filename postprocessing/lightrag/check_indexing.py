from pathlib import Path
import pickle
import faiss
import numpy as np

index_path = Path("C:/Users/paolo/Desktop/Ontology-Induction/outputs/graphragbench_medical/postprocessing/indexing/entities.index")
meta_path = Path("C:/Users/paolo/Desktop/Ontology-Induction/outputs/graphragbench_medical/postprocessing/indexing/entity_metadata.pkl")

index = faiss.read_index(str(index_path))
with open(meta_path, "rb") as f:
    metadata = pickle.load(f)

print("n vectors: ", index.ntotal)
print("metadata records :", len(metadata))
