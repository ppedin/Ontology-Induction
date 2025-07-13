from pathlib import Path
import pickle
import faiss
import numpy as np

index_path = Path("C:/Users/paolo/Desktop/Ontology-Induction/outputs/graphragbench_medical/postprocessing/indexing/relationships.index")
meta_path = Path("C:/Users/paolo/Desktop/Ontology-Induction/outputs/graphragbench_medical/postprocessing/indexing/relationship_metadata.pkl")

index = faiss.read_index(str(index_path))
with open(meta_path, "rb") as f:
    metadata = pickle.load(f)

for i in range(len(metadata)):
    print(metadata[i])
    break