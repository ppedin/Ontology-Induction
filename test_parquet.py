import pyarrow.dataset as ds
from pathlib import Path
import json
import pandas as pd
BASE_PATH = Path("C:/Users/paolo/Desktop/Ontology-Induction/outputs/exp_7_13/graphragbench_medical")
dataset_path = BASE_PATH / "pragmarag_prompt" / "extraction" / "parts" / "part-0bcac479f0c4-1752446098105862.parquet"

dataset = ds.dataset(dataset_path, format="parquet")

df = dataset.to_table().to_pandas()

for index, row in df.iterrows():
    print(row["payload_json"])
    break






