import pyarrow.dataset as ds
from pathlib import Path
import json
import pandas as pd
dataset_path = Path("C:/Users/paolo/Desktop/Ontology-Induction/outputs/extraction/graphragbench_medical/graphragbench_medical_extraction/parts")

dataset = ds.dataset(dataset_path, format="parquet")
df = dataset.to_table().to_pandas()
df["parsed"] = df["payload_json"].map(json.loads)

for index, row in df.iterrows():
    print(row["parsed"]["entities"])
    break