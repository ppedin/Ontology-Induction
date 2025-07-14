import pyarrow.dataset as ds
from pathlib import Path
import json
import pandas as pd
BASE_PATH = Path("C:/Users/paolo/Desktop/Ontology-Induction/outputs/exp_7_13/graphragbench_medical")
dataset_path = BASE_PATH / "lightrag" / "postprocessing" / "embedding" / "relationships.parquet"

dataset = ds.dataset(dataset_path, format="parquet")
df = dataset.to_table().to_pandas()
#  df["parsed"] = df["payload_json"].map(json.loads)

for index, row in df.iterrows():
    print(row)
    break