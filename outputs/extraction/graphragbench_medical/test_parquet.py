import pandas as pd

# Carica il Parquet
df = pd.read_parquet("C:/Users/paolo/Desktop/Ontology-Induction/outputs/extraction/graphragbench_medical/graphragbench_medical_extraction.parquet")

# Mostra lo schema (colonne e tipi)
print(df.dtypes)

# Mostra le prime 5 righe
print(df.head())
