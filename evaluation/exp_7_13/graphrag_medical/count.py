import pandas as pd
from pathlib import Path

# Percorso del file CSV
csv_path = Path("C:/Users/paolo/Desktop/Ontology-Induction/outputs/exp_7_13/graphragbench_medical/evaluation/evaluation_static.csv")

# Carica il CSV
df = pd.read_csv(csv_path)

# Converte le colonne in float
df["lightrag_coherence"] = df["lightrag_coherence"].astype(float)
df["pragmarag_coherence"] = df["pragmarag_coherence"].astype(float)

# Calcola le somme
lrt_sum = df["lightrag_coherence"].sum()
prg_sum = df["pragmarag_coherence"].sum()

# Stampa il confronto
print(f"LightRAG  coherence sum: {lrt_sum:.2f}")
print(f"PragmaRAG coherence sum: {prg_sum:.2f}")

lrt_avg = df["lightrag_coherence"].mean()
prg_avg = df["pragmarag_coherence"].mean()

print(f"LightRAG  average coherence: {lrt_avg:.2f}")
print(f"PragmaRAG average coherence: {prg_avg:.2f}")