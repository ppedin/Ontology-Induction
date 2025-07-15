import pandas as pd

eval_results_path = "C:/Users/paolo/Desktop/Ontology-Induction/outputs/exp_7_13/graphragbench_medical/evaluation/evaluation_refined.csv"

df = pd.read_csv(eval_results_path)


print(df["lightrag_information_coverage"].mean())
print(df["pragmarag_information_coverage"].mean())


