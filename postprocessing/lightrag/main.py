from pathlib import Path
from graph_formatter import format_graph

if __name__ == "__main__":
    format_graph(
        input_dir=Path("C:/Users/paolo/Desktop/Ontology-Induction/outputs/graphragbench_medical/"
                        "extraction/parts"),
        output_dir=Path("C:/Users/paolo/Desktop/Ontology-Induction/outputs/graphragbench_medical/postprocessing/"
                        "graph_formatter")
        )