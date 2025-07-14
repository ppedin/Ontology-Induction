"""
graph_builder.py – crea un grafo iGraph da Parquet di entità & relazioni
------------------------------------------------------------------------
• Entità  ➜  attributi:  "type", "description"
• Relazioni ➜ attributi: "keyword" (relation_keywords), "description"

Paths (default):
  entities_parquet      = outputs/graphragbench_medical/postprocessing/deduplication/entities.parquet
  relationships_parquet = outputs/graphragbench_medical/postprocessing/deduplication/relationships.parquet
  output_dir            = outputs/graphragbench_medical/postprocessing/graph_builder
  output files          = medical_graph.igraph  +  medical_graph.graphml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import igraph as ig


# --------------------------------------------------------------------------- #
# ---------------------------  CORE FUNCTION  -------------------------------- #
# --------------------------------------------------------------------------- #
def build_igraph(
    entities_pq: Path,
    relationships_pq: Path,
    output_dir: Path,
    graph_name: str = "graphragbench_medical",
) -> None:
    """Costruisce un grafo iGraph e lo persiste in formato binario + GraphML."""
    entities_pq, relationships_pq, output_dir = map(
        Path, (entities_pq, relationships_pq, output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Carica Parquet -------------------------------------------------- #
    ents = pd.read_parquet(entities_pq)
    rels = pd.read_parquet(relationships_pq)

    required_ent_cols = {"entity_name", "entity_type", "entity_description"}
    required_rel_cols = {
        "head",
        "tail",
        "relation_keywords",
        "relation_description",
    }
    if not required_ent_cols.issubset(ents.columns):
        missing = required_ent_cols - set(ents.columns)
        raise ValueError(f"Mancano colonne entità: {missing}")
    if not required_rel_cols.issubset(rels.columns):
        missing = required_rel_cols - set(rels.columns)
        raise ValueError(f"Mancano colonne relazioni: {missing}")

    # --- 2. Crea grafo vuoto ------------------------------------------------ #
    g = ig.Graph(directed=True)

    # Mappa nome ➜ indice nodo per aggiungere archi più tardi
    name_to_idx: dict[str, int] = {}

    # --- 2a. Aggiungi nodi unici ------------------------------------------- #
    for row in ents.itertuples(index=False):
        idx = g.vcount()
        g.add_vertex(name=row.entity_name)
        name_to_idx[row.entity_name] = idx
        g.vs[idx]["type"] = row.entity_type
        g.vs[idx]["description"] = row.entity_description

    # --- 2b. Aggiungi archi ------------------------------------------------- #
    edges_list: list[tuple[int, int]] = []
    edge_keywords: list[str] = []
    edge_descriptions: list[str] = []

    for row in rels.itertuples(index=False):
        # assicura che head/tail esistano come nodi
        for node_name in (row.head, row.tail):
            if node_name not in name_to_idx:
                idx = g.vcount()
                g.add_vertex(name=node_name)
                name_to_idx[node_name] = idx
                # nodi non presenti nel parquet entità → tipo/descrizione vuoti
                g.vs[idx]["type"] = ""
                g.vs[idx]["description"] = ""

        h_idx = name_to_idx[row.head]
        t_idx = name_to_idx[row.tail]

        edges_list.append((h_idx, t_idx))
        edge_keywords.append(row.relation_keywords)
        edge_descriptions.append(row.relation_description)

    # aggiunge tutti gli archi in un colpo solo (più efficiente)
    g.add_edges(edges_list)
    g.es["keyword"] = edge_keywords
    g.es["description"] = edge_descriptions

    # --- 3. Persiste il grafo ---------------------------------------------- #
    bin_path = output_dir / f"{graph_name}.igraph"
    xml_path = output_dir / f"{graph_name}.graphml"

    g.write_pickle(str(bin_path))    # <-- cast a str
    g.write_graphml(str(xml_path))   # <-- cast a str

    print(f"✅ Grafo salvato:\n  • {bin_path}\n  • {xml_path}")
    print(f"Nodi: {g.vcount():,} | Archi: {g.ecount():,}")


# --------------------------------------------------------------------------- #
# -------------------------------  CLI  ------------------------------------- #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    BASE_PATH = Path("C:/Users/paolo/Desktop/Ontology-Induction/outputs/exp_7_13/graphragbench_medical")
    entities_pq = BASE_PATH / "pragmarag_prompt" / "postprocessing" / "deduplication" / "entities.parquet"
    relationships_pq = BASE_PATH / "pragmarag_prompt" / "postprocessing" / "deduplication" / "relationships.parquet"
    output_dir = BASE_PATH / "pragmarag_prompt" / "postprocessing" / "graph_builder"

    build_igraph(entities_pq, relationships_pq, output_dir)

