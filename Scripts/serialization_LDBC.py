import pandas as pd
import json
from pathlib import Path

# CONFIG
basePath = "your_base_path"
edgePath = basePath + "Edges/"
nodePath = basePath + "Nodes/"

node_files = {
    "Person": nodePath + "Person.csv",
    "Forum": nodePath + "Forum.csv",
    "Comment": nodePath + "Comment.csv",
    "Company": nodePath + "Organisation.csv",
    "Place": nodePath + "Place.csv",
    "Post": nodePath + "post.csv"
}

edge_files = {
    "Forum_Person": edgePath + "F_to_p.csv",
    "Company_Place": edgePath + "Organisation_isLocatedIn_Place.csv",
    "Person_Comment": edgePath + "p_like_comment.csv",
    "Person_Post": edgePath + "P_like_post.csv",
    "Person_Person": edgePath + "Person_knows_Person.csv",
    "Person_Company": edgePath + "Person_workAt_Company.csv",
    "Place_Place": edgePath + "Place_isPartOf_Place.csv",
}

output_dir = Path("../output_json")
output_dir.mkdir(exist_ok=True)

# LOAD NODE DATA
nodes = {
    name: pd.read_csv(path, on_bad_lines="skip", dtype={"id": str})
    for name, path in node_files.items()
}

print("--- Node Columns --- \n")
for name, df in nodes.items():
    print(f"{name}: {df.columns.tolist()}")

# PROCESS EDGE FILES
all_relations = []

for edge_name, edge_path in edge_files.items():
    edges = pd.read_csv(edge_path, on_bad_lines="skip", sep=",")

    print(f"\nProcessing edges for: {edge_name}")
    print(f"Columns: {edges.columns.tolist()}")

    source_type, target_type = edge_name.split("_")
         # if source_type == "Place":
         #    source_type = source_type + "1"


    relations = []

    # Normalize column names (case-insensitive)
    edges.columns = [c.strip() for c in edges.columns]

    for idx, edge in edges.iterrows():
        # Extract correct edge ID values (case-insensitive)
        def get_edge_val(colname):
            for c in edge.index:
                if c.lower() == colname.lower():
                    return str(edge[c])
            raise KeyError(f"{colname} not found in {edge_name} columns: {edge.index.tolist()}")

        if source_type == target_type:
            source_id_val = get_edge_val(f"{source_type}1Id")
            target_id_val = get_edge_val(f"{target_type}2Id")
        else:
            source_id_val = get_edge_val(f"{source_type}Id")
            target_id_val = get_edge_val(f"{target_type}Id")

        # Find matching nodes (convert to string for safety)
        src_df = nodes[source_type]
        tgt_df = nodes[target_type]

        src_match = src_df.loc[src_df["id"].astype(str) == source_id_val]
        tgt_match = tgt_df.loc[tgt_df["id"].astype(str) == target_id_val]

        if src_match.empty or tgt_match.empty:
            print(f" Skipping edge {idx}: could not find node(s)")
            continue

        source_node = src_match.iloc[0].to_dict()
        target_node = tgt_match.iloc[0].to_dict()

        relation = {
            "Relation": edge_name,
            "Source node": {"Label": source_type, **source_node},
            "Edge": {
                "type": edge.get("type", "DATE_CREATED"),
                **{k: v for k, v in edge.items() if k not in [f"{source_type}Id", f"{target_type}Id"]}
            },
            "Target node": {"Label": target_type, **target_node}
        }

        relations.append(relation)
        all_relations.append(relation)

    # Write per-edge JSON
    out_path = output_dir / f"{edge_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(relations, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(relations)} relations to {out_path}")

# WRITE COMBINED FILE
combined_path = output_dir / "all_relations.json"
with open(combined_path, "w", encoding="utf-8") as f:
    json.dump(all_relations, f, indent=2, ensure_ascii=False)

print(f"\nCombined file written to {combined_path}")
