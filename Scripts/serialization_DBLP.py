import pandas as pd
import json
from pathlib import Path

# CONFIG
basePath = "your_base_path"
edgePath = basePath + "edges/"
nodePath = basePath + "nodes/"

node_files = {
    "articles": nodePath + "articles.csv",
    "authors": nodePath + "authors.csv",
    "keywords": nodePath + "keywords.csv",
    "venues": nodePath + "venues.csv"
}

edge_files = {
    "article_article": edgePath + "article_to_article.csv",
    "article_author": edgePath + "article_to_author.csv",
    "article_keyword": edgePath + "article_to_keyword.csv",
    "article_venue": edgePath + "article_to_venue.csv"
}

output_dir = Path("your_output_dir")
output_dir.mkdir(exist_ok=True)

# LOAD NODE DATA
nodes = {
    name: pd.read_csv(path, on_bad_lines="skip", dtype={"id": str})
    for name, path in node_files.items()
}

print("\n--- Node Columns ---")
for name, df in nodes.items():
    print(f"{name}: {df.columns.tolist()}")


# PROCESSING EDGE FILES
all_relations = []

for edge_name, edge_path in edge_files.items():
    edges = pd.read_csv(edge_path, on_bad_lines="skip", sep=",")
    print(f"\nProcessing edges for: {edge_name}")
    print(f"Columns: {edges.columns.tolist()}")

    source_type, target_type = edge_name.split("_")
    source_type = source_type + "s"
    target_type = target_type + "s"


    relations = []

    edges.columns = [c.strip() for c in edges.columns]

    for idx, edge in edges.iterrows():
        def get_edge_val(colname):
            for c in edge.index:
                if c.lower() == colname.lower():
                    return str(edge[c])
            raise KeyError(f"{colname} not found in {edge_name} columns: {edge.index.tolist()}")

        source_id_val = get_edge_val("source")
        target_id_val = get_edge_val("target")

        src_df = nodes[source_type]
        tgt_df = nodes[target_type]

        # print(src_df.head())
        # print(tgt_df.head())
        # exit(0)

        src_match = src_df.loc[src_df["id"].astype(str) == source_id_val]
        tgt_match = tgt_df.loc[tgt_df["id"].astype(str) == target_id_val]

        if src_match.empty or tgt_match.empty:
            print(f"Skipping edge {idx}: could not find node(s)")
            continue

        source_node = src_match.iloc[0].to_dict()
        target_node = tgt_match.iloc[0].to_dict()

        relation = {
            "Source node": {"Label": source_type, **source_node},
            "Edge": {
                "type": edge.get("label", "id"),
                **{k: v for k, v in edge.items() if k not in [f"{source_type}", f"{target_type}"]}
            },
            "Target node": {"Label": target_type, **target_node}
        }

        relations.append(relation)
        all_relations.append(relation)

    out_path = output_dir / f"{edge_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(relations, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(relations)} relations to {out_path}")

combined_path = output_dir / "all_relations.json"
with open(combined_path, "w", encoding="utf-8") as f:
    json.dump(all_relations, f, indent=2, ensure_ascii=False)

print(f"\nCombined file written to {combined_path}")

