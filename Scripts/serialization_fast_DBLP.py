import pandas as pd
import json
from pathlib import Path

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

output_dir = Path("processed_fast")
output_dir.mkdir(exist_ok=True)

nodes = {}
for name, path in node_files.items():
    df = pd.read_csv(path, dtype={"id": "string"}, on_bad_lines="skip")
    df.columns = [c.strip() for c in df.columns]
    if "id" not in df.columns:
        raise ValueError(f"{name}: missing 'id' column")
    nodes[name] = df.set_index("id", drop=False)

for edge_name, edge_path in edge_files.items():
    print(f"\nProcessing: {edge_name}")
    source_type, target_type = edge_name.split("_")
    source_type += "s"
    target_type += "s"

    edges = pd.read_csv(edge_path, on_bad_lines="skip", dtype={"source": "string", "target": "string"})
    edges.columns = [c.strip() for c in edges.columns]

    colmap = {c.lower(): c for c in edges.columns}
    if "source" not in colmap or "target" not in colmap:
        raise ValueError(f"{edge_name}: need source/target columns, got {edges.columns.tolist()}")

    src_col = colmap["source"]
    tgt_col = colmap["target"]

    src_df = nodes[source_type]
    tgt_df = nodes[target_type]

    merged = edges.merge(
        src_df.add_prefix("src__"),
        how="left",
        left_on=src_col,
        right_on="src__id"
    )

    merged = merged.merge(
        tgt_df.add_prefix("tgt__"),
        how="left",
        left_on=tgt_col,
        right_on="tgt__id"
    )

    before = len(merged)
    merged = merged[merged["src__id"].notna() & merged["tgt__id"].notna()].copy()
    print(f"Kept {len(merged)}/{before} edges after node join")

    out_path = output_dir / f"{edge_name}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        edge_cols = [c for c in edges.columns]

        for row in merged.itertuples(index=False):
            rowd = row._asdict()

            src_node = {k.replace("src__", ""): v for k, v in rowd.items() if k.startswith("src__")}
            tgt_node = {k.replace("tgt__", ""): v for k, v in rowd.items() if k.startswith("tgt__")}

            edge_obj = {k: rowd[k] for k in edge_cols if k in rowd}
            rel = {
                "Source node": {"Label": source_type, **src_node},
                "Edge": {"type": edge_obj.get("label", "id"), **edge_obj},
                "Target node": {"Label": target_type, **tgt_node},
            }
            f.write(json.dumps(rel, ensure_ascii=False) + "\n")

    print(f"Saved: {out_path}")