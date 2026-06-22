import csv
import json
from pathlib import Path

NODES_CSV = "nodes.csv"
EDGES_CSV = "edge.csv"
OUT_JSON = "graph_edges.json"

NODE_LABEL = "Person"
EDGE_TYPE = "knows"

def read_nodes(nodes_path: str) -> dict:
    nodes = {}
    with open(nodes_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            nid = int(row["node_id"])
            nodes[nid] = {
                "Label": NODE_LABEL,
                "id": str(nid),
                "firstName": row.get("fname", ""),
                "lastName": row.get("lname", "")

            }
    return nodes

def build_records(edges_path: str, nodes_by_id: dict) -> list:
    out = []
    missing = []

    with open(edges_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            sid = int(row["source_id"])
            tid = int(row["target_id"])
            createdate = row.get("createdate", "")

            src = nodes_by_id.get(sid)
            tgt = nodes_by_id.get(tid)

            if src is None or tgt is None:
                missing.append({
                    "edg_id": row.get("edg_id"),
                    "missing_source": src is None,
                    "missing_target": tgt is None,
                    "source_id": sid,
                    "target_id": tid,
                })
                continue

            out.append({
                "Source node": src,
                "Edge": {
                    "type": EDGE_TYPE,
                    "creationDate": createdate
                },
                "Target node": tgt
            })

    if missing:
        print("skipping edges because node id(s) were not found in nodes.csv:")
        for m in missing:
            print(m)

    return out

def main():
    nodes_by_id = read_nodes(NODES_CSV)
    records = build_records(EDGES_CSV, nodes_by_id)

    Path(OUT_JSON).write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"Wrote {OUT_JSON} with {len(records)} edge-records.")

if __name__ == "__main__":
    main()