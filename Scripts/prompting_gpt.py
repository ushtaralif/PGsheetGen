"""
This code is for generating summaries for LDBC and Knows dataset iteratively multiple times using LLAMA, Mistral and GPT5
"""

import json
import os
import time
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from collections import defaultdict
from openai import OpenAI
# from HalluFunctions import HalluFunctions
from stratified_sampling import Sampling

# hallu = HalluFunctions()
sampler = Sampling()


device = "cuda" if torch.cuda.is_available() else "cpu"

parent_dir = os.path.dirname(os.path.abspath(__file__))
key_file = "your_gpt_key"
with open(key_file, 'r', encoding='utf-8') as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)

chunk_size = 50

basePath = "base_path"
datasets = {"ldbc": "your_dataset_file.json",
            "knows": "your_dataset_file.json"}

def build_property_graph_schema(data):
    node_schema = defaultdict(set)
    edge_schema = defaultdict(set)

    for item in data:
        src = item.get("Source node", {})
        src_label = src.get("Label", "Unknown")
        for k in src.keys():
            node_schema[src_label].add(k)

        tgt = item.get("Target node", {})
        tgt_label = tgt.get("Label", "Unknown")
        for k in tgt.keys():
            node_schema[tgt_label].add(k)

        edge = item.get("Edge", {})
        edge_type = edge.get("type", "Unknown")
        for k in edge.keys():
            edge_schema[edge_type].add(k)

    return node_schema, edge_schema


def schema_to_prompt(node_schema, edge_schema):
    txt = "PROPERTY GRAPH SCHEMA (FOLLOW STRICTLY)\n\n"

    txt += "=== NODE TYPES ===\n"
    for label, fields in node_schema.items():
        txt += f"\nNODE TYPE: {label}\n"
        for f in sorted(fields):
            txt += f"  • {f}\n"

    txt += "\n=== EDGE TYPES ===\n"
    for rel, fields in edge_schema.items():
        txt += f"\nEDGE TYPE: {rel}\n"
        for f in sorted(fields):
            txt += f"  • {f}\n"

    txt += (
        "\nRULES:\n"
        "• Each entry is a triple: Source Node → Edge → Target Node\n"
        "• Never invent new fields\n"
        "• Missing fields mean NULL\n"
        "• Follow schema strictly\n"
    )

    return txt


def progress_file_for_model(base_dir: str, model_name: str) -> str:
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in model_name)
    return os.path.join(base_dir, f".chunk_progress_{safe}.json")


def load_progress(prog_fp: str):
    if not os.path.exists(prog_fp):
        return None
    try:
        with open(prog_fp, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_progress(prog_fp: str, dataset_key: str, run_i: int, chunk_i: int):
    with open(prog_fp, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": dataset_key,
                "run": run_i,
                "chunk": chunk_i
            },
            f
        )


models = {

    "gpt": {
        "type": "openai",
        "model": "gpt-5"
    },
}

for i in range(1, 4):

    for key, fname in datasets.items():
        in_file = basePath + fname

        with open(in_file, "r", encoding="utf-8") as f:
            full_dataset = json.load(f)
        # full_dataset = full_dataset[:200]

        print(f"Loaded {(key)} dataset graph triples. for {i} time \n")

        node_schema, edge_schema = build_property_graph_schema(full_dataset)
        schema_text = schema_to_prompt(node_schema, edge_schema)
        print("Schema Extraction: Done \n")

        random.shuffle(full_dataset)
        print("Data Shuffling: Done \n")

        chunks, records_used = sampler.build_stratified_bins_from_file(
            input_path=in_file,
            num_chunks=200,
            chunk_size=50,
            seed=42,
            allow_replacement=False
        )
        # sampler.print_stats(records_used, chunks, top_k=30)

        for model_name, model_info in models.items():
            print(f"\n=== Starting model: {model_name} ===")

            prog_fp = progress_file_for_model(basePath, model_name)
            progress = load_progress(prog_fp)

            if progress is not None:
                saved_dataset = progress.get("dataset")
                saved_run = progress.get("run")
                saved_chunk = progress.get("chunk", -1)

                if i < saved_run:
                    continue
                if i == saved_run and key != saved_dataset:
                    continue

                if key == saved_dataset and i == saved_run:
                    start_chunk_i = saved_chunk + 1
                else:
                    start_chunk_i = 0
            else:
                start_chunk_i = 0

            if start_chunk_i >= len(chunks):
                if os.path.exists(prog_fp):
                    os.remove(prog_fp)
                print(f" {model_name}: already completed. Skipping.")
                continue

            master_summary_file = basePath + f"Data/input/{model_name}_Master_Summary_Input.txt"
            master_summary_output_file = basePath + f"Data/input/{model_name}_Master_Summary_Output_{key}_{i}.txt"
            llm_output_file = basePath + f"Data/input/{model_name}_LLM_Outputs_{key}_{i}.txt"

            model_type = model_info["type"]

            for chunk_i in range(start_chunk_i, len(chunks)):
                ch = chunks[chunk_i]

                chunk_text = json.dumps(ch, ensure_ascii=False, indent=2)
                print(f"=== Chunk {chunk_i} ===")

                if chunk_i == 0:
                    prev_file = master_summary_file
                else:
                    prev_file = master_summary_output_file

                if os.path.exists(prev_file):
                    with open(prev_file, "r", encoding="utf-8") as f:
                        previous_summary = f.read().strip()
                else:
                    previous_summary = "None"
                    print("Exiting because previous %s is empty" % prev_file)
                    exit(0)

                gpt_prompt = f"""
                    You are given:
                    • A Previous PG-Sheet Summary (keep structure EXACTLY)
                    • A New Dataset Chunk 
                    • A Schema Representing the Dataset

                    INSTRUCTIONS:
                        Update the PG-Sheet Summary (ONLY the answer fields) according to New Dataset Chunk (follow schema strictly).
                        Do not change headings, bullets, spacing, or order.

                    ### SCHEMA (STRICT):
                    {schema_text}

                    ### Previous Summary:
                    {previous_summary}

                    ### New Dataset Chunk:
                    {chunk_text}

                    Return ONLY the updated PG-Sheet Summary.
                    """

                print("SUMMARY \n*2")
                print(previous_summary)
                print("\n*2")

                if model_type == "openai":
                    response = client.chat.completions.create(
                        model=model_info["model"],
                        messages=[
                            {"role": "system", "content": "You are an expert in dataset summary generation from property graphs"},
                            {"role": "user", "content": gpt_prompt}
                        ]
                    )
                    result = response.choices[0].message.content.strip()


                with open(llm_output_file, "a") as f:
                    f.write(f"\n=== {model_name} | Batch {chunk_i + 1} ===\n{result}\n")

                with open(master_summary_output_file, "w") as f:
                    f.write(result)

                print(f"Batch {chunk_i + 1} complete")
                save_progress(prog_fp, key, i, chunk_i)
                time.sleep(1)

            if os.path.exists(prog_fp):
                os.remove(prog_fp)

        print(f" Dataset {key} completed for run {i} (processed {len(chunks)} chunks).")