
import json
import os
import time
import gc
import random
import torch
from collections import defaultdict, deque
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from openai import OpenAI
# from HalluFunctions import HalluFunctions
from stratified_sampling import Sampling
# hallu = HalluFunctions()
sampler = Sampling()

device = "cuda" if torch.cuda.is_available() else "cpu"

basePath = "your_base_path"
datasets = {
    "ldbc": "all_relations_shuffled.json",
    "knows": "graph_edges.json",
}

chunk_size = 50
num_chunks = 200
reload_every = 4
RUNS = [1,2,3]

parent_dir = os.path.dirname(os.path.abspath(__file__))

models = { 
        "mist-7b": {
            "path": "model_path",
            "type": "transformer",
            "quantized_4bit": False,
            "quantized_8bit": False,
        },
        "llama-8b": {
            "path": "model_path",
            "type": "transformer",
            "quantized_4bit": False,
            "quantized_8bit": False,
        },
        "qwen-7b": {
            "path": "model_path",
            "type": "transformer",
            "quantized_4bit": False,
            "quantized_8bit": False,
        },
    }

def build_property_graph_schema(data):
    node_schema = defaultdict(set)
    edge_schema = defaultdict(set)

    for item in data:
        src = item.get("Source node", {}) or {}
        src_label = src.get("Label", "Unknown")
        for k in src.keys():
            node_schema[src_label].add(k)

        tgt = item.get("Target node", {}) or {}
        tgt_label = tgt.get("Label", "Unknown")
        for k in tgt.keys():
            node_schema[tgt_label].add(k)

        edge = item.get("Edge", {}) or {}
        edge_type = edge.get("type", "Unknown")
        for k in edge.keys():
            edge_schema[edge_type].add(k)

    return node_schema, edge_schema


def schema_to_prompt(node_schema, edge_schema) -> str:
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

def load_model_and_tokenizer(model_name: str, model_info: dict):
    print(f"\n Loading model: {model_name}")

    if model_info["type"] != "transformer":
        raise ValueError(f"Unsupported model type: {model_info['type']}")

    model_path = model_info["path"]
    quant_4bit = model_info.get("quantized-4bit", False)
    quant_8bit = model_info.get("quantized-8bit", False)

    if quant_8bit:
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant_cfg,
            device_map="auto",
            trust_remote_code=True,
        )
    elif quant_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant_cfg,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def unload_model(model, tokenizer):
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(" Model unloaded and CUDA cache cleared.")


def run_inference(
    model_name: str,
    model_info: dict,
    model,
    tokenizer,
    schema_text: str,
    previous_summary: str,
    chunk_text: str,
) -> str:
    prompt = f"""
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

    system_msg = "You are an expert in dataset summary generation from property graphs."

    if model_info["type"] == "openai":
        response = client.chat.completions.create(
            model=model_info["model"],
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
    input_device = model.get_input_embeddings().weight.device
    inputs = {k: v.to(input_device) for k, v in inputs.items()}

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = out_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()

def hierarchical_summarization(
    items: list,
    model_name: str,
    model_info: dict,
    schema_text: str,
    seed_summary_file: str,
    out_dir: str,
    run_id: int,
    reload_every: int,
    sleep: float,
    llm_output_file: str,
) -> str:

    level = 0
    current_items = items  

    while len(current_items) > 1:
        level += 1
        num_groups = (len(current_items) + reload_every - 1) // reload_every
        print(f"\n=== {model_name} | Run {run_id} | LEVEL {level} | "
              f"items={len(current_items)} | groups={num_groups} ===")

        new_items = []

        for g in range(num_groups):
            group = current_items[g * reload_every: (g + 1) * reload_every]
            print(f"[{model_name}] Run {run_id} | Level {level} | "
                  f"Group {g + 1}/{num_groups} | size={len(group)}")

            model, tokenizer = load_model_and_tokenizer(model_name, model_info)

            prev_summary_file = seed_summary_file
            result = None

            for step_i, item in enumerate(group):
                chunk_text = (
                    item if isinstance(item, str)
                    else json.dumps(item, ensure_ascii=False, indent=2)
                )

                if os.path.exists(prev_summary_file):
                    with open(prev_summary_file, "r", encoding="utf-8", errors="replace") as f:
                        previous_summary = f.read().strip()
                else:
                    previous_summary = "None"

                result = run_inference(
                    model_name=model_name,
                    model_info=model_info,
                    model=model,
                    tokenizer=tokenizer,
                    schema_text=schema_text,
                    previous_summary=previous_summary,
                    chunk_text=chunk_text,
                )

                with open(llm_output_file, "a", encoding="utf-8") as f:
                    f.write(
                        f"\n=== {model_name} | Run {run_id} | "
                        f"Level {level} | Group {g + 1} | Step {step_i + 1} ===\n"
                    )
                    f.write(result)
                    f.write("\n")

                tmp_path = os.path.join(
                    out_dir,
                    f"{model_name}_tmp_run{run_id}_L{level}_G{g + 1}_S{step_i + 1}.txt",
                )
                with open(tmp_path, "w", encoding="utf-8") as f:
                    f.write(result)

                prev_summary_file = tmp_path
                time.sleep(sleep)

            group_out = os.path.join(
                out_dir,
                f"{model_name}_level{level}_group{g + 1}_run{run_id}.txt",
            )
            with open(group_out, "w", encoding="utf-8") as f:
                f.write(result if result is not None else "")

            new_items.append(result if result is not None else "")
            unload_model(model, tokenizer)

        current_items = new_items

    return current_items[0] if current_items else ""


def main():
    for dataset_key, fname in datasets.items():

        dataset_out_dir = os.path.join(basePath, f"outputs_{dataset_key}")
        os.makedirs(dataset_out_dir, exist_ok=True)
        print(f"\n{'=' * 60}")
        print(f"DATASET: {dataset_key}  →  output dir: {dataset_out_dir}")
        print(f"{'=' * 60}")

        in_file = os.path.join(basePath, fname)
        with open(in_file, "r", encoding="utf-8") as f:
            full_dataset = json.load(f)
        print(f"Loaded {len(full_dataset):,} graph triples from '{dataset_key}'")

        node_schema, edge_schema = build_property_graph_schema(full_dataset)
        schema_text = schema_to_prompt(node_schema, edge_schema)
        print("Schema Extraction: Done\n")

        random.shuffle(full_dataset)
        print("Data Shuffling: Done\n")

        chunks, records_used = sampler.build_stratified_bins_from_file(
            input_path=in_file,
            num_chunks=num_chunks,
            chunk_size=chunk_size,
            seed=42,
            allow_replacement=False,
        )
        print(f"Chunks built: {len(chunks)}  |  records used: {records_used}\n")

        for model_name, model_info in models.items():
            print(f"\n=== Starting model: {model_name} ===")

            for run_id in RUNS:
                print(f"\n--- {model_name} | Dataset: {dataset_key} | Run {run_id} ---")

                seed_summary_file = os.path.join(
                    dataset_out_dir, f"{model_name}_Master_Summary_Input.txt"
                )
                if not os.path.exists(seed_summary_file):
                    fallback = os.path.join(basePath + "/input/", f"{model_name}_Master_Summary_Input.txt")
                    if os.path.exists(fallback):
                        seed_summary_file = fallback
                    else:
                        raise FileNotFoundError(
                            f"Seed summary template not found.\n"
                            f"  Tried: {seed_summary_file}\n"
                            f"  Tried: {fallback}"
                        )
                    
                master_out_file = os.path.join(
                    dataset_out_dir,
                    f"{model_name}_Master_Summary_Output_{dataset_key}_{run_id}.txt",
                )
                llm_out_file = os.path.join(
                    dataset_out_dir,
                    f"{model_name}_LLM_Outputs_{dataset_key}_{run_id}.txt",
                )

                with open(llm_out_file, "w", encoding="utf-8") as f:
                    f.write(f"=== {model_name} | {dataset_key} | run {run_id} ===\n")

                final_summary = hierarchical_summarization(
                    items=chunks,
                    model_name=model_name,
                    model_info=model_info,
                    schema_text=schema_text,
                    seed_summary_file=seed_summary_file,
                    out_dir=dataset_out_dir,
                    run_id=run_id,
                    reload_every=reload_every,
                    sleep=1.0,
                    llm_output_file=llm_out_file,
                )

                with open(master_out_file, "w", encoding="utf-8") as f:
                    f.write(final_summary)

                final_path = os.path.join(
                    dataset_out_dir,
                    f"{model_name}_FINAL_{dataset_key}_run{run_id}.txt",
                )
                with open(final_path, "w", encoding="utf-8") as f:
                    f.write(final_summary)

                print(f"  {model_name} | {dataset_key} | Run {run_id} DONE")
                print(f"   Final summary → {final_path}")


if __name__ == "__main__":
    main()