import json
import os
import time
import torch
import random
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from collections import defaultdict
from HalluFunctions import HalluFunctions
from stratified_sampling import Sampling

hallu = HalluFunctions()
sampler = Sampling()

device = "cuda" if torch.cuda.is_available() else "cpu"

basePath = "your_base_path"
datasets = {"ldbc": "all_relations_shuffled.json",
            "knows": "graph_edges.json"}

chunk_size = 50
num_chunks = 200

def load_model_and_tokenizer(model_name, model_info):

    print(f"\n Loading model: {model_name}")

    model_type = model_info["type"]
    if model_type != "transformer":
        raise ValueError(f"Unsupported model type: {model_type}")

    model_path = model_info["path"]
    quantized_4bit = model_info.get("quantized-4bit", False)
    quantized_8bit = model_info.get("quantized-8bit", False)

    if quantized_8bit:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant_config,
            device_map="auto"
        )

    elif quantized_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant_config,
            device_map="cuda:0",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


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


models = {
    "mist-7b": {
        "type": "transformer",
        "path": "model_path",
        "quantized-4bit": False,
        "quantized-8bit": False
    },
    "qwen-7b": {
        "type": "transformer",
        "path": "model_path",
        "quantized-4bit": False,
        "quantized-8bit": False
    },
    # "gpt": {
    #     "type": "openai",
    #     "model": "gpt-5"
    # },

    # "llama-8b": {
    #     "type": "transformer",
    #     "id": model_id,
    #     "path": model_path,
    #     "quantized-4bit": False,
    #     "quantized-8bit": False
    # },
}

RELOAD_EVERY_N_CHUNKS = 3

for key, fname in datasets.items():
    in_file = basePath + fname
    with open(in_file, "r", encoding="utf-8") as f:
        full_dataset = json.load(f)
        # full_dataset = full_dataset[:200]

    print(f"Loaded {len(full_dataset)} graph triples from {key} dataset")

    node_schema, edge_schema = build_property_graph_schema(full_dataset)
    schema_text = schema_to_prompt(node_schema, edge_schema)
    print("Schema Extraction: Done \n")

    random.shuffle(full_dataset)
    print("Data Shuffling: Done \n")

    chunks, records_used = sampler.build_stratified_bins_from_file(
        input_path=in_file,
        num_chunks=num_chunks,
        chunk_size=chunk_size,
        seed=42,
        allow_replacement=False
    )

    for model_name, model_info in models.items():
        runs = [1, 2, 3]
        print(f"\n=== Starting model: {model_name} ===")

        for i in runs:
            master_summary_file = "your_path" + f"{model_name}_Master_Summary_Input.txt"
            master_summary_output_file = basePath + f"{model_name}_Master_Summary_Output_{key}_PR_{i}.txt"
            llm_output_file = basePath + f"{model_name}_LLM_Outputs_{key}_PR_{i}.txt"

            prog_fp = basePath + f".progress_{model_name}_{key}_{i}.txt"

            if os.path.exists(prog_fp):
                with open(prog_fp, "r", encoding="utf-8") as f:
                    start_chunk = int(f.read().strip()) + 1
            else:
                start_chunk = 0

            model = None
            tokenizer = None
            model_type = model_info["type"]

            for chunk_i in range(start_chunk, len(chunks)):
                ch = chunks[chunk_i]

                if chunk_i == start_chunk or (chunk_i % RELOAD_EVERY_N_CHUNKS == 0):
                    if model is not None:
                        print(f"\n Reinitializing {model_name} at chunk {chunk_i}")
                        del model
                        del tokenizer
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        time.sleep(1)

                    model, tokenizer = load_model_and_tokenizer(model_name, model_info)

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

                if model_type == "openai":
                    print("OPENAI")
                    # response = client.chat.completions.create(
                    #     model=model_info["model"],
                    #     messages=[
                    #         {"role": "system",
                    #          "content": "You are an expert in dataset summary generation from property graphs."},
                    #         {"role": "user", "content": prompt}
                    #     ]
                    # )
                    # result = response.choices[0].message.content.strip()

                else:
                    messages = [
                        {"role": "system",
                         "content": "You are an expert in dataset summary generation from property graphs."},
                        {"role": "user", "content": prompt}
                    ]
                    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    # if model == "nemotron-30b":
                    #     text = tokenizer.apply_chat_template(
                    #         messages,
                    #         tokenize=True,
                    #         enable_thinking=False,
                    #         add_generation_prompt=True,
                    #         return_tensors="pt"
                    #     ).to(model.device)
                    inputs = tokenizer([text], return_tensors="pt").to(model.device)

                    with torch.no_grad():
                        # if model == "nemotron-30b":
                        #     outputs = model.generate(
                        #         **inputs,
                        #         max_new_tokens=2048,
                        #         temperature=1,
                        #         top_p=1,
                        #         do_sample=True,
                        #         pad_token_id=tokenizer.eos_token_id
                        #     )

                        output = model.generate(**inputs, max_new_tokens=2048)

                    new_ids = output[0][len(inputs.input_ids[0]):]
                    result = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

                with open(llm_output_file, "a") as f:
                    f.write(f"\n=== {model_name} | Batch {chunk_i + 1} ===\n{result}\n")

                with open(master_summary_output_file, "w") as f:
                    f.write(result)

                with open(prog_fp, "w", encoding="utf-8") as f:
                    f.write(str(chunk_i))

                print(f"✓ Batch {chunk_i + 1} complete")
                time.sleep(1)

            if os.path.exists(prog_fp):
                os.remove(prog_fp)

            print(f" Completed dataset '{key}' for run {i}. Processed {len(chunks)} chunks across all models.")

