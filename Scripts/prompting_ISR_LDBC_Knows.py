"""
This code is for generating PG-Sheet content under ISR for LDBC and Knows dataset iteratively multiple times using LLAMA, Mistral and GPT5
"""

import json
import os
import time
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from collections import defaultdict
from HalluFunctions import HalluFunctions
from stratified_sampling import Sampling

hallu = HalluFunctions()
sampler = Sampling()


device = "cuda" if torch.cuda.is_available() else "cpu"

parent_dir = os.path.dirname(os.path.abspath(__file__))

chunk_size = 50

basePath = "base_path"
datasets = {"ldbc": "input_file.json",
            "knows":"input_file.json"}


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
    return os.path.join(base_dir, f".chunk_progress_{safe}.txt")


models = {
    "qwen-7b": {
        "type": "transformer",
        "path": "model_path",
        "quantized-4bit": False,
        "quantized-8bit": False
    },
    "mist-7b": {
        "type": "transformer",
        "path": "model_path",
        "quantized-4bit": False,
        "quantized-8bit": False
    },

    "llama-8b": {
        "type": "transformer",
        "id": "model_id",
        "path": "model_path",
        "quantized-4bit": False,
        "quantized-8bit": False
    },

    # "gpt": {
    #     "type": "openai",
    #     "model": "gpt-5"
    # },
    #
}

for i in range (1,4):

    for key,fname in datasets.items():
        in_file = basePath + fname

        with open(in_file, "r", encoding="utf-8") as f:
            full_dataset = json.load(f)
        # full_dataset = full_dataset[:200]

        print(f"Loaded {(key)} dataset graph triples. for f{i} time \n")

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


        for model_name, model_info in models.items():
            print(f"\n=== Starting model: {model_name} ===")

            prog_fp = progress_file_for_model(basePath, model_name)

            if os.path.exists(prog_fp):
                try:
                    start_chunk_i = int(open(prog_fp, "r", encoding="utf-8").read().strip()) + 1
                except Exception:
                    start_chunk_i = 0
            else:
                start_chunk_i = 0

            if start_chunk_i >= len(chunks):
                if os.path.exists(prog_fp):
                    os.remove(prog_fp)
                print(f"✓ {model_name}: already completed. Skipping.")
                continue

            master_summary_file = "base_path"  + f"{model_name}_Master_Summary_Input.txt"
            master_summary_output_file = basePath + f"{model_name}_Master_Summary_Output_{key}_{i}.txt"
            llm_output_file = basePath + f"{model_name}_LLM_Outputs_{key}_{i}.txt"

            model_type = model_info["type"]

            if model_type == "transformer":
                model_path = model_info["path"]
                quantized_4bit = model_info.get("quantized-4bit", False)
                quantized_8bit = model_info.get("quantized-8bit", False)

                if quantized_8bit:
                    quant_config = BitsAndBytesConfig(load_in_8bit=True)
                    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                                 quantization_config=quant_config,
                                                                 device_map="auto")
                if quantized_4bit:
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
                    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                                 torch_dtype=torch.float16,
                                                                 device_map="auto")

                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

            for chunk_i in range(start_chunk_i, len(chunks)):
                ch = chunks[chunk_i]

                chunk_text = json.dumps(ch, ensure_ascii=False, indent=2)  # indent optional
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
                    print("Exiting because previous %s is empty"%prev_file)
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

                # =====================================================
                # GPT MODEL CALL
                # =====================================================
                print("SUMMARY \n*2")
                print(previous_summary)
                print("\n*2")
                # exit(0)
                if model_type == "openai":
                    print("OPENAI")
                    # response = client.chat.completions.create(
                    #     model=model_info["model"],
                    #     messages=[
                    #         {"role": "system", "content": "You are an expert in dataset summary generation from property graphs"},
                    #         {"role": "user", "content": gpt_prompt}
                    #     ]
                    # )
                    # result = response.choices[0].message.content.strip()


                else:
                    messages = [
                        {"role": "system", "content": "You are an expert in dataset summary generation from property graphs."},
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

                print(f" Batch {chunk_i + 1} complete")
                with open(prog_fp, "w", encoding="utf-8") as f:
                    f.write(str(chunk_i))
                time.sleep(1)

            if os.path.exists(prog_fp):
                os.remove(prog_fp)

        print(f" Dataset {key} completed for run {i} (processed {len(chunks)} chunks).")

