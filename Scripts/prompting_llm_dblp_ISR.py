import argparse
import gc
import json
import os
import random
import time
from collections import defaultdict, deque
from typing import Any, Dict, Iterator, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ======================================================
# STREAMING / COUNTING
# ======================================================
def stream_jsonl(files: List[str]) -> Iterator[Dict[str, Any]]:
    for fp in files:
        print(f"[stream] {fp}")
        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue


def count_jsonl_lines(files: List[str]) -> Tuple[Dict[str, int], int]:
    counts: Dict[str, int] = {}
    total = 0

    print("\n=== COUNTING FILE LENGTHS ===")
    for fp in files:
        c = 0
        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            for _ in f:
                c += 1
        counts[fp] = c
        total += c
        pct = (c / total * 100) if total > 0 else 0.0

    for fp, c in counts.items():
        pct = (c / total * 100) if total > 0 else 0.0
        print(f"{os.path.basename(fp)}: {c:,} ({pct:.2f}%)")

    print(f"TOTAL: {total:,}\n")
    return counts, total


# ======================================================
# SCHEMA
# ======================================================
def build_property_graph_schema_stream(files: List[str], log_every: int = 1_000_000):
    node_schema = defaultdict(set)
    edge_schema = defaultdict(set)

    count = 0
    for item in stream_jsonl(files):
        count += 1

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

        if log_every and (count % log_every == 0):
            print(f"[schema] processed {count:,} rows...")

    print(f"[schema] done. total rows scanned: {count:,}")
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


# ======================================================
# PROPORTIONAL 10K SAMPLING
# ======================================================
def compute_sample_targets(
    counts: Dict[str, int],
    total: int,
    sample_total: int = 10_000
) -> Dict[str, int]:
    raw = {}
    floors = {}
    remainders = []

    for fp, c in counts.items():
        exact = (c / total) * sample_total if total > 0 else 0
        raw[fp] = exact
        floors[fp] = int(exact)
        remainders.append((fp, exact - floors[fp]))

    assigned = sum(floors.values())
    leftover = sample_total - assigned

    remainders.sort(key=lambda x: x[1], reverse=True)
    for i in range(leftover):
        floors[remainders[i][0]] += 1

    print("=== SAMPLE TARGETS ===")
    for fp, k in floors.items():
        print(f"{os.path.basename(fp)} -> {k}")
    print(f"TOTAL TARGET: {sum(floors.values())}\n")

    return floors


def reservoir_sample_jsonl(fp: str, k: int, seed: int = 42) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    reservoir: List[Dict[str, Any]] = []
    fname = os.path.basename(fp)

    with open(fp, "r", encoding="utf-8", errors="replace") as f:
        seen = 0
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
                item["_source_file"] = fname
            except Exception:
                continue

            if len(reservoir) < k:
                reservoir.append(item)
            else:
                j = rng.randint(0, seen)
                if j < k:
                    reservoir[j] = item

            seen += 1
            if seen % 500_000 == 0:
                print(f"[reservoir] {fname}: processed {seen:,}")

    print(f"[reservoir] {fname}: final sample = {len(reservoir)}")
    return reservoir


def build_10k_sample(files: List[str], file_targets: Dict[str, int], seed: int = 42) -> List[Dict[str, Any]]:
    all_samples: List[Dict[str, Any]] = []

    print("\n=== BUILDING PROPORTIONAL SAMPLE ===")
    for fp in files:
        k = file_targets.get(fp, 0)
        if k <= 0:
            continue
        sampled = reservoir_sample_jsonl(fp, k, seed=seed)
        all_samples.extend(sampled)

    print(f"TOTAL SAMPLED: {len(all_samples):,}\n")
    return all_samples


# ======================================================
# STRATIFICATION / CHUNKING
# ======================================================
def _strat_key(item: Dict[str, Any]) -> Tuple[str, str, str]:
    src = item.get("Source node", {}) or {}
    tgt = item.get("Target node", {}) or {}
    edge = item.get("Edge", {}) or {}

    src_label = str(src.get("Label", "Unknown"))
    edge_type = str(edge.get("type", "Unknown"))
    tgt_label = str(tgt.get("Label", "Unknown"))
    return (src_label, edge_type, tgt_label)


def compute_per_chunk_file_targets(
    sampled_items: List[Dict[str, Any]],
    chunk_size: int,
    num_chunks: int,
) -> Dict[str, int]:
    file_counts = defaultdict(int)
    for item in sampled_items:
        file_counts[item["_source_file"]] += 1

    total_sampled = len(sampled_items)
    file_names = list(file_counts.keys())

    raw_targets = {}
    base_targets = {}

    for fname in file_names:
        ratio = file_counts[fname] / total_sampled if total_sampled > 0 else 0
        raw = ratio * chunk_size
        tgt = int(raw)

        if file_counts[fname] >= num_chunks:
            tgt = max(1, tgt)

        raw_targets[fname] = raw
        base_targets[fname] = tgt

    current = sum(base_targets.values())

    if current < chunk_size:
        leftovers = sorted(
            ((fname, raw_targets[fname] - base_targets[fname]) for fname in file_names),
            key=lambda x: x[1],
            reverse=True
        )
        idx = 0
        while current < chunk_size and leftovers:
            fname = leftovers[idx % len(leftovers)][0]
            base_targets[fname] += 1
            current += 1
            idx += 1

    elif current > chunk_size:
        reducible = sorted(
            file_names,
            key=lambda f: (base_targets[f] - 1, raw_targets[f] - base_targets[f]),
            reverse=True
        )
        idx = 0
        while current > chunk_size and reducible:
            fname = reducible[idx % len(reducible)]
            min_allowed = 1 if file_counts[fname] >= num_chunks else 0
            if base_targets[fname] > min_allowed:
                base_targets[fname] -= 1
                current -= 1
            idx += 1
            if idx > 100000:
                break

    print("=== PER-CHUNK FILE DISTRIBUTION ===")
    print(dict(sorted(base_targets.items())))
    print()

    return base_targets


def build_chunks_with_file_balance_and_stratification(
    sampled_items: List[Dict[str, Any]],
    num_chunks: int,
    chunk_size: int,
    seed: int = 42,
) -> List[List[Dict[str, Any]]]:
    rng = random.Random(seed)

    per_chunk_file_targets = compute_per_chunk_file_targets(
        sampled_items=sampled_items,
        chunk_size=chunk_size,
        num_chunks=num_chunks,
    )

    grouped = defaultdict(lambda: defaultdict(list))
    for item in sampled_items:
        fname = item["_source_file"]
        grouped[fname][_strat_key(item)].append(item)

    file_strata = {}
    for fname, strata in grouped.items():
        strata_list = []
        for skey, items in strata.items():
            rng.shuffle(items)
            strata_list.append((skey, deque(items)))
        rng.shuffle(strata_list)
        file_strata[fname] = strata_list

    chunks: List[List[Dict[str, Any]]] = [[] for _ in range(num_chunks)]

    for chunk_idx in range(num_chunks):
        chunk: List[Dict[str, Any]] = []

        for fname, target_count in per_chunk_file_targets.items():
            if target_count <= 0:
                continue

            picked = 0
            strata_list = file_strata.get(fname, [])
            if not strata_list:
                continue

            round_robin_idx = 0
            fail_guard = 0

            while picked < target_count and fail_guard < 100000:
                if not strata_list:
                    break

                _, q = strata_list[round_robin_idx % len(strata_list)]

                if q:
                    chunk.append(q.popleft())
                    picked += 1

                round_robin_idx += 1
                fail_guard += 1

                if all(len(q2) == 0 for _, q2 in strata_list):
                    break

        if len(chunk) < chunk_size:
            remaining_pool = []
            for _, strata_list in file_strata.items():
                for _, q in strata_list:
                    remaining_pool.extend(list(q))

            rng.shuffle(remaining_pool)
            needed = chunk_size - len(chunk)
            fillers = remaining_pool[:needed]
            chunk.extend(fillers)

            used_ids = set(id(x) for x in fillers)
            for fname, strata_list in file_strata.items():
                for idx, (skey, q) in enumerate(strata_list):
                    new_q = deque([x for x in q if id(x) not in used_ids])
                    strata_list[idx] = (skey, new_q)

        chunks[chunk_idx] = chunk[:chunk_size]

    chunks = [ch for ch in chunks if ch]
    print(f"TOTAL CHUNKS: {len(chunks)}")
    return chunks


def print_chunk_file_distribution(chunks: List[List[Dict[str, Any]]], top_n: int = 5):
    print("\n=== CHUNK FILE DISTRIBUTION PREVIEW ===")
    for i, ch in enumerate(chunks[:top_n], start=1):
        cnt = defaultdict(int)
        for item in ch:
            cnt[item.get("_source_file", "UNKNOWN")] += 1
        print(f"Chunk {i}: {dict(sorted(cnt.items()))}")
    print()


def load_transformer(model_path: str, quant_4bit: bool, quant_8bit: bool):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    if not os.path.exists(os.path.join(model_path, "config.json")):
        raise FileNotFoundError(f"Missing config.json in model path: {model_path}")

    if quant_8bit:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant_config,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=True,
        )
    elif quant_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            local_files_only=True,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=True,
        )

    tok = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model.eval()
    return model, tok


def run_transformer(model, tokenizer, prompt: str, max_new_tokens: int = 2048) -> str:
    messages = [
        {"role": "system", "content": "You are an expert in dataset summary generation from property graphs."},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)

    input_device = model.get_input_embeddings().weight.device
    inputs = {k: v.to(input_device) for k, v in inputs.items()}

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    new_ids = out_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def unload_model(model, tokenizer):
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ======================================================
# MAIN
# ======================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_path", required=True, help="Directory containing processed edge jsonl files.")
    ap.add_argument("--chunk_size", type=int, default=50)
    ap.add_argument("--num_chunks", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--out_dir", required=True, help="Directory to save summaries and outputs.")
    ap.add_argument("--sample_total", type=int, default=10000, help="Total proportional sample size before chunking.")
    args = ap.parse_args()

    base_path = args.base_path
    out_dir = args.out_dir
    in_file = "input_file_path"
    os.makedirs(out_dir, exist_ok=True)

    files = [
        os.path.join(base_path, "article_article.jsonl"),
        os.path.join(base_path, "article_author.jsonl"),
        os.path.join(base_path, "article_keyword.jsonl"),
        os.path.join(base_path, "article_venue.jsonl"),
    ]
    files = [f for f in files if os.path.exists(f)]
    if not files:
        raise FileNotFoundError(f"No input jsonl files found under {base_path}.")

    print("\n=== 1) Building schema (streaming) ===")
    node_schema, edge_schema = build_property_graph_schema_stream(files)
    schema_text = schema_to_prompt(node_schema, edge_schema)
    print("Schema Extraction: Done\n")

    print("\n=== 2) Counting + proportional sampling ===")
    counts, total = count_jsonl_lines(files)
    targets = compute_sample_targets(counts, total, sample_total=args.sample_total)
    sampled_items = build_10k_sample(files, targets, seed=args.seed)

    print("\n=== 3) Building balanced chunks ===")
    chunks = build_chunks_with_file_balance_and_stratification(
        sampled_items=sampled_items,
        num_chunks=args.num_chunks,
        chunk_size=args.chunk_size,
        seed=args.seed,
    )
    print_chunk_file_distribution(chunks, top_n=10)

    models = {

        "qwen-7b": {
            "path": "model_path",
            "type": "transformer",
            "quantized_4bit": False,
            "quantized_8bit": False,
        },
        
        "mist-7b": {
            "path": "model_path",
            "quantized_4bit": False,
            "quantized_8bit": False,
        },
        "llama-8b": {
            "path": "model_path",
            "quantized_4bit": False,
            "quantized_8bit": False,
        },

    }

    runs = [1, 2, 3]

    for model_name, model_info in models.items():
        print(f"\n=== Starting model: {model_name} ===")
        print(f"Loading from: {model_info['path']}")

        for i in runs:
            model, tokenizer = load_transformer(
                model_path=model_info["path"],
                quant_4bit=model_info.get("quantized_4bit", False),
                quant_8bit=model_info.get("quantized_8bit", False),
            )

            master_summary_file = os.path.join(out_dir, f"{model_name}_Master_Summary_Input.txt")
            master_summary_output_file = os.path.join(out_dir, f"{model_name}_Master_Summary_Output_dblp_{i}.txt")
            llm_output_file = os.path.join(out_dir, f"{model_name}_LLM_Outputs_dblp_{i}.txt")

            if not os.path.exists(master_summary_file):
                input_template_fallback = os.path.join(in_file, f"{model_name}_Master_Summary_Input.txt")
                if os.path.exists(input_template_fallback):
                    master_summary_file = input_template_fallback
                else:
                    raise FileNotFoundError(
                        f"Missing initial summary template file: {os.path.join(out_dir, f'{model_name}_Master_Summary_Input.txt')} "
                        f"and fallback not found: {input_template_fallback}"
                    )

            with open(llm_output_file, "w", encoding="utf-8") as f:
                f.write(f"=== {model_name} | DBLP | run {i} ===\n")

            for chunk_i, ch in enumerate(chunks):
                chunk_text = json.dumps(ch, ensure_ascii=False, indent=2)
                print(f"=== {model_name} | Run {i} | Chunk {chunk_i + 1}/{len(chunks)} ===")

                prev_file = master_summary_file if chunk_i == 0 else master_summary_output_file

                if os.path.exists(prev_file):
                    with open(prev_file, "r", encoding="utf-8", errors="replace") as f:
                        previous_summary = f.read().strip()
                else:
                    previous_summary = "None"

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
                            """.strip()

                result = run_transformer(model, tokenizer, prompt, max_new_tokens=2048)

                with open(llm_output_file, "a", encoding="utf-8") as f:
                    f.write(f"\n=== {model_name} | Chunk {chunk_i + 1} ===\n{result}\n")

                with open(master_summary_output_file, "w", encoding="utf-8") as f:
                    f.write(result)

                print(f" {model_name} | Run {i} | Chunk {chunk_i + 1} complete")
                time.sleep(args.sleep)

            unload_model(model, tokenizer)
            print(f"=== {model_name} | RUN {i} DONE ===")

        print(f"=== {model_name} DONE ===")

    print(f"\noutputs saved under: {out_dir}")


if __name__ == "__main__":
    main()