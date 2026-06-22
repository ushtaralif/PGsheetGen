
import json
import random
import argparse

def process_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must contain a list of objects.")

    for entry in data:
        entry.pop("Relation", None)

    random.shuffle(data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Processing complete. Output written to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Randomize JSON objects and remove 'Relation' key.")
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    args = parser.parse_args()

    process_file(args.input, args.output)