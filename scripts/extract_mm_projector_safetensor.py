"""
Extracts mm_projector weights from sharded .safetensors model files and saves them to a single output file.
"""

import os
import argparse
import torch
import json
from safetensors import safe_open

def parse_args():
    parser = argparse.ArgumentParser(description='Extract MMProjector weights from sharded .safetensors models')
    parser.add_argument('--model-path', type=str, help='Path to the model folder')
    parser.add_argument('--output', type=str, help='Output file name for the extracted weights')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    # Load the index file to determine shard mapping
    index_file = os.path.join(args.model_path, "model.safetensors.index.json")
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"Index file 'model.safetensors.index.json' not found in {args.model_path}")

    with open(index_file, 'r') as f:
        index_data = json.load(f)

    # Filter for keys matching 'mm_projector'
    keys_to_match = "mm_projector"
    shard_to_keys = {}
    for key, shard in index_data['weight_map'].items():
        if keys_to_match in key:
            if shard not in shard_to_keys:
                shard_to_keys[shard] = []
            shard_to_keys[shard].append(key)

    # Extract and consolidate mm_projector weights
    mm_projector = {}
    for shard, keys in shard_to_keys.items():
        shard_path = os.path.join(args.model_path, shard)
        with safe_open(shard_path, framework="pt") as f:
            for key in keys:
                mm_projector[key] = f.get_tensor(key)

    # Save the extracted weights to the specified output file
    torch.save(mm_projector, args.output)
    print(f"MMProjector weights extracted and saved to {args.output}")
