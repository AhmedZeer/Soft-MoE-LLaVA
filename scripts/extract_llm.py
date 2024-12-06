import os
import argparse
from safetensors.torch import load_file, save_file
import math


def extract_and_shard_weights(input_path, num_shards, output_path):
    """
    Extract weights with keys containing 'model.layers' and save them in sharded format.
    """
    os.makedirs(output_path, exist_ok=True)

    # Load all weights from the input directory
    all_weights = {}
    shard_files = [f for f in os.listdir(input_path) if f.endswith(".safetensors")]
    for shard_file in shard_files:
        print("Shared File:", shard_file)
        shard_path = os.path.join(input_path, shard_file)
        shard_weights = load_file(shard_path)
        all_weights.update(shard_weights)

    # Filter weights containing 'model.layers'
    filtered_weights = {k: v for k, v in all_weights.items() if "model.layers" in k}

    # Split weights into specified number of shards
    shard_keys = list(filtered_weights.keys())
    shard_size = math.ceil(len(shard_keys) / num_shards)
    print("Saving Shards ...")
    for shard_idx in range(num_shards):
        start_idx = shard_idx * shard_size
        end_idx = min((shard_idx + 1) * shard_size, len(shard_keys))
        shard_dict = {k: filtered_weights[k] for k in shard_keys[start_idx:end_idx]}

        # Save the shard
        shard_file = os.path.join(output_path, f"model-{shard_idx + 1:05d}-of-{num_shards:05d}.safetensors")
        print(" - Saving : ", shard_file)
        save_file(shard_dict, shard_file, metadata={"format": "pt"})
        print(f"Saved shard: {shard_file}")


def main(input_path, num_shards, output_path):
    print("Extracting weights containing 'model.layers'...")
    extract_and_shard_weights(input_path, num_shards, output_path)
    print("Process completed successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract and shard weights with keys containing 'model.layers'")
    parser.add_argument('--input-path', type=str, required=True, help='Path to the input directory with .safetensors files')
    parser.add_argument('--num-shards', type=int, required=True, help='Number of shards for the output')
    parser.add_argument('--output-path', type=str, required=True, help='Path to the output directory for the sharded weights')
    args = parser.parse_args()

    main(args.input_path, args.num_shards, args.output_path)
