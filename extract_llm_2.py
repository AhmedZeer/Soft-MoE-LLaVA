import os
from safetensors.torch import load_file, save_file

def load_all_weights(directory):
    """Load all safetensor files from a directory."""
    file_paths = sorted(
        [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".safetensors")]
    )
    return [load_file(path) for path in file_paths], file_paths

def update_weights(source_weights, target_weights):
    """
    Update weights in target_weights with common keys from source_weights.
    """
    for src_weight in source_weights:
        for key, value in src_weight.items():
            for tgt_weight in target_weights:
                if key in tgt_weight:
                    print("Found Key:", key)
                    tgt_weight[key] = value
    return target_weights

def save_updated_weights(weights_list, original_file_paths, output_directory, metadata=None):
    """
    Save updated weights with the same filenames to the output directory.
    """
    os.makedirs(output_directory, exist_ok=True)
    for original_path, weights in zip(original_file_paths, weights_list):
        file_name = os.path.basename(original_path)
        output_path = os.path.join(output_directory, file_name)
        print(f"Saving: {output_path} ...")
        save_file(weights, output_path, metadata=metadata)
        print(f"Saved: {output_path}")

def process_weights(llava_dir, llama_dir, output_dir):
    """
    Process LLAVA and LLAMA weights, updating LLAMA weights with LLAVA weights,
    and save the updated weights to the output directory.
    """
    # Load LLAVA and LLAMA weights
    llava_weights, _ = load_all_weights(llava_dir)
    llama_weights, llama_file_paths = load_all_weights(llama_dir)

    # Update LLAMA weights
    updated_llama_weights = update_weights(llava_weights, llama_weights)

    # Save updated LLAMA weights
    save_updated_weights(updated_llama_weights, llama_file_paths, output_dir, metadata={"format": "pt"})

# Define input and output directories
llava_dir = "/home/azeer/uhem/test/checkpoints/ocr-llava-batch1-llm"
llama_dir = "./checkpoints/COSMOS-DPO/"
output_dir = "./LLMs/batch1-llm/"

# Process and update weights
process_weights(llava_dir, llama_dir, output_dir)
