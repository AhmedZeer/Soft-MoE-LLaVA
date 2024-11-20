import os
import shutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file
import json

def convert_bf16_to_float16(input_repo_path):
    # Remove trailing slash if present
    input_repo_path = input_repo_path.rstrip('/')
    output_repo_path = input_repo_path + '__float16/'

    # Step 1: Create output directory (overwrite if exists)
    if os.path.exists(output_repo_path):
        shutil.rmtree(output_repo_path)
        print(f"Removed existing output directory: {output_repo_path}")

    os.makedirs(output_repo_path)
    print(f"Created output directory: {output_repo_path}")

    # Step 2: Copy over non-safetensors files
    print("Copying non-safetensors files...")
    for filename in os.listdir(input_repo_path):
        if not filename.endswith('.safetensors') and filename != 'model.safetensors.index.json':
            src_file = os.path.join(input_repo_path, filename)
            dst_file = os.path.join(output_repo_path, filename)
            shutil.copy2(src_file, dst_file)
            print(f"Copied {filename}")

    # Step 3: Process .safetensors shard files
    print("Processing .safetensors shard files...")
    for filename in os.listdir(input_repo_path):
        if filename.endswith('.safetensors'):
            src_file = os.path.join(input_repo_path, filename)
            dst_file = os.path.join(output_repo_path, filename)
            print(f"Processing {filename}...")

            # Load tensors and metadata using safe_open
            tensors = {}
            with safe_open(src_file, framework="pt", device="cpu") as f:
                metadata = f.metadata()
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    tensors[key] = tensor

            # Convert tensors to float16
            for tensor_name in tensors:
                tensor = tensors[tensor_name]
                if tensor.dtype == torch.bfloat16:
                    tensors[tensor_name] = tensor.to(torch.float16)
                else:
                    # If tensor is not bf16, keep it as is
                    tensors[tensor_name] = tensor

            # Update metadata
            if 'format' not in metadata or metadata['format'] != 'pt':
                metadata['format'] = 'pt'

            # Save tensors with updated metadata
            save_file(tensors, dst_file, metadata=metadata)
            print(f"Saved converted tensors to {dst_file}")

    # Step 4: Update model.safetensors.index.json
    index_file_src = os.path.join(input_repo_path, 'model.safetensors.index.json')
    index_file_dst = os.path.join(output_repo_path, 'model.safetensors.index.json')

    with open(index_file_src, 'r') as f:
        index_data = json.load(f)

    # Update dtype fields for each tensor
    if 'tensors' in index_data:
        for tensor_name in index_data['tensors']:
            tensor_info = index_data['tensors'][tensor_name]
            if tensor_info['dtype'] == 'bfloat16':
                tensor_info['dtype'] = 'float16'

    # Save updated index file
    with open(index_file_dst, 'w') as f:
        json.dump(index_data, f, indent=2)
        print(f"Updated index file saved to {index_file_dst}")

    print("Conversion complete.")

# Example usage:
input_repo_path = "checkpoints/llava-v1.5-8b-2e-2p-cosmosdpo/"
convert_bf16_to_float16(input_repo_path)
