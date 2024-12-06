MODEL_PATH="./checkpoints/llava-v1.5-8b-2e-2p-cosmosdpo-clamp_layerNorm_tanh_lowerDim-FULL"
MODEL_NAME="llava-v1.5-8b-2e-2p-cosmosdpo-clamp_layerNorm_tanh_lowerDim-FULL"

python ./scripts/extract_llm.py \
  --input-path ${MODEL_PATH} \
  --num-shards 4 \
  --output-path ./LLMs/${MODEL_NAME}/
