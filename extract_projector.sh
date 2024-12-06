# MODEL_NAME="llava-v1.5-8b-2e-2p-cosmosdpo-clamp_layerNorm_tanh_lowerDim-FULL"

# MODEL_PATH="/home/ahmed4/uhem/test/checkpoints"
MODEL_PATH="${HOME}/test/checkpoints/ocr-llava-batch1-llm"

python ./scripts/extract_mm_projector_safetensor.py \
  --model-path ${MODEL_PATH} \
  --output ./projectors/batch1-llm/mm_projector.bin
