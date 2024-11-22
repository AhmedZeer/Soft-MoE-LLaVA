#!/bin/bash

# --eval_data_path /ari/users/azeer/llava++/LLaVA-pp/LLaVA/playground/data/LLaVa-pretrain/chat-5k-lines.json \

deepspeed /ari/users/azeer/Soft-MoE-LLaVA/llava/train/train_mem.py \
  --deepspeed ./scripts/zero2.json \
  --model_name_or_path ytu-ce-cosmos/Turkish-Llama-8b-DPO-v0.1 \
  --version plain \
  --train_data_path /ari/users/azeer/llava++/LLaVA-pp/LLaVA/playground/data/LLaVa-pretrain/chat-4k-lines.json \
  --eval_data_path /ari/users/azeer/llava++/LLaVA-pp/LLaVA/playground/data/LLaVa-pretrain/chat-1k-lines.json \
  --image_folder /ari/users/azeer/llava++/LLaVA-pp/LLaVA/playground/data/LLaVa-pretrain/images \
  --eval_image_folder /ari/users/azeer/llava++/LLaVA-pp/LLaVA/playground/data/LLaVa-pretrain/images \
  --soft_moe True \
  --vision_tower openai/clip-vit-large-patch14-336 \
  --mm_projector_type mlp2x_gelu \
  --tune_mm_mlp_adapter True \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --fp16 True \
  --output_dir ./checkpoints/llava-v1.5-8b-2e-2p-pretrain-cosmosdpo-clamp_layerNorm_tanh_lowerDim-eval-2 \
  --num_train_epochs 500 \
  --per_device_train_batch_size 8 \
  --moe_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --evaluation_strategy "steps" \
  --save_strategy "steps" \
  --save_steps 50000 \
  --eval_steps 30 \
  --save_total_limit 3 \
  --learning_rate 1e-3 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --lazy_preprocess True \
  --report_to wandb
