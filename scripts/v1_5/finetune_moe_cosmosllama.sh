#!/bin/bash

# --data_path /ari/users/azeer/test/LLaVA-pp/LLaVA/playground/ocr-data/batch-1-200K/ocrllava_batch1_200k_training_data.json \
# --image_folder /ari/users/azeer/test/LLaVA-pp/LLaVA/playground/ocr-data/batch-1-200K/imgs \

deepspeed llava/train/train_mem.py \
  --deepspeed ./scripts/zero3.json \
  --model_name_or_path ytu-ce-cosmos/Turkish-Llama-8b-DPO-v0.1 \
  --version llama3 \
  --soft_moe True \
  --data_path /ari/users/azeer/llava++/LLaVA-pp/LLaVA/playground/data/LLaVa-finetune/2_toUHEM_Final_noLang_WithBooks_2.json \
  --image_folder /ari/users/azeer/llava++/LLaVA-pp/LLaVA/playground/data/llava_images/ \
  --vision_tower openai/clip-vit-large-patch14-336 \
  --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-8b-2e-2p-pretrain-cosmosdpo-clamp_layerNorm_tanh-FULL/mm_projector.bin \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length True \
  --fp16 True \
  --output_dir ./checkpoints/llava-v1.5-8b-2e-2p-cosmosdpo-clamp_layerNorm_tanh-FULL/ \
  --num_train_epochs 1 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 50000 \
  --save_total_limit 1 \
  --learning_rate 2e-5 \
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
