#!/bin/bash

# --data_path /ari/users/azeer/test/LLaVA-pp/LLaVA/playground/ocr-data/batch-1-200K/ocrllava_batch1_200k_training_data.json \
# --image_folder /ari/users/azeer/test/LLaVA-pp/LLaVA/playground/ocr-data/batch-1-200K/imgs \

deepspeed llava/train/train_mem.py \
  --deepspeed ./scripts/zero3.json \
  --model_name_or_path ytu-ce-cosmos/Turkish-Llama-8b-DPO-v0.1 \
  --version llama3 \
  --soft_moe True \
  --experts_n 32 \
  --train_data_path ./playground/ocr-data/batch-2-200K/ocr-75K-finetune-train.json \
  --eval_data_path ./playground/ocr-data/batch-2-200K/ocr-75K-finetune-test.json \
  --image_folder ./playground/ocr-data/batch-2-200K/imgs/ \
  --eval_image_folder ./playground/ocr-data/batch-2-200K/imgs/ \
  --vision_tower openai/clip-vit-large-patch14-336 \
  --pretrain_mm_mlp_adapter ./checkpoints/llava-pretrain-32_ExpertsMoE-clip-ocr75K/mm_projector.bin \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length True \
  --fp16 True \
  --output_dir ./checkpoints/llava-32_ExpertsMoE-clip-ocr75K/ \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 4 \
  --moe_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --evaluation_strategy "steps" \
  --save_strategy "steps" \
  --save_steps 2000 \
  --eval_steps 100 \
  --save_total_limit 4 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing False \
  --dataloader_num_workers 4 \
  --lazy_preprocess True \
  --report_to wandb
