#!/bin/bash

#SBATCH --job-name=multinode
#SBATCH -A sdmmtv                           # account / proje adi
#SBATCH -p a100x4q                          # kuyruk (partition/queue) adi
#SBATCH -n 64                            # cekirdek / islemci sayisi
#SBATCH -N 1                              # bilgisayar sayisi
#SBATCH --gres=gpu:4                # number of GPUs per node

######################
#### Set network #####
######################
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
######################

export LAUNCHER="accelerate launch \
    --num_processes 8 \
    --num_machines 2 \
    --rdzv_backend c10d \
    --main_process_ip a147 \
    --main_process_port 29500 \
    --config_file ./scripts/v1_5/multi_node.yaml \
    "

export PYTHON_FILE="llava/train/train_mem.py"
export ARGS=" \
  --model_name_or_path ytu-ce-cosmos/Turkish-Llama-8b-DPO-v0.1 \
  --version plain \
  --train_data_path ./playground/ocr-data/batch-1+2-400K/ocr-pretrain-400K-train.json \
  --eval_data_path ./playground/ocr-data/batch-1+2-400K/ocr-pretrain-400K-test.json \
  --image_folder ./playground/ocr-data/batch-1+2-400K/imgs/ \
  --eval_image_folder ./playground/ocr-data/batch-1+2-400K/imgs/ \
  --soft_moe False \
  --vision_tower openai/clip-vit-large-patch14-336 \
  --mm_projector_type mlp2x_gelu \
  --tune_mm_mlp_adapter True \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --fp16 True \
  --output_dir ./checkpoints/llava-pretrain-clip-batch1+2-multinode \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --moe_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 8 \
  --evaluation_strategy "steps" \
  --save_strategy "epoch" \
  --eval_steps 200 \
  --save_total_limit 2 \
  --learning_rate 2e-6 \
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
  "

# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $PYTHON_FILE $ARGS"
# srun $CMD
echo $CMD
$CMD
