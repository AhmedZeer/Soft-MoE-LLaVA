#!/bin/bash

#SBATCH -J "Llava_finetune"                         # isin adi

#SBATCH -A sdmmtv                           # account / proje adi
#SBATCH -p a100q                          # kuyruk (partition/queue) adi

#SBATCH -n 64                            # cekirdek / islemci sayisi
#SBATCH -N 1                              # bilgisayar sayisi
#SBATCH --gres=gpu:1                      # ilave kaynak (1 gpu gerekli)

# CUDA module'u yukleyelim
# module load cuda/cuda-11.7-a100q

#Programiniz birden cok GPU kullanacak sekilde yazilmis olmalidir.
#calisacak gpu isi

#/ari/users/azeer/llava++/LLaVA-pp/LLaVA/LLaMA3-V_pretrain.sh

#$HOME/test.sh

# ./playground/data/LLaVA-Pretrain/chat.json
sh ./scripts/v1_5/pretrain_moe_cosmosllama.sh
# sh ./scripts/v1_5/finetune_moe_cosmosllama.sh
# sh ./scripts/v1_5/finetune_moe_vicuna.sh
