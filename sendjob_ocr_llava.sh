#!/bin/bash

#SBATCH -J "Llava"                         # isin adi

#SBATCH -A sdmmtv                           # account / proje adi
#SBATCH -p a100x4q                          # kuyruk (partition/queue) adi

#SBATCH -n 64                            # cekirdek / islemci sayisi
#SBATCH -N 1                              # bilgisayar sayisi
#SBATCH --gres=gpu:4                      # ilave kaynak (1 gpu gerekli)

# CUDA module'u yukleyelim
# module load cuda/cuda-11.7-a100q

#Programiniz birden cok GPU kullanacak sekilde yazilmis olmalidir.
#calisacak gpu isi

#/ari/users/azeer/llava++/LLaVA-pp/LLaVA/LLaMA3-V_pretrain.sh

#$HOME/test.sh

# NPROC_PER_NODE=4 xtuner train /ari/users/azeer/xtuner/cosmos_configs/1_epoch_pretrain_cosmosLLaVA_99eren.py --deepspeed deepspeed_zero2 --seed 1024
# ./playground/data/LLaVA-Pretrain/chat.json
sh ./scripts/v1_5/pretrain_moe.sh