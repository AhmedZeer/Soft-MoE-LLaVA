#!/bin/bash

#SBATCH -J "Llava_SigLIP"                         # isin adi

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

# ./playground/data/LLaVA-Pretrain/chat.json
# sh ./scripts/v1_5/pretrain_moe_cosmosllama.sh
# sh ./scripts/v1_5/finetune_moe_cosmosllama_overfit.sh
# sh ./scripts/v1_5/finetune_moe_vicuna.sh
# sh ./scripts/v1_5/pretrain_cosmosllama_siglip.sh

# sh ./scripts/v1_5/finetune_moe_cosmosllama.sh

# sh ./scripts/v1_5/finetune_cosmosllama_siglip.sh
# sh ./scripts/v1_5/pretrain_moe_cosmosllama_eval_test.sh
# sh ./scripts/v1_5/pretrain_cosmosllama_siglip_eval_test.sh
# sh ./scripts/v1_5/hypr/finetune_moe_cosmosllama_lr1e-8.sh

# sh ./scripts/v1_5/hypr/finetune_moe_cosmosllama_ga-16.sh
# sh ./scripts/v1_5/hypr/finetune_moe_cosmosllama_lr1e-8_ga-16.sh
# sh ./scripts/v1_5/hypr/siglip/finetune-1-default.sh
# sh ./scripts/v1_5/overfit-encoders/overfit-clip.sh
# sh ./scripts/v1_5/overfit-encoders/overfit-clip-2.sh
# sh ./scripts/v1_5/overfit-encoders/overfit-moe.sh
# sh ./scripts/v1_5/overfit-encoders/overfit-siglip.sh

sh ./scripts/v1_5/finetune_cosmosllama_internVit.sh

