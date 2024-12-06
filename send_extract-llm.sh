#!/bin/bash

#SBATCH -J "CPU"                         # isin adi

#SBATCH -A sdmmtv                           # account / proje adi
#SBATCH -p defq                          # kuyruk (partition/queue) adi

#SBATCH -n 128                            # cekirdek / islemci sayisi
#SBATCH -N 1                              # bilgisayar sayisi

# CUDA module'u yukleyelim
# module load cuda/cuda-11.7-a100q

#Programiniz birden cok GPU kullanacak sekilde yazilmis olmalidir.
#calisacak gpu isi

# sh ./extract_llm.sh

python ./extract_llm_2.py
