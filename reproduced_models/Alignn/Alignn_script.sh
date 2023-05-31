#!/bin/sh
conda activate alignn-segnn
# cd /home/holywater2/2023/_Reproduce/script
python3 ../../script/intergrated_train.py \
--project="Alignn_02" \
--model="Alignn" \
--weight_decay=0.00001 \
--learning_rate=0.001 \
--batch_size=64 \
--mode="n" \
--data_pdirname="../../dataset/mp_megnet" \
--GPU="2" \
--n_epochs=300 \
--det="y" \
# --data_dirname="mp_megnet_sample001" \
# --wandb_disabled=True \
