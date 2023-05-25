#!/bin/sh
conda activate alignn-segnn
# cd /home/holywater2/2023/_Reproduce/script
python3 ../../script/intergrated_train.py \
--project="Alignn_02" \
--model="Alignn" \
--weight_decay=0.00001 \
--learning_rate=0.0005 \
--batch_size=64 \
--mode="n" \
--data_dirname="mp_megnet_sample001" \
--data_pdirname="../../dataset/mp_megnet" \
--GPU="3" \
--n_epochs=200 \
--det="y" \
# --wandb_disabled=True \
