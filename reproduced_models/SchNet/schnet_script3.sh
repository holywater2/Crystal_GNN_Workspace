#!/bin/sh
conda activate alignn
# cd /home/holywater2/2023/_Reproduce/script
python3 ../../script/intergrated_train.py \
--project="SchNet_Periodic_02" \
--model="SchNet" \
--weight_decay=0.00001 \
--learning_rate=0.0005 \
--n_epochs=500 \
--batch_size=64 \
--data_pdirname="../../dataset/mp_megnet" \
--GPU="7" \
--mode="n" \
# --save_loader=True \
# --loader_dirname="dataloader/mp_megent_dataloader" \
# --data_dirname="mp_megnet_sample001" \
# --wandb_disabled=True \
