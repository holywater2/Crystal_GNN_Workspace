#!/bin/sh
conda activate alignn
cd /home/holywater2/2023/_Reproduce/script/
CUDA_VISIBLE_DEVICES="2" python3 train_schnet.py --weight_decay=0.00001 --learning_rate=0.0005 --n_epochs=500 --batch_size=64