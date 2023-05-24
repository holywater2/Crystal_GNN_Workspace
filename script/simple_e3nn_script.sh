#!/bin/sh
conda activate alignn-segnn
cd /home/holywater2/2023/_Reproduce/script/
CUDA_VISIBLE_DEVICES="1" python3 train_simple_e3nn.py --weight_decay=0.00001 --learning_rate=0.0005 --n_epochs=200 --batch_size=64 --mode=n