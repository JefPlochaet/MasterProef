#!/bin/bash

#Script to run rcgan

python3 src/main.py \
	--train 1 \
	--batch_size 1 \
	--input_length 3 \
	--total_length 4 \
	--img_height 160 \
	--img_width 240 \
	--img_ch 1 \
	--max_itr 15 \
	--test_interval 15 \
	--snapshot_interval 20 \
	--lr 0.0003 \
	--results_dir results/moretesteen \
	--train_path data/auto-train-front-3-1.npy \
	--valid_path data/auto-validatie-front-3-1.npy
