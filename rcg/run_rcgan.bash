#!/bin/bash

#Script to run rcgan

python3 src/main.py \
	--train 0 \
	--batch_size 1 \
	--input_length 5 \
	--total_length 6 \
	--img_height 160 \
	--img_width 240 \
	--img_ch 1 \
	--max_itr 15 \
	--test_interval 15 \
	--snapshot_interval 20 \
	--lr 0.001 \
	--results_dir results/test-5-1-80000 \
	--train_path data/auto-train-front-5-1.npy \
	--valid_path data/auto-validatie-front-5-1.npy \
	--test_path data/auto-test-front-5-1.npy \
	--pretrained_model results/5-1-80000/checkpoints/80000/generator_80000.pkl | tee output-5-1-80000.txt
