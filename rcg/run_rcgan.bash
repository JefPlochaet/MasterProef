#!/bin/bash

#Script to run rcgan

python3 src/main.py \
	--train 1 \
	--batch_size 1 \
	--input_length 3 \
	--total_length 5 \
	--img_height 160 \
	--img_width 240 \
	--img_ch 1 \
	--max_itr 15 \
	--test_interval 15 \
	--snapshot_interval 20 \
	--lr 0.0003 \
	--results_dir results/moretest \
	--pretrained_model /home/jef/Documents/masterproef/rcg/results/3-1-240000/checkpoints/240000/generator_240000.pkl
