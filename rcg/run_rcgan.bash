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
	--max_itr 5 \
	--test_interval 20 \
	--snapshot_interval 10 \
	--lr 0.0003 \
	--pretrained_model checkpoints/generator_5000.pkl | tee otput.txt
