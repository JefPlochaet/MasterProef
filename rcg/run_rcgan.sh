#!/bin/sh

#Script to run rcgan

python3 src/main.py \
	--train 1 \
	--batch_size 2 \
	--input_length 3 \
	--total_length 4 \
	--img_height 160 \
	--img_width 240 \
	--img_ch 1 \
	--max_itr 1 \
	--test_interval 1 \
	--snapshot_interval 1
