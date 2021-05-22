#!/bin/bash

#Run python script to make the dataset for predrnn/predrnn++

python3 datasetmaken.py \
		--network predrnn \
		--seq_length 4 \
		--input_length 3 \
		--view side \
		--extension 3-1 \
		--add_dummy_data 0
