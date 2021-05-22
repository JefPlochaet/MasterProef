#!/bin/bash

#Run script to make the dataset for rcgan

python3 datasetmaken.py \
		--network rcgan \
		--seq_length 7 \
		--input_length 5 \
		--view front \
		--extension 5-2 \
