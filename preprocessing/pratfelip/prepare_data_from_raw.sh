#!/bin/sh
sbatch -n 1 -c 1 --mem=32G -t 120 -o prepare_data_from_raw.log --wrap="python prepare_data_from_raw.py"