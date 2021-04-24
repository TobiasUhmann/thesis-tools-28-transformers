#!/bin/bash

PYTHONPATH=src/ \
nohup python src/grid_search_sent_len.py \
> logs/grid_search_sent_len_$(date +'%Y-%m-%d_%H-%M-%S').stdout &
