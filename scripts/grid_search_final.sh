#!/bin/bash

PYTHONPATH=src/ \
nohup python src/grid_search_final.py \
> logs/grid_search_final_$(date +'%Y-%m-%d_%H-%M-%S').stdout &
