#!/bin/bash

PYTHONPATH=src/ \
nohup python src/grid_search_lr.py \
> logs/grid_search_lr_$(date +'%Y-%m-%d_%H-%M-%S').stdout &
