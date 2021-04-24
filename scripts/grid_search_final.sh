#!/bin/bash

PYTHONPATH=src/ \
nohup python src/grid_search_datasets.py \
> logs/grid_search_datasets_$(date +'%Y-%m-%d_%H-%M-%S').stdout &
