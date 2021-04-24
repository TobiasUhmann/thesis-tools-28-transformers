#!/bin/bash

PYTHONPATH=src/ \
nohup python src/grid_search_use_embs.py \
> logs/grid_search_use_embs_$(date +'%Y-%m-%d_%H-%M-%S').stdout &
