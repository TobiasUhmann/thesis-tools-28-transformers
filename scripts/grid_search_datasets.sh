#!/bin/bash

PYTHONPATH=src/ \
nohup python src/gird_search_datasets.py \
> log/build-baseline_$(date +'%Y-%m-%d_%H-%M-%S').stdout &
