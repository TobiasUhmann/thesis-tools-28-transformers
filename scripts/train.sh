#!/bin/bash

PYTHONPATH=src/ \
nohup python src/train.py \
  data/ower/ower-v4-fb-irt-100-5/ \
  100 \
  5 \
> log/train_$(date +'%Y-%m-%d_%H-%M-%S').stdout &
