#!/bin/bash

PYTHONPATH=src/ \
nohup python src/train.py \
  data/power/samples-v5/cde-irt-5-marked/ \
  100 \
  5 \
  data/power/split-v2/cde-100/ \
  data/power/texter-v2/cde-irt-5-marked_power.pkl \
  data/power/eval-v1/cde-irt-5-marked_power.yml \
  --epoch-count 2 \
  --model power \
  --test \
> logs/train_$(date +'%Y-%m-%d_%H-%M-%S').stdout &
