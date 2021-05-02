#!/bin/bash

PYTHONPATH=src/ \
nohup python src/eval_texter.py \
  data/power/texter/cde-irt-5-clean.pkl \
  5 \
  data/power/split/cde-0/ \
  data/irt/text/cde-irt-5-clean/ \
> logs/eval_texter_$(date +'%Y-%m-%d_%H-%M-%S').stdout &
