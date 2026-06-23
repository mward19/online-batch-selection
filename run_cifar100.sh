#!/bin/bash

# Usage: ./run.sh

SEEDS=(1 2 3 4 5)

CONFIG_DIR="configs/cifar100"

DIAGNOSTICS="$CONFIG_DIR/diagnostics/all_log_interval.yaml"
METHODS=("$CONFIG_DIR/method/"*)
MODELS=("$CONFIG_DIR/model/resnet18.yaml")
OPTIMS=("$CONFIG_DIR/optim/adamw-320-0.001-0.01.yaml")
DATAS=("$CONFIG_DIR/data/cifar100.yaml")

for SEED in "${SEEDS[@]}"; do
  for data in "${DATAS[@]}"; do
    for model in "${MODELS[@]}"; do
      for optim in "${OPTIMS[@]}"; do
        for method in "${METHODS[@]}"; do
          echo "Running: --method $method --data $data --model $model --optim $optim --diagnostics $DIAGNOSTICS --seed $SEED"
          CUDA_VISIBLE_DEVICES=1 uv run main.py --method "$method" --data "$data" --model "$model" --optim "$optim" --diagnostics "$DIAGNOSTICS" --seed "$SEED" --wandb_not_upload
        done
      done
    done
  done
done