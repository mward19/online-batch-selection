#!/bin/bash

# Usage: ./run.sh

SEEDS=(1 2 3 4 5)

CONFIG_DIR="configs/mnist"

DIAGNOSTICS="$CONFIG_DIR/diagnostics/feature_learning_log_interval.yaml"
METHODS=("$CONFIG_DIR/method/"*)
MODELS=("$CONFIG_DIR/model/lenet.yaml")
OPTIMS=("$CONFIG_DIR/optim/sgd-320-0.01-0.0.yaml")
DATAS=("$CONFIG_DIR/data/mnist_noise.yaml")

for SEED in "${SEEDS[@]}"; do
  for data in "${DATAS[@]}"; do
    for model in "${MODELS[@]}"; do
      for optim in "${OPTIMS[@]}"; do
        for method in "${METHODS[@]}"; do
          echo "Running: --method $method --data $data --model $model --optim $optim --diagnostics $DIAGNOSTICS --seed $SEED"
          CUDA_VISIBLE_DEVICES=0 uv run main.py --method "$method" --data "$data" --model "$model" --optim "$optim" --diagnostics "$DIAGNOSTICS" --seed "$SEED" --wandb_not_upload
        done
      done
    done
  done
done