#!/bin/bash
# 三组实验：除了训练数据和名字，其他设置完全相同
# 用法：bash scripts/run_exp.sh 种子数字，例如 bash scripts/run_exp.sh 1
set -e
SEED=${1:-0}

COMMON=(
  --model ViT-B-32
  --dataset-type webdataset
  --dataset-resampled
  --train-num-samples 494018
  --batch-size 256
  --epochs 1
  --lr 5e-4
  --warmup 200
  --wd 0.1
  --precision amp
  --workers 6
  --seed "$SEED"
  --val-data "$HOME/work/sets/val/{0000..0010}.tar"
  --val-num-samples 10082
  --logs "$HOME/work/runs"
)

python -m open_clip_train.main "${COMMON[@]}" --train-data "$HOME/work/cc12m/cc12m-train-{0000..0097}.tar" --name "exp_A_full_s$SEED"
python -m open_clip_train.main "${COMMON[@]}" --train-data "$HOME/work/sets/random30/{0000..0148}.tar" --name "exp_B_random30_s$SEED"
python -m open_clip_train.main "${COMMON[@]}" --train-data "$HOME/work/sets/clip30/{0000..0148}.tar" --name "exp_C_clip30_s$SEED"
