#!/usr/bin/env bash
set -euo pipefail

DATA_NAME="moses_dataset_v2"
BATCH_SIZE=384
MAX_EPOCHS=10

# Run two jobs in parallel on two GPUs. Adjust CUDA_VISIBLE_DEVICES if needed.
CUDA_VISIBLE_DEVICES=0 python train/train.py --run_name "scaf_logp/run" --data_name "$DATA_NAME" --scaffold --props logp --num_props 1 --batch_size $BATCH_SIZE --max_epochs $MAX_EPOCHS &
CUDA_VISIBLE_DEVICES=1 python train/train.py --run_name "scaf_sas/run" --data_name "$DATA_NAME" --scaffold --props sas --num_props 1 --batch_size $BATCH_SIZE --max_epochs $MAX_EPOCHS &
wait

CUDA_VISIBLE_DEVICES=0 python train/train.py --run_name "scaf_tpsa/run" --data_name "$DATA_NAME" --scaffold --props tpsa --num_props 1 --batch_size $BATCH_SIZE --max_epochs $MAX_EPOCHS &
CUDA_VISIBLE_DEVICES=1 python train/train.py --run_name "scaf_qed/run" --data_name "$DATA_NAME" --scaffold --props qed --num_props 1 --batch_size $BATCH_SIZE --max_epochs $MAX_EPOCHS &
wait
