#!/bin/bash

RUN_NAME="attn-shift-ffn-mse-ce"
DATASET="ok_vqa"
NUM_QUERY_SAMPLES=("300" "500" "1000")
NUM_EXPS=${#NUM_QUERY_SAMPLES[@]}
CUDA_DEVICES=("3" "4" "5" "6" "7")


for (( DEVICE=0; DEVICE<NUM_EXPS; DEVICE++ )); do
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICES[$DEVICE]}" python eval.py runname="$RUN_NAME-${NUM_QUERY_SAMPLES[$DEVICE]}" &
    if [ $? -ne 0 ]; then
        echo "Evaluation failed for runname: $RUN_NAME-${NUM_QUERY_SAMPLES[$DEVICE]}. Exiting."
        exit 1
    fi
done

wait
