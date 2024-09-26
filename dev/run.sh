#!/bin/bash

get_cuda_devices() {
    local free_gpus
    free_gpus=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{if ($1 > 20000) print NR-1}' | tr '\n' ',')
    free_gpus="${free_gpus%,}"
    echo "$free_gpus"
}

RUN_NAME="attn-shift-ffn-after_lm_kl-ce"
DATASET="vqav2"
NUM_QUERY_SAMPLES=("500" "1000" "2000" "4000" "8000")
NUM_EXPS=${#NUM_QUERY_SAMPLES[@]}

for SAMPLES in "${NUM_QUERY_SAMPLES[@]}"; do
    CUDA_DEVICES=$(get_cuda_devices)
    
    if [ -z "$CUDA_DEVICES" ]; then
        echo "No available GPUs with more than 20GB free memory."
        exit 1
    fi
    
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" python train.py runname="$RUN_NAME-$DATASET-$SAMPLES" data.num_query_samples="$SAMPLES" data.name="$DATASET"
    if [ $? -ne 0 ]; then
        echo "Training failed for runname: $RUN_NAME-$SAMPLES. Exiting."
        exit 1
    fi
done

CUDA_DEVICES=$(get_cuda_devices)
IFS=',' read -r -a CUDA_DEVICE_ARRAY <<< "$CUDA_DEVICES"

for (( DEVICE=0; DEVICE<NUM_EXPS; DEVICE++ )); do
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_ARRAY[$DEVICE]}" python eval.py runname="$RUN_NAME-$DATASET-${NUM_QUERY_SAMPLES[$DEVICE]}" data.name="$DATASET" &
    if [ $? -ne 0 ]; then
        echo "Evaluation failed for runname: $RUN_NAME-${NUM_QUERY_SAMPLES[$DEVICE]}. Exiting."
        exit 1
    fi
done

wait
