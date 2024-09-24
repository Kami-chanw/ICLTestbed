#!/bin/bash

RUN_NAME="attn-shift-ffn-mse-500"
CUDA_VISIBLE_DEVICES=2 python train.py --runname "RUN_NAME"
python eval.py --runname "RUN_NAME"