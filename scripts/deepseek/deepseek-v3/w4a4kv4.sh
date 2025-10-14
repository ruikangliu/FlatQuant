#!/bin/bash

RANK=${RANK:-0}
ADDR=${ADDR:-"127.0.0.1"}
PORT=${PORT:-29500}

torchrun --nnodes 2 --nproc-per-node 8 --node-rank $RANK --master-addr $ADDR --master-port $PORT \
    main_dpskv3.py --ckpt-path ./modelzoo/deepseek/deepseek_v3 --config deepseek_v3/configs/config_671B.json \
    --w_bits 4 --a_bits 4 \
    --cali_bsz 8 --epoch 15 --flat_lr 5e-3 \
    --lwc --lac --cali_trans --add_diag --v3_not_last \
    --output_dir ./outputs --save_matrix \
    --cali_seqlen 4096 --cali_dataset wikitext2 \

