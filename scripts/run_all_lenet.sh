#!/usr/bin/env bash

set -Ceu

arch=lenet
exp_id=00
exp_root_dir=experiments/${arch}/exp${exp_id}/
for test_id in {0..14}; do
    CUDA_VISIBLE_DEVICES=0 python -u train.py \
        --config configs/${arch}_train.yaml \
        model.name ${arch} \
        train.test_id ${test_id} \
        train.output_dir ${exp_root_dir}

    exp_dir=${exp_root_dir}/$(printf %02d ${test_id})
    CUDA_VISIBLE_DEVICES=0 python -u evaluate.py \
        --config configs/${arch}_eval.yaml \
        test.test_id ${test_id} \
        test.checkpoint "${exp_dir}"/checkpoint_0040.pth \
        test.output_dir "${exp_dir}"/eval
done
