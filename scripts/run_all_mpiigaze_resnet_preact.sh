#!/usr/bin/env bash

set -Ceu

arch=resnet_preact
exp_id=${1:-00}
devices=${2:-0}
exp_root_dir=experiments/mpiigaze/${arch}/exp${exp_id}/
for test_id in {0..14}; do
    CUDA_VISIBLE_DEVICES=${devices} python -u train.py \
        --config configs/mpiigaze/${arch}_train.yaml \
        train.test_id ${test_id} \
        train.output_dir "${exp_root_dir}"

    exp_dir=${exp_root_dir}/$(printf %02d ${test_id})
    CUDA_VISIBLE_DEVICES=${devices} python -u evaluate.py \
        --config configs/mpiigaze/${arch}_eval.yaml \
        test.test_id ${test_id} \
        test.checkpoint "${exp_dir}"/checkpoint_0040.pth \
        test.output_dir "${exp_dir}"/eval
done
