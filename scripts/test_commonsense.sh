#!/bin/bash

declare -A BATCH_SIZE_MULTIPLIER
BATCH_SIZE_MULTIPLIER[boolq]=4
BATCH_SIZE_MULTIPLIER[piqa]=1
BATCH_SIZE_MULTIPLIER[social_i_qa]=2
BATCH_SIZE_MULTIPLIER[hellaswag]=1
BATCH_SIZE_MULTIPLIER[winogrande]=2
BATCH_SIZE_MULTIPLIER[arc_easy]=2
BATCH_SIZE_MULTIPLIER[arc_challenge]=1
BATCH_SIZE_MULTIPLIER[openbookqa]=2


EVAL_DATASETS=(boolq piqa social_i_qa hellaswag winogrande arc_easy arc_challenge openbookqa)
for EVAL_DATASET in ${EVAL_DATASETS[@]}; do
    MUL_FAC=${BATCH_SIZE_MULTIPLIER[$EVAL_DATASET]}
    if [ ! -f $SAVED_PATH/test_$EVAL_DATASET\_beams4.json.gz ]; then
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:=0} \
        OMP_NUM_THREADS=2 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        python test_generation.py \
            saved_path=$SAVED_PATH \
            dataset=$EVAL_DATASET \
            results_filename=test_$EVAL_DATASET\_beams4 \
            num_beams=4 \
            max_new_tokens=32 \
            test_batch_size=$((TEST_BATCH_SIZE*MUL_FAC))
    else
        echo "$SAVED_PATH/test_$EVAL_DATASET\_beams4.json.gz exists!"
    fi

    # if [ ! -f $SAVED_PATH/test_$EVAL_DATASET.json.gz ]; then
    #     CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:=0} \
    #     OMP_NUM_THREADS=2 \
    #     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    #     CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    #     python test_generation.py \
    #         saved_path=$SAVED_PATH \
    #         dataset=$EVAL_DATASET \
    #         results_filename=test_$EVAL_DATASET \
    #         num_beams=1 \
    #         max_new_tokens=32 \
    #         test_batch_size=$((TEST_BATCH_SIZE*4))
    # fi
done
