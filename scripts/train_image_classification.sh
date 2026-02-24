#!/bin/bash

MODEL_BASE_NAME=`basename $MODEL_NAME`
MODEL_BASE_NAME=${MODEL_BASE_NAME//-/_}
SAVE_PATH=output/${CUSTOM_SUBDIR:=vision}/$MODEL_BASE_NAME/$DATASET/$TRAIN_MODE$CUSTOM_NAME/$LR/$SEED


if [ -d $SAVE_PATH ]; then
    echo "$SAVE_PATH already exists. Skipping!"
else
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:=0} \
    OMP_NUM_THREADS=2 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python train_image_classification.py \
        model_name_or_path=$MODEL_NAME \
        save_path=$SAVE_PATH \
        dataset=$DATASET \
        train_batch_size=128 \
        training_mode=$TRAIN_MODE \
        lr=$LR \
        seed=$SEED $OPTS
fi

if [ -f $SAVE_PATH/validation.json ]; then
    if [ -f $SAVE_PATH/test_$DATASET.json.gz ]; then
        echo "$SAVE_PATH/test_$DATASET.json.gz already exists. Skipping!"
    else
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:=0} \
        OMP_NUM_THREADS=2 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python test_image_classification.py \
            saved_path=$SAVE_PATH \
            dataset=$DATASET \
            test_batch_size=128 \
            results_filename=test_$DATASET
    fi
else
    echo "$SAVE_PATH/validation.json does not exist. Is training complete?"
fi
