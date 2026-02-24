#!/bin/bash

LR=${LR:=1e-4}
SEED=${SEED:=42}

MODEL_BASE_NAME=`basename $MODEL_NAME`
MODEL_BASE_NAME=${MODEL_BASE_NAME/-/_}
SAVE_PATH=output/${CUSTOM_SUBDIR:=glue}/$MODEL_BASE_NAME/$TASK/$TRAIN_MODE$CUSTOM_NAME/$LR/$SEED

if [[ "$TASK" == "qnli" || "$TASK" == "rte" ]]; then
    if [ "$MODEL_NAME" == "roberta-large" ]; then
        OPTS="$OPTS accumulate_grad_batches=2"
    fi
fi

if [ -d $SAVE_PATH ]; then
    echo "$SAVE_PATH already exists. Skipping!"
else
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:=0} \
    OMP_NUM_THREADS=2 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python train_glue.py \
        model_type=$MODEL_TYPE \
        model_name_or_path=$MODEL_NAME \
        save_path=$SAVE_PATH \
        task=$TASK \
        train_batch_size=64 \
        training_mode=$TRAIN_MODE \
        lr=$LR \
        seed=$SEED $OPTS
fi
