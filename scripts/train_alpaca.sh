#!/bin/bash

if [ "$MODEL" == "mistral" ]; then
    MODEL_NAME=mistralai/Mistral-7B-v0.1
else
    echo "Model '$MODEL' not supported!"
    exit
fi

MODEL_BASE_NAME=`basename $MODEL_NAME`
MODEL_BASE_NAME=${MODEL_BASE_NAME//-/_}
SAVE_PATH=output/${CUSTOM_SUBDIR:=alpaca}/$MODEL_BASE_NAME/$TRAIN_MODE$CUSTOM_NAME/$LR/$SEED
RUN_NAME=$MODEL\_$TRAIN_MODE$CUSTOM_NAME\_$LR\_$SEED

if [ ! -d $SAVE_PATH ]; then
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    OMP_NUM_THREADS=2 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python train_alpaca.py \
        model_name_or_path=$MODEL_NAME \
        save_path=$SAVE_PATH \
        training_mode=$TRAIN_MODE \
        lr=$LR \
        run_name=$RUN_NAME \
        seed=$SEED $OPTS
else
    echo "$SAVE_PATH already exists!"
fi

if [ -d $SAVE_PATH ]; then
    python test_alpaca.py saved_path=$SAVE_PATH ckpt_name=last dataset_file=datasets/mt-bench/dev_set.json
fi

