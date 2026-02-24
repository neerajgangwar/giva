source ~/.miniconda3/etc/profile.d/conda.sh
conda activate peft

if [ "$MODEL" = "Qwen2" ]; then
    MODEL_NAME=Qwen/Qwen2-0.5B-Instruct
elif [ "$MODEL" = "Phi3" ]; then
    MODEL_NAME=microsoft/Phi-3-mini-4k-instruct
elif [ "$MODEL" == "olmo" ]; then
    MODEL_NAME=allenai/OLMo-2-1124-7B
fi

MODEL_BASE_NAME=`basename $MODEL_NAME`
MODEL_BASE_NAME=${MODEL_BASE_NAME/-/_}
SAVE_PATH=output/${OUT_DIR:=$DATASET}/$MODEL_BASE_NAME/$TRAIN_MODE$CUSTOM_NAME/$LR/$SEED
RUN_NAME=$MODEL\_$TRAIN_MODE$CUSTOM_NAME\_$LR\_$SEED

if [ ! -d $SAVE_PATH ]; then
    OMP_NUM_THREADS=2 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python train_generation.py \
        model_name_or_path=$MODEL_NAME \
        dataset=$DATASET \
        save_path=$SAVE_PATH \
        train_batch_size=$TRAIN_BATCH_SIZE \
        training_mode=$TRAIN_MODE \
        lr=$LR \
        run_name=$RUN_NAME \
        seed=$SEED $OPTS
else
    echo "$SAVE_PATH already exists!"
fi

if [ -f $SAVE_PATH/validation.json ]; then
    if [ "$DATASET" == "commonsense_15k" ]; then
        SAVED_PATH=$SAVE_PATH TEST_BATCH_SIZE=$TEST_BATCH_SIZE sh scripts/test_commonsense.sh
    elif [ "$DATASET" == "metamath" ]; then
        if [ ! -f $SAVE_PATH/test_gsm8k.json.gz ]; then
                OMP_NUM_THREADS=2 \
                PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
                CUBLAS_WORKSPACE_CONFIG=:4096:8 \
                python test_generation.py \
                    saved_path=$SAVE_PATH \
                    test_batch_size=$TEST_BATCH_SIZE \
                    num_beams=1 \
                    max_new_tokens=256 \
                    dataset=gsm8k \
                    results_filename=test_gsm8k
        else
            echo "$SAVE_PATH/test_gsm8k.json.gz already exists. Skipping!"
        fi
    elif [ "$DATASET" == "code_feedback" ]; then
        SAVED_PATH=$SAVE_PATH sh scripts/test_humaneval.sh
    else
        echo "$DATASET not supported"
    fi
else
    echo "$SAVE_PATH/validation.json does not exist! Is the training complete?"
fi
