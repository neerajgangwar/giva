#!/bin/bash

# eval "$(~/bin/micromamba shell hook -s posix)"
source ~/.miniconda3/etc/profile.d/conda.sh

HUMANEVAL_DIR=../human-eval/human_eval
RESULTS_FILENAME=test_humaneval_beams4

if [ -f $SAVED_PATH/validation.json ]; then
    if [ ! -f $SAVED_PATH/$RESULTS_FILENAME\.json.gz ]; then
        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
        OMP_NUM_THREADS=2 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        python test_generation.py \
            saved_path=$SAVED_PATH \
            test_batch_size=16 \
            ckpt_name=best \
            dataset=humaneval \
            num_beams=4 \
            max_new_tokens=256 \
            results_filename=$RESULTS_FILENAME
    else
        echo "$SAVED_PATH/$RESULTS_FILENAME\.json.gz already exists!"
    fi

    if [[ -f $SAVED_PATH/$RESULTS_FILENAME\_output.jsonl && ! -f $SAVED_PATH/$RESULTS_FILENAME\_scores.txt ]]; then
        conda activate humaneval

        python $HUMANEVAL_DIR/evaluate_functional_correctness.py \
            --sample_file=$SAVED_PATH/$RESULTS_FILENAME\_output.jsonl > $SAVED_PATH/$RESULTS_FILENAME\_scores.txt
    else
        echo "$SAVED_PATH/$RESULTS_FILENAME\_output.jsonl or $SAVED_PATH/$RESULTS_FILENAME\_scores.txt already exists!"
    fi

else
    echo "Is training complete?"
fi
