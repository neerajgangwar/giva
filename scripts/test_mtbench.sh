#!/bin/bash

eval "$(~/bin/micromamba shell hook -s posix)"
micromamba activate mtbench

# GPT: gpt-4-0613
JUDGE_MODEL=gpt-4
LLM_JUDGE_DIR=../FastChat/fastchat/llm_judge
cd $LLM_JUDGE_DIR

MODEL_BASE_NAME=`basename $MODEL_PATH`
MODEL_ID=${MODEL_BASE_NAME//peft_/}
echo "MODEL_PATH: $MODEL_PATH"
echo "MODEL_ID: $MODEL_ID"

if [ "$MODE" == "gen_answer" ]; then
    if [ ! -f data/mt_bench/model_answer/$MODEL_ID.jsonl ]; then
        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
        python gen_model_answer.py \
            --model-path $MODEL_PATH \
            --model-id $MODEL_ID \
            --dtype bfloat16
    else
        echo "Skipping! data/mt_bench/model_answer/$MODEL_ID.jsonl exists!"
    fi
elif [ "$MODE" == "gen_judgment" ]; then
    OUTPUT_FILE=data/mt_bench/model_judgment/$JUDGE_MODEL\_single.jsonl
    OUTPUT_MV_FILE=data/mt_bench/model_judgment/$JUDGE_MODEL\_single_$MODEL_ID\.jsonl
    if [ -f $OUTPUT_FILE ]; then
        echo "Skipping! $OUTPUT_FILE exists!"
    elif [ -f $OUTPUT_MV_FILE ]; then
        echo "Skipping! $OUTPUT_MV_FILE exists!"
    else
        python gen_judgment.py \
            --model-list $MODEL_ID \
            --judge-model $JUDGE_MODEL \
            --parallel ${PARALLEL_CALLS:=1}

        if [ "$?" == "0" ]; then
            mv $OUTPUT_FILE $OUTPUT_MV_FILE
        else
            echo "gen_judgment command exit status non-zero!"
        fi
    fi
elif [ "$MODE" == "view_results" ]; then
    OUTPUT_FILE=data/mt_bench/model_judgment/$JUDGE_MODEL\_single_$MODEL_ID\.jsonl
    python show_result.py --input-file $OUTPUT_FILE
else
    echo "MODE '$MODE' not supported. Exiting!"
fi
