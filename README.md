GiVA: Gradient-Informed Bases for Vector-Based Adaptation
=========================================================

This repository contains the official implementation for GiVA.

Environment Setup
-----------------
To set up the environment, first create a new Python enviroment with `mamba` (or `conda`) as follows:
```bash
mamba env create -n peft -f environment.yaml
```
Next, install `peft` with changes required for this work:
```bash
pip install -e peft/
```
Changes in `peft` are based on v0.15.2.

Datasets
--------
We use HuggingFace's [`datasets`](https://github.com/huggingface/datasets) library for most datasets. For commonsense reasoning, we use datasets from [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters). Download them to [`datasets/`](datasets/) directory.

Experiments
-----------
### GLUE
Use [`scripts/train_glue.sh`](scripts/train_glue.sh) to train RoBERTa on GLUE tasks. For example,
```bash
MODEL_TYPE=bert \
MODEL_NAME=roberta-base \
TASK=sst2 \
TRAIN_MODE=giva \
LR=1e-2 \
SEED=0 \
SAVE_PATH=output/sst2_giva \
OPTS="training_mode.config.r=8 training_mode.config.init_weights=Vr training_mode.num_batches=1" \
sh scripts/train_glue.sh
```

### Commonsense / Math Reasoning / Code Generation
Use [`scripts/train_generation.sh`](scripts/train_generation.sh) to train models on commonsense reasoning. For example,
```bash
MODEL=Qwen2 \
DATASET=commonsense_15k \
TRAIN_MODE=giva \
LR=1e-2 \
SEED=0 \
TRAIN_BATCH_SIZE=16 \
TEST_BATCH_SIZE=16 \
OPTS="training_mode.config.r=8 training_mode.config.init_weights=Vr training_mode.num_batches=1" \
sh scripts/train_generation.sh
```
Replace `DATASET` with `metamath` or `code_feedback` for math reasoning and code generation, respectively.


### Instruction Tuning
Use [`scripts/train_alpaca.sh`](scripts/train_alpaca.sh) to train Mistral on the Alpaca dataset. For example,
```bash
MODEL=mistral \
SEED=0 \
TRAIN_MODE=giva \
LR=1e-2 \
OPTS="training_mode.config.giva_dropout=0. training_mode.config.r=32 training_mode.config.init_weights=Vr training_mode.num_batches=1" \
sh scripts/train_alpaca.sh
```
[`test_alpaca.py`](test_alpaca.py) may be used to decide the best checkpoint. Save the model in HF format using [`scripts/save_alpaca_hf_model.py`](scripts/save_alpaca_hf_models.py) and use [MT-Bench](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) for evaluating the model.


### Image Classification
Use [`scripts/train_image_classification.sh`](scripts/train_image_classification.sh) to train vision backbones on the image classification datasets. For example,
```bash
MODEL_NAME=facebook/dinov2-base \
DATASET=cifar100 \
TRAIN_MODE=giva \
LR=1e-2 \
SEED=0 \
OPTS="num_epochs=10 training_mode.config.r=32 training_mode.config.init_weights=Vr training_mode.num_batches=1 accumulate_grad_batches=1" \
sh scripts/train_image_classification.sh
```

### General Notes
- Check the corresponding Python scripts for more details on training and testing.
- Note that `training_mode.num_batches` must be equal to the number of batches to be used for gradient estimation multiplied by the number of gradient accumulation steps.
