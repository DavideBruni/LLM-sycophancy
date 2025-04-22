#!/bin/bash

# Optional: log files manually
mkdir -p logs/run_llama
source ~/miniconda3/etc/profile.d/conda.sh

MODEL_NAME="${1:-meta-llama/Llama-3.2-3B}"

export HF_HOME="/root/hf_cache" 

python run_llama_logit_cot.py \
  --model_name "$MODEL_NAME" \
  --dataset "mmlu" \
  --max_retries 1 \
  --question_type "plain" \
  --prefix_type "" \
  --prefix_subtype "" \
  --academic_level "" \
  --input_filename "prefix_full_question/mmlu_sample/mmlu_subset.pkl" \
  --full_question_column "full_question_plain" | tee logs/run_llama/$(date +%s).log

deactivate
echo "Run completed."