#!/bin/bash
#SBATCH --job-name=run_llama_logit_cot_job
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G 
#SBATCH --output=./logs/run_llama_logit_cot/%j.out
#SBATCH --error=./logs/run_llama_logit_cot/%j.err

set -e

source ~/syco/bin/activate


DEFAULT_MODEL="meta-llama/Llama-3.2-3B"

if [[ -n "$1" ]]; then
  MODEL_NAME="$1"
else
  MODEL_NAME="$DEFAULT_MODEL"
fi

# Set huggingface cache dir
export HF_HOME="/ibex/user/wangk0e/cache/huggingface"

python run_llama_logit_cot.py \
  --model_name "$MODEL_NAME" \
  --dataset "mmlu_sample" \
  --max_retries 1 \
  --question_type "plain" \
  --prefix_type "" \
  --prefix_subtype "" \
  --academic_level "" \
  --input_filename "prefix_full_question/mmlu_sample/mmlu_subset.pkl" \
  --full_question_column "full_question_plain" 

deactivate

echo "Run completed."