#!/bin/bash
#SBATCH --job-name=run_llama_logit_cot_job
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=200G
#SBATCH --output=./logs/run_llama_logit/%j.out
#SBATCH --error=./logs/run_llama_logit/%j.err

set -e

source ~/syco/bin/activate

DEFAULT_MODEL="meta-llama/Llama-3.2-3B"

if [[ -n "$1" ]]; then
  MODEL_NAME="$1"
else
  MODEL_NAME="$DEFAULT_MODEL"
fi

# set huggingface cache dir
export HF_HOME="/ibex/user/wangk0e/cache/huggingface"

python run_llama_logit_cot.py \
  --model_name "$MODEL_NAME" \
  --prefix_type "" \
  --difficulty "" \
  --question_style "plain" \
  --input_filename "prefix_full_question/mmlu_sample/mmlu_subset.pkl" \
  --output_dir "output/mmlu/logit_cot" \
  --max_retries 1 \
  --full_question_col "full_question_plain"

deactivate

echo "Run completed."