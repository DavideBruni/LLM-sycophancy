#!/bin/bash
#SBATCH --job-name=run_llama_logit_job
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
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

python run_llama_logit.py \
  --model_name "$MODEL_NAME" \
  --prefix_type "" \
  --difficulty "" \
  --question_style "plain" \
  --input_filename "prefix_full_question/mmlu/mmlu_plain.pkl" \
  --output_dir "output/mmlu/logit_redo_full" \
  --max_retries 1 \
  --full_question_col "full_question"

deactivate

echo "Run completed."