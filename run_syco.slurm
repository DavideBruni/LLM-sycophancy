#!/bin/bash
#SBATCH --job-name=run_llama_job
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G 
#SBATCH --output=./logs/run_llama/%j.out
#SBATCH --error=./logs/run_llama/%j.err

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

python run_llama.py \
  --model_name "$MODEL_NAME" \
  --dataset "mmlu" \
  --max_retries 1 \
  --question_type "opinion_only" \
  --prefix_type "" \
  --prefix_subtype "" \
  --academic_level "" \
  --input_filename "prefix_full_question/mmlu/mmlu_opinion_only.pkl" \
  --full_question_column "full_question" 

deactivate

echo "Run completed."