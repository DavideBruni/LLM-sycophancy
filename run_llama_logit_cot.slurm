#!/bin/bash
#SBATCH --job-name=run_llama_logit_cot_job_v100
#SBATCH --time=00:59:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=256G
#SBATCH --output=./logs/run_llama_logit_cot_v100/%j.out
#SBATCH --error=./logs/run_llama_logit_cot_v100/%j.err

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
  --dataset "mmlu" \
  --max_retries 1 \
  --question_type "opinion_only" \
  --prefix_type "" \
  --prefix_subtype "" \
  --academic_level "" \
  --input_filename "prefix_full_question/mmlu/mmlu_opinion_only.pkl" \
  --full_question_column "full_question" \
  --inference_mode "logit_only" 

deactivate
echo "Run completed."