#!/bin/bash -l
#SBATCH --job-name=run_logit_llama
#SBATCH --partition=yamadau
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --output=./logs/run_logit_llama/%j.out

set -e

conda activate syco

DEFAULT_MODEL="meta-llama/Llama-3.1-8B-Instruct"

if [[ -n "$1" ]]; then
  MODEL_NAME="$1"
else
  MODEL_NAME="$DEFAULT_MODEL"
fi

python run_syco_logit_cot.py \
  --model_name "$MODEL_NAME" \
  --dataset "mmlu" \
  --max_retries 1 \
  --question_type "prefix_and_opinion" \
  --prefix_type "academic" \
  --prefix_subtype "third_pov" \
  --academic_level "advanced" \
  --input_filename "../../lib/pov/prefix/third_pov/mmlu_academic_opinion_advanced.pkl" \
  --full_question_column "full_question" \
  --inference_mode "logit_only" \
  --inference_layer "all"

deactivate
echo "Run completed."