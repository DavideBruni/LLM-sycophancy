#!/bin/bash -l
#SBATCH --job-name=run_early_decoding
#SBATCH --partition=yamadau
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --output=./logs/run_early_decoding/%j.out

set -e
source ~/miniforge3/etc/profile.d/conda.sh
conda activate syco

DEFAULT_MODEL="Qwen/Qwen2.5-1.5B-Instruct"

if [[ -n "$1" ]]; then
  MODEL_NAME="$1"
else
  MODEL_NAME="$DEFAULT_MODEL"
fi

echo "----------------------------------------------------"
echo "Starting LLM sycophancy analysis job..."
echo "Job ID: ${SLURM_JOB_ID}"
echo "Run Time: $(date)"
echo "Model: ${MODEL_NAME}"
echo "----------------------------------------------------"

python run_early_decoding_qwen.py \
  --model_name "$MODEL_NAME" \
  --dataset "mmlu" \
  --max_retries 1 \
  --question_type "plain" \
  --prefix_type "" \
  --prefix_subtype "" \
  --academic_level "" \
  --input_filename "prefix_full_question/mmlu/mmlu_plain.pkl" \
  --full_question_column "full_question" \
  --inference_mode "logit_only" \
  --inference_layer "all"

deactivate

echo "Run completed."