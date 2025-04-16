#!/bin/bash
#SBATCH --job-name=run_llama_job_8b
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G  # Increased memory to be safe
#SBATCH --output=./logs/run_llama/%j.out
#SBATCH --error=./logs/run_llama/%j.err

set -e

source ~/syco/bin/activate

# Default values
DEFAULT_MODEL="meta-llama/Llama-3.2-3B"
INPUT_FILENAME="prefix_full_question/mmlu_opinion_only.pkl"  # Updated to match your error message
FULL_QUESTION_COL="full_question"
DIFFICULTY=""


# Set huggingface cache dir
export HF_HOME="/ibex/user/wangk0e/cache/huggingface"

python run_llama.py \
  --model_name "$MODEL_NAME" \
  --prefix_type "academic" \
  --question_style "prefix_and_opinion" \
  --difficulty "$DIFFICULTY" \
  --input_filename "$INPUT_FILENAME" \
  --output_dir "output/mmlu" \
  --max_retries 1 \
  --full_question_col "$FULL_QUESTION_COL"

deactivate

echo "Run completed."