#!/bin/bash
#SBATCH --job-name=run_llama_job_8b
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
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

# set huggingface cache dir
export HF_HOME="/ibex/user/wangk0e/cache/huggingface"

DIFFICULTY="intermediate"
INPUT_FILENAME="output/mmlu/mmlu_academic_opinion_${DIFFICULTY}.pkl"

python run_llama.py \
  --model_name "$MODEL_NAME" \
  --difficulty "$DIFFICULTY" \
  --prefix_type "academic" \
  --question_style "prefix_and_opinion" \
  --input_filename "$INPUT_FILENAME" \
  --output_dir "output/mmlu" \
  --max_retries 1

deactivate

echo "Run completed."