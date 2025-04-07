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

python run_llama.py \
  --model_name "$MODEL_NAME" \
  --prefix_type "" \
  --input_filename "output/mmlupro/mmlupro_no_prefix.pkl" \
  --output_dir "output/mmlupro"

deactivate

echo "Run completed."