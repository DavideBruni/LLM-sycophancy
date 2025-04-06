#!/bin/bash
#SBATCH --job-name=run_llama_job
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --output=./logs/run_llama/%j.out
#SBATCH --error=./logs/run_llama/%j.err

set -e

# If needed, change directory (optional)
# cd path/to/your/script

# Just run your Python script
python run_llama.py

echo "Run completed."
