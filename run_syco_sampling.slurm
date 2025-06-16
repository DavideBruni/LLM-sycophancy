#!/bin/bash -l
#SBATCH --job-name=run_syco_sampling
#SBATCH --partition=yamadau
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --output=./logs/run_syco_sampling_llama/%j.out

echo "Start time: $(date)"
module purge
module load cuda/12.2


#module load cuda/11.3
#module load cuDNN/8.8.4-cuda11


source /home/k/keyu-wang/miniforge3/etc/profile.d/conda.sh
conda activate syco

# Set environment variables
export HF_TOKEN=$(python -c 'import config; print(config.HF_TOKEN)')
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
 
DEFAULT_MODEL="meta-llama/Llama-3.1-8B-Instruct"
#DEFAULT_MODEL="Qwen/Qwen2.5-7B-Instruct"

if [[ -n "$1" ]]; then
  MODEL_NAME="$1"
else
  MODEL_NAME="$DEFAULT_MODEL"
fi

python run_syco_sampling.py \
  --model_name "$MODEL_NAME" \
  --dataset "mmlu" \
  --max_retries 1 \
  --question_type "plain" \
  --prefix_type "" \
  --prefix_subtype "" \
  --academic_level "" \
  --input_filename "prefix_full_question/mmlu/mmlu_plain.pkl" \
  --full_question_column "full_question" \
  --sampling 30

conda deactivate

echo "End time: $(date)"
echo "Run completed."