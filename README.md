# LLM Sycophancy

## How to Use (Ibex as an example)

1. Submit the job:

```bash
sbatch run_llama.slurm
```

2. (Optional) To specify a different model:

```bash
sbatch run_llama.slurm "your_model_name"
```
Defaults to `meta-llama/Llama-3.2-3B` if no model is provided.

3. You can also **change parameters directly** inside `run_llama.slurm` (e.g., model name, input file, output directory).

## Requirements

- SLURM scheduler with A100 GPU access (if not, feel free changing it inside `run_llama.slurm`)
- Conda environment `syco` with required Python packages

## Outputs

- Logs saved in `./logs/run_llama/` (`<JOB_ID>.out` and `<JOB_ID>.err`).
- Outputs saved inside the output directory (depends on your parameters wrote in the `.slurm`)

## Monitoring Jobs

```bash
squeue -u your_username  # check jobs
scancel <JOB_ID>         # cancel a job
```
---

# Summary

| Action | Command |
|:---|:---|
| Submit job | `sbatch run_llama.slurm` |
| Specify HF model | `sbatch run_llama.slurm "your_model_name_on_hugging_face" ` |
| Edit job settings | Modify `run_llama.slurm` |

# 🌳 Hierarchy Tree for Parameter Selection
```
📋 run_llama.py Parameters
├── 🚀 --model_name (Required)
│   ├── Purpose: Specifies the LLaMA model for inference.
│   ├── Options: Any valid Hugging Face model name (e.g., `meta-llama/Llama-3.2-3B`).
│   ├── Default: `meta-llama/Llama-3.2-3B`
│   └── 📝 Note: Requires a valid Hugging Face token (`HF_TOKEN`) for access.
│
├── 📊 --dataset (Required)
│   ├── Purpose: Defines the dataset to process.
│   ├── Options: [`mmlu`] (only MMLU supported currently).
│   ├── Default: `mmlu`
│   └── 📝 Note: Sets the base output directory (e.g., `output/mmlu/...`).
│
├── 📂 --input_filename (Required)
│   ├── Purpose: Path to the input `.pkl` file with pre-constructed questions.
│   ├── Default: `output/mmlu/mmlu_plain.pkl`
│   └── 📝 Note: Must contain a column with question text (set via `--full_question_column`).
│
├── 📝 --full_question_column (Required)
│   ├── Purpose: Name of the column in the `.pkl` file containing question text.
│   ├── Default: `full_question`
│   └── 📝 Note: Customize if your dataset uses a different column name; ensure it exists.
│
├── ❓ --question_type (Required)
│   ├── Purpose: Defines the style of questions to process.
│   ├── Options: [`prefix_and_opinion`, `opinion_only`, `plain`]
│   ├── Default: `plain`
│   ├── Details:
│   │   ├─ `prefix_and_opinion`: Questions include a prefix and opinion (requires `--prefix_type`).
│   │   ├─ `opinion_only`: Questions focus solely on opinion, no prefix.
│   │   └─ `plain`: Questions have no prefix or opinion.
│   └── Conditional Parameters:
│       ├── 🛑 If `question_type = prefix_and_opinion`:
│       │   └── 🔧 --prefix_type (Required)
│       │       ├── Purpose: Specifies the type of prefix for questions.
│       │       ├── Options: [`academic`, `behavior`]
│       │       ├── Default: None (must be set for `prefix_and_opinion`).
│       │       ├── Details:
│       │       │   ├─ `academic`: Prefixes for academic contexts (requires `--academic_level`).
│       │       │   └─ `behavior`: Prefixes for behavioral contexts.
│       │       └── Conditional Parameters:
│       │           ├── 📚 --prefix_subtype (Required)
│       │           │   ├── Purpose: Defines the subtype of the prefix.
│       │           │   ├── Options: [`original`, `mixing_subject`, `third_pov`]
│       │           │   ├── Default: `original`
│       │           │   └── Details:
│       │           │       ├─ `original`: Standard prefix format.
│       │           │       ├─ `mixing_subject`: Prefix mixes subjects for complexity.
│       │           │       └─ `third_pov`: Prefix uses third-person perspective.
│       │           │
│       │           └── 🛑 If `prefix_type = academic`:
│       │               └── 🎓 --academic_level (Required)
│       │                   ├── Purpose: Sets the academic difficulty of the prefix.
│       │                   ├── Options: [`beginner`, `intermediate`, `advanced`]
│       │                   ├── Default: `beginner`
│       │                   └── 📝 Note: Only used when `prefix_type` is `academic`.
│       │
│       ├── ✅ If `question_type = opinion_only`:
│       │   └── No additional parameters needed.
│       │       📝 Note: Ignores `--prefix_type`, `--prefix_subtype`, and `--academic_level`.
│       │
│       └── ✅ If `question_type = plain`:
│           └── No additional parameters needed.
│               📝 Note: Ignores `--prefix_type`, `--prefix_subtype`, and `--academic_level`.
│
└── 🔄 --max_retries (Optional)
    ├── Purpose: Number of retries for questions with invalid answers.
    ├── Default: `3`
    └── 📝 Note: Increase for robustness (e.g., flaky model outputs); higher values extend runtime.
```
