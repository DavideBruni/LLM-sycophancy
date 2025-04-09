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
- Outputs saved inside the output directory you specified.

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
