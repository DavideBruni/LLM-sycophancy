from huggingface_hub import snapshot_download
import config
snapshot_download(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",  # or your target model
    local_dir="./modelhub/meta-llama/Llama-3.1-8B-Instruct",         
    resume_download=True,
    local_dir_use_symlinks=False,
    token=config.HF_TOKEN,  
)
