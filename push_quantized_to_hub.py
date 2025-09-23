from huggingface_hub import HfApi

local_path = "./quantized_model_awq"
repo_id = "jmcinern/qwen3-8B-cpt-sft-awq"

print(f"Pushing quantized model files from {local_path} to the hub at {repo_id}...")

HfApi().upload_folder(
    folder_path=local_path,
    repo_id=repo_id,
    repo_type="model",
    commit_message="Upload AWQ files",
    ignore_patterns=[".DS_Store", "*.tmp", ".ipynb_checkpoints"]
)

print(f"Quantized model files pushed to the hub at {repo_id}")