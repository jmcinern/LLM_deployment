from huggingface_hub import HfApi, HfFileSystem

api = HfApi()
fs = HfFileSystem()  # uses your HF token from `huggingface-cli login`

src = "jmcinern/qwen3-8B-cpt-sft"
dst_full = "jmcinern/qwen3-8B-cpt-sft-full"
dst_lora = "jmcinern/qwen3-8B-cpt-sft-lora"

# 1) Ensure targets exist
api.create_repo(dst_full, repo_type="model", exist_ok=True)
api.create_repo(dst_lora, repo_type="model", exist_ok=True)

# 2) List and copy each subfolder's files to the new repo ROOT
files = api.list_repo_files(src, repo_type="model", revision="main")

def copy_subfolder(sub, dst):
    for f in files:
        if f.startswith(sub) and not f.endswith("/"):
            rel = f[len(sub):]                      # strip subfolder prefix
            fs.cp_file(f"{src}/{f}", f"{dst}/{rel}", revision="main")

copy_subfolder("qwen3-8B-cpt-sft-full/", dst_full)
copy_subfolder("qwen3-8B-cpt-sft-lora/", dst_lora)
