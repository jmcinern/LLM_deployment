from huggingface_hub import HfApi, CommitOperationCopy

api = HfApi()
src = "jmcinern/qwen3-8B-cpt-sft"
dst_full = "jmcinern/qwen3-8B-cpt-sft-full"
dst_lora = "jmcinern/qwen3-8B-cpt-sft-lora"

# 1) create target repos
api.create_repo(dst_full, repo_type="model", exist_ok=True)
api.create_repo(dst_lora, repo_type="model", exist_ok=True)

# 2) copy each subfolder to target repo ROOT
files = api.list_repo_files(src, repo_type="model", revision="main")
for sub, dst in [("qwen3-8B-cpt-sft-full/", dst_full),
                 ("qwen3-8B-cpt-sft-lora/", dst_lora)]:
    ops = [CommitOperationCopy(src_repo=src, src_revision="main",
                               src_path=f, path_in_repo=f[len(sub):])
           for f in files if f.startswith(sub)]
    api.create_commit(repo_id=dst, repo_type="model",
                      operations=ops, commit_message=f"Import {sub} to root")

# 3) optional: delete old repo
# api.delete_repo(src, repo_type="model")
