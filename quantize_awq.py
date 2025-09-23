import os, json, torch
from datasets import Dataset
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor import oneshot  # corrected import

def load_calibration_data(fp):
    return [json.loads(l)["text"] for l in open(fp, "r", encoding="utf-8")]

def main():
    repo = "jmcinern/qwen3-8B-cpt-sft"
    sub  = "qwen3-8B-cpt-sft-full"
    calibration_file = "./awq_calibration_data.jsonl"
    output_dir = "./quantized_model_awq"
    cache_dir = os.path.expanduser("~/.cache/hf_qwen")  # your custom cache

    # 1) Ensure subfolder is cached (first run downloads once; then reuse)
    snap = snapshot_download(
        repo_id=repo,
        allow_patterns=f"{sub}/*",
        cache_dir=cache_dir,
        local_files_only=False,   # set True if you know it's already cached
        revision="main",
    )
    local_sub = os.path.join(snap, sub)

    # 2) Load tokenizer/model from LOCAL path; force offline reuse
    tokenizer = AutoTokenizer.from_pretrained(local_sub, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        local_sub,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )
    # Prevent hub lookups during save
    model.config._name_or_path = local_sub

    # 3) Dataset + recipe
    calibration_texts = load_calibration_data(calibration_file)
    calibration_dataset = Dataset.from_dict({"text": calibration_texts})
    recipe = """
default_stage:
default_modifiers:
  AWQModifier:
    targets: [Linear]
    ignore: [lm_head]
    scheme: W4A16
"""

    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=calibration_dataset,
        recipe=recipe,
        output_dir=output_dir,
        num_calibration_samples=min(256, len(calibration_texts)),
        max_seq_length=2048,
    )

if __name__ == "__main__":
    main()
