import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login, HfApi
import torch

def push_to_hub():
    # Configuration
    local_model_dir = "./quantized_model_awq"
    hub_repo_name = "jmcinern/qwen3-8B-cpt-sft-awq"  # Your new repo name
    
    # Login to HuggingFace (you'll need a token)
    # Get your token from: https://huggingface.co/settings/tokens
    hf_token = input("Enter your HuggingFace token: ")
    login(token=hf_token)
    
    print("Logged in to HuggingFace successfully!")
    
    # Check if the quantized model exists
    if not os.path.exists(local_model_dir):
        print(f"Error: {local_model_dir} not found. Run quantization first!")
        return
    
    print(f"Loading quantized model from {local_model_dir}...")
    
    # Load the quantized model and tokenizer
    try:
        model = AutoModelForCausalLM.from_pretrained(
            local_model_dir,
            torch_dtype=torch.float16,
            device_map="cpu",  # Load on CPU for uploading
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            local_model_dir,
            trust_remote_code=True
        )
        print("✓ Model and tokenizer loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create repository on Hub if it doesn't exist
    api = HfApi()
    try:
        api.create_repo(
            repo_id=hub_repo_name,
            repo_type="model",
            exist_ok=True,  # Don't error if repo already exists
            private=False   # Set to True if you want a private repo
        )
        print(f"✓ Repository {hub_repo_name} ready")
    except Exception as e:
        print(f"Repository creation info: {e}")
    
    # Push model to hub
    print("Pushing model to HuggingFace Hub...")
    try:
        model.push_to_hub(
            repo_id=hub_repo_name,
            commit_message="Add AWQ quantized model",
            safe_serialization=True
        )
        print("✓ Model pushed successfully")
    except Exception as e:
        print(f"Error pushing model: {e}")
        return
    
    # Push tokenizer to hub
    print("Pushing tokenizer to HuggingFace Hub...")
    try:
        tokenizer.push_to_hub(
            repo_id=hub_repo_name,
            commit_message="Add tokenizer for AWQ quantized model"
        )
        print("✓ Tokenizer pushed successfully")
    except Exception as e:
        print(f"Error pushing tokenizer: {e}")
        return
    
    # Create and upload README
    readme_content = f"""---
license: apache-2.0
base_model: jmcinern/qwen3-8B-cpt-sft
quantized_by: llmcompressor
quantization_method: awq
quantization_config:
  bits: 4
  group_size: 128
  scheme: W4A16
---

# Qwen3-8B-CPT-SFT AWQ Quantized

This is an AWQ quantized version of `jmcinern/qwen3-8B-cpt-sft/qwen3-8B-cpt-sft-full`.

## Quantization Details
- **Method**: AWQ (Activation-aware Weight Quantization)
- **Precision**: 4-bit weights, 16-bit activations (W4A16)
- **Quantization Tool**: llmcompressor
- **Model Size**: ~2GB (reduced from ~15GB)

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "{hub_repo_name}"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Generate text
inputs = tokenizer("Hello, how are you?", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50, do_sample=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Original Model
- Base model: [jmcinern/qwen3-8B-cpt-sft](https://huggingface.co/jmcinern/qwen3-8B-cpt-sft)
"""
    
    try:
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=hub_repo_name,
            commit_message="Add README"
        )
        print("✓ README uploaded successfully")
    except Exception as e:
        print(f"Error uploading README: {e}")
    
    print(f"\n🎉 Model successfully pushed to: https://huggingface.co/{hub_repo_name}")
    print(f"You can now use it with: AutoModelForCausalLM.from_pretrained('{hub_repo_name}')")

if __name__ == "__main__":
    push_to_hub()