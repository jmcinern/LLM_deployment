import json
from datasets import Dataset
from llmcompressor import oneshot
from transformers import AutoTokenizer, AutoModelForCausalLM  
import torch

def load_calibration_data(file_path):
    """Load calibration data from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item['text'])
    return data

def main():
    # Configuration
    model_id = "jmcinern/qwen3-8B-cpt-sft"
    model_subfolder = "qwen3-8B-cpt-sft-full"
    calibration_file = "./awq_calibration_data.jsonl"
    output_dir = "./quantized_model_awq"
    
    # Load calibration data
    print("Loading calibration data...")
    calibration_texts = load_calibration_data(calibration_file)
    print(f"Loaded {len(calibration_texts)} calibration samples")
    
    # Create dataset
    calibration_dataset = Dataset.from_dict({"text": calibration_texts})
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder=model_subfolder, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        subfolder=model_subfolder,    
        device_map="auto",
        torch_dtype=torch.float16,    
        trust_remote_code=True,
    )
    
    # CRITICAL FIX: Clear the problematic subfolder path for save_pretrained()
    # save_pretrained() expects a clean model name, not a path with subfolders
    model.config._name_or_path = output_dir  # Point to where it will be saved
    if hasattr(model, 'name_or_path'):
        model.name_or_path = output_dir
    
    print(f"Model path cleared for save_pretrained compatibility")
    
    # AWQ quantization recipe
    recipe = """
default_stage:
default_modifiers:
  AWQModifier:
    targets: [Linear]
    ignore: [lm_head]
    scheme: W4A16
"""
    
    # Apply quantization using oneshot
    print("Applying AWQ quantization...")
    oneshot(
        model=model,  
        dataset=calibration_dataset,
        recipe=recipe,
        tokenizer=tokenizer,
        output_dir=output_dir,
        num_calibration_samples=min(256, len(calibration_texts)),
        max_seq_length=2048
    )
    
    print("Quantization completed successfully!")

if __name__ == "__main__":
    main()