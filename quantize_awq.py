# load calibration data from ./awq_calibration_data.jsonl which has the "text" field
# the model is on HF at: jmcinern/qwen3-8B-cpt-sft/qwen3-8B-cpt-sft-full
# use llmcompressor to quantize the model with awq using the calibration data
import json
from datasets import Dataset
from llmcompressor.transformers import SparseAutoModelForCausalLM
from transformers import AutoTokenizer
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
    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder=model_subfolder)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # AWQ quantization recipe
    recipe = """
    quant_stage:
      quant_modifiers:
        QuantizationModifier:
          ignore: ["lm_head"]
          config_groups:
            group_0:
              weights:
                num_bits: 4
                type: "int"
                symmetric: true
                strategy: "channel"
              input_activations:
                num_bits: 16
                type: "float"
              targets: ["Linear"]
    """
    
    # Load and quantize model
    print("Loading and quantizing model...")
    model = SparseAutoModelForCausalLM.from_pretrained(
        model_id,
        subfolder=model_subfolder,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # Apply quantization
    model.apply(
        recipe=recipe,
        dataloader=calibration_dataset,
        tokenizer=tokenizer,
        num_calibration_samples=min(256, len(calibration_texts)),
        max_seq_length=2048
    )
    
    # Save quantized model
    print(f"Saving quantized model to {output_dir}...")
    model.save_pretrained(output_dir, save_compressed=True)
    tokenizer.save_pretrained(output_dir)
    
    print("Quantization completed successfully!")

if __name__ == "__main__":
    main()