import json
import re
from transformers import AutoTokenizer
from datasets import load_dataset

def preprocess_awq_calibration_data(dataset_path, output_file, model_id, max_samples=None):
    """
    Preprocess calibration data for AWQ quantization by applying chat template
    
    Args:
        dataset_path: Path or name of the dataset to load
        output_file: Path to output JSONL file  
        model_id: Model ID for tokenizer
        subfolder: Optional subfolder for tokenizer
        max_samples: Optional limit on number of samples
    """
    
    # Load tokenizer (same as your fine-tuned model)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        trust_remote_code=True, 
    )
    
    # Load your calibration dataset
    # Replace with your actual dataset loading method
    ds = load_dataset(dataset_path)
    
    # Use train split or adjust as needed
    data = ds["train"] if "train" in ds else ds["validation"] if "validation" in ds else list(ds.values())[0]
    
    if max_samples:
        data = data.select(range(min(max_samples, len(data))))
    
    # Regex to remove think tags (consistent with your training)
    THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)
    
    processed_samples = []
    
    for i, sample in enumerate(data):
        try:
            # Convert to messages format (same as your training preprocessing)
            user_content = sample["instruction"]
            if sample.get("context"):
                user_content += "\n\n" + sample["context"]
            
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": sample["response"]},
            ]
            
            # Apply chat template with thinking disabled (consistent with your training)
            formatted_text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False, 
                enable_thinking=False
            )
            
            # Strip any remaining think tags (safety measure)
            if "<think>" in formatted_text:
                formatted_text = THINK_RE.sub("", formatted_text)
            
            # Create sample for AWQ calibration
            processed_sample = {
                "text": formatted_text,
                "group_id": sample.get("group_id"),
                "category": sample.get("category"),
                "lang": sample.get("lang")
            }
            
            processed_samples.append(processed_sample)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} samples...")
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    # Save as JSONL (one sample per line)
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in processed_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(processed_samples)} processed samples to {output_file}")
    
    # Print a sample for verification
    if processed_samples:
        print("\nSample processed text:")
        print("-" * 50)
        print(processed_samples[0]["text"][:500] + "..." if len(processed_samples[0]["text"]) > 500 else processed_samples[0]["text"])
        print("-" * 50)

# Example usage
if __name__ == "__main__":
    # Configure these paths according to your setup
    DATASET_PATH = "jmcinern/Instruction_Ga_En_for_LoRA"  # Replace with your dataset
    OUTPUT_FILE = "awq_calibration_data.jsonl"
    MODEL_ID = "Qwen/Qwen3-8B" # for tokenizer  
    MAX_SAMPLES = 1000  # Typically 128-1000 samples is enough for AWQ calibration
    
    preprocess_awq_calibration_data(
        dataset_path=DATASET_PATH,
        output_file=OUTPUT_FILE,
        model_id=MODEL_ID,
        max_samples=MAX_SAMPLES
    )

# Alternative: If you have data as a list of dictionaries
def preprocess_from_dict_list(data_list, output_file, tokenizer):
    """
    If you have your calibration data as a list of dicts (like your sample)
    """
    THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)
    
    processed_samples = []
    
    for i, sample in enumerate(data_list):
        user_content = sample["instruction"]
        if sample.get("context"):
            user_content += "\n\n" + sample["context"]
        
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": sample["response"]},
        ]
        
        formatted_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False, 
            enable_thinking=False
        )
        
        if "<think>" in formatted_text:
            formatted_text = THINK_RE.sub("", formatted_text)
        
        processed_sample = {
            "text": formatted_text,
            "group_id": sample.get("group_id"),
            "category": sample.get("category"),
            "lang": sample.get("lang")
        }
        
        processed_samples.append(processed_sample)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in processed_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(processed_samples)} samples to {output_file}")
    return processed_samples