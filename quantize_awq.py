from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import os
import logging

# to log awq progress
logging.basicConfig(level=logging.INFO)


# Paths
folder = "jmcinern/qwen3-8B-cpt-sft/"   
SFT_subfolder = "qwen3-8B-cpt-sft-full"
local_quant_path = "./8B-sft-full-awq-quant"        
repo_id = "jmcinern/qwen3-8B-sft-awq"   
hf_token = os.environ["HF_TOKEN"]


# Load model + tokenizer
model = AutoAWQForCausalLM.from_pretrained(folder, subfolder= SFT_subfolder, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(folder, subfolder= SFT_subfolder, trust_remote_code=True)

# data is a mix of Irish and English sentences split by \n
with open("calibration_mix.txt", "r", encoding="utf-8") as f:
    ga_en_calib_data = f.read().splitlines()
    # filter by > 30 chars
    ga_en_calib_data = [line for line in ga_en_calib_data if len(line) > 30]
print(f"Calibration sentences: {len(ga_en_calib_data)}")
quant_config = {
    "w_bit": 4,
    "q_group_size": 128,
    "zero_point": True,
    "duo_scaling": True,
    "max_calib_samples": 256,
    "max_calib_seq_len": 512
}
# Run AWQ quantization (defaults: 4-bit, group size 128)
model.quantize(tokenizer, 
               calib_data = ga_en_calib_data,
               quant_config=quant_config,
               )

# save quantized model to huggingface
model.save_quantized(local_quant_path, safetensors=True)
tokenizer.save_pretrained(local_quant_path)

# push to hf
model.push_to_hub(repo_id, token=hf_token)
tokenizer.push_to_hub(repo_id, token=hf_token)