import re
import random

# Paths
en_path = "D:/VS-code-projects/CPT-Dáil/wikipedia_sample.txt"
ga_path = "D:/VS-code-projects/CPT-Dáil/data/NCI_ga.txt"

def load_sentences(path, n=1000):
    # read file
    with open(path, "r", encoding="utf-8") as f:
        text = " ".join([line.strip() for line in f if line.strip()])
    
    # regex-based sentence tokenizer
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # take first n
    return sentences[:n]

# load 1k sentences from each
en_calib_data = load_sentences(en_path, 1000)
ga_calib_data = load_sentences(ga_path, 1000)

# combine and shuffle
calib_data = en_calib_data + ga_calib_data
random.shuffle(calib_data)

print(f"Total sentences: {len(calib_data)}")
print(calib_data[:50])  # preview

# write to txt file
with open("calibration_mix.txt", "w", encoding="utf-8") as f:
    for line in calib_data:
        f.write(line + "\n")
