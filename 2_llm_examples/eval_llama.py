import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

# Define the absolute path to your downloaded folder
model_path = os.path.abspath("Llama-3.1-8B-Instruct")

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# 1. Clear memory before loading
if torch.backends.mps.is_available():
    torch.mps.empty_cache()

# 2. Load the model WITHOUT device_map="auto"
# We use float16 because it's better supported on Mac Metal than bfloat16
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    local_files_only=True
).to(device) # Force it to the GPU here

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

# 3. Create the pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Confirm you are running from local files."},
]

outputs = pipe(messages, max_new_tokens=100, max_length=2048)
print(outputs[0]["generated_text"][-1])

