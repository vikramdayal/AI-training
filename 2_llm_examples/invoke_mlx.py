from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Confirm you are running from local files."},
]

# 1. Apply the chat template to format the string for Llama 3.1
prompt = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)

# 2. Pass the formatted string to generate
response = generate(
    model, 
    tokenizer, 
    prompt=prompt, 
    max_tokens=500,
    verbose=True  # This shows the output in real-time
)
print()
print("*******************************")
print(response)

