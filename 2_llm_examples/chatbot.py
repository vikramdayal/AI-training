from mlx_lm import load, generate

# 1. Setup - Using the 4-bit model for speed on 16GB RAM
model_id = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
model, tokenizer = load(model_id)

# 2. Initialize history with a System Prompt
messages = [
    {"role": "system", "content": "You are a helpful, concise AI assistant."}
]

print("--- Llama 3.1 Chatbot (Type 'exit' to quit) ---")

while True:
    # Get user input
    user_input = input("\nUser: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    # Add user message to history
    messages.append({"role": "user", "content": user_input})

    # 3. Format history into the Llama 3.1 chat template
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    # 4. Generate response
    # 'verbose=True' streams the text to the terminal as it generates
    print("\nAssistant: ", end="", flush=True)
    response = generate(
        model, 
        tokenizer, 
        prompt=prompt, 
        max_tokens=500,
        verbose=True 
    )

    # 5. Save the assistant's reply to history for the next turn
    messages.append({"role": "assistant", "content": response})

