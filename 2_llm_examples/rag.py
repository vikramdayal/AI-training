import chromadb
from mlx_lm import load, generate

# 1. Setup MLX and ChromaDB
model_id = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
model, tokenizer = load(model_id)

# Connect to your local ChromaDB folder
db_client = chromadb.PersistentClient(path="./my_local_rag_db")
collection = db_client.get_or_create_collection(name="knowledge_base")

def get_context(query, n_results=2):
    """Search ChromaDB for the most relevant documents."""
    results = collection.query(query_texts=[query], n_results=n_results)
    # Flatten the list of documents into a single string
    return " ".join(results['documents'][0]) if results['documents'] else ""

print("--- RAG Chatbot Ready (Type 'exit' to quit) ---")

while True:
    user_input = input("\nUser: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    # 2. Retrieval Step
    relevant_context = get_context(user_input)

    # 3. Augmenting the System Prompt
    # We provide the facts directly to the model here
    messages = [
        {
            "role": "system", 
            "content": f"You are a helpful assistant. Use the following context to answer the user's question. If the answer isn't in the context, say you don't know.\n\nContext: {relevant_context}"
        },
        {"role": "user", "content": user_input}
    ]

    # 4. Format and Generate
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print("\nAssistant: ", end="", flush=True)
    response = generate(model, tokenizer, prompt=prompt, verbose=True)


