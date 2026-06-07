import chromadb
from mlx_lm import load, generate

# 1. Setup MLX and ChromaDB
model_id = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
model, tokenizer = load(model_id)

db_client = chromadb.PersistentClient(path="./my_local_rag_db")
collection = db_client.get_or_create_collection(name="knowledge_base")

def get_rag_content(query, n_results=2):
    """Search ChromaDB and return both text and source metadata."""
    results = collection.query(query_texts=[query], n_results=n_results)
    
    context_text = ""
    sources = []
    
    if results['documents']:
        context_text = " ".join(results['documents'][0])
        # Extract the 'source' filename from metadata
        sources = [meta.get('source', 'Unknown') for meta in results['metadatas'][0]]
    
    return context_text, list(set(sources)) # Return unique sources

print("--- RAG Chatbot with Sources (Type 'exit' to quit) ---")

while True:
    user_input = input("\nUser: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    # 2. Retrieval Step
    relevant_context, source_list = get_rag_content(user_input)

    # 3. Augmenting the System Prompt
    messages = [
        {
            "role": "system", 
            "content": f"Use the provided context to answer. Context: {relevant_context}"
        },
        {"role": "user", "content": user_input}
    ]

    # 4. Generate
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print("\nAssistant: ", end="", flush=True)
    response = generate(model, tokenizer, prompt=prompt, verbose=True)

    # 5. Print Sources
    if source_list:
        print(f"\n\n[Sources: {', '.join(source_list)}]")
    else:
        print("\n\n[No local sources found for this query.]")

