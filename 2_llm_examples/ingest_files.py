import os
import fitz  # PyMuPDF
import chromadb

# Setup ChromaDB client
client = chromadb.PersistentClient(path="./my_local_rag_db")
collection = client.get_or_create_collection(name="knowledge_base")

def chunk_text(text, size=1000, overlap=100):
    """Simple sliding window chunking."""
    return [text[i:i + size] for i in range(0, len(text), size - overlap)]

def ingest_files(directory_path):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        text = ""

        # Extract text based on file type
        if filename.endswith(".pdf"):
            with fitz.open(file_path) as doc:
                text = " ".join([page.get_text() for page in doc])
        elif filename.endswith(".txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

        if text:
            # Chunk the text to fit within model limits
            chunks = chunk_text(text)
            ids = [f"{filename}_{i}" for i in range(len(chunks))]
            metadatas = [{"source": filename} for _ in range(len(chunks))]

            # Add to ChromaDB in batches
            collection.add(documents=chunks, ids=ids, metadatas=metadatas)
            print(f"Ingested {len(chunks)} chunks from {filename}")

# Run ingestion for a 'docs' folder
ingest_files("./my_documents")

