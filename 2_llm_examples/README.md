# Building a Small RAG Chatbot in Hugging Face Spaces

## Goal

Build a simple chatbot with lightweight RAG capabilities that can run in a Hugging Face Space.

The desired chatbot should:

- Use a Gradio chat interface
- Load a small set of local documents
- Retrieve relevant document chunks using vector search
- Send retrieved context to a language model
- Generate answers grounded in the documents
- Avoid unnecessary external services where possible

---

## Basic RAG Architecture

```text
User
  |
  v
Gradio Chat UI
  |
  v
User Question
  |
  +--> Embedding Model
  |        |
  |        v
  |   Vector Search
  |   FAISS
  |
  +--> Retrieved Document Chunks
           |
           v
      Prompt Builder
           |
           v
         LLM
           |
           v
       Answer
           |
           v
         User
```

## Recommended Hugging Face Space Structure

```text
├── app.py
├── requirements.txt
├── docs
│   ├── faq.txt
│   └── handbook.txt
└── README.md
```

## Initial RAG stack
| Component     | Tool                 |
| ------------- | -------------------- |
| UI            | Gradio               |
| Embeddings    | SentenceTransformers |
| Vector Search | FAISS                |
| LLM           | Hugging Face model   |
| Hosting       | Hugging Face Spaces  |

## Models That Do Not Require Special Connectivity or Authentication
To avoid API calls, the model should run locally inside the Hugging Face Space using transformers.
Recommended local public models included:
| Model                                 |  Size | Notes                                         |
| ------------------------------------- | ----: | --------------------------------------------- |
| `Qwen/Qwen2.5-1.5B-Instruct`          |  1.5B | Best practical quality for small local RAG    |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0`  |  1.1B | Faster, weaker than Qwen                      |
| `HuggingFaceTB/SmolLM2-360M-Instruct` |  360M | Fastest, weaker answers                       |
| `distilgpt2`                          | Small | Useful only for testing public model download |

## What FAISS Is
FAISS stands for: Facebook AI Similarity Search

It is used to quickly find vectors that are closest to another vector.
In the chatbot, FAISS acts as the vector search engine.

```text
Documents
   |
   v
Split into chunks
   |
   v
Convert chunks into embeddings
   |
   v
Store embeddings in FAISS
   |
   v
User asks a question
   |
   v
Convert question into embedding
   |
   v
FAISS finds most similar chunks
   |
   v
Send those chunks to the LLM
   |
   v
LLM answers using retrieved context
```

FAISS does not generate answers. It only retrieves the most relevant chunks.

The LLM then uses those chunks to generate the final response.
