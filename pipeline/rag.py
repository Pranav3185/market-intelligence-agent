from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

# ── Lazy globals ───────────────────────────────────────────────────────────────
_embedding_model = None
_groq_client = None
_collection = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print("Loading embedding model...")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Embedding model loaded.")
    return _embedding_model


def get_groq_client():
    global _groq_client
    if _groq_client is None:
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("GROQ_API_KEY not found. Check your .env file.")
        _groq_client = Groq(api_key=groq_key)
    return _groq_client


def get_collection():
    global _collection
    if _collection is None:
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        _collection = chroma_client.get_or_create_collection(
            name="market_intelligence",
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Connected. Articles in ChromaDB: {_collection.count()}")
    return _collection


# ── Core functions ─────────────────────────────────────────────────────────────
def retrieve_context(query: str, n_results: int = 10):
    """Retrieve most relevant articles."""
    query_embedding = get_embedding_model().encode(query).tolist()

    results = get_collection().query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    if not results["documents"]:
        return []

    context_chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        context_chunks.append({
            "text": doc,
            "metadata": meta,
            "relevance": round(1 - dist, 4)
        })

    return context_chunks


def ask(query: str) -> str:
    """RAG Query — retrieve context then generate answer."""
    context_chunks = retrieve_context(query)

    if not context_chunks:
        return "No relevant articles found in the database."

    context = "\n\n---\n\n".join([
        f"Article (relevance: {c['relevance']}):\n{c['text']}"
        for c in context_chunks
    ])

    prompt = f"""
You are a Senior Market Intelligence Analyst.

Analyze the provided news articles and answer the user's question.

Requirements:
- Provide a detailed response (300-500 words when possible).
- Start with an Executive Summary.
- Explain the major themes and trends.
- Discuss positive and negative signals separately.
- Mention companies, sources, and developments when relevant.
- Use evidence from the articles.
- Do NOT simply count positive vs negative articles.
- Do NOT provide one-line answers.
- Write like a professional analyst report.

CONTEXT:
{context}

QUESTION:
{query}

Structure your response as:

## Executive Summary

## Key Trends

## Positive Signals

## Risks & Concerns

## Overall Assessment

Answer:
"""

    response = get_groq_client().chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are an expert Market Intelligence Analyst who creates concise executive reports from news and business data."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.4,
        max_tokens=1000
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    print("\nRunning test queries...\n")
    ask("What is the current sentiment around artificial intelligence companies?")
    ask("What are the latest developments in startup funding?")