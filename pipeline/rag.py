from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
import os
from dotenv import load_dotenv
import os

print("rag.py loaded")
print("RUNNING FILE:", os.path.abspath(__file__))

print("Starting RAG...")

# Load environment variables
load_dotenv()

# Check API key
groq_key = os.getenv("GROQ_API_KEY")

if not groq_key:
    raise ValueError(
        "GROQ_API_KEY not found. Check your .env file."
    )

print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded.")

print("MODEL BEING USED:", "llama-3.1-8b-instant")
print("Creating Groq client...")

groq_client = Groq(api_key=groq_key)
print("Groq client created.")

print("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_or_create_collection(
    name="market_intelligence",
    metadata={"hnsw:space": "cosine"}
)

print(f"Connected. Articles in ChromaDB: {collection.count()}")


def retrieve_context(query: str, n_results: int = 10):
    """Retrieve most relevant articles."""

    print("Generating query embedding...")
    query_embedding = embedding_model.encode(query).tolist()

    print("Searching ChromaDB...")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    context_chunks = []

    if not results["documents"]:
        return []

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        context_chunks.append(
            {
                "text": doc,
                "metadata": meta,
                "relevance": round(1 - dist, 4)
            }
        )

    print(f"Retrieved {len(context_chunks)} articles.")
    return context_chunks


def ask(query: str):
    """RAG Query"""

    print("\n" + "=" * 60)
    print(f"QUERY: {query}")
    print("=" * 60)

    context_chunks = retrieve_context(query)

    if not context_chunks:
        print("No relevant articles found.")
        return

    context = "\n\n---\n\n".join(
        [
            f"Article (relevance: {c['relevance']}):\n{c['text']}"
            for c in context_chunks
        ]
    )

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

    print("Sending request to Groq...")

    response = groq_client.chat.completions.create(
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

    answer = response.choices[0].message.content

    print("\nANSWER:")
    print(answer)

    return answer


if __name__ == "__main__":
    print("\nRunning test queries...\n")

    ask(
        "What is the current sentiment around artificial intelligence companies?"
    )

    ask(
        "What are the latest developments in startup funding?"
    )