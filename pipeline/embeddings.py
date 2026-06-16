from sentence_transformers import SentenceTransformer
from db.database import get_engine
import pandas as pd
import chromadb
import os


# Load embedding model (cached after first download)
print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded.")

# Initialize ChromaDB (persistent local vector store)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="market_intelligence",
    metadata={"hnsw:space": "cosine"}
)


def get_articles_for_embedding() -> pd.DataFrame:
    """Fetch articles not yet embedded."""
    engine = get_engine()

    # Get IDs already in ChromaDB
    existing_ids = set(collection.get()["ids"])

    query = """
        SELECT id, article_id, title, description, content,
               source_name, category, published_at, sentiment_label, sentiment_score
        FROM articles
        WHERE is_processed = TRUE
    """
    df = pd.read_sql(query, engine)

    # Filter out already embedded articles
    df = df[~df["article_id"].isin(existing_ids)]
    print(f"Found {len(df)} new articles to embed.")
    return df


def embed_and_store(df: pd.DataFrame):
    """Generate embeddings and store in ChromaDB."""
    if df.empty:
        print("No new articles to embed.")
        return

    documents = []
    embeddings = []
    ids = []
    metadatas = []

    for _, row in df.iterrows():
        # Combine fields for richer embedding context
        text = f"""
        Title: {row['title']}
        Summary: {row['description'] or ''}
        Content: {row['content'] or ''}
        Source: {row['source_name']}
        Category: {row['category']}
        Sentiment: {row['sentiment_label']} ({row['sentiment_score']})
        """.strip()

        embedding = embedding_model.encode(text).tolist()

        documents.append(text)
        embeddings.append(embedding)
        ids.append(str(row["article_id"]))
        metadatas.append({
            "title": str(row["title"]),
            "source": str(row["source_name"]),
            "category": str(row["category"]),
            "sentiment": str(row["sentiment_label"]),
            "score": float(row["sentiment_score"]),
            "published_at": str(row["published_at"])
        })

    # Store in ChromaDB in batches of 50
    batch_size = 50
    for i in range(0, len(documents), batch_size):
        collection.add(
            documents=documents[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            ids=ids[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )
        print(f"Embedded {min(i+batch_size, len(documents))}/{len(documents)} articles...")

    print(f"Embedding complete. {len(documents)} articles stored in ChromaDB.")


def run_embedding_pipeline():
    """Full embedding pipeline."""
    df = get_articles_for_embedding()
    embed_and_store(df)


if __name__ == "__main__":
    run_embedding_pipeline()