from transformers import pipeline
from sqlalchemy import text
from db.database import get_engine
import pandas as pd

# Load HuggingFace sentiment model once (downloads on first run ~250MB)
print("Loading sentiment model...")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    truncation=True,
    max_length=512
)
print("Model loaded.")


def get_unprocessed_articles() -> pd.DataFrame:
    """Fetch articles that haven't been sentiment-analyzed yet."""
    engine = get_engine()
    query = """
        SELECT id, title, description, content
        FROM articles
        WHERE is_processed = FALSE
    """
    df = pd.read_sql(query, engine)
    print(f"Found {len(df)} unprocessed articles.")
    return df


def analyze_sentiment(text: str) -> dict:
    """Run sentiment analysis on a piece of text."""
    # Use title + description as input (content can be truncated by NewsAPI)
    result = sentiment_pipeline(text[:512])[0]
    label = result["label"]        # POSITIVE or NEGATIVE
    score = result["score"]        # confidence 0-1

    # Convert to -1 to +1 scale
    normalized_score = score if label == "POSITIVE" else -score

    return {
        "sentiment_label": label,
        "sentiment_score": round(normalized_score, 4)
    }


def process_articles():
    """Run sentiment on all unprocessed articles and update DB."""
    df = get_unprocessed_articles()

    if df.empty:
        print("No articles to process.")
        return

    engine = get_engine()
    processed = 0

    with engine.connect() as conn:
        for _, row in df.iterrows():
            try:
                # Combine title + description for better context
                text_input = f"{row['title']}. {row['description'] or ''}"
                sentiment = analyze_sentiment(text_input)

                conn.execute(text("""
                    UPDATE articles
                    SET
                        sentiment_score = :score,
                        sentiment_label = :label,
                        is_processed = TRUE
                    WHERE id = :id
                """), {
                    "score": sentiment["sentiment_score"],
                    "label": sentiment["sentiment_label"],
                    "id": row["id"]
                })

                processed += 1

                if processed % 10 == 0:
                    print(f"Processed {processed}/{len(df)} articles...")

            except Exception as e:
                print(f"Error processing article {row['id']}: {e}")

        conn.commit()

    print(f"\nSentiment analysis complete. Processed {processed} articles.")


if __name__ == "__main__":
    process_articles()