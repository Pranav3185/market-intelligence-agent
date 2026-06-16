from transformers import pipeline
from sqlalchemy import text
from db.database import get_engine
import pandas as pd

# ── Lazy global ────────────────────────────────────────────────────────────────
_sentiment_pipeline = None


def get_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        print("Loading sentiment model...")
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            max_length=512
        )
        print("Sentiment model loaded.")
    return _sentiment_pipeline


# ── Core functions ─────────────────────────────────────────────────────────────
def get_unprocessed_articles() -> pd.DataFrame:
    """Fetch articles that haven't been sentiment-analyzed yet."""
    engine = get_engine()

    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT id, title, description, content
            FROM articles
            WHERE is_processed = FALSE
        """))
        rows = result.fetchall()

    df = pd.DataFrame(rows, columns=["id", "title", "description", "content"])
    print(f"Found {len(df)} unprocessed articles.")
    return df


def analyze_sentiment(text_input: str) -> dict:
    """Run sentiment analysis on a piece of text."""
    result = get_sentiment_pipeline()(text_input[:512])[0]
    label = result["label"]
    score = result["score"]

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