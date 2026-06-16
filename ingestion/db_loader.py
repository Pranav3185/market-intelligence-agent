import pandas as pd
from sqlalchemy import text
from db.database import get_engine
from datetime import datetime


def insert_articles(df: pd.DataFrame) -> dict:
    """
    Insert cleaned articles into PostgreSQL.
    Skips duplicates using article_id.
    Returns a summary dict.
    """
    if df.empty:
        print("No articles to insert.")
        return {"fetched": 0, "inserted": 0, "skipped": 0}

    engine = get_engine()
    inserted = 0
    skipped = 0

    with engine.connect() as conn:
        for _, row in df.iterrows():
            try:
                conn.execute(text("""
                    INSERT INTO articles (
                        article_id, title, description, content,
                        source_name, author, url, published_at,
                        category, is_processed
                    )
                    VALUES (
                        :article_id, :title, :description, :content,
                        :source_name, :author, :url, :published_at,
                        :category, :is_processed
                    )
                    ON CONFLICT (article_id) DO NOTHING
                """), {
                    "article_id": row["article_id"],
                    "title": row["title"],
                    "description": row["description"],
                    "content": row["content"],
                    "source_name": row["source_name"],
                    "author": row["author"],
                    "url": row["url"],
                    "published_at": row["published_at"],
                    "category": row["category"],
                    "is_processed": row["is_processed"]
                })
                inserted += 1

            except Exception as e:
                print(f"Error inserting article {row['article_id']}: {e}")
                skipped += 1

        conn.commit()

    print(f"Inserted: {inserted} | Skipped (duplicates): {skipped}")
    return {"fetched": len(df), "inserted": inserted, "skipped": skipped}


def log_pipeline_run(stats: dict, status: str = "success", error: str = None):
    """Log each ETL run into pipeline_logs table."""
    engine = get_engine()

    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO pipeline_logs (
                run_at, articles_fetched, articles_inserted, status, error_message
            )
            VALUES (
                :run_at, :fetched, :inserted, :status, :error
            )
        """), {
            "run_at": datetime.now(),
            "fetched": stats.get("fetched", 0),
            "inserted": stats.get("inserted", 0),
            "status": status,
            "error": error
        })
        conn.commit()

if __name__ == "__main__":
    print("db_loader imported successfully")    

    print(f"Pipeline run logged — status: {status}")