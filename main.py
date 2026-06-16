import os
from db.database import init_db
from ingestion.news_fetcher import run_fetcher
from ingestion.db_loader import insert_articles, log_pipeline_run
from analytics.sentiment import process_articles
from analytics.entity_extractor import run_entity_extraction
from pipeline.embeddings import run_embedding_pipeline

def run_pipeline():
    print("=== Market Intelligence Pipeline ===\n")

    init_db()

    df = run_fetcher()
    stats = insert_articles(df)

    print("\n--- Sentiment Analysis ---")
    process_articles()

    print("\n--- Entity Extraction ---")
    run_entity_extraction()

    print("\n--- Embedding Pipeline ---")
    run_embedding_pipeline()

    log_pipeline_run(stats, status="success")

    print("\n=== Pipeline Complete ===")
    print(f"Fetched: {stats['fetched']} | Inserted: {stats['inserted']} | Skipped: {stats['skipped']}")


if __name__ == "__main__":
    run_pipeline()