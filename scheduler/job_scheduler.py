from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime
from db.database import init_db
from ingestion.news_fetcher import run_fetcher
from ingestion.db_loader import insert_articles, log_pipeline_run
from analytics.sentiment import process_articles
from pipeline.embeddings import run_embedding_pipeline
from analytics.entity_extractor import run_entity_extraction

scheduler = BlockingScheduler()


def run_pipeline():
    """Full pipeline run — fetch, clean, store, analyze."""
    print(f"\n{'='*45}")
    print(f"Pipeline started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*45}\n")

def run_pipeline():
    """Full pipeline run — fetch, clean, store, analyze."""
    print(f"\n{'='*45}")
    print(f"Pipeline started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*45}\n")

    try:
        df = run_fetcher()
        stats = insert_articles(df)

        process_articles()
        run_entity_extraction()
        run_embedding_pipeline()

        log_pipeline_run(stats, status="success")
        print("\nPipeline complete. Next run in 6 hours.")

    except Exception as e:
        print(f"Pipeline failed: {e}")
        log_pipeline_run(
            {"fetched": 0, "inserted": 0},
            status="failed",
            error=str(e)
        )


def start_scheduler():
    print("Initializing database...")
    init_db()

    # Run immediately on startup
    print("Running pipeline now on startup...\n")
    run_pipeline()

    # Then schedule every 6 hours
    scheduler.add_job(
        run_pipeline,
        trigger=IntervalTrigger(hours=6),
        id="market_intelligence_pipeline",
        name="Market Intelligence ETL",
        replace_existing=True
    )

    print(f"\nScheduler started. Pipeline runs every 6 hours.")
    print("Press Ctrl+C to stop.\n")

    try:
        scheduler.start()
    except KeyboardInterrupt:
        print("\nScheduler stopped.")
        scheduler.shutdown()


if __name__ == "__main__":
    start_scheduler()