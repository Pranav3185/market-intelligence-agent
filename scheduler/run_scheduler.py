from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime
import main

scheduler = BlockingScheduler()

def scheduled_job():
    print(f"\nScheduled run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    main.run_pipeline()

if __name__ == "__main__":
    # Run immediately on startup
    scheduled_job()

    # Then every 6 hours
    scheduler.add_job(
        scheduled_job,
        trigger=IntervalTrigger(hours=6),
        id="pipeline",
        replace_existing=True
    )

    print("\nScheduler running. Press Ctrl+C to stop.")
    try:
        scheduler.start()
    except KeyboardInterrupt:
        scheduler.shutdown()