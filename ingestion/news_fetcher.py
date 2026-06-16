import requests
import hashlib
import os
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
BASE_URL = "https://newsapi.org/v2/everything"

TOPICS = [
    "artificial intelligence",
    "stock market",
    "technology companies",
    "startup funding",
    "mergers acquisitions"
]

def fetch_articles(topic: str, days_back: int = 1) -> list:
    """Fetch articles for a given topic from NewsAPI."""
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    params = {
        "q": topic,
        "from": from_date,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": 20,
        "apiKey": NEWS_API_KEY
    }

    response = requests.get(BASE_URL, params=params)

    if response.status_code != 200:
        print(f"Error fetching '{topic}': {response.status_code}")
        return []

    articles = response.json().get("articles", [])
    print(f"Fetched {len(articles)} articles for topic: '{topic}'")
    return articles


def clean_articles(articles: list, topic: str) -> pd.DataFrame:
    """Clean and normalize raw articles into a DataFrame."""
    cleaned = []

    for article in articles:
        # Skip articles with missing critical fields
        if not article.get("title") or not article.get("url"):
            continue

        # Generate a unique ID from the URL
        article_id = hashlib.md5(article["url"].encode()).hexdigest()

        # Clean and extract fields
        cleaned.append({
            "article_id": article_id,
            "title": article["title"].strip(),
            "description": (article.get("description") or "").strip(),
            "content": (article.get("content") or "").strip(),
            "source_name": article.get("source", {}).get("name", "Unknown"),
            "author": (article.get("author") or "Unknown").strip(),
            "url": article["url"],
            "published_at": article.get("publishedAt"),
            "category": topic,
            "is_processed": False
        })

    df = pd.DataFrame(cleaned)

    if df.empty:
        return df

    # Drop duplicates within this batch
    df = df.drop_duplicates(subset=["article_id"])

    # Parse published_at to datetime
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")

    # Drop rows where title is [Removed] (NewsAPI quirk)
    df = df[~df["title"].str.contains(r"\[Removed\]", na=False)]

    print(f"Cleaned down to {len(df)} valid articles for topic: '{topic}'")
    return df


def run_fetcher() -> pd.DataFrame:
    """Run fetcher for all topics and return combined DataFrame."""
    all_articles = []

    for topic in TOPICS:
        raw = fetch_articles(topic)
        cleaned = clean_articles(raw, topic)
        if not cleaned.empty:
            all_articles.append(cleaned)

    if not all_articles:
        print("No articles fetched.")
        return pd.DataFrame()

    combined = pd.concat(all_articles, ignore_index=True)

    # Final dedup across all topics by article_id
    combined = combined.drop_duplicates(subset=["article_id"])
    print(f"\nTotal unique articles ready for DB: {len(combined)}")
    return combined


if __name__ == "__main__":
    df = run_fetcher()
    if not df.empty:
        print("\nSample data:")
        print(df[["title", "source_name", "category", "published_at"]].head())