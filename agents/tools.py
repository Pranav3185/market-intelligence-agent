from langchain.tools import tool
from pipeline.rag import ask as rag_ask
from db.database import get_engine
from sqlalchemy import text
import logging

logger = logging.getLogger(__name__)


@tool
def search_articles(query: str) -> str:
    """
    Search for relevant articles using semantic similarity.
    Use this when the user asks about news, events, or topics.
    """
    try:
        return rag_ask(query)
    except Exception as e:
        return f"Search unavailable: {str(e)}. Try using get_sentiment_summary or get_trending_topics instead."


@tool
def get_sentiment_summary(category: str = "all") -> str:
    """
    Get sentiment summary statistics from the database.
    Use this when the user asks about sentiment trends, scores, or mood.
    Input should be a category like 'artificial intelligence' or 'all'.
    """
    category = str(category).strip()
    if "=" in category:
        category = category.split("=")[-1]
    category = category.strip("'\" ")

    engine = get_engine()

    with engine.connect() as conn:
        if category.lower() == "all":
            result = conn.execute(text("""
                SELECT
                    category,
                    ROUND(AVG(sentiment_score)::numeric, 3) AS avg_sentiment,
                    SUM(CASE WHEN sentiment_label='POSITIVE' THEN 1 ELSE 0 END) AS positive_count,
                    SUM(CASE WHEN sentiment_label='NEGATIVE' THEN 1 ELSE 0 END) AS negative_count
                FROM articles
                WHERE is_processed = TRUE
                GROUP BY category
                ORDER BY avg_sentiment ASC
            """))
        else:
            result = conn.execute(text("""
                SELECT
                    category,
                    ROUND(AVG(sentiment_score)::numeric, 3) AS avg_sentiment,
                    SUM(CASE WHEN sentiment_label='POSITIVE' THEN 1 ELSE 0 END) AS positive_count,
                    SUM(CASE WHEN sentiment_label='NEGATIVE' THEN 1 ELSE 0 END) AS negative_count
                FROM articles
                WHERE is_processed = TRUE
                AND LOWER(category) LIKE LOWER(:category)
                GROUP BY category
                ORDER BY avg_sentiment ASC
            """), {"category": f"%{category}%"})

        rows = result.fetchall()

    if not rows:
        return f"No sentiment data found for category: {category}"

    return "\n".join([
        f"{row.category}: sentiment={row.avg_sentiment}, "
        f"positive={row.positive_count}, negative={row.negative_count}"
        for row in rows
    ])


@tool
def get_top_articles(input_str: str = "NEGATIVE,5") -> str:
    """
    Get the most positive or negative articles from the database.
    Input should be a plain string like 'NEGATIVE,5' or 'POSITIVE,3'.
    """
    parts = input_str.replace("'", "").replace('"', "").split(",")
    sentiment = parts[0].strip().upper() if parts else "NEGATIVE"
    sentiment = "NEGATIVE" if "NEGATIVE" in sentiment else "POSITIVE"

    try:
        limit = int(parts[1].strip()) if len(parts) > 1 else 5
        limit = max(1, min(limit, 20))  # clamp between 1 and 20
    except ValueError:
        limit = 5

    order = "ASC" if sentiment == "NEGATIVE" else "DESC"

    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text(f"""
            SELECT title, source_name, category, sentiment_score
            FROM articles
            WHERE sentiment_label = :sentiment
            AND is_processed = TRUE
            ORDER BY sentiment_score {order}
            LIMIT :limit
        """), {"sentiment": sentiment, "limit": limit})
        rows = result.fetchall()

    if not rows:
        return f"No {sentiment} articles found."

    return "\n\n".join([
        f"{i}. {row.title}\n"
        f"   Source: {row.source_name} | Category: {row.category} | Score: {row.sentiment_score}"
        for i, row in enumerate(rows, 1)
    ])


@tool
def get_trending_topics(days: str = "7") -> str:
    """
    Get trending topics and article counts from the last N days.
    Use this when the user asks what's trending.
    Input should be number of days as a plain string like '7'.
    """
    days_str = str(days).strip()
    if "=" in days_str:
        days_str = days_str.split("=")[-1]
    days_str = days_str.strip("'\" ")

    try:
        days_int = int(days_str)
        days_int = max(1, min(days_int, 30))  # clamp between 1 and 30
    except ValueError:
        days_int = 7

    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT
                category,
                COUNT(*) AS article_count,
                ROUND(AVG(sentiment_score)::numeric, 3) AS avg_sentiment
            FROM articles
            WHERE published_at >= NOW() - ((:days || ' days')::interval)
            GROUP BY category
            ORDER BY article_count DESC
        """), {"days": days_int})
        rows = result.fetchall()

    if not rows:
        return f"No articles found in the last {days_int} days."

    return "\n".join([
        f"{row.category}: {row.article_count} articles, avg sentiment={row.avg_sentiment}"
        for row in rows
    ])