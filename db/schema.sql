CREATE TABLE IF NOT EXISTS articles (
    id SERIAL PRIMARY KEY,
    article_id VARCHAR(64) UNIQUE NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    content TEXT,
    source_name VARCHAR(255),
    author VARCHAR(255),
    url TEXT,
    published_at TIMESTAMP,
    category VARCHAR(100),
    sentiment_score FLOAT,
    sentiment_label VARCHAR(20),
    entities TEXT,
    is_processed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS pipeline_logs (
    id SERIAL PRIMARY KEY,
    run_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    articles_fetched INTEGER,
    articles_inserted INTEGER,
    status VARCHAR(50),
    error_message TEXT
);