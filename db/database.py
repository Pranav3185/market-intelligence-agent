import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

def get_engine():
    """Create and return a SQLAlchemy engine."""
    db_url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )

    engine = create_engine(
        db_url,
        future=True,
        pool_pre_ping=True,
    )
    return engine


def init_db():
    """Run schema.sql to create tables if they don't exist."""
    engine = get_engine()
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")

    with open(schema_path, "r") as f:
        schema_sql = f.read()

    with engine.connect() as conn:
        conn.execute(text(schema_sql))
        conn.commit()
if __name__ == "__main__":
    init_db()
    
    print("Database initialized successfully.")