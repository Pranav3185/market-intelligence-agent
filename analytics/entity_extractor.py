from transformers import pipeline
from sqlalchemy import text
from db.database import get_engine
import json
import logging

logger = logging.getLogger(__name__)

print("Loading NER model...")
ner_pipeline = pipeline(
    "ner",
    model="dbmdz/bert-large-cased-finetuned-conll03-english",
    aggregation_strategy="simple",
    device=-1  # CPU
)
print("NER model loaded.")


def extract_entities(text_input: str) -> dict:
    """Extract named entities from text and return grouped by type."""
    if not text_input or len(text_input.strip()) < 10:
        return {}

    try:
        results = ner_pipeline(text_input[:512])

        entities = {}
        for entity in results:
            label = entity["entity_group"]
            word = entity["word"].strip()

            # Only keep meaningful entity types
            if label not in ["ORG", "PER", "GPE", "MONEY"]:
                continue

            # Skip very short or noisy tokens
            if len(word) < 2:
                continue

            if label not in entities:
                entities[label] = []

            if word not in entities[label]:
                entities[label].append(word)

        return entities

    except Exception as e:
        logger.error(f"NER error: {e}")
        return {}


def get_articles_without_entities():
    """Fetch articles that haven't had entity extraction run yet."""
    engine = get_engine()

    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT id, title, description
            FROM articles
            WHERE is_processed = TRUE
            AND (entities IS NULL OR entities = '')
        """))
        rows = result.fetchall()

    print(f"Found {len(rows)} articles needing entity extraction.")
    return rows


def run_entity_extraction():
    """Run NER on all unprocessed articles and update DB."""
    rows = get_articles_without_entities()

    if not rows:
        print("No articles need entity extraction.")
        return

    engine = get_engine()
    processed = 0

    with engine.connect() as conn:
        for row in rows:
            try:
                # Combine title + description for better coverage
                text_input = f"{row.title}. {row.description or ''}"
                entities = extract_entities(text_input)

                conn.execute(text("""
                    UPDATE articles
                    SET entities = :entities
                    WHERE id = :id
                """), {
                    "entities": json.dumps(entities),
                    "id": row.id
                })

                processed += 1

                if processed % 10 == 0:
                    print(f"Extracted entities from {processed}/{len(rows)} articles...")

            except Exception as e:
                logger.error(f"Error processing article {row.id}: {e}")

        conn.commit()

    print(f"Entity extraction complete. Processed {processed} articles.")


if __name__ == "__main__":
    run_entity_extraction()