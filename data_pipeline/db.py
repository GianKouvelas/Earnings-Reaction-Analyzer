"""
db.py — Database connection setup for Earnings Reaction Analyzer.

Uses SQLAlchemy to manage the connection to PostgreSQL and expose
a session factory that the rest of the pipeline can use to read/write data.
"""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Load variables from .env into the environment
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found. Did you create a .env file?")

# The engine manages the actual connection pool to PostgreSQL
engine = create_engine(DATABASE_URL)

# SessionLocal is a factory: each call gives us a new DB session (like opening a "conversation" with the DB)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_session():
    """
    Returns a new database session.
    Call session.close() when done, or use as a context manager.
    """
    return SessionLocal()


def test_connection():
    """Quick sanity check that the DB is reachable."""
    try:
        with engine.connect() as conn:
            print("✅ Connected to the database successfully.")
    except Exception as e:
        print(f"❌ Could not connect to the database: {e}")


if __name__ == "__main__":
    test_connection()