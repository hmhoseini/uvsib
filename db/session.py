from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from uvsib.db.db_url import DB_URL

Base = declarative_base()
engine = create_engine(DB_URL, echo=False)
# Create a session factory (not a single session instance!)
SessionLocal = sessionmaker(bind=engine) #, autoflush=False, autocommit=False, future=True)

@contextmanager
def get_session():
    """
    Yields a SQLAlchemy session.
    """

    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()
