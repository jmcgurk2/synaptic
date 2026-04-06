import os
from datetime import datetime
from uuid import uuid4

from sqlmodel import Field, Session, SQLModel, create_engine, select


class Entry(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    type: str
    title: str
    tags: str  # JSON-serialised list
    summary: str
    raw_text: str
    source: str  # @synaptic | @orex | future agents
    confidence: float
    status: str  # stored | pending_fix
    project: str | None = None  # Optional project association
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ReceiptLog(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    raw_text: str
    source: str
    classified_as: str
    confidence: float
    disposition: str  # stored | bounced | fixed
    entry_id: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


_engine = None


def get_engine():
    global _engine
    if _engine is None:
        db_path = os.getenv("SQLITE_PATH", "/data/synaptic.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        _engine = create_engine(f"sqlite:///{db_path}", echo=False)
    return _engine


def init_db():
    SQLModel.metadata.create_all(get_engine())


def get_session():
    with Session(get_engine()) as session:
        yield session


async def check_sqlite() -> str:
    try:
        engine = get_engine()
        with Session(engine) as session:
            session.exec(select(Entry).limit(1))
        return "ok"
    except Exception:
        return "error"
