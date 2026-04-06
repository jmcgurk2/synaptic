from pydantic import BaseModel, Field


class CaptureRequest(BaseModel):
    text: str
    source: str = "@synaptic"
    channel_id: str | None = None
    project: str | None = None


class CaptureResponse(BaseModel):
    id: str
    status: str  # stored | held_for_review
    type: str
    title: str
    tags: list[str]
    summary: str
    confidence: float
    project: str | None = None


class SearchResult(BaseModel):
    id: str
    type: str
    title: str
    tags: list[str]
    summary: str
    source: str
    confidence: float
    score: float = 0.0
    project: str | None = None


class EntryDetail(BaseModel):
    id: str
    type: str
    title: str
    tags: list[str]
    summary: str
    raw_text: str
    source: str
    confidence: float
    status: str
    project: str | None = None
    created_at: str
    updated_at: str


class FixRequest(BaseModel):
    type: str


class HealthResponse(BaseModel):
    status: str
    sqlite: str
    qdrant: str
    version: str


class ContextResponse(BaseModel):
    recent: list[EntryDetail]
    pending_fix: list[EntryDetail]
