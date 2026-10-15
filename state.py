from typing import Literal, TypedDict

from pydantic import BaseModel, ConfigDict


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document: str
    section: str
    page: int


class RAGResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recommendation: str
    evidence: str
    citations: list[Citation]
    confidence: Literal["high", "medium", "low", "insufficient"]


class GraphState(TypedDict, total=False):
    question: str
    clarified_question: str
    classification: str  # "safe" | "needs_clarification" | "clinical_request" | "emergency"
    retrieved: list
    response: RAGResponse
    final_answer: dict
    verification_status: str
