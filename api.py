"""FastAPI backend wrapper around the existing LangGraph medical RAG flow."""

import uuid
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langgraph.types import Command

from graph import build_graph

app = FastAPI(title="Medical RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: Optional[str] = Field(default=None, min_length=1)
    thread_id: Optional[str] = None
    user_reply: Optional[str] = None


class ChatResponse(BaseModel):
    status: str
    thread_id: str
    classification: Optional[str] = None
    question_to_user: Optional[str] = None
    final_answer: Optional[dict[str, Any]] = None
    verification_status: Optional[str] = None
    retrieved: Optional[list[dict[str, Any]]] = None
    structured_output: Optional[dict[str, Any]] = None
    error: Optional[str] = None


def _serialize_response(response: Any) -> Optional[dict[str, Any]]:
    if response is None:
        return None
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    if isinstance(response, dict):
        return response
    return None


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if not request.question and request.user_reply is None:
        raise HTTPException(status_code=400, detail="Provide a question or a clarification reply.")

    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    graph = build_graph()

    try:
        if request.user_reply is not None:
            result = graph.invoke(Command(resume=request.user_reply), config=config)
        else:
            result = graph.invoke({"question": request.question}, config=config)
    except Exception as exc:  # pragma: no cover - defensive API error handling
        raise HTTPException(status_code=500, detail=f"Graph execution failed: {exc}") from exc

    interrupt_items = result.get("__interrupt__", [])
    if interrupt_items:
        prompt_to_user = interrupt_items[0].value.get("question_to_user")
        return ChatResponse(
            status="clarification_needed",
            thread_id=thread_id,
            classification=result.get("classification"),
            question_to_user=prompt_to_user,
            retrieved=result.get("retrieved"),
            structured_output=_serialize_response(result.get("response")),
            verification_status=result.get("verification_status"),
        )

    final_answer = result.get("final_answer")
    if final_answer is None:
        final_answer = _serialize_response(result.get("response"))

    return ChatResponse(
        status="ok",
        thread_id=thread_id,
        classification=result.get("classification"),
        final_answer=final_answer,
        verification_status=result.get("verification_status"),
        retrieved=result.get("retrieved"),
        structured_output=_serialize_response(result.get("response")),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
