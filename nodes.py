import json
import logging
import re

from langgraph.types import interrupt
from config import (
    GROQ_CLASSIFY_MODEL,
    GROQ_GENERATE_MODEL,
    GENERATION_CONTEXT_MAX_CHARS,
    TOP_K,
    get_embed_model,
    get_collection,
    get_groq_client,
)
from state import Citation, GraphState, RAGResponse
from prompts import (
    SAFETY_REDIRECT_TEXT,
    CLASSIFY_SYSTEM_PROMPT,
    CLARIFY_SYSTEM_PROMPT,
    GENERATE_SYSTEM_PROMPT,
)

from chunking import CLINICAL_KEYWORD_RE

logger = logging.getLogger(__name__)

VALID_LABELS = {
    "emergency",
    "clinical_request",
    "needs_clarification",
    "safe",
}


def classify_query(state: GraphState) -> dict:
    q = state.get("clarified_question") or state["question"]

    resp = get_groq_client().chat.completions.create(
        model=GROQ_CLASSIFY_MODEL,
        max_tokens=10,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": CLASSIFY_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": q,
            },
        ],
    )

    label = (resp.choices[0].message.content or "").strip().lower()

    # Never let the classifier route an explicit treatment or dosing request
    # into retrieval when the safety keyword guard already identifies it.
    if CLINICAL_KEYWORD_RE.search(q):
        label = "clinical_request"

    if label not in VALID_LABELS:
        print(f"[classify_query] invalid label: {label!r}")
        label = "needs_clarification"

    print(f"[classify_query] question: {q}")
    print(f"[classify_query] classification: {label}")

    return {"classification": label}

def route_after_classify(state: GraphState) -> str:
    c = state["classification"]
    if c in ("emergency", "clinical_request"):
        return "safety_redirect"
    if c == "needs_clarification":
        return "ask_clarifying_question"
    return "retrieve"


def ask_clarifying_question(state: GraphState) -> dict:
    question = state["question"]
    try:
        resp = get_groq_client().chat.completions.create(
            model=GROQ_CLASSIFY_MODEL,
            max_tokens=80,
            temperature=0,
            messages=[
                {"role": "system", "content": CLARIFY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Original user question:\n{question}",
                },
            ],
        )
        clarification = " ".join((resp.choices[0].message.content or "").split())
        if not clarification:
            raise ValueError("empty clarification question")
        if not clarification.endswith("?"):
            clarification += "?"
    except Exception as exc:
        logger.error("Dynamic clarification generation failed: %s", exc)
        clarification = (
            "What is the main symptom or situation you are asking about, and what "
            "important detail would help explain it?"
        )

    user_reply = interrupt({"question_to_user": clarification})
    combined = f"{question}\nAdditional info from parent: {user_reply}"
    return {"clarified_question": combined}


def retrieve(state: GraphState) -> dict:
    """Semantic-only retrieval: pulls the top candidates purely by
    embedding similarity (cosine), no keyword re-ranking.
    """
    q = state.get("clarified_question") or state["question"]
    print("1. Getting embedding model...")
    model = get_embed_model()
    print("2. Embedding model loaded successfully!")
    q_emb = model.encode([q], prompt_name="query", convert_to_numpy=True)[0]

    collection = get_collection()
    results = collection.query(
        query_embeddings=[q_emb.tolist()],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    if not docs:
        return {"retrieved": []}

    semantic_scores = [1.0 - d for d in dists]

    retrieved = [
        {
            "text": doc,
            "metadata": meta,
            "distance": dist,
            "semantic_score": round(sem, 4),
        }
        for doc, meta, dist, sem in zip(docs, metas, dists, semantic_scores)
    ]
    return {"retrieved": retrieved}

def _insufficient_response(message: str) -> RAGResponse:
    return RAGResponse(
        recommendation=message,
        evidence="",
        citations=[],
        confidence="insufficient",
    )


def _citations_from_retrieved(retrieved: list, limit: int = 3) -> list[Citation]:
    citations = []
    seen = set()
    for result in retrieved:
        metadata = result.get("metadata") or {}
        document = metadata.get("source")
        section = metadata.get("section")
        page = metadata.get("page")

        if document is None or page in (None, "") or isinstance(page, bool):
            continue
        if isinstance(page, str):
            if not page.strip().isdigit():
                continue
            page = int(page.strip())
        elif not isinstance(page, int):
            continue

        citation_key = (str(document), str(section or ""), page)
        if citation_key not in seen:
            seen.add(citation_key)
            citations.append(
                Citation(
                    document=citation_key[0],
                    section=citation_key[1],
                    page=page,
                )
            )
            if len(citations) >= limit:
                break
    return citations


def _response_from_generated_text(content: str, retrieved: list) -> RAGResponse:
    """Normalize either an accidental JSON response or the normal text response."""
    content = content.strip()
    if not content:
        raise ValueError("empty generation")

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return RAGResponse(
            recommendation=_collapse_repeated_bullets(content),
            evidence="",
            citations=_citations_from_retrieved(retrieved),
            confidence="medium",
        )

    return RAGResponse.model_validate(parsed)


def _collapse_repeated_bullets(content: str) -> str:
    """Remove exact duplicate bullet items from a degenerate model response."""
    if "•" not in content:
        return content

    parts = content.split("•")
    result = [parts[0]]
    seen = set()
    for item in parts[1:]:
        normalized = " ".join(item.split()).casefold()
        if normalized and normalized in seen:
            continue
        if normalized:
            seen.add(normalized)
        result.append(item)
    return "•".join(result)


def _generation_context(retrieved: list, max_chars: int) -> str:
    """Build a bounded context while keeping retrieved excerpts intact."""
    excerpts = []
    used_chars = 0
    for result in retrieved:
        metadata = result.get("metadata") or {}
        excerpt = (
            f"[{metadata.get('source')} - page {metadata.get('page')}] "
            f"{result.get('text', '')}"
        )
        separator_chars = 2 if excerpts else 0
        if used_chars + separator_chars + len(excerpt) > max_chars:
            break
        excerpts.append(excerpt)
        used_chars += separator_chars + len(excerpt)
    return "\n\n".join(excerpts)


def generate(state: GraphState) -> dict:
    if not state["retrieved"]:
        return {
            "response": _insufficient_response(
                "The available information is not sufficient to answer this question. "
                "Please check with a doctor or pharmacist."
            )
        }

    context = _generation_context(
        state["retrieved"], GENERATION_CONTEXT_MAX_CHARS
    )
    q = (state.get("clarified_question") or state["question"])[:2000]

    try:
        resp = get_groq_client().chat.completions.create(
            model=GROQ_GENERATE_MODEL,
            max_tokens=800,
            temperature=0,
            messages=[
                {"role": "system", "content": GENERATE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Available excerpts:\n{context}\n\nParent's question: {q}",
                },
            ],
        )
        content = resp.choices[0].message.content
        response = _response_from_generated_text(content or "", state["retrieved"])
    except Exception as exc:
        logger.error("Answer generation failed: %s", exc)
        return {
            "response": _insufficient_response(
                "The system could not produce a reliable answer from the available "
                "information. Please check with a doctor or pharmacist."
            )
        }

    if response.confidence == "insufficient":
        response = _insufficient_response(
            "The available information is not sufficient to answer this question. "
            "Please check with a doctor or pharmacist."
        )
    else:
        response = response.model_copy(
            update={"citations": _citations_from_retrieved(state["retrieved"])}
        )
    return {"response": response}


def verify_output(state: GraphState) -> dict:
    # Content-level safety net on the MODEL'S OWN generated text, not just
    # the retrieved chunks -- catches inference/hallucination of dosing or
    # invasive-procedure language that wasn't explicit in the source text.
    response = state["response"]
    generated_text = f"{response.recommendation}\n{response.evidence}"
    if CLINICAL_KEYWORD_RE.search(generated_text):
        response = _redirect_response()
        return {
            "final_answer": response.model_dump(mode="json"),
            "verification_status": "redirected",
        }
    return {
        "final_answer": response.model_dump(mode="json"),
        "verification_status": "passed",
    }


def _redirect_response() -> RAGResponse:
    return _insufficient_response(SAFETY_REDIRECT_TEXT)


def safety_redirect(state: GraphState) -> dict:
    return {
        "final_answer": _redirect_response().model_dump(mode="json"),
        "verification_status": "redirected",
    }
