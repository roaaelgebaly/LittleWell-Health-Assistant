"""Streamlit chat UI for the existing LangGraph medical RAG application."""

import json
import uuid

import streamlit as st
from langgraph.types import Command

from graph import build_graph


CONFIDENCE_LABELS = {
    "high": "High confidence",
    "medium": "Moderate confidence",
    "low": "Low confidence",
    "insufficient": "Insufficient evidence",
}


def _new_conversation() -> None:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = [
        {
            "role": "assistant",
            "kind": "welcome",
            "content": "How can I help you with your child's health today?",
        }
    ]
    st.session_state.pending_interrupt = False
    st.session_state.active_question = None
    st.session_state.debug_entries = []


@st.cache_resource
def _get_graph():
    return build_graph()


def _confidence_text(value: str) -> str:
    return CONFIDENCE_LABELS.get(value, "Confidence unavailable")


def _render_answer(answer: dict) -> None:
    recommendation = answer.get("recommendation", "")
    evidence = answer.get("evidence", "")
    citations = answer.get("citations") or []
    confidence = answer.get("confidence", "insufficient")

    if recommendation:
        st.markdown("**Recommendation**")
        st.write(recommendation)
    if evidence:
        st.markdown("**Why this information?**")
        st.write(evidence)
    if citations:
        st.markdown("**Sources**")
        for citation in citations:
            document = citation.get("document", "")
            section = citation.get("section", "")
            page = citation.get("page")
            section_text = f" — {section}" if section else ""
            page_text = f" — p. {page}" if page is not None else ""
            st.markdown(f"- {document}{section_text}{page_text}")
    st.caption(_confidence_text(confidence))


def _debug_snapshot(result: dict, original_question: str) -> dict:
    parsed_response = result.get("response")
    if hasattr(parsed_response, "model_dump"):
        parsed_response = parsed_response.model_dump(mode="json")
    return {
        "original_question": original_question,
        "classification": result.get("classification"),
        "retrieved_chunks": result.get("retrieved", []),
        "structured_output": parsed_response or result.get("final_answer"),
        "verification_status": result.get("verification_status", "not run"),
    }


def _run_question(question: str) -> None:
    graph = _get_graph()
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    original_question = st.session_state.active_question or question
    if st.session_state.pending_interrupt:
        result = graph.invoke(Command(resume=question), config=config)
    else:
        st.session_state.active_question = question
        result = graph.invoke({"question": question}, config=config)

    interrupt_items = result.get("__interrupt__", [])
    if interrupt_items:
        clarification = interrupt_items[0].value["question_to_user"]
        st.session_state.messages.append(
            {"role": "assistant", "kind": "answer", "content": clarification}
        )
        st.session_state.pending_interrupt = True
    else:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "kind": "answer",
                "answer": result["final_answer"],
            }
        )
        st.session_state.pending_interrupt = False
        st.session_state.active_question = None

    st.session_state.debug_entries.append(_debug_snapshot(result, original_question))


def _render_debug() -> None:
    with st.expander("Developer / RAG Debug"):
        if not st.session_state.debug_entries:
            st.caption("Debug details will appear after the first question.")
            return
        for index, entry in enumerate(reversed(st.session_state.debug_entries), start=1):
            st.markdown(f"**Run {index}**")
            st.json(entry)


st.set_page_config(page_title="Child Health Information", page_icon="+")
st.title("Child health information")
st.caption("Grounded answers from the medical knowledge base")

if "messages" not in st.session_state:
    _new_conversation()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("kind") == "welcome":
            st.write(message["content"])
        elif "answer" in message:
            _render_answer(message["answer"])
        else:
            st.write(message["content"])

question = st.chat_input("Ask about your child's health")
if question and question.strip():
    question = question.strip()
    st.session_state.messages.append({"role": "user", "content": question})
    _run_question(question)
    st.rerun()

_render_debug()