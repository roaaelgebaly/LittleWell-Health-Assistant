from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from state import GraphState
from nodes import (
    classify_query,
    route_after_classify,
    ask_clarifying_question,
    retrieve,
    generate,
    verify_output,
    safety_redirect,
)


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("classify_query", classify_query)
    graph.add_node("ask_clarifying_question", ask_clarifying_question)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("verify_output", verify_output)
    graph.add_node("safety_redirect", safety_redirect)

    graph.set_entry_point("classify_query")
    graph.add_conditional_edges(
        "classify_query",
        route_after_classify,
        {
            "safety_redirect": "safety_redirect",
            "ask_clarifying_question": "ask_clarifying_question",
            "retrieve": "retrieve",
        },
    )
  
    graph.add_edge("ask_clarifying_question", "classify_query")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "verify_output")
    graph.add_edge("verify_output", END)
    graph.add_edge("safety_redirect", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
