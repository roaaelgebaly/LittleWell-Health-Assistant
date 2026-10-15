import json
import uuid

from langgraph.types import Command

from graph import build_graph


def run_cli():
    app = build_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    question = input("Your question: ")
    result = app.invoke({"question": question}, config=config)

    while "__interrupt__" in result:
        interrupt_obj = result["__interrupt__"][0]
        prompt_to_user = interrupt_obj.value["question_to_user"]
        user_reply = input(f"\n{prompt_to_user}\n> ")
        result = app.invoke(Command(resume=user_reply), config=config)

    print("\n" + json.dumps(result["final_answer"], ensure_ascii=False))


if __name__ == "__main__":
    run_cli()
