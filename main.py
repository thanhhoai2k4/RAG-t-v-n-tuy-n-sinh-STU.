from src.generator import generate_answer, get_context, LLM_MODEL
from src.memory import ChatMemory
from src.retriever import get_retriever
from src.generator import build_rag_chain


def main() -> None:
    """
    Simple CLI chat interface for the STU admissions assistant.

    - Uses the RAG pipeline in `src.generator.generate_answer`
    - Maintains short-term history via `src.memory.ChatMemory`
    - Prints retrieved context for each answer
    """
    print("=" * 60)
    print("Trợ lý tư vấn tuyển sinh STU")
    print(f"Sử dụng mô hình: {LLM_MODEL}")
    print("Gõ câu hỏi của bạn và nhấn Enter.")
    print("Gõ 'exit', 'quit' hoặc 'q' để thoát.")
    print("=" * 60)


    
    try:
        retriever = get_retriever()
        chain = build_rag_chain(retriever)
        print("Thực hiện load thành công retriever và chain LLM")
    except:
        print("Không thể load thành công retriever và chain LLM")

    

    memory = ChatMemory(max_history=4)

    while True:
        try:
            user_input = input("\nBạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\End the conversation. Goodbye!")
            break

        # input == ""
        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "q"}:
            print("End the conversation. Goodbye!")
            break

        memory.add_user_message(user_input)
        chat_history_str = memory.get_history_string()

        # region agent log
        import json as _json, time as _time
        with open("debug-4fc04b.log", "a", encoding="utf-8") as _f:
            _f.write(_json.dumps({
                "sessionId": "4fc04b",
                "runId": "pre-fix",
                "hypothesisId": "H3",
                "location": "main.py:before_generate_answer",
                "message": "Before generate_answer call",
                "data": {
                    "user_input": user_input,
                },
                "timestamp": int(_time.time() * 1000),
            }) + "\n")
        # endregion

        answer = generate_answer(
            user_input,
            chat_history=chat_history_str,
            chain=chain,
        )

        print("\nAI:")
        print(answer)


        memory.add_ai_message(answer if isinstance(answer, str) else str(answer))


if __name__ == "__main__":
    main()

