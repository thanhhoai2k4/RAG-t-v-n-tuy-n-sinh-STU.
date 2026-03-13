from src.generator import generate_answer, get_context, LLM_MODEL
from src.memory import ChatMemory
from src.retriever import get_retriever
from src.generator import build_rag_chain
import os
from dotenv import load_dotenv


load_dotenv()


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


    retriever = get_retriever()
    chain = build_rag_chain(retriever)
    

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

