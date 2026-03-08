from langchain_core.documents.base import Document
from langchain_ollama import ChatOllama #  local chat
from langchain_core.prompts import PromptTemplate # template 
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.retriever import get_retriever
from operator import itemgetter
import sys
import os

LLM_MODEL = "deepseek-r1:1.5b" 

# ==========================================
# KHUÔN MẪU PROMPT (Định hình tính cách AI)
# ==========================================
PROMPT_TEMPLATE = """Bạn là một chuyên viên tư vấn tuyển sinh và học vụ thân thiện, nhiệt tình của Trường Đại học Công nghệ Sài Gòn (STU).
Nhiệm vụ của bạn là trả lời câu hỏi của sinh viên một cách chính xác, DỰA HOÀN TOÀN VÀO các tài liệu được cung cấp bên dưới.

QUY TẮC QUAN TRỌNG:
1. Nếu thông tin để trả lời không có trong tài liệu, hãy thành thật nói rằng bạn chưa có thông tin chính xác và khuyên sinh viên liên hệ trực tiếp Phòng Đào tạo. Tuyệt đối KHÔNG tự bịa ra thông tin.
2. Trả lời ngắn gọn, súc tích, dễ hiểu. Có thể dùng gạch đầu dòng để làm rõ ý.
3. Luôn giữ thái độ lịch sự, xưng "mình" hoặc "Thầy/Cô" và gọi người hỏi là "bạn" hoặc "em".

LỊCH SỬ TRÒ CHUYỆN GẦN ĐÂY:
{chat_history}

TÀI LIỆU THAM KHẢO (Context):
{context}

CÂU HỎI CỦA SINH VIÊN:
{question}

CÂU TRẢ LỜI CỦA BẠN:"""

def format_docs(docs: list[Document]):
    """ 
        Standardize the document. Source/ Page(if any)/ content
        Args:
            - docs: List[Document]. medatata(source, pagem,..) 
    """
    formatted_chunks = []
    for i, doc in enumerate(docs):
        content = doc.page_content
        source = doc.metadata.get("source", "The documentation is unclear")
        page = doc.metadata.get("page", "unclear")
        if isinstance(page, int):
            page += 1
        chunk_info = f"--- DOCUMENT {i} ---\nSource: {source} (Trang {page})\nContent:\n{content}"
        formatted_chunks.append(chunk_info)
    return ".".join(formatted_chunks)


def get_context(question: str) -> str:
    """
    Trả về chuỗi context đã được format cho một câu hỏi.
    """
    retriever = get_retriever()
    docs = retriever.invoke(question)
    return format_docs(docs)


def build_rag_chain():
    """Construct a RAG (Retriever-Augmented Generation) pipeline using LCEL"""
    retriever = get_retriever()

    # init model LLM
    llm = ChatOllama(model=LLM_MODEL, temperature=0.1)

    # init PromptTemplate
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    # contruct chain 
    # flow: get quesion , pass into retriever find the docs, then format_docs function
    # pass throun into PromptTemplate: quesion, context.

    RAG_CHAIN = (
        {
            "context": itemgetter("question") | retriever | format_docs, 
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history")
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return RAG_CHAIN

def generate_answer(question: str, chat_history: str)->str:
    """
        predict     
    """
    # print(f"finding document and response for: '{question}'...")

    chain = build_rag_chain()

    inputs = {
        "question": question,
        "chat_history": chat_history if chat_history else "No conversations yet."
    }

    response = chain.invoke(inputs)

    return response



if __name__ == "__main__":
    # Test toàn bộ RAG Pipeline
    test_question = "Điểm xét tuyển chưa cộng điểm ưu tiên được tính như thế nào?"
    test_history = "Chưa có lịch sử trò chuyện."
    
    print("="*50)
    # Gọi hàm với cả 2 tham số
    answer = generate_answer(test_question, chat_history=test_history)
    print("\n=== CÂU TRẢ LỜI CỦA AI ===")
    print(answer)
    print("="*50)