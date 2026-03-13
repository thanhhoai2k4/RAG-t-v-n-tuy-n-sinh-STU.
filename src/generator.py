from langchain_core.documents.base import Document
from langchain_ollama import ChatOllama #  local chat
from langchain_core.prompts import PromptTemplate # template 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_core.runnables.base import RunnableSerializable
from src.retriever import get_retriever
from operator import itemgetter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _TimeoutError

from src.config import LLM_MODEL
from langchain_core.runnables import RunnableLambda

# ==========================================
# KHUÔN MẪU PROMPT (Định hình tính cách AI)
# ==========================================
SYSTEM_PROMPT = """Bạn là một Giáo viên và Hướng dẩn, nhiệt tình của môn toán học.
Nhiệm vụ của bạn là trả lời câu hỏi của sinh viên một cách chính xác, DỰA HOÀN TOÀN VÀO các tài liệu được cung cấp bên dưới.

QUY TẮC QUAN TRỌNG:
1. Nếu thông tin để trả lời không có trong tài liệu, hãy thành thật nói rằng bạn chưa có thông tin chính xác và khuyên sinh viên liên hệ trực tiếp Phòng Đào tạo. 
2. Trả lời ngắn gọn, súc tích, dễ hiểu. Có thể dùng gạch đầu dòng để làm rõ ý.
3. Luôn giữ thái độ lịch sự, xưng "mình" hoặc "Thầy/Cô" và gọi người hỏi là "bạn" hoặc "em".
4. Luốn show công thức ra để người dùng hình dung được công thức của toán học
"""

HUMAN_PROMPT = """LỊCH SỬ TRÒ CHUYỆN GẦN ĐÂY:
{chat_history}

TÀI LIỆU THAM KHẢO (Context):
{context}

CÂU HỎI CỦA SINH VIÊN:
{question}
"""

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
    return "\n\n".join(formatted_chunks)


def get_context(question: str, retriever : VectorStoreRetriever) -> str:
    """
    Trả về chuỗi context đã được format cho một câu hỏi.
    """

    docs = retriever.invoke(question)
    return format_docs(docs)


def build_rag_chain(retriever : VectorStoreRetriever):
    """Construct a RAG (Retriever-Augmented Generation) pipeline using LCEL"""

    # init model LLM
    llm = ChatOllama(
        model=LLM_MODEL, temperature=0.1, verbose=True  
    )

    # test LLM
    # resp = llm.stream("who are you?")
    # for i in resp:
    #     print(i, end=" ", flush=True)
    # return
    # test LLM


    # init PromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT)
    ])

    # contruct chain 
    # flow: get quesion , pass into retriever find the docs, then format_docs function
    # pass throun into PromptTemplate: quesion, context.

    RAG_CHAIN = (
        {
            "context": itemgetter("question") | retriever  | format_docs, 
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history")
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # test for none retriver 
    # empty_retriever = RunnableLambda(lambda _question: [])  # trả về list docs rỗng
    # RAG_CHAIN = (
    #     {
    #         "context": itemgetter("question") | empty_retriever  | format_docs, 
    #         "question": itemgetter("question"),
    #         "chat_history": itemgetter("chat_history")
    #     }
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    # )


    # test with docuement
    # DOC_TEXT = """--- DOCUMENT 0 ---
    #     Source: TEST_DOC (Trang 1)
    #     Content:
    #     Trường STU có mã bí mật để kiểm tra là: STU-VERIFY-7391.
    #     Học phí ngành X trong ví dụ này là 12.345.000 VNĐ/học kỳ.
    # """

    # static_context = RunnableLambda(lambda _question: DOC_TEXT)

    # RAG_CHAIN = (
    #     {
    #         "context": itemgetter("question") | static_context,
    #         "question": itemgetter("question"),
    #         "chat_history": itemgetter("chat_history"),
    #     }
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    # )

    return RAG_CHAIN

def generate_answer(
    question: str,
    chat_history: str,
    chain,
) -> any:
    """
    """
    inputs = {
        "question": question,
        "chat_history": chat_history if chat_history else "No conversations yet.",
    }


    return chain.invoke(inputs)

