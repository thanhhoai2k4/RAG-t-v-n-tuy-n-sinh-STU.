import os
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings


FAISS_PATH = "vector_db/faiss_index" # path into vector database.
EMBEDDING_MODEL = "qwen3-embedding:0.6b" # name model. ollama pull qwen3-embedding:0.6b 600mb

def get_retriever(k: int = 4):
    """
        load vector database and return retriever object.
    """
    if not os.path.exists(FAISS_PATH):
        raise FileNotFoundError(f"dont have :{FAISS_PATH}")
    
    # load model embeddings
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # load FAISS
    vector_db = FAISS.load_local(
        folder_path=FAISS_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True  # reques
    )

    # 
    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    
    return retriever


if __name__ == "__main__":
    my_retriever = get_retriever(k=3)
    try:
    # Câu hỏi test (Giả lập câu hỏi của sinh viên)
        test_query = "Trường có cộng điểm ưu tiên khu vực không?"
        print(f"\n❓ Câu hỏi của sinh viên: '{test_query}'\n")
        print("🔍 Đang tìm kiếm trong kho dữ liệu...\n")
        # Dùng hàm invoke() để tìm kiếm
        retrieved_docs = my_retriever.invoke(test_query)
        # In kết quả
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"--- 📄 TÀI LIỆU SỐ {i} ---")
            print(f"Nguồn: {doc.metadata.get('source', 'Không rõ')}")
            # Xóa bớt khoảng trắng/xuống dòng thừa cho dễ nhìn
            clean_content = " ".join(doc.page_content.split()) 
            print(f"Nội dung: {clean_content}\n")
    except Exception as e:
        print(e)