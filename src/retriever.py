import os
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores.base import VectorStoreRetriever
from src.config import FAISS_PATH, model_embeddings


def get_retriever(k: int = 4)->VectorStoreRetriever:
    """
        load vector database and return retriever object.
        Load once.
        Args:
            -k: GET K Retriever.
        Returns:
            - VectorStoreRetriever.
    """
    if not os.path.exists(FAISS_PATH):
        raise FileNotFoundError(f"dont have :{FAISS_PATH}")
    
    # load model embeddings
    embeddings = OllamaEmbeddings(model=model_embeddings)

    # load FAISS
    vector_db = FAISS.load_local(
        folder_path=FAISS_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True  # reques
    )

    # 
    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 7}
    )
    
    return retriever