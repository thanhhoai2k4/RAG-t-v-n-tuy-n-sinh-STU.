from src.retriever import get_retriever




retriever = get_retriever(2)
kq = retriever.invoke("Cách giải phương trình bật nhất?")
print(kq)



# from src.data_ingestion import build_vector_database
# build_vector_database()