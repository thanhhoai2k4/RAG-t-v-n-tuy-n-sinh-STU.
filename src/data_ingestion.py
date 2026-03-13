import os # sys
import unstructured_pytesseract
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader, TextLoader, PyPDFDirectoryLoader, UnstructuredPDFLoader,PyMuPDFLoader # load data 
from langchain_text_splitters import RecursiveCharacterTextSplitter # chunk
from langchain_ollama import OllamaEmbeddings # load local embedings model from ollama 
from langchain_community.vectorstores import FAISS # Facebook AI similary search
from src.config import model_embeddings
from src.config import FAISS_PATH
import logging
logging.getLogger("unstructured").setLevel(logging.ERROR)

# basic colors
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
RESET = "\033[0m"

DATA_DIR_TXT = "data/processed" # path into processed data
DATA_DIR_PDF_IMAGE = "data/pdf_images" # path into PDF data raw
DATA_DIR_PDF_TEXT = "data/pdf_text" # path into text PDF

unstructured_pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ["OCR_AGENT"] = "unstructured.partition.utils.ocr_models.tesseract_ocr.OCRAgentTesseract"

def build_vector_database():
    """  this is function is only use once when new data is add."""
    print(RED + "1.Loading data from the folder" + RESET)
    documents = []

    # load all file txt.
    if os.path.exists(DATA_DIR_TXT):
        # load all data txt in DATA_DIR_TXT. by for element in DATA_DIR_TXT: Textloader(element)
        txt_loader = DirectoryLoader(
            path=DATA_DIR_TXT, 
            glob="*.txt", 
            loader_cls=lambda p: TextLoader(p, encoding="utf-8"))
        txt_docs = txt_loader.load() # Replacement: txt_loader.lazy_load()
        documents.extend(txt_docs)
        print(f"\t-loaded {len(txt_docs)} file txt.")
    else:
        print(f"\t-folder { DATA_DIR_TXT } dont exists. Ignore TXT.")



    if os.path.exists(DATA_DIR_PDF_TEXT):
        textpdf_loader = DirectoryLoader(
            path=DATA_DIR_PDF_TEXT,
            glob="*.pdf",
            loader_cls=PyMuPDFLoader
        )
        textpdf_docs = textpdf_loader.load()
        documents.extend(textpdf_docs)
        print(f"\t- Loaded {len(textpdf_docs)} file from text pdf")

    else:
        print(f"\t-Folder {DATA_DIR_PDF_TEXT} dont exist.Ignore pdf text.")


    # load special PDF: image
    if os.path.exists(DATA_DIR_PDF_IMAGE):
        print(f"\t-prepare pdf and run ORC...")
        loader_kwargs = {
            "strategy": "hi_res",         # Must use 'hi_res' so it can analyze the layout and scanned images
            "languages": ["vie", "eng"],  # Prioritize Vietnamese recognition, then English.
        }
        pdf_loader = DirectoryLoader(
            path=DATA_DIR_PDF_IMAGE, 
            glob="*.pdf", 
            loader_cls=UnstructuredPDFLoader,
            loader_kwargs=loader_kwargs
        )

        pdf_docs = pdf_loader.load()
        documents.extend(pdf_docs)
        print(f"\t- Loaded {len(pdf_docs)} chunks from PDF (processed OCR).")


    else:
        print(f"\t-Folder {DATA_DIR_PDF_IMAGE} dont exists. Ignore PDF.")


    if not documents:
        print("\t-dont have loaded document. please add file into data folder.")
        return
    
    print(f"\t-Total: need to process: {len(documents)} document/page.")

    print(RED + "2. Chunking the text" + RESET)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50, 
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    print(f"\t-created {len(chunks)} chunks.")


    print(RED  + "3.Create Embedings and save to FAISS " + RESET)
    # use ollama model
    embeddings = OllamaEmbeddings(model=model_embeddings)
    # create vector database into your disk
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(FAISS_PATH)
    print("✅ The Vector Database has been successfully built using Ollama and FAISS (including both PDF and TXT)!")

def load_and_search_faiss(query: str, k: int = 3):
    """"""
    print(BLUE + f" loading FAISS and search {k} stament nearst for {query}." + RESET)

    # use ollama model
    embeddings = OllamaEmbeddings(model=model_embeddings)

    # load FAISS
    vector_db = FAISS.load_local(
            folder_path=FAISS_PATH, 
            embeddings=embeddings, 
            allow_dangerous_deserialization=True # request
        )

    top_k_results = vector_db.similarity_search(query, k=k)

    for e in top_k_results:
        id = e.id
        source = e.metadata.get("source")
        page_content = e.page_content
        print(f"source from : {source} with content: {page_content}")
        print("*"*100)



if __name__ == "__main__":
    # buil data vector
    build_vector_database()

    # test
    load_and_search_faiss("điểm xét tuyển chưa cộng điểm ưu tiên, điểm cộng")