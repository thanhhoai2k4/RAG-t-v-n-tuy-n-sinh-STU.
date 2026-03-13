# Trợ lý Tư vấn Tuyển sinh STU (STU Admissions Assistant)

## 📖 Tổng quan
Dự án này là một chatbot dựa trên mô hình RAG (Retrieval-Augmented Generation) được thiết kế để hỗ trợ tư vấn tuyển sinh cho trường Đại học Công nghệ Sài Gòn (STU). Hệ thống cung cấp một giao diện dòng lệnh (CLI) cho phép người dùng đặt câu hỏi và nhận câu trả lời dựa trên các tài liệu tuyển sinh được cung cấp.

## ✨ Tính năng chính
- **Tích hợp LLM Cục bộ**: Sử dụng Ollama với mô hình `deepseek-r1:1.5b` để tạo câu trả lời và `qwen3-embedding:0.6b` để tạo vector nhúng (embeddings).
- **Xử lý dữ liệu đa dạng**: Hỗ trợ đọc file Text thông thường, file PDF và cả PDF dạng scan bằng công cụ nhận dạng ký tự quang học Tesseract OCR thông qua thư viện `unstructured`.
- **Tìm kiếm Vector**: Sử dụng cơ sở dữ liệu FAISS để lưu trữ và truy xuất tài liệu một cách nhanh chóng, hiệu quả.
- **Ghi nhớ ngữ cảnh**: Hệ thống sử dụng bộ nhớ (`ChatMemory`) để duy trì lịch sử trò chuyện ngắn hạn, giúp AI hiểu được các câu hỏi tiếp nối của người dùng.

## 🛠️ Công nghệ sử dụng
- **Ngôn ngữ**: Python >= 3.10
- **Framework chính**: LangChain
- **Cơ sở dữ liệu Vector**: FAISS
- **LLM Engine**: Ollama
- **OCR**: Tesseract & thư viện Unstructured

## 🚀 Cài đặt và Khởi chạy

### Yêu cầu hệ thống
1. **Ollama**: Cài đặt Ollama và tải các mô hình cần thiết:
   ```bash
   ollama run deepseek-r1:1.5b
   ollama pull qwen3-embedding:0.6b
    ```

Cài đặt các gói:
    ````
        uv sync
    ````

Cấu trúc thư mục
data/processed/: Nơi chứa các file văn bản .txt đã được xử lý.

data/pdf_text/: Nơi chứa các file .pdf chứa văn bản gốc.

data/pdf_images/: Nơi chứa các file .pdf dạng ảnh hoặc scan (cần chạy OCR để trích xuất).

vector_db/faiss_index/: Thư mục lưu trữ cơ sở dữ liệu vector FAISS sau khi xử lý nhúng.

src/: Thư mục mã nguồn chính.

config.py: Cấu hình tên mô hình và đường dẫn.

data_ingestion.py: Mã nguồn đọc dữ liệu, chia nhỏ đoạn (chunking) và lưu vào Vector DB.

generator.py: Khởi tạo LangChain RAG pipeline và định nghĩa khuôn mẫu Prompt.

retriever.py: Load cơ sở dữ liệu FAISS và khởi tạo đối tượng truy xuất.

memory.py: Quản lý bộ nhớ cuộc hội thoại.

main.py: File chính để chạy ứng dụng chatbot CLI.




## Xây dựng dữ liệu (Vector Database):
````
python -m src.data_ingestion
````

## chạy chương trình:

````
.venv/Script/activate
python main.py
````



