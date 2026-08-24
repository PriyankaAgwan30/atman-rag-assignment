# Atman Document RAG Assistant

## Overview

This project implements a Retrieval-Augmented Generation (RAG) based document question-answering system.

The application allows users to ask questions about the provided PDF documents. Relevant document chunks are retrieved from a ChromaDB vector database and provided as context to a local Gemma language model through Ollama. The application returns a grounded answer along with the source document, page number, and chunk information.

## Architecture

PDF Documents
    ↓
PDF Text Extraction
    ↓
Text Chunking
    ↓
Sentence Transformer Embeddings
    ↓
ChromaDB Vector Database
    ↓
Semantic Retrieval
    ↓
Relevant Document Context
    ↓
Gemma via Ollama
    ↓
Grounded Answer + Sources

## Technologies Used

- Python
- Streamlit
- ChromaDB
- Sentence Transformers
- all-MiniLM-L6-v2
- Ollama
- Gemma
- PDF processing
- Retrieval-Augmented Generation (RAG)

## Project Structure

```text
atman-rag-assignment/
│
├── app/
│   └── streamlit_app.py
│
├── src/
│   ├── pdf_loader.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── llm.py
│   └── rag_pipeline.py
│
├── data/
│   └── PDF documents
│
├── chroma_db/
│   └── chroma.sqlite3
│
├── evaluation/
│   └── sample_questions.md
│
├── requirements.txt
├── README.md
├── .gitignore
└── .env.example
