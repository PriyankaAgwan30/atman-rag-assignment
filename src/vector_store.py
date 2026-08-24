from pathlib import Path
import sys

import chromadb

# Allow importing modules from src
sys.path.append(str(Path(__file__).parent))

from pdf_loader import load_all_pdfs
from chunker import create_chunks
from embeddings import EmbeddingModel


CHROMA_PATH = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "atman_documents"


class VectorStore:
    def __init__(self):
        print("Initializing ChromaDB...")

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_PATH)
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                "description": "Atman RAG document collection"
            }
        )

        self.embedding_model = EmbeddingModel()

        print("ChromaDB initialized successfully.")

    def add_chunks(self, chunks):
        """
        Generate embeddings and store chunks in ChromaDB.
        """

        if not chunks:
            print("No chunks to add.")
            return

        texts = [chunk["text"] for chunk in chunks]

        print(f"\nGenerating embeddings for {len(texts)} chunks...")

        embeddings = self.embedding_model.embed_documents(texts)

        ids = [
            f"chunk_{chunk['chunk_id']}"
            for chunk in chunks
        ]

        metadatas = [
            {
                "document": chunk["document"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"]
            }
            for chunk in chunks
        ]

        print("Storing chunks in ChromaDB...")

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )

        print("Chunks stored successfully.")

    def count(self):
        """
        Return the number of stored chunks.
        """
        return self.collection.count()


if __name__ == "__main__":
    data_directory = (
        Path(__file__).parent.parent
        / "data"
        / "documents"
    )

    print("=" * 60)
    print("ATMAN RAG - VECTOR DATABASE SETUP")
    print("=" * 60)

    # Load PDFs
    print("\n1. Loading PDFs...")
    documents = load_all_pdfs(data_directory)

    # Create chunks
    print("\n2. Creating chunks...")
    chunks = create_chunks(documents)

    print(f"Created {len(chunks)} chunks.")

    # Initialize vector store
    print("\n3. Initializing vector store...")
    vector_store = VectorStore()

    # Store chunks
    print("\n4. Indexing chunks...")
    vector_store.add_chunks(chunks)

    # Verify
    print("\n" + "=" * 60)
    print("VECTOR DATABASE READY")
    print("=" * 60)

    print(f"Total chunks in ChromaDB: {vector_store.count()}")
    print(f"Database location: {CHROMA_PATH}")
