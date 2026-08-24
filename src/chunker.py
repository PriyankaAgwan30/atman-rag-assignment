from pathlib import Path
import sys

# Allow importing pdf_loader from the same src directory
sys.path.append(str(Path(__file__).parent))

from pdf_loader import load_all_pdfs


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def create_chunks(documents):
    """
    Split extracted PDF pages into overlapping chunks.

    Each chunk keeps the source document and page number.
    """

    chunks = []
    chunk_id = 0

    for document in documents:
        text = document["text"]
        doc_name = document["document"]
        page = document["page"]

        start = 0

        while start < len(text):
            end = start + CHUNK_SIZE

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "chunk_id": chunk_id,
                    "document": doc_name,
                    "page": page,
                    "text": chunk_text
                })

                chunk_id += 1

            if end >= len(text):
                break

            start = end - CHUNK_OVERLAP

    return chunks


if __name__ == "__main__":
    data_directory = Path(__file__).parent.parent / "data" / "documents"

    print("Loading PDF documents...\n")

    documents = load_all_pdfs(data_directory)

    print("\nCreating chunks...\n")

    chunks = create_chunks(documents)

    print("=" * 60)
    print("CHUNKING COMPLETE")
    print("=" * 60)

    print(f"Total pages: {len(documents)}")
    print(f"Total chunks: {len(chunks)}")

    print("\nSample chunks:\n")

    for chunk in chunks[:5]:
        print("-" * 60)
        print(f"Chunk ID : {chunk['chunk_id']}")
        print(f"Document : {chunk['document']}")
        print(f"Page     : {chunk['page']}")
        print(f"Length   : {len(chunk['text'])}")
        print(f"Text     : {chunk['text'][:300]}...")
