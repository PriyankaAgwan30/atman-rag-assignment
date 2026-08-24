from pathlib import Path
import sys

import chromadb

sys.path.append(str(Path(__file__).parent))

from embeddings import EmbeddingModel


CHROMA_PATH = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "atman_documents"


class Retriever:
    def __init__(self, top_k=5):
        self.top_k = top_k

        print("Initializing retriever...")

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_PATH)
        )

        self.collection = self.client.get_collection(
            name=COLLECTION_NAME
        )

        self.embedding_model = EmbeddingModel()

        print(
            f"Retriever ready. "
            f"Documents in database: {self.collection.count()}"
        )

    def retrieve(self, query):
        """
        Retrieve the most semantically relevant chunks
        for a user query.
        """

        query_embedding = self.embedding_model.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved_chunks = []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances
        ):
            retrieved_chunks.append({
                "text": document,
                "document": metadata["document"],
                "page": metadata["page"],
                "chunk_id": metadata["chunk_id"],
                "distance": distance
            })

        return retrieved_chunks


if __name__ == "__main__":

    retriever = Retriever(top_k=5)

    test_questions = [
        "How many paid sick days do employees receive?",
        "How long are API refresh tokens valid?",
        "What is the uptime guarantee for the Standard plan?"
    ]

    for question in test_questions:

        print("\n" + "=" * 70)
        print(f"QUESTION: {question}")
        print("=" * 70)

        results = retriever.retrieve(question)

        for rank, result in enumerate(results, start=1):

            print(f"\nRank {rank}")
            print(f"Document : {result['document']}")
            print(f"Page     : {result['page']}")
            print(f"Chunk ID  : {result['chunk_id']}")
            print(f"Distance : {result['distance']:.4f}")
            print(f"Text     : {result['text'][:500]}")
