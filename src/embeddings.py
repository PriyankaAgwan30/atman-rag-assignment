from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingModel:
    def __init__(self):
        print(f"Loading embedding model: {MODEL_NAME}")
        self.model = SentenceTransformer(MODEL_NAME)
        print("Embedding model loaded successfully.")

    def embed_documents(self, texts):
        """
        Generate embeddings for multiple documents.
        """
        return self.model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True
        )

    def embed_query(self, query):
        """
        Generate an embedding for a single query.
        """
        return self.model.encode(
            query,
            normalize_embeddings=True
        )


if __name__ == "__main__":
    model = EmbeddingModel()

    test_texts = [
        "How do I reset my password?",
        "What is the company's PTO policy?"
    ]

    embeddings = model.embed_documents(test_texts)

    print("\n" + "=" * 60)
    print("EMBEDDING TEST")
    print("=" * 60)

    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Embedding dimensions: {len(embeddings[0])}")

    query_embedding = model.embed_query(
        "How can I change my password?"
    )

    print(f"Query embedding dimensions: {len(query_embedding)}")
