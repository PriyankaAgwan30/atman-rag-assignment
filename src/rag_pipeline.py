from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from retriever import Retriever
from llm import LLM


class RAGPipeline:
    def __init__(self, top_k=5):
        print("Initializing RAG pipeline...")

        self.retriever = Retriever(top_k=top_k)
        self.llm = LLM()

        print("RAG pipeline ready.")

    def build_prompt(self, question, retrieved_chunks):
        """
        Build a grounded prompt using only retrieved document context.
        """

        context_parts = []

        for i, chunk in enumerate(retrieved_chunks, start=1):
            context_parts.append(
                f"""
SOURCE {i}
Document: {chunk['document']}
Page: {chunk['page']}
Chunk ID: {chunk['chunk_id']}

Content:
{chunk['text']}
"""
            )

        context = "\n".join(context_parts)

        prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the information provided
in the document context below.

IMPORTANT RULES:
1. Do not use outside knowledge.
2. Do not invent or assume information.
3. If the answer is not supported by the provided context,
   clearly say that the information is not available in the
   provided documents.
4. Give a concise and direct answer.
5. When possible, include the relevant numbers, dates,
   names, or technical details from the context.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

        return prompt

    def answer(self, question):
        """
        Retrieve relevant chunks and generate a grounded answer.
        """

        retrieved_chunks = self.retriever.retrieve(question)

        prompt = self.build_prompt(
            question,
            retrieved_chunks
        )

        answer = self.llm.generate(prompt)

        return {
            "question": question,
            "answer": answer,
            "sources": retrieved_chunks
        }


if __name__ == "__main__":

    rag = RAGPipeline(top_k=5)

    test_questions = [
        "How many paid sick days do employees receive?",
        "How long are API refresh tokens valid?",
        "What is the uptime guarantee for the Standard plan?",
        "What is the company's stock price?"
    ]

    for question in test_questions:

        print("\n" + "=" * 70)
        print(f"QUESTION: {question}")
        print("=" * 70)

        result = rag.answer(question)

        print("\nANSWER:")
        print(result["answer"])

        print("\nSOURCES:")

        for source in result["sources"]:
            print(
                f"- {source['document']} "
                f"(Page {source['page']}, "
                f"Chunk {source['chunk_id']})"
            )
