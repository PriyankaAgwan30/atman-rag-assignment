import sys
from pathlib import Path

import streamlit as st

# Add src directory to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.append(str(SRC_DIR))

from rag_pipeline import RAGPipeline


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Atman RAG Assistant",
    page_icon="📚",
    layout="wide"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("📚 Atman Document RAG Assistant")

st.write(
    "Ask questions about the provided company documents. "
    "The system answers only using information retrieved from the documents."
)


# --------------------------------------------------
# Initialize RAG pipeline
# --------------------------------------------------

@st.cache_resource
def load_rag_pipeline():
    return RAGPipeline(top_k=5)


with st.spinner("Loading RAG pipeline..."):
    rag = load_rag_pipeline()


# --------------------------------------------------
# Question input
# --------------------------------------------------

question = st.text_input(
    "Ask a question",
    placeholder="Example: How many paid sick days do employees receive?"
)


# --------------------------------------------------
# Ask button
# --------------------------------------------------

if st.button("🔍 Ask Question"):

    if not question.strip():
        st.warning("Please enter a question.")

    else:
        with st.spinner("Searching documents and generating answer..."):

            try:
                result = rag.answer(question)

                # ------------------------------
                # Answer
                # ------------------------------

                st.subheader("Answer")

                st.write(result["answer"])

                # ------------------------------
                # Sources
                # ------------------------------

                st.subheader("Sources")

                sources = result.get("sources", [])

                if sources:

                    for source in sources:

                        document = source.get("document", "Unknown document")
                        page = source.get("page", "Unknown")
                        chunk_id = source.get("chunk_id", "Unknown")

                        st.write(
                            f"📄 **{document}** — "
                            f"Page {page}, Chunk {chunk_id}"
                        )

                else:
                    st.info("No sources were returned.")

            except Exception as e:

                st.error("An error occurred while processing the question.")

                with st.expander("Technical details"):
                    st.exception(e)


# --------------------------------------------------
# Example questions
# --------------------------------------------------

st.divider()

st.subheader("Example Questions")

st.markdown("""
- How many paid sick days do employees receive?
- How long are API refresh tokens valid?
- What is the uptime guarantee for the Standard plan?
- What is the company's stock price?
""")
