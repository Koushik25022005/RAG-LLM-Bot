import streamlit as st
from pathlib import Path

from unstructured_loader import load_documents
from vectorstore import build_vectorstore
from rag_chain import get_rag_chain
from splitters import split_documents  # or inline

from config import VECTORSTORE_DIR

st.title("📄 RAG with Unstructured + LangChain")

with st.sidebar:
    if st.button("Build Index"):
        docs = load_documents()
        chunks = split_documents(docs)
        build_vectorstore(chunks)
        st.success("Index built!")

if not Path(VECTORSTORE_DIR).exists():
    st.info("Build the index first.")
else:
    question = st.text_input("Ask a question")
    if question:
        chain = get_rag_chain()
        result = chain({"query": question})

        st.write(result["result"])

        with st.expander("Sources"):
            for doc in result["source_documents"]:
                st.write(doc.metadata["source"])
