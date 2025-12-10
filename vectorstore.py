from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from config import VECTORSTORE, EMBEDDING_MODEL

def build_vectorstore(docs):
    embeddings=OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectordb = Chroma.from_documents(
        documents=docs,
        embeddings=embeddings,
        persist_directory=VECTORSTORE
    )
    vectordb.persist()
    return vector db

def load_vectorstore():
    embeddings=