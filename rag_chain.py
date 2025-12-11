# rag_chain.py

from typing import Any, Dict

from langchain_openai import ChatOpenAI  # or OpenAI if you really want completions
from langchain_classic.chains.llm import LLMChain
from langchain_classic.chains.conversational_retrieval import ConversationalRetrievalChain
from langchain_classic.chains.qa_with_sources import load_qa_with_sources_chain
from langchain_core.prompts.prompt import PromptTemplate

from vectorstore import load_vectorstore
from config import MODEL_NAME  # e.g. "gpt-4.1-mini"


# Prompt used to turn follow-up questions into standalone questions
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
    """You are a helpful AI assistant.

Given the following chat history and a follow-up question,
rewrite the follow-up question to be a standalone question.

Chat history:
{chat_history}

Follow-up question:
{question}

Standalone question:"""
)


def get_rag_chain(k: int = 5) -> ConversationalRetrievalChain:
    """Build a Conversational RAG chain using a retriever + map_reduce QA chain."""
    # 1) Load vectorstore & retriever
    vectordb = load_vectorstore()
    retriever = vectordb.as_retriever(search_kwargs={"k": k})

    # 2) LLM
    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0,
    )
    # If you really want the completions-style LLM instead:
    # from langchain_openai import OpenAI
    # llm = OpenAI(temperature=0)

    # 3) Chain that answers questions using retrieved docs (with sources)
    doc_chain = load_qa_with_sources_chain(
        llm=llm,
        chain_type="map_reduce",  # or "stuff" / "refine"
    )

    # 4) Chain that takes chat history + follow-up question → standalone question
    question_generator_chain = LLMChain(
        llm=llm,
        prompt=CONDENSE_QUESTION_PROMPT,
    )  

    # 5) Conversational Retrieval Chain
    qa_chain = ConversationalRetrievalChain(
        retriever=retriever,
        question_generator=question_generator_chain,
        combine_docs_chain=doc_chain,
        return_source_documents=True,
    )

    return qa_chain
