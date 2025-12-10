from langchain_openai import ChatOpenAI
from langchain_core.chains import ConversationalRetrievelvalChain, LLMChain
from langchain_core.prompts import PromptTemplate


from config import CHAT_MODEL
from vectorstore import load_vectorstore


def get_rag_chain(k: int = 3):
    vectordb = load_vectorstore()
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k":k})

    template = """
    Your helpful AI Assistant that weill help you decode and understand any topic
    from any given topic from the provided context. So how can i help you today ??
    
    Question: {question}
    ======================
    Context: {context}
    
    """

    prompt = PromptTemplate(
        input_variables=["question", "coontext"],
        template=template
    )


