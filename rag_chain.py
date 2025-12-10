from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

from config import MODEL_NAME
from vectorstore import load_vectorstore


class SimpleRetrievalChain:
    """A lightweight retrieval+LLM chain that does not depend on `langchain.chains`.

    Usage:
        chain = SimpleRetrievalChain(llm, retriever, prompt_template, k=3)
        answer = chain.run("Your question")
    """

    def __init__(self, llm, retriever, prompt_template: PromptTemplate, k: int = 3):
        self.llm = llm
        self.retriever = retriever
        self.prompt = prompt_template
        self.k = k

    def _build_context(self, docs):
        parts = []
        for d in docs[: self.k]:
            # LangChain Documents often have `page_content`; fallback to `content` or str(d)
            text = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
            parts.append(text)
        return "\n\n".join(parts)

    def run(self, question: str) -> str:
        # retrieve relevant documents
        # common retriever interface supports `get_relevant_documents(question)`
        if hasattr(self.retriever, "get_relevant_documents"):
            docs = self.retriever.get_relevant_documents(question)
        else:
            # fallback to `.retrieve` or `.get_relevant_documents`
            docs = self.retriever.retrieve(question) if hasattr(self.retriever, "retrieve") else []

        context = self._build_context(docs)

        system_text = self.prompt.format(question=question, context=context)
        system_msg = SystemMessage(content=system_text)
        human_msg = HumanMessage(content=question)

        # ChatOpenAI.generate expects a batch: list[list[BaseMessage]]
        result = self.llm.generate([[system_msg, human_msg]])

        # extract text from LLMResult -> generations
        try:
            text = result.generations[0][0].text
        except Exception:
            # fallback - return the raw result for debugging
            text = str(result)
        return text


def get_rag_chain(k: int = 3):
    vectordb = load_vectorstore()
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": k})

    template = """
    Your helpful AI Assistant that will help you decode and understand any topic
    from the provided PDFs and presentations.

    Question: {question}
    =====================
    Context: {context}
    """

    prompt = PromptTemplate(input_variables=["question", "context"], template=template)

    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0)
    chain = SimpleRetrievalChain(llm=llm, retriever=retriever, prompt_template=prompt, k=k)
    return chain

