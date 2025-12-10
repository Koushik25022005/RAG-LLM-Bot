from pathlib import Path
from typing import List

from unstructured_client import UnstructuredClient
from unstructured_client.models import shared, operations
from unstructured_client.models.errors import SDKError
from unstructured.staging.base import dict_to_elements
from unstructured.partition.pptx import partition_pptx
from unstructured.chunking.title import chunk_by_title

from langchain_core.documents import Document

from langchain_text_splitters import RecursiveTextCharacterSplitter
from config import UNSTRUCTURED_API_KEY, DATA_DIR

def load_documents() -> List[Document]:
    """ Load documents from Unstrutured API
    and partition them into elements."""
    client = UnstructuredClient(api_auth_key=UNSTRUCTURED_API_KEY)

    document: List[Document] =  []

    for file in Path(DATA_DIR).glob("*"):
        if not file.is_file():
            continue

        if file.suffix.lower == ".pdf":
            with open(file, "rb") as f:
                files = shared.Files(
                    content=f.read(),
                    filename=file.name,
                )
            request = operations.PartitionRequest(
                shared.PartitionParamters(
                    files=files,
                    strategy="hi_res",
                    hi_res_model_name="yolox",
                    skip_infer_table_type=[],
                    pdf_infer_table_structure=True
                )
            )
            try:
                resp = client.general.paritition(request)
                elements = dict_to_elements(resp)
            except SDKError as e:
                print(f"Error partitioning file: {e}")

        elif file.suffix.lower() == ".pptx":
            elements = partition_pptx(filename=str(file))

        elements = chunk_by_title(elements)
        for element in elements:
            metadata = element.metadata.to_dict()
            metadata["source"] = str(file)
            document.append(Document(page_content=element.text, metadata=metadata))

    return document

def split_documents(docs: List[Document]) -> List[Document]:
    """ Split documents into smaller chunks."""
    text_splitter = RecursiveTextCharacterSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = []
    for doc in docs:
        splits = text_splitter.split_text(doc.page_content)
        for split in splits:
            split_docs.append(Document(page_content=split, metadata=doc.metadata))
    return split_docs
        

