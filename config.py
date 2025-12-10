import os
from dotenv import load_dotenv

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
UNSTRUCTURED_API_KEY=os.getenv("UNSTRUCTURED_API_KEY")

DATA_DIR="data/docs"
VECTORSTORE_DIR="vectorstore"
# Backwards-compatible name expected elsewhere in the code
VECTORSTORE = VECTORSTORE_DIR
 
EMBEDDING_MODEL="text-embedding-3-small"
MODEL_NAME="gpt-3.5-turbo-1106"