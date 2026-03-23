# type: ignore
# retriever.py
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

# Config
INDEX_NAME = os.getenv("PINECONE_INDEX", "employee-handbook")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east1-gcp")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    encode_kwargs={"normalize_embeddings": True}
)

# Connect to existing index
vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embedding_model
)

# Create retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
# k=5 means top 5 documents will be returned

print("Retriever ready.")