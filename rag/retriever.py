# # retriever.py
# #from langchain.vectorstores import FAISS
# from langchain_community.vectorstores import FAISS

# #from langchain.embeddings import HuggingFaceEmbeddings
# #from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# from pathlib import Path
# import os

# # Path to local FAISS index
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# faiss_local_path = os.path.join(BASE_DIR, "data")

# #faiss_local_path = Path(__file__).parent / "data"

# # Load embeddings
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # Load FAISS index
# #vectorstore = FAISS.load_local(faiss_local_path, embedding_model)
# vectorstore = FAISS.load_local(
#     faiss_local_path, 
#     embedding_model, 
#     allow_dangerous_deserialization=True
# )

# # Create a retriever for querying
# retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  








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