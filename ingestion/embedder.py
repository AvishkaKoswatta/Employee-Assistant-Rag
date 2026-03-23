# type: ignore
# embedder.py
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
from document_loader import load_from_s3
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# -----------------------------
# Config
# -----------------------------
BUCKET = os.getenv("BUCKET")
JSON_KEY = os.getenv("JSON_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "employee-handbook")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east1-gcp")  # region

# -----------------------------
# Initialize Pinecone client (new SDK)
# -----------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

# List existing indexes
existing_indexes = [idx.name for idx in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,  # must match embedding size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  # adjust region
    )

# Connect to the index
index = pc.Index(INDEX_NAME)

# -----------------------------
# Load JSON from S3
# -----------------------------
json_data = load_from_s3(bucket=BUCKET, key=JSON_KEY)

# Convert to Documents with metadata
# Load Documents from S3 (already returns list[Document])
documents = load_from_s3(bucket=BUCKET, key=JSON_KEY)

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    encode_kwargs={"normalize_embeddings": True}
)

# Upsert to Pinecone
vectorstore = PineconeVectorStore.from_documents(
    documents=documents,
    embedding=embedding_model,
    index_name=INDEX_NAME
)

print(f"Inserted {len(documents)} documents into Pinecone index: {INDEX_NAME}")