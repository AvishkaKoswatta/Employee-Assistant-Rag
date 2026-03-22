# # embedder_documents.py
# #from langchain.embeddings import HuggingFaceEmbeddings
# #from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings
# #from langchain.vectorstores import FAISS
# #from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# #from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from document_loader import load_from_s3
# from dotenv import load_dotenv
# import boto3
# import os

# load_dotenv()

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# faiss_local_path = os.path.join(BASE_DIR, "data")

# os.makedirs(faiss_local_path, exist_ok=True)

# print(f"Saving FAISS to: {faiss_local_path}")

# bucket = os.getenv("BUCKET")
# json_key = os.getenv("JSON_KEY")
# faiss_key = os.getenv("FAISS_KEY")

# # Load documents from S3
# documents = load_from_s3(bucket=bucket, key=json_key)

# # Split into chunks
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500,
#     chunk_overlap=50
# )

# chunked_documents = []
# for doc in documents:
#     chunked_documents.extend(text_splitter.split_documents([doc]))

# print(f"Original docs: {len(documents)}, Chunked docs: {len(chunked_documents)}")

# # Create embeddings and vectorstore
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# vectorstore = FAISS.from_documents(chunked_documents, embedding_model)

# # Save locally
# vectorstore.save_local(faiss_local_path)

# # Upload FAISS index to S3
# s3 = boto3.client("s3")
# for file in ["index.faiss", "index.pkl"]: 
#     s3.upload_file(os.path.join(faiss_local_path, file), bucket, f"{faiss_key}/{file}")

# print("Embedding completed and FAISS index saved to S3.")












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