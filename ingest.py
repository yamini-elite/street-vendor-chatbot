"""
ingest.py – Document ingestion script for creating vector embeddings
Compatible with HuggingFace embeddings for Streamlit Cloud deployment
"""

import glob
import pathlib
import chromadb
import pdfplumber
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize embedding model (same as in rag_chain.py)
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create/connect to persistent ChromaDB instance
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("vendor_docs")

def pdf_to_text(file_path):
    """Extract text from a PDF file."""
    with pdfplumber.open(file_path) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]
    return "\n".join(pages)

# Process all PDFs in the data folder
docs = []
ids = []

for file_path in glob.glob("data/*.pdf"):
    text = pdf_to_text(file_path)
    # Split text into 1000-character chunks
    for i in range(0, len(text), 1000):
        chunk = text[i:i+1000]
        if chunk.strip():  # Only add non-empty chunks
            docs.append(chunk)
            ids.append(f"{pathlib.Path(file_path).stem}_{i}")

print(f"📄 Found {len(docs)} text chunks to embed...")

if docs:
    # Create embeddings for all document chunks
    embeddings = embedder.embed_documents(docs)
    
    # Add documents, embeddings, and IDs to ChromaDB
    collection.add(
        ids=ids, 
        documents=docs, 
        embeddings=embeddings
    )
    
    print("✅ Ingestion complete.")
else:
    print("❌ No documents found in the data/ folder. Please add PDF files and try again.")
