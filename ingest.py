"""
ingest.py – Document ingestion using OpenAI embeddings
"""

import glob
import pathlib
import chromadb
import pdfplumber
from langchain_openai import OpenAIEmbeddings
import os
import streamlit as st

def get_openai_api_key():
    """Get OpenAI API key"""
    try:
        return st.secrets["OPENAI_API_KEY"]
    except:
        return os.getenv("OPENAI_API_KEY")

# Initialize OpenAI embeddings
api_key = get_openai_api_key()
if not api_key:
    print("❌ OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
    exit(1)

embedder = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=api_key
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

print("📁 Scanning for PDF files...")
pdf_files = glob.glob("data/*.pdf")

if not pdf_files:
    print("❌ No PDF files found in the data/ folder.")
else:
    print(f"📄 Found {len(pdf_files)} PDF files")
    
    for file_path in pdf_files:
        print(f"📖 Processing: {pathlib.Path(file_path).name}")
        text = pdf_to_text(file_path)
        
        # Split text into 1000-character chunks
        for i in range(0, len(text), 1000):
            chunk = text[i:i+1000]
            if chunk.strip():
                docs.append(chunk)
                ids.append(f"{pathlib.Path(file_path).stem}_{i}")

    print(f"📄 Found {len(docs)} text chunks to embed...")

    if docs:
        # Create embeddings using OpenAI
        collection.add(
            ids=ids, 
            documents=docs, 
            embeddings=embedder.embed_documents(docs)
        )
        
        print("✅ Ingestion complete.")
    else:
        print("❌ No text content found in PDF files.")
