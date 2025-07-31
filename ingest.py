"""
ingest.py – Document ingestion script using SimpleEmbeddings class
for better Streamlit Cloud compatibility
"""

import glob
import pathlib
import chromadb
import pdfplumber
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# ─── Custom SimpleEmbeddings Class (same as in rag_chain.py) ──────────
class SimpleEmbeddings:
    def __init__(self):
        print("🔄 Loading embedding model for ingestion...")
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        print("✅ Embedding model loaded successfully")
    
    def embed_documents(self, texts):
        """Embed a list of documents"""
        if not texts:
            return []
        
        # Process in batches to avoid memory issues
        batch_size = 32
        all_embeddings = []
        
        print(f"📊 Processing {len(texts)} documents in batches...")
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, 
                                  return_tensors="pt", max_length=512)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
                all_embeddings.extend(embeddings)
            
            print(f"✅ Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        
        return all_embeddings
    
    def embed_query(self, text):
        """Embed a single query"""
        inputs = self.tokenizer([text], padding=True, truncation=True, 
                               return_tensors="pt", max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).numpy()[0]
        
        return embedding

# Initialize embedding model
embedder = SimpleEmbeddings()

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
    print("❌ No PDF files found in the data/ folder. Please add PDF files and try again.")
else:
    print(f"📄 Found {len(pdf_files)} PDF files")
    
    for file_path in pdf_files:
        print(f"📖 Processing: {pathlib.Path(file_path).name}")
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
        print("❌ No text content found in PDF files.")
