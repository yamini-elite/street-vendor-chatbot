"""
ingest.py  –  Chunk every PDF in ./data, create sentence-transformer
embeddings, and store them in a persistent Chroma vector database.
Run this once each time you add or change PDFs.
"""

import glob, pathlib, chromadb, pdfplumber
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"          # small, RAM-friendly
embedder    = SentenceTransformer(EMBED_MODEL)

client      = chromadb.PersistentClient(path="chroma_db")
collection  = client.get_or_create_collection("vendor_docs")

def pdf_to_text(path):
    with pdfplumber.open(path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

docs, ids = [], []

for fpath in glob.glob("data/*.pdf"):
    text = pdf_to_text(fpath)
    for i in range(0, len(text), 1000):            # 1 000-char chunks
        chunk = text[i:i+1000]
        docs.append(chunk)
        ids.append(f"{pathlib.Path(fpath).stem}_{i}")

print(f"📄 Found {len(docs)} text chunks – embedding …")
embeddings = embedder.encode(docs, batch_size=32, show_progress_bar=True)
collection.add(ids=ids, documents=docs, embeddings=embeddings.tolist())
print("✅ Ingestion complete.")
