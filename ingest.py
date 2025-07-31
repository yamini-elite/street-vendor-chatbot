"""
ingest.py – Simple document ingestion (can be enhanced later)
For now, this is a placeholder since we're using Gemini's built-in knowledge
"""

import glob
import pathlib
import streamlit as st

def simple_pdf_reader():
    """Simple PDF reader - can be enhanced with vector embeddings later"""
    try:
        import pdfplumber
        
        pdf_files = glob.glob("data/*.pdf")
        if not pdf_files:
            print("❌ No PDF files found in the data/ folder.")
            return
        
        print(f"📄 Found {len(pdf_files)} PDF files")
        
        for file_path in pdf_files:
            print(f"📖 Processing: {pathlib.Path(file_path).name}")
            
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
                
                # For now, just print summary
                print(f"✅ Extracted {len(text)} characters from {pathlib.Path(file_path).name}")
        
        print("ℹ️  Note: This is a basic PDF reader. Vector embeddings can be added later for advanced document search.")
        
    except ImportError:
        print("❌ pdfplumber not installed. Run: pip install pdfplumber")
    except Exception as e:
        print(f"❌ Error processing PDFs: {e}")

if __name__ == "__main__":
    simple_pdf_reader()
