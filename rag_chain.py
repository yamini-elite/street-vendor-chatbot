"""
rag_chain.py – Ultra-minimal RAG pipeline using transformers directly 
for maximum Streamlit Cloud compatibility with reduced memory usage
"""

from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import numpy as np
from langchain.vectorstores import Chroma
from langdetect import detect, LangDetectException
import logging
import os

# Configure logging to reduce noise
logging.getLogger("transformers").setLevel(logging.WARNING)

# Language detection and prompts
LANG_PROMPTS = {
    'en': "English", 'hi': "Hindi", 'mr': "Marathi", 'ta': "Tamil",
    'te': "Telugu", 'kn': "Kannada", 'gu': "Gujarati", 'bn': "Bengali",
    'pa': "Punjabi", 'ml': "Malayalam", 'ur': "Urdu"
}

GREETINGS_LIST = [
    "hi", "hello", "hii", "hey",
    "नमस्ते", "हाय", "नमस्कार", "வணக்கம்", "ஹாய்",
    "హాయ్", "ഹായ്", "നമസ്കാരം", "ಹಾಯ್", "હાય", "হ্যালো", "ਸਤਿ ਸ਼੍ਰੀ ਅਕਾਲ"
]

GREETINGS_REPLY = {
    'en': "Hello! 👋 How can I help you today with street vendor digitalization?",
    'hi': "नमस्ते! मैं आपके स्ट्रीट वेंडर से जुड़े सवालों में कैसे मदद कर सकता हूँ?",
    'mr': "नमस्कार! मी स्ट्रीट वेंडर विषयी आपल्याला कशी मदत करू शकतो?",
    'ta': "வணக்கம்! தெரு வியாபாரிகள் தொடர்பான எந்த உதவியும் கேளுங்கள்.",
    'te': "హాయ్! స్ట్రీట్ వెండర్ సంబంధించిన మీ ప్రశ్నలకు సహాయం చేస్తాను.",
    'gu': "હાય! સ્ટ્રીટ વેન્ડર પ્રશ્નો માટે હું મદદ કરી શકું છું.",
    'bn': "হ্যালো! রাস্তার বিক্রেতা সংক্রান্ত যেকোনো প্রশ্ন করুন।",
    'pa': "ਸਤਿ ਸ਼੍ਰੀ ਅਕਾਲ! ਤੁਸੀਂ ਸਟਰੀਟ ਵੈਂਡਰ ਸੰਬੰਧੀ ਕੋਈ ਸਵਾਲ ਪੁੱਛੋ।",
    'ml': "ഹായ്! സ്ട്രീറ്റ് വെണ്ടർ സംബന്ധിച്ചുള്ള നിങ്ങളുടെ ചോദ്യങ്ങളിൽ സഹായിക്കാം।",
}

# ─── Custom SimpleEmbeddings Class ──────────────────────
class SimpleEmbeddings:
    def __init__(self):
        print("🔄 Loading embedding model...")
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
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, 
                                  return_tensors="pt", max_length=512)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
                all_embeddings.extend(embeddings)
        
        return all_embeddings
    
    def embed_query(self, text):
        """Embed a single query"""
        inputs = self.tokenizer([text], padding=True, truncation=True, 
                               return_tensors="pt", max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).numpy()[0]
        
        return embedding

# ─── Initialize Embeddings ──────────────────────
embedder = SimpleEmbeddings()

# ─── Vector DB ────────────────────────
try:
    vectordb = Chroma(
        persist_directory="chroma_db",
        embedding_function=embedder
    )
    print("✅ Vector database loaded successfully")
except Exception as e:
    print(f"Warning: Could not load vector database: {e}")
    vectordb = None

# ─── HuggingFace Text Generation Pipeline ────────
try:
    print("🔄 Loading text generation model...")
    # Use the smaller model for better memory efficiency
    llm_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",  # Smaller model for Streamlit Cloud
        max_length=400,
        do_sample=True,
        temperature=0.7,
        device=-1,  # Use CPU
        model_kwargs={"torch_dtype": torch.float32}
    )
    print("✅ Text generation pipeline loaded successfully")
except Exception as e:
    print(f"❌ Error loading HuggingFace pipeline: {e}")
    llm_pipeline = None

def detect_user_language(text):
    """Detect language of user input"""
    try:
        return detect(text)
    except LangDetectException:
        return "en"

def get_greeting_reply(lang_code):
    """Get localized greeting response"""
    return GREETINGS_REPLY.get(lang_code, GREETINGS_REPLY['en'])

def search_documents(question, k=4):
    """Search for relevant documents"""
    if vectordb is None:
        return []
    
    try:
        retriever = vectordb.as_retriever(search_kwargs={"k": k})
        docs = retriever.get_relevant_documents(question)
        return [doc.page_content for doc in docs]
    except Exception as e:
        print(f"Document search error: {e}")
        return []

def generate_answer(question, context_docs, user_lang):
    """Generate answer using HuggingFace pipeline"""
    if llm_pipeline is None:
        return "Sorry, the AI model is not available right now. Please try again later."
    
    # Create context from retrieved documents
    context = "\n".join(context_docs[:3]) if context_docs else ""
    
    # Create a comprehensive prompt
    lang_name = LANG_PROMPTS.get(user_lang, "English")
    
    if context:
        prompt = f"""Context about Indian street vendor policies and schemes:
{context}

Question: {question}

Based on the context above, provide a helpful answer about street vendor digitalization, government schemes, or digital payments in India. If context doesn't fully cover the question, add general knowledge about Indian street vendor policies. Answer in {lang_name}."""
    else:
        prompt = f"""Question: {question}

Provide a helpful answer about street vendor digitalization, government schemes like PM-SVANidhi, digital payments, UPI setup, or related topics for Indian street vendors. Answer in {lang_name}."""
    
    try:
        # Generate response
        response = llm_pipeline(prompt, max_length=350, min_length=30)
        answer = response[0]['generated_text'] if response else "I apologize, but I'm having trouble generating a response right now."
        return answer
    except Exception as e:
        print(f"Generation error: {e}")
        return f"I apologize, but I encountered an error while generating the response. Please try rephrasing your question."

def rag_chain(question, forced_language=None):
    """Main RAG function - handles greetings and questions"""
    
    # Detect or use forced language
    user_lang = forced_language or detect_user_language(question)
    
    # Handle greetings
    question_clean = question.strip().lower()
    if any(greeting in question_clean for greeting in GREETINGS_LIST):
        return {"answer": get_greeting_reply(user_lang)}
    
    # Search for relevant documents
    context_docs = search_documents(question)
    
    # Generate answer
    answer = generate_answer(question, context_docs, user_lang)
    
    return {"answer": answer}

# Test function (for debugging)
def test_pipeline():
    """Test if the pipeline is working"""
    try:
        print("🧪 Testing pipeline...")
        test_response = rag_chain("Hello")
        print(f"Test successful: {test_response}")
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    test_pipeline()
