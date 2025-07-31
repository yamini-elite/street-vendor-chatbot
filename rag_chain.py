"""
rag_chain.py – RAG pipeline using HuggingFace transformers pipeline instead of Ollama
for Streamlit Cloud deployment compatibility
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from transformers import pipeline
from langdetect import detect, LangDetectException
import logging

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

# ─── Embeddings ──────────────────────
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

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
    # Use a multilingual model that works well for QA
    llm_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=512,
        do_sample=True,
        temperature=0.7,
        device=-1  # Use CPU (GPU not available on Streamlit Cloud)
    )
    print("✅ HuggingFace pipeline loaded successfully")
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

Based on the context above, please provide a helpful answer about street vendor digitalization, government schemes, or digital payments in India. If the context doesn't fully answer the question, supplement with general knowledge about Indian street vendor policies. Respond in {lang_name}."""
    else:
        prompt = f"""Question: {question}

Please provide a helpful answer about street vendor digitalization, government schemes like PM-SVANidhi, digital payments, UPI setup, or related topics for Indian street vendors. Respond in {lang_name}."""
    
    try:
        # Generate response
        response = llm_pipeline(prompt, max_length=300, min_length=50)
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
        test_response = rag_chain("Hello")
        print(f"Test successful: {test_response}")
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    test_pipeline()
