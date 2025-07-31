"""
rag_chain.py – RAG pipeline using OpenAI API via LangChain
Streamlit Cloud compatible with minimal memory footprint
"""

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langdetect import detect, LangDetectException
import os

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

def get_openai_api_key():
    """Get OpenAI API key from Streamlit secrets or environment"""
    import streamlit as st
    
    # Try to get from Streamlit secrets first
    try:
        return st.secrets["OPENAI_API_KEY"]
    except:
        # Fallback to environment variable
        return os.getenv("OPENAI_API_KEY")

def init_openai_models():
    """Initialize OpenAI models with API key"""
    api_key = get_openai_api_key()
    
    if not api_key:
        return None, None
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",  # Cost-effective embedding model
        openai_api_key=api_key
    )
    
    # Initialize chat model
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",  # Cost-effective chat model
        temperature=0.7,
        openai_api_key=api_key
    )
    
    return embeddings, llm

# ─── Initialize models ────────────────────────
embeddings, llm = init_openai_models()

# ─── Vector DB ────────────────────────
try:
    if embeddings:
        vectordb = Chroma(
            persist_directory="chroma_db",
            embedding_function=embeddings
        )
        print("✅ Vector database loaded successfully")
    else:
        vectordb = None
        print("⚠️ OpenAI API key not found - vector DB unavailable")
except Exception as e:
    print(f"Warning: Could not load vector database: {e}")
    vectordb = None

# ─── Memory and Chain ────────────────────────
if llm:
    memory = ConversationBufferMemory(
        return_messages=True,
        input_key="question",
        output_key="answer",
        memory_key="chat_history"
    )
    
    if vectordb:
        base_rag_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
            memory=memory
        )
    else:
        base_rag_chain = None
else:
    base_rag_chain = None

def detect_user_language(text):
    """Detect language of user input"""
    try:
        return detect(text)
    except LangDetectException:
        return "en"

def get_greeting_reply(lang_code):
    """Get localized greeting response"""
    return GREETINGS_REPLY.get(lang_code, GREETINGS_REPLY['en'])

def rag_chain(question, forced_language=None):
    """Main RAG function with OpenAI API"""
    
    # Check if OpenAI is available
    if not llm:
        return {"answer": "⚠️ Please add your OpenAI API key to use the chatbot. Go to 'Manage app' → 'Settings' → 'Secrets' and add: OPENAI_API_KEY='your-key-here'"}
    
    # Detect or use forced language
    user_lang = forced_language or detect_user_language(question)
    
    # Handle greetings
    question_clean = question.strip().lower()
    if any(greeting in question_clean for greeting in GREETINGS_LIST):
        return {"answer": get_greeting_reply(user_lang)}
    
    # Use RAG chain if available, otherwise direct LLM
    if base_rag_chain:
        try:
            # Create system prompt for language
            lang_name = LANG_PROMPTS.get(user_lang, "English")
            enhanced_question = f"Please answer in {lang_name}. Focus on street vendor digitalization, government schemes, and digital payments in India.\n\nQuestion: {question}"
            
            response = base_rag_chain.invoke({"question": enhanced_question})
            return response
        except Exception as e:
            print(f"RAG chain error: {e}")
            # Fallback to direct LLM
    
    # Direct LLM fallback
    try:
        lang_name = LANG_PROMPTS.get(user_lang, "English")
        prompt = f"""You are a helpful assistant for Indian street vendors. Answer in {lang_name}.
        
        Question: {question}
        
        Provide helpful information about street vendor digitalization, government schemes like PM-SVANidhi, digital payments, UPI setup, or related topics for Indian street vendors."""
        
        response = llm.invoke(prompt)
        return {"answer": response.content}
    except Exception as e:
        return {"answer": f"Sorry, I encountered an error: {str(e)}"}

# Export functions for app.py
__all__ = ['rag_chain', 'LANG_PROMPTS', 'detect_user_language']
