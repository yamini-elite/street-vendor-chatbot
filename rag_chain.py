"""
rag_chain.py – RAG pipeline using Google Gemini API instead of OpenAI
100% FREE - No payment required!
"""

import google.generativeai as genai
import streamlit as st
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

def get_gemini_api_key():
    """Get Gemini API key from Streamlit secrets or environment"""
    try:
        return st.secrets["GEMINI_API_KEY"]
    except:
        return os.getenv("GEMINI_API_KEY")

def init_gemini():
    """Initialize Google Gemini API"""
    api_key = get_gemini_api_key()
    
    if not api_key:
        return None
    
    try:
        genai.configure(api_key=api_key)
        # Use Gemini 1.5 Flash - it's free and excellent for multilingual tasks
        model = genai.GenerativeModel('gemini-1.5-flash')
        print("✅ Google Gemini initialized successfully")
        return model
    except Exception as e:
        print(f"❌ Error initializing Gemini: {e}")
        return None

# Initialize Gemini model
gemini_model = init_gemini()

def detect_user_language(text):
    """Detect language of user input"""
    try:
        return detect(text)
    except LangDetectException:
        return "en"

def get_greeting_reply(lang_code):
    """Get localized greeting response"""
    return GREETINGS_REPLY.get(lang_code, GREETINGS_REPLY['en'])

def search_documents_simple(question):
    """Simple document search - can be enhanced with vector search later"""
    # This is a placeholder - you can add vector search here if you have ingested documents
    # For now, return empty list to use pure Gemini responses
    return []

def generate_gemini_response(question, context_docs, user_lang):
    """Generate response using Google Gemini API"""
    if not gemini_model:
        return "⚠️ Please add your Gemini API key to use the chatbot. Go to 'Manage app' → 'Settings' → 'Secrets' and add: GEMINI_API_KEY='your-key-here'"
    
    # Create context from documents if available
    context = "\n".join(context_docs[:3]) if context_docs else ""
    
    # Create language-specific prompt
    lang_name = LANG_PROMPTS.get(user_lang, "English")
    
    if context:
        prompt = f"""You are a helpful assistant for Indian street vendors. Please respond in {lang_name}.

Context from government documents:
{context}

User Question: {question}

Based on the context above and your knowledge, provide a helpful answer about street vendor digitalization, government schemes like PM-SVANidhi, digital payments, UPI setup, street vendor registration, or related topics for Indian street vendors. If the context doesn't fully answer the question, supplement with your general knowledge about Indian street vendor policies and digital initiatives.

Important: Always respond in {lang_name} language."""
    else:
        prompt = f"""You are a helpful assistant for Indian street vendors. Please respond in {lang_name}.

User Question: {question}

Provide a helpful and detailed answer about street vendor digitalization, government schemes like PM-SVANidhi, digital payments, UPI QR code setup, street vendor registration, or related topics for Indian street vendors. Use your knowledge of Indian government policies and digital initiatives for street vendors.

Important: Always respond in {lang_name} language."""
    
    try:
        # Generate response with Gemini
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini generation error: {e}")
        return f"I apologize, but I encountered an error while generating the response. Please try rephrasing your question. Error: {str(e)}"

def rag_chain(question, forced_language=None):
    """Main RAG function using Google Gemini API"""
    
    # Detect or use forced language
    user_lang = forced_language or detect_user_language(question)
    
    # Handle greetings
    question_clean = question.strip().lower()
    if any(greeting in question_clean for greeting in GREETINGS_LIST):
        return {"answer": get_greeting_reply(user_lang)}
    
    # Search for relevant documents (placeholder for now)
    context_docs = search_documents_simple(question)
    
    # Generate answer using Gemini
    answer = generate_gemini_response(question, context_docs, user_lang)
    
    return {"answer": answer}

# Test function
def test_gemini():
    """Test if Gemini is working"""
    try:
        test_response = rag_chain("Hello")
        print(f"Gemini test successful: {test_response}")
        return True
    except Exception as e:
        print(f"Gemini test failed: {e}")
        return False

# Export functions for app.py
__all__ = ['rag_chain', 'LANG_PROMPTS', 'detect_user_language']

if __name__ == "__main__":
    test_gemini()
