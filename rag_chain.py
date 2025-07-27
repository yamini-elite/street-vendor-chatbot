"""
rag_chain.py – Universal pipeline: if user says 'hi' or any message, answer from Ollama in user's language; blend document search Q&A with open chat for off-topic, chit-chat, or unsupported queries.
"""

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langdetect import detect, LangDetectException

# Multilingual greeting triggers for bonus handling
GREETINGS_LIST = [
    "hi", "hello", "hii", "hey",
    "नमस्ते", "हाय", "नमस्कार", "வணக்கம்", "ஹாய்",
    "హాయ్", "ഹായ്", "നമസ്കാരം", "ಹಾಯ್", "હાય", "হ্যালো", "ਸਤਿ ਸ਼੍ਰੀ ਅਕਾਲ"
]

LANG_PROMPTS = {
    'en': "English", 'hi': "Hindi (हिंदी)", 'mr': "Marathi (मराठी)", 'ta': "Tamil (தமிழ்)",
    'te': "Telugu (తెలుగు)", 'kn': "Kannada (ಕನ್ನಡ)", 'gu': "Gujarati (ગુજરાતી)", 'bn': "Bengali (বাংলা)",
    'pa': "Punjabi (ਪੰਜਾਬੀ)", 'ml': "Malayalam (മലയാളം)", 'ur': "Urdu (اردو)"
}

BASE_SYSTEM_PROMPT = (
    "You are a helpful, friendly assistant for Indian street vendors. "
    "Always reply in the same language as the user. If you can answer from your document knowledge, do so; "
    "otherwise, reply as a friendly LLM chat assistant in the user's language."
)

# Langchain vector DB/LLM/memory setup (unchanged)
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="chroma_db", embedding_function=embedder)
llm = ChatOllama(base_url="http://localhost:11434", model="qwen2.5:0.5b")
memory = ConversationBufferMemory(return_messages=True, input_key="question", output_key="answer", memory_key="chat_history")

# Core RAG chain for "knowledge" questions
base_rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
    memory=memory
)

def detect_user_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "en"

def rag_chain(question, forced_language=None):
    # Detect language or use forced
    user_lang = forced_language or detect_user_language(question)
    lang_label = LANG_PROMPTS.get(user_lang, "English")

    # If input is a short greeting, reply in user's language
    if question.strip().lower() in GREETINGS_LIST:
        hello_replies = {
            'en': "Hello! 👋 How can I help you today?",          'hi': "नमस्ते! कैसे मदद कर सकता हूँ?",
            'mr': "नमस्कार! मी कशी मदत करू?",                     'ta': "வணக்கம்! நான் எப்படி உதவலாம்?",
            'te': "హాయ్! ఎలా సహాయం చేయగలను?",                  'gu': "હાય! હું કેમ મદદ કરી શકું?",
            'bn': "হ্যালো! কীভাবে সাহায্য করতে পারি?",            'pa': "ਸਤਿ ਸ਼੍ਰੀ ਅਕਾਲ! ਮੈਂ ਤੁਹਾਡੀ ਕਿਵੇਂ ਮਦਦ ਕਰ ਸਕਦਾ ਹਾਂ?",
            'ml': "ഹായ്! എനിക്ക് നിങ്ങളെ എങ്ങനെ സഹായിക്കാം?",       'ur': "ہیلو! میں آپ کی مدد کیسے کرسکتا ہوں؟"
        }
        return {"answer": hello_replies.get(user_lang, hello_replies['en'])}

    # Try to answer from knowledge base (RAG)
    # Enhance the prompt with explicit "reply in X language" instruction
    system_prompt = (
        f"{BASE_SYSTEM_PROMPT} Reply in {lang_label}."
    )
    run_question = f"{system_prompt}\nUser Question: {question}"

    # Call chain (RAG)
    result = base_rag_chain.invoke({"question": run_question})

    answer = result.get("answer") if isinstance(result, dict) else str(result)
    # If RAG returns a poor/no answer, escalate to open LLM for generic chat
    insufficient_answers = [
        "", None,
        "Sorry, I don't know.", "I don’t know.", "I'm not sure.",
        "Sorry, I could not find an answer.",
        "I'm not able to answer that question."
    ]
    if not answer or any(x.lower() in (answer or "").lower() for x in insufficient_answers):
        # Use ChatOllama LLM for open domain response
        completion = llm.invoke(f"{system_prompt}\nUser Message: {question}")
        # This returns a LangChain ChatMessage object (check for draft/AI reply)
        if hasattr(completion, "content"):
            answer = completion.content
    return {"answer": answer}

# Optional: export detect_user_language and LANG_PROMPTS if you want language display in Streamlit
"""
rag_chain.py – Universal pipeline: if user says 'hi' or any message, answer from Ollama in user's language; blend document search Q&A with open chat for off-topic, chit-chat, or unsupported queries.
"""

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langdetect import detect, LangDetectException

# Multilingual greeting triggers for bonus handling
GREETINGS_LIST = [
    "hi", "hello", "hii", "hey",
    "नमस्ते", "हाय", "नमस्कार", "வணக்கம்", "ஹாய்",
    "హాయ్", "ഹായ്", "നമസ്കാരം", "ಹಾಯ್", "હાય", "হ্যালো", "ਸਤਿ ਸ਼੍ਰੀ ਅਕਾਲ"
]

LANG_PROMPTS = {
    'en': "English", 'hi': "Hindi (हिंदी)", 'mr': "Marathi (मराठी)", 'ta': "Tamil (தமிழ்)",
    'te': "Telugu (తెలుగు)", 'kn': "Kannada (ಕನ್ನಡ)", 'gu': "Gujarati (ગુજરાતી)", 'bn': "Bengali (বাংলা)",
    'pa': "Punjabi (ਪੰਜਾਬੀ)", 'ml': "Malayalam (മലയാളം)", 'ur': "Urdu (اردو)"
}

BASE_SYSTEM_PROMPT = (
    "You are a helpful, friendly assistant for Indian street vendors. "
    "Always reply in the same language as the user. If you can answer from your document knowledge, do so; "
    "otherwise, reply as a friendly LLM chat assistant in the user's language."
)

# Langchain vector DB/LLM/memory setup (unchanged)
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="chroma_db", embedding_function=embedder)
llm = ChatOllama(base_url="http://localhost:11434", model="qwen2.5:0.5b")
memory = ConversationBufferMemory(return_messages=True, input_key="question", output_key="answer", memory_key="chat_history")

# Core RAG chain for "knowledge" questions
base_rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
    memory=memory
)

def detect_user_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "en"

def rag_chain(question, forced_language=None):
    # Detect language or use forced
    user_lang = forced_language or detect_user_language(question)
    lang_label = LANG_PROMPTS.get(user_lang, "English")

    # If input is a short greeting, reply in user's language
    if question.strip().lower() in GREETINGS_LIST:
        hello_replies = {
            'en': "Hello! 👋 How can I help you today?",          'hi': "नमस्ते! कैसे मदद कर सकता हूँ?",
            'mr': "नमस्कार! मी कशी मदत करू?",                     'ta': "வணக்கம்! நான் எப்படி உதவலாம்?",
            'te': "హాయ్! ఎలా సహాయం చేయగలను?",                  'gu': "હાય! હું કેમ મદદ કરી શકું?",
            'bn': "হ্যালো! কীভাবে সাহায্য করতে পারি?",            'pa': "ਸਤਿ ਸ਼੍ਰੀ ਅਕਾਲ! ਮੈਂ ਤੁਹਾਡੀ ਕਿਵੇਂ ਮਦਦ ਕਰ ਸਕਦਾ ਹਾਂ?",
            'ml': "ഹായ്! എനിക്ക് നിങ്ങളെ എങ്ങനെ സഹായിക്കാം?",       'ur': "ہیلو! میں آپ کی مدد کیسے کرسکتا ہوں؟"
        }
        return {"answer": hello_replies.get(user_lang, hello_replies['en'])}

    # Try to answer from knowledge base (RAG)
    # Enhance the prompt with explicit "reply in X language" instruction
    system_prompt = (
        f"{BASE_SYSTEM_PROMPT} Reply in {lang_label}."
    )
    run_question = f"{system_prompt}\nUser Question: {question}"

    # Call chain (RAG)
    result = base_rag_chain.invoke({"question": run_question})

    answer = result.get("answer") if isinstance(result, dict) else str(result)
    # If RAG returns a poor/no answer, escalate to open LLM for generic chat
    insufficient_answers = [
        "", None,
        "Sorry, I don't know.", "I don’t know.", "I'm not sure.",
        "Sorry, I could not find an answer.",
        "I'm not able to answer that question."
    ]
    if not answer or any(x.lower() in (answer or "").lower() for x in insufficient_answers):
        # Use ChatOllama LLM for open domain response
        completion = llm.invoke(f"{system_prompt}\nUser Message: {question}")
        # This returns a LangChain ChatMessage object (check for draft/AI reply)
        if hasattr(completion, "content"):
            answer = completion.content
    return {"answer": answer}

# Optional: export detect_user_language and LANG_PROMPTS if you want language display in Streamlit
