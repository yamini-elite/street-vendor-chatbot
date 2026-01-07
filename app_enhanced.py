"""
app.py – Streamlit UI for Google Gemini-powered multilingual street vendor chatbot
100% FREE - No OpenAI costs!
"""

import streamlit as st
from rag_chain import rag_chain, LANG_PROMPTS, detect_user_language

st.set_page_config(
    page_title="Street Vendor Digitalisation Agent",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    .chat-container {
        background: #f8f9fa;
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
    .sidebar-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border: 1px solid #e9ecef;
    }
    .feature-list {
        background: #e8f5e8;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stChatMessage {
        border-radius: 15px;
        margin: 0.5rem 0;
        padding: 1rem;
    }
    .stChatMessage[data-testid="user"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .stChatMessage[data-testid="assistant"] {
        background: white;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .footer {
        text-align: center;
        color: #6c757d;
        font-size: 0.9rem;
        margin-top: 2rem;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Check for Gemini API key
def check_gemini_key():
    try:
        # Try Streamlit secrets first
        api_key = st.secrets.get("GEMINI_API_KEY")
        if api_key and len(api_key) > 10:
            return True

        # Fallback to environment variable
        import os
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        return bool(api_key and len(api_key) > 10)
    except:
        return False

# Main title with enhanced styling
st.markdown("""
<div class="main-header">
    <div class="main-title">🏪 Street Vendor Digitalisation Agent</div>
    <div class="subtitle">🌍 Multilingual AI Assistant for Indian Street Vendors</div>
    <div style="font-size: 0.9rem; opacity: 0.8; margin-top: 0.5rem;">
        Powered by FREE Google Gemini • Supporting 12+ Indian Languages
    </div>
</div>
""", unsafe_allow_html=True)

# API Key warning
if not check_gemini_key():
    st.error("""
    🔑 **Google Gemini API Key Required**
    
    To use this chatbot, you need to add your FREE Gemini API key:
    1. Click **"Manage app"** (bottom right)
    2. Go to **Settings** → **Secrets**
    3. Add: `GEMINI_API_KEY="your-key-here"`
    4. Get your FREE API key from: https://aistudio.google.com/
    
    ✅ **No credit card required - Completely FREE!**
    """)
    st.stop()

# Create two columns for main layout
col1, col2 = st.columns([3, 1])

with col2:
    # Language support sidebar with cards
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.header("🌐 Language Support")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    # Language selector
    st.subheader("🎯 Force Language (Optional)")
    force_language = st.selectbox(
        "Choose a specific language:",
        options=["Auto-detect"] + list(LANG_PROMPTS.keys()),
        format_func=lambda x: "🤖 Auto-detect" if x == "Auto-detect" else f"{LANG_PROMPTS[x]}",
        help="Select 'Auto-detect' to let the AI detect your language automatically, or choose a specific language."
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    # Show supported languages
    st.subheader("📋 Supported Languages")
    st.write("**Currently supported:**")
    for code, name in LANG_PROMPTS.items():
        st.write(f"• {name}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    # Usage tips
    st.subheader("💡 Usage Tips")
    st.info("""
    **🎯 How to use:**
    
    1. **Auto-detect**: Just type in any supported language
    2. **Force language**: Select a language above
    3. **Mix languages**: Switch languages anytime
    
    **📝 Example questions:**
    - "How to get PM-SVANidhi loan?"
    - "मुझे UPI QR कोड कैसे बनाना है?"
    - "પીએમ સ્વનિધિ લોન કેવી રીતે મેળવવી?"
    - "நான் சென்னையில் பழம் விற்கிறேன். UPI எப்படி செட்டப் பண்ணுவது?"
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    # Free API info
    st.subheader("🆓 100% Free Service")
    st.success("""
    **This chatbot is completely FREE!**
    
    ✅ No subscription fees
    ✅ No per-message charges  
    ✅ Powered by Google Gemini
    ✅ Generous daily limits
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col1:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        welcome_msg = """
        **🙏 Welcome to Street Vendor Digitalisation Agent!**
        
        I can help you with:
        • 💳 PM-SVANidhi loan applications
        • 📱 UPI QR code setup
        • 📋 Street vendor registration
        • 🏛️ Government schemes and benefits
        • 💰 Digital payment solutions
        • 📚 Street vendor rights and policies
        
        **Ask me anything in your preferred language 
        """
        st.session_state.messages.append({
            "role": "assistant", 
            "content": welcome_msg,
            "detected_lang": "en"
        })

    # Chat container with styling
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    chat_container = st.container()
    
    with chat_container:
        # Render chat history
        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
                # Show language detection info for user messages
                if msg["role"] == "user" and "detected_lang" in msg:
                    detected_lang_name = LANG_PROMPTS.get(msg["detected_lang"], "Unknown")
                    st.caption(f"🌍 Detected: {detected_lang_name}")
    st.markdown('</div>', unsafe_allow_html=True)

    # Chat input
    user_input = st.chat_input(
        "Ask in any language... / किसी भी भाषा में पूछें... / કોઈપણ ભાષામાં પૂછો...",
        key="chat_input"
    )

    if user_input:
        # Determine language to use
        forced_lang = None if force_language == "Auto-detect" else force_language
        
        # Detect or use forced language
        if forced_lang:
            detected_lang = forced_lang
            lang_display = f"🎯 Forced: {LANG_PROMPTS[forced_lang]}"
        else:
            detected_lang = detect_user_language(user_input)
            lang_display = f"🤖 Auto-detected: {LANG_PROMPTS.get(detected_lang, 'Unknown')}"
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
            st.caption(lang_display)
        
        # Save user message to session
        st.session_state.messages.append({
            "role": "user", 
            "content": user_input,
            "detected_lang": detected_lang
        })

        # Get AI response
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("_🤔 Thinking with Google Gemini..._")
            
            try:
                # Call the RAG chain with Gemini
                response = rag_chain(user_input, forced_lang)
                answer_text = response.get("answer", "Sorry, I couldn't generate a response.")
                
                # Display the response
                placeholder.markdown(answer_text)
                
                # Save assistant response
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer_text,
                    "response_lang": detected_lang
                })
                
            except Exception as e:
                error_msg = f"❌ **Error occurred:** {str(e)}\n\nPlease try again or contact support."
                placeholder.markdown(error_msg)
                
                # Save error message
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg
                })

# Footer with enhanced styling
st.markdown("""
<div class="footer">
    <div style="margin-bottom: 0.5rem;">
        🏪 <strong>Street Vendor Digitalisation Agent</strong> | Built with ❤️ for Indian Entrepreneurs
    </div>
    <div>
        🤖 Powered by FREE Google Gemini API | 🌍 Supporting 12+ Indian Languages
    </div>
</div>
""", unsafe_allow_html=True)
