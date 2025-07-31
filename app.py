"""
app.py – Streamlit UI for the multilingual street vendor chatbot
Compatible with HuggingFace pipeline backend
"""

import streamlit as st
from rag_chain import rag_chain, LANG_PROMPTS, detect_user_language

st.set_page_config(
    page_title="Street Vendor Digitalisation Agent",
    page_icon="🏪",
    layout="wide"
)

# Main title
st.title("🏪 Street Vendor Digitalisation Agent")
st.caption("🌍 Multilingual AI Assistant for Indian Street Vendors")

# Create two columns for main layout
col1, col2 = st.columns([3, 1])

with col2:
    # Language support sidebar
    st.header("🌐 Language Support")
    
    # Language selector
    st.subheader("🎯 Force Language (Optional)")
    force_language = st.selectbox(
        "Choose a specific language:",
        options=["Auto-detect"] + list(LANG_PROMPTS.keys()),
        format_func=lambda x: "🤖 Auto-detect" if x == "Auto-detect" else f"{LANG_PROMPTS[x]}",
        help="Select 'Auto-detect' to let the AI detect your language automatically, or choose a specific language."
    )
    
    # Show supported languages
    st.subheader("📋 Supported Languages")
    st.write("**Currently supported:**")
    for code, name in LANG_PROMPTS.items():
        st.write(f"• {name}")
    
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
        
        **Ask me anything in your preferred language!**
        """
        st.session_state.messages.append({
            "role": "assistant", 
            "content": welcome_msg,
            "detected_lang": "en"
        })

    # Chat container
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
            placeholder.markdown("_🤔 Thinking..._")
            
            try:
                # Call the RAG chain with HuggingFace pipeline
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
                error_msg = f"❌ **Error occurred:** {str(e)}\n\nPlease try again or rephrase your question."
                placeholder.markdown(error_msg)
                
                # Save error message
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg
                })

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
        🏪 Street Vendor Digitalisation Agent | Built with ❤️ for Indian Entrepreneurs<br>
        🤖 Powered by HuggingFace Transformers | 🌍 Supporting 12+ Indian Languages
    </div>
    """, 
    unsafe_allow_html=True
)
