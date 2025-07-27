import streamlit as st
from rag_chain import rag_chain

st.set_page_config(page_title="Street Vendor Digitalisation Agent")
st.title("🏪 Street Vendor Digitalisation Agent")

# Optional: Simple language code-to-name map for display
LANG_PROMPTS = {
    'en': "English", 'hi': "Hindi (हिंदी)", 'mr': "Marathi (मराठी)", 'ta': "Tamil (தமிழ்)",
    'te': "Telugu (తెలుగు)", 'kn': "Kannada (ಕನ್ನಡ)", 'gu': "Gujarati (ગુજરાતી)", 
    'bn': "Bengali (বাংলা)", 'pa': "Punjabi (ਪੰਜਾਬੀ)", 'ml': "Malayalam (മലയാളം)", 'ur': "Urdu (اردو)"
}

if "messages" not in st.session_state:
    st.session_state.messages = []
    # Optionally, add a welcome assistant message at start
    st.session_state.messages.append({
        "role": "assistant",
        "content": "🙏 Welcome! Ask me anything about digital schemes, PM-SVANidhi loan, UPI, or government policies in your language."
    })

# Render chat so far
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # If you store/detect language, could display language info here

user_input = st.chat_input("Ask me in English, Hindi, Marathi, Tamil, Telugu...")

if user_input:
    # Show the user's bubble
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get a reply -- always interactive, even for chit-chat/greetings
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("_Thinking..._")

        response = rag_chain(user_input)   # universal signature in rag_chain.py
        answer_text = response.get("answer", str(response))
        placeholder.markdown(answer_text)
        st.session_state.messages.append({"role": "assistant", "content": answer_text})
