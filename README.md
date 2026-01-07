# 🏪 Street Vendor Digitalisation Agent

A multilingual AI chatbot designed to help Indian street vendors with digital transformation, government schemes, and business guidance.

## 🌟 Features

- **Multilingual Support**: Supports 12+ Indian languages including Hindi, Tamil, Telugu, Gujarati, Bengali, Punjabi, Malayalam, Urdu, Marathi, Kannada
- **FREE AI**: Powered by Groq API (no costs, no subscriptions)
- **Smart Language Detection**: Automatically detects user language or allows manual selection
- **Comprehensive Guidance**: Covers PM-SVANidhi loans, UPI setup, vendor registration, digital payments, and government schemes

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd street-vendor-chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Create .env file
   echo "GROQ_API_KEY=your-groq-api-key-here" > .env
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

## ☁️ Deploy to Streamlit Cloud

### Step 1: Prepare Your Repository

1. **Ensure your repository is on GitHub**
   - Push your code to a GitHub repository
   - Make sure `app.py` is in the root directory

2. **Required Files** (already configured):
   - `app.py` - Main application file
   - `requirements.txt` - Python dependencies
   - `rag_chain.py` - AI logic and language processing

### Step 2: Deploy on Streamlit Cloud

1. **Go to [Streamlit Cloud](https://streamlit.io/cloud)**

2. **Connect your GitHub account**
   - Click "New app" → "Connect GitHub"

3. **Configure deployment**:
   - **Repository**: Select your repository
   - **Branch**: `main` (or your default branch)
   - **Main file path**: `app.py`
   - **Python version**: `3.9` or higher

4. **Set up secrets**:
   - In Streamlit Cloud dashboard, go to your app
   - Click "⋮" (three dots) → "Settings" → "Secrets"
   - Add your Groq API key:
     ```
     GROQ_API_KEY = "your-groq-api-key-here"
     ```

5. **Deploy**
   - Click "Deploy!" and wait for the build to complete

### Step 3: Get Your Groq API Key

1. **Visit [Groq Console](https://console.groq.com/)**
2. **Sign up** (free, no credit card required)
3. **Create API Key** in the dashboard
4. **Copy the key** and add it to Streamlit Cloud secrets

## 📋 Supported Languages

- English (en)
- Hindi (hi) - हिंदी
- Marathi (mr) - मराठी
- Tamil (ta) - தமிழ்
- Telugu (te) - తెలుగు
- Gujarati (gu) - ગુજરાતી
- Bengali (bn) - বাংলা
- Punjabi (pa) - ਪੰਜਾਬੀ
- Malayalam (ml) - മലയാളം
- Urdu (ur) - اردو
- Kannada (kn) - ಕನ್ನಡ

## 💡 Usage Examples

**English**: "How do I apply for PM-SVANidhi loan?"
**Hindi**: "मुझे PM-SVANidhi लोन कैसे मिलेगा?"
**Tamil**: "நான் PM-SVANidhi கடன் எப்படி பெறுவது?"
**Gujarati**: "મને PM-SVANidhi લોન કેવી રીતે મળશે?"

## 🛠️ Technical Details

- **Frontend**: Streamlit
- **AI Engine**: Groq API (Llama 3.1 8B Instant)
- **Language Detection**: langdetect library
- **Architecture**: RAG (Retrieval-Augmented Generation) pipeline

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Built for Indian street vendors and entrepreneurs
- Powered by free AI technology from Groq
- Inspired by the PM-SVANidhi scheme and digital India initiatives

---

**Made with ❤️ for Indian Entrepreneurs**