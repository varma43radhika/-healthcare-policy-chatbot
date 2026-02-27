# 🏥 Healthcare Policy Chatbot

A RAG-powered chatbot that answers questions over insurance and healthcare policy PDFs in plain English.

## What it does
- Upload any insurance policy PDF
- Ask questions in plain English
- Get answers grounded in the actual document with source citations
- Remembers conversation history

## Tech Stack
- Python
- LangChain
- Groq (LLaMA 3.3)
- ChromaDB
- HuggingFace Embeddings
- Streamlit

## How to run locally
1. Clone the repo
2. Install dependencies: `pip3 install -r requirements.txt`
3. Add your Groq API key to `.env`: `GROQ_API_KEY=your_key`
4. Run: `streamlit run app.py`


## 🚀 Live Demo
👉 [Click here to try the app](https://9ntaxbb2u8qkepfhwlsik5.streamlit.app/)
