import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq


load_dotenv()

st.set_page_config(
    page_title="Healthcare Policy Chatbot",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Healthcare Policy Chatbot")
st.caption("Upload an insurance policy PDF and ask questions in plain English")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

with st.sidebar:
    st.header("📄 Upload Policy Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file:
        with st.spinner("Processing PDF..."):
            pdf_reader = PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = splitter.split_text(text)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            st.session_state.vectorstore = Chroma.from_texts(
                chunks,
                embeddings
            )
        st.success(f"✅ Ready! {len(chunks)} chunks processed.")
        st.info(f"📑 Pages: {len(pdf_reader.pages)}")

if st.session_state.vectorstore is None:
    st.info("👈 Upload a policy PDF from the sidebar to get started")
else:
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.2
    )

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if question := st.chat_input("Ask anything about the policy..."):
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Retrieve relevant chunks
                retriever = st.session_state.vectorstore.as_retriever(
                    search_kwargs={"k": 3}
                )
                docs = retriever.invoke(question)
                context = "\n\n".join([doc.page_content for doc in docs])

                # Build prompt with chat history
                history_text = ""
                for msg in st.session_state.chat_history[:-1]:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    history_text += f"{role}: {msg['content']}\n"

                prompt = f"""You are a helpful assistant that answers questions about insurance and healthcare policy documents.
Use the following context from the policy document to answer the question.
If the answer is not in the context, say "I couldn't find that information in the policy document."

Context:
{context}

Chat History:
{history_text}

Question: {question}

Answer:"""

                response = llm.invoke(prompt)
                answer = response.content
                st.write(answer)

                with st.expander("📚 Sources"):
                    for i, doc in enumerate(docs):
                        st.caption(f"Chunk {i+1}: {doc.page_content[:200]}...")

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })