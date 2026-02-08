import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# 1. Load Environment Variables (for local development)
load_dotenv()

# --- ⚙️ UI CONFIGURATION ---
st.set_page_config(
    page_title="LitisFarm Intelligence", 
    page_icon="🌿", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- 📁 SETTINGS ---
# These pull from Streamlit Secrets (Cloud) or .env (Local)
DB_PATH = os.getenv("VECTOR_DB_PATH", "faiss_index")
EMB_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
INF_MODEL = os.getenv("INFERENCE_MODEL", "models/gemini-2.5-flash")

# --- 🔑 AUTHENTICATION ---
# Mapping Streamlit Secrets to Environment Variables
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
elif not os.getenv("GOOGLE_API_KEY"):
    st.error("❌ GOOGLE_API_KEY missing. Please add it to Streamlit Secrets.")
    st.stop()

# --- 🧠 RAG INITIALIZATION ---
@st.cache_resource
def load_rag_chain():
    if not os.path.exists(DB_PATH):
        st.error(f"❌ Database folder '{DB_PATH}' not found. Did you push it to GitHub?")
        st.stop()

    # Initialize Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMB_MODEL,
        task_type="RETRIEVAL_QUERY"
    )
    
    # Load the local FAISS index
    db = FAISS.load_local(
        DB_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    # Initialize LLM with streaming enabled
    llm = ChatGoogleGenerativeAI(
        model=INF_MODEL, 
        temperature=0.2,
        streaming=True
    )
    
    # Create the QA Chain
    return RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True 
    )

# Load the chain
try:
    qa_chain = load_rag_chain()
except Exception as e:
    st.error(f"Failed to initialize RAG: {e}")
    st.stop()

# --- 💬 UI HEADER ---
header_col, button_col = st.columns([3, 1])

with header_col:
    st.title("🌿 LitisFarm AI")
    st.caption(f"Engine: {INF_MODEL.split('/')[-1]}")

with button_col:
    st.write("##") 
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.divider()

# --- 💬 CHAT SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 Sources"):
                for source in message["sources"]:
                    st.markdown(f"📌 {source}")

# --- 🚀 CHAT INTERACTION ---
if prompt := st.chat_input("Ask about LitisFarm..."):
    # 1. User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Assistant Message with Streaming
    with st.chat_message("assistant"):
        # We fetch sources first (fast, non-streamed)
        with st.spinner("Searching documents..."):
            source_docs = qa_chain.retriever.invoke(prompt)
            
        # Define a generator for the stream
        def response_generator():
            # qa_chain.stream returns a dict; we yield the 'result' chunk
            for chunk in qa_chain.stream({"query": prompt}):
                if "result" in chunk:
                    yield chunk["result"]
        
        # Stream the text to the UI
        answer = st.write_stream(response_generator())

        # 3. Handle Citations
        citations = []
        if source_docs:
            with st.expander("📚 View Citations"):
                for i, doc in enumerate(source_docs):
                    source_path = doc.metadata.get('source', 'Unknown')
                    file_name = os.path.basename(source_path)
                    page_num = doc.metadata.get('page', 0) + 1
                    
                    cite = f"{file_name} (Page {page_num})"
                    citations.append(cite)
                    
                    st.markdown(f"**Source {i+1}:** {cite}")
                    st.caption(f"\"{doc.page_content[:150]}...\"")
                    if i < len(source_docs) - 1: st.divider()

        # 4. Save to session state
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer,
            "sources": citations
        })