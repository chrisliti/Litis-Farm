import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.callbacks.base import BaseCallbackHandler

# 1. Load Environment Variables (local .env)
load_dotenv()

# --- ⚙️ UI CONFIGURATION ---
st.set_page_config(
    page_title="LitisFarm Intelligence", 
    page_icon="🌿", 
    layout="centered"
)

# --- 📁 SETTINGS ---
# Pull from Streamlit Secrets or environment
DB_PATH = os.getenv("VECTOR_DB_PATH", "faiss_index")
EMB_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
INF_MODEL = os.getenv("INFERENCE_MODEL", "models/gemini-2.5-flash")

# --- 🔑 AUTHENTICATION ---
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
elif not os.getenv("GOOGLE_API_KEY"):
    st.error("❌ GOOGLE_API_KEY missing. Please add it to Streamlit Secrets.")
    st.stop()

# --- 🛠️ STREAMING HANDLER ---
class StreamHandler(BaseCallbackHandler):
    """Custom handler to catch tokens and update a Streamlit container."""
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        # Add a cursor '▌' to make it look like typing
        self.container.markdown(self.text + "▌")

# --- 🧠 RAG INITIALIZATION ---
@st.cache_resource
def load_rag_chain():
    if not os.path.exists(DB_PATH):
        st.error(f"❌ Database folder '{DB_PATH}' not found on GitHub.")
        st.stop()

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMB_MODEL,
        task_type="RETRIEVAL_QUERY"
    )
    
    db = FAISS.load_local(
        DB_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    # We create the LLM here but we will pass the callback during the call
    llm = ChatGoogleGenerativeAI(
        model=INF_MODEL, 
        temperature=0.2,
        streaming=True
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True 
    )

qa_chain = load_rag_chain()

# --- 💬 UI HEADER ---
header_col, button_col = st.columns([3, 1])
with header_col:
    st.title("🌿 LitisFarm AI")
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
    # 1. Show User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Assistant Message with Streaming
    with st.chat_message("assistant"):
        # Create a placeholder for the streaming text
        response_placeholder = st.empty()
        stream_handler = StreamHandler(response_placeholder)
        
        with st.spinner("Consulting farm records..."):
            # We call the chain and pass the callback to the 'config'
            response = qa_chain.invoke(
                {"query": prompt},
                {"callbacks": [stream_handler]}
            )
            
        answer = response["result"]
        # Remove the cursor '▌' for the final display
        response_placeholder.markdown(answer)

        # 3. Handle Citations
        source_docs = response.get("source_documents", [])
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

        # 4. Save to history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer,
            "sources": citations
        })