import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# 1. Load Environment Variables
load_dotenv()

# --- ⚙️ UI CONFIGURATION ---
st.set_page_config(
    page_title="LitisFarm Intelligence", 
    page_icon="🌿", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- 📁 SETTINGS ---
DB_PATH = os.getenv("VECTOR_DB_PATH", "faiss_index")
EMB_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
INF_MODEL = os.getenv("INFERENCE_MODEL", "models/gemini-2.5-flash")

# --- 🧠 RAG INITIALIZATION ---
@st.cache_resource
def load_rag_chain():
    if not os.path.exists(DB_PATH):
        st.error(f"❌ Database not found at '{DB_PATH}'. Please run 'python ingest.py' first.")
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
    
    llm = ChatGoogleGenerativeAI(model=INF_MODEL, temperature=0.2)
    
    return RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True 
    )

qa_chain = load_rag_chain()

# --- 💬 UI HEADER ---
# Using columns to place the Clear button on the same line as the title
header_col, button_col = st.columns([3, 1])

with header_col:
    st.title("🌿 LitisFarm AI")
    st.caption(f"Engine: {INF_MODEL.split('/')[-1]}")

with button_col:
    # Adding some vertical space so the button aligns with the title text
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
        if "sources" in message:
            with st.expander("📚 Sources"):
                for source in message["sources"]:
                    st.markdown(f"📌 {source}")

# --- 🚀 CHAT INTERACTION ---
if prompt := st.chat_input("Ask about LitisFarm..."):
    # 1. User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Assistant Message
    with st.chat_message("assistant"):
        with st.spinner("Analyzing docs..."):
            response = qa_chain.invoke({"query": prompt})
            answer = response["result"]
            source_docs = response["source_documents"]
            
            st.markdown(answer)
            
            # 3. Handle Citations
            citations = []
            if source_docs:
                with st.expander("📚 View Citations"):
                    for i, doc in enumerate(source_docs):
                        file_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
                        # +1 to convert computer 0-index to human page numbers
                        page_num = doc.metadata.get('page', 0) + 1
                        
                        cite = f"{file_name} (Page {page_num})"
                        citations.append(cite)
                        
                        st.markdown(f"**Source {i+1}:** {cite}")
                        st.caption(f"\"{doc.page_content[:150]}...\"")
                        if i < len(source_docs) - 1: st.divider()

            # 4. Save to session
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "sources": citations
            })