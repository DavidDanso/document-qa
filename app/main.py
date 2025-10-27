import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
import os
import tempfile
from langchain_google_genai import ChatGoogleGenerativeAI

MODEL_NAME = "gemini-2.5-flash"

# page config - must be first streamlit command
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .upload-text {
        font-size: 1.1rem;
        color: #555;
    }
    .success-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .info-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
    </style>
""", unsafe_allow_html=True)

# session state to store vector db and chain
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None
if 'retrieval_chain' not in st.session_state:
    st.session_state.retrieval_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# function to process uploaded PDF
def process_pdf(pdf_file):
    """Load, split, and embed PDF into vector database"""
    # save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_path = tmp_file.name
    
    # load PDF using langchain
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    
    # split into chunks - 1000 char chunks with 20 char overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20
    )
    documents = text_splitter.split_documents(docs)
    
    # create embeddings using google gemini
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # build FAISS vector database from chunks
    vector_db = FAISS.from_documents(documents, embeddings)
    
    # cleanup temp file
    os.unlink(tmp_path)
    
    return vector_db, len(documents)

# function to create RAG chain
def create_rag_chain(vector_db):
    """Build retrieval chain with LLM"""
    # init local llama model
    # llm = OllamaLLM(model="llama3.2")

    llm = ChatGoogleGenerativeAI(model=MODEL_NAME)
    
    # prompt template with context placeholder
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context. 
    Think step by step before providing a detailed answer. 
    Be concise and clear.
    
    <context>
    {context}
    </context>
    
    Question: {input}
    """)
    
    # chain that stuffs docs into prompt
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # convert vector db to retriever
    retriever = vector_db.as_retriever()
    
    # full RAG pipeline: retrieval + document chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain

# main app UI
st.title("📚 RAG Document Q&A System")
st.markdown("Upload a PDF and ask questions about its content")

# sidebar for PDF upload and settings
with st.sidebar:
    st.header("⚙️ Setup")
    
    # google api key input
    google_api_key = st.text_input(
        "Google API Key",
        type="password",
        help="Get your API key from https://makersuite.google.com/app/apikey"
    )
    
    # set api key as env variable if provided
    if google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key
    
    st.divider()
    
    # file uploader
    uploaded_file = st.file_uploader(
        "Upload PDF Document",
        type=['pdf'],
        help="Upload a PDF to analyze"
    )
    
    # process button
    if uploaded_file:
        if st.button("🔄 Process Document"):
            # check if api key is set
            if not google_api_key:
                st.error("⚠️ Please enter your Google API key first!")
            else:
                with st.spinner("Processing PDF..."):
                    try:
                        # process PDF and create vector db
                        vector_db, num_chunks = process_pdf(uploaded_file)
                        st.session_state.vector_db = vector_db
                        
                        # create RAG chain
                        retrieval_chain = create_rag_chain(vector_db)
                        st.session_state.retrieval_chain = retrieval_chain
                        
                        # clear chat history on new doc
                        st.session_state.chat_history = []
                        
                        st.success(f"✅ Processed {num_chunks} chunks!")
                        st.balloons()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    # show status
    st.divider()
    st.subheader("📊 Status")
    if st.session_state.vector_db:
        st.markdown('<div class="success-box">✅ Document loaded</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">ℹ️ No document loaded</div>', unsafe_allow_html=True)
    
    # clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

# main chat area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Ask Questions")
    
    # check if system is ready
    if not st.session_state.retrieval_chain:
        st.info("👈 Upload and process a PDF to get started!")
    else:
        # question input
        question = st.text_input(
            "Your Question:",
            placeholder="What is this document about?",
            key="question_input"
        )
        
        # ask button
        if st.button("🔍 Ask") and question:
            with st.spinner("Thinking..."):
                try:
                    # run RAG pipeline
                    response = st.session_state.retrieval_chain.invoke({
                        "input": question
                    })
                    
                    # extract answer
                    answer = response['answer']
                    
                    # store in chat history
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": answer
                    })
                    
                    # clear input
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # display chat history (reverse order - newest first)
        st.divider()
        if st.session_state.chat_history:
            st.subheader("📝 Conversation History")
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.expander(f"Q: {chat['question']}", expanded=(i==0)):
                    st.markdown(f"**Answer:**\n\n{chat['answer']}")
        else:
            st.info("No questions asked yet. Start by asking something!")

with col2:
    st.subheader("ℹ️ How It Works")
    st.markdown("""
    **RAG Pipeline:**
    
    1. 📄 **Upload PDF** - Your document
    2. ✂️ **Split** - Break into chunks
    3. 🧮 **Embed** - Convert to vectors
    4. 💾 **Store** - Save in FAISS database
    5. 🔍 **Retrieve** - Find relevant chunks
    6. 🤖 **Generate** - LLM creates answer
    
    ---
    
    **Tech Stack:**
    - LLM: gemini-2.5-flash
    - Embeddings: Google Gemini
    - Vector DB: FAISS
    - Framework: LangChain
    """)
    
    st.divider()
    st.caption("Built with Streamlit & LangChain")
