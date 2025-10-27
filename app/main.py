# Load PDF document using langchain
from langchain_community.document_loaders import PyPDFLoader
# Break documents into smaller chunks for better retrieval
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Set up vector database with embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
# load Gemini via ChatGoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI
# build prompt template with placeholders for context and question
from langchain_core.prompts import ChatPromptTemplate
# create chain that stuffs docs into prompt and sends to LLM
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# wire up the full RAG pipeline
from langchain_classic.chains.retrieval import create_retrieval_chain

from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")


# point to the resume PDF in current directory
loader = PyPDFLoader("cover_letter.pdf")
# extract all pages - returns list of Document objects with metadata
docs = loader.load()

# create text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
# store all split chunks for vector db
documents = text_splitter.split_documents(docs)

# use google's gemini model to convert text to vectors
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GOOGLE_API_KEY)
# FAISS vector store creation
# FAISS = Similarity Search - fast vector lookup
db=FAISS.from_documents(documents,embeddings)


# test the vector search - find similar chunks to query
query="Whose cover letter is this?"

# search db for chunks most similar to query
result=db.similarity_search(query)

# gemini-2.5-flash model
llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")


# {context} gets replaced with retrieved docs
# {input} gets replaced with user's question
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context. 
Think step by step before providing a detailed answer. 
I will tip you $1000 if the user finds the answer helpful. 
<context>
{context}
</context>
Question: {input}""")

# chain takes retrieved docs + prompt, formats them, sends to llm
document_chain=create_stuff_documents_chain(llm,prompt)

# convert vector db into retriever interface
# retriever = wrapper that searches and returns relevant docs
retriever=db.as_retriever()

# flow: question -> retriever finds docs -> document_chain formats + sends to LLM -> answer
retrieval_chain=create_retrieval_chain(retriever,document_chain)

# run the full pipeline with a question
response=retrieval_chain.invoke({"input":"what is she applying for?"})["answer"]