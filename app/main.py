# Load PDF document using langchain
from langchain_community.document_loaders import PyPDFLoader
# Break documents into smaller chunks for better retrieval
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Set up vector database with embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
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

# show the most relevant chunk found
print(result[0].page_content)