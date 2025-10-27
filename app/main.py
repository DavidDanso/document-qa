# Load PDF document using langchain
from langchain_community.document_loaders import PyPDFLoader

# point to the resume PDF in current directory
loader = PyPDFLoader("resume.pdf")

# extract all pages - returns list of Document objects with metadata
docs = loader.load()
docs