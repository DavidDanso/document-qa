# Document Q&A with RAG

Ask questions about your PDF documents using Google's Gemini AI. Built with Streamlit, LangChain, and FAISS vector search.

## What This Does

Upload a PDF, ask questions about it, and get answers based on the actual content. The system splits your document into chunks, converts them to vectors, and uses semantic search to find relevant parts before generating an answer.

## Tech Stack

- **LLM**: Google Gemini 2.5 Flash
- **Embeddings**: Google Gemini Embedding Model
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Framework**: LangChain
- **UI**: Streamlit

## Prerequisites

- Python 3.10+
- Google API Key ([Get one here](https://makersuite.google.com/app/apikey))

## App Preview:

<table width="100%"> 
<tr>
<td width="50%">      
&nbsp; 
<br>
<p align="center">
  Main Feed
</p>
<img src="https://github.com/DavidDanso/document-qa/blob/main/ui/main-feed.png" />
</td> 
<td width="50%">
<br>
<p align="center">
  Q&A Interface
</p>
<img src="https://github.com/DavidDanso/document-qa/blob/main/ui/q%26a.png" />
</td>
</table>

## Setup

1. **Clone the repo**

```bash
git clone <your-repo-url>
cd document-qa
```

2. **Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

Create a `.env` file in the root directory:

```bash
GOOGLE_API_KEY=your_api_key_here
```

Or enter it directly in the app's sidebar when running.

## Running the App

```bash
streamlit run app/main.py
```

The app will open in your browser at `http://localhost:8501`

## How to Use

1. Enter your Google API key in the sidebar
2. Upload a PDF document
3. Click "Process Document" and wait for it to finish
4. Ask questions in the text input
5. Get answers based on your document's content

## How It Works (RAG Pipeline)

1. **Document Loading**: PDF is loaded using PyPDFLoader
2. **Text Splitting**: Document is split into 1000-character chunks with 20-char overlap
3. **Embeddings**: Each chunk is converted to a vector using Google's embedding model
4. **Vector Storage**: Vectors are stored in FAISS for fast similarity search
5. **Retrieval**: When you ask a question, the system finds the most relevant chunks
6. **Generation**: Gemini LLM uses the retrieved context to generate an answer

## Project Structure

```
document-qa/
├── app/
│   ├── main.py           # Streamlit app with RAG pipeline
├── venv/                 # Virtual environment
├── .env                  # Environment variables (API keys)
├── .gitignore
├── requirements.txt      # Python dependencies
└── README.md
```

## Requirements

Key dependencies:

- `streamlit` - Web UI
- `langchain-community` - Document loaders and vector stores
- `langchain-google-genai` - Google Gemini integration
- `faiss-cpu` - Vector similarity search
- `pypdf` - PDF parsing
- `python-dotenv` - Environment variable management

See `requirements.txt` for full list.

## Common Issues

**"Your default credentials were not found"**

- Make sure you've set `GOOGLE_API_KEY` in `.env` or entered it in the sidebar

**"Module not found"**

- Activate your virtual environment: `source venv/bin/activate`
- Reinstall dependencies: `pip install -r requirements.txt`

**Slow processing**

- Large PDFs take time to process
- Processing happens once per document, then cached

## Features

- ✅ PDF upload and processing
- ✅ Semantic search with FAISS
- ✅ Chat history tracking
- ✅ Clean, responsive UI
- ✅ Real-time question answering
- ✅ Context-aware responses

## Contributing

Feel free to fork and submit PRs. Keep it simple.

Happy Coding 🎉
