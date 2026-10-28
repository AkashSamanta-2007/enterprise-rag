# EnterpriseRAG - Document-based AI Assistant

EnterpriseRAG is a document-based AI assistant that can ingest company PDFs, retrieve relevant passages, and answer user questions using an LLM.

## Technology Stack
- **Python 3.11+** - Core language
- **Groq API** - LLM generation
- **ChromaDB** - Vector storage
- **Sentence Transformers** - Local embeddings
- **PyMuPDF** - PDF text extraction
- **Streamlit** - Frontend
- **python-dotenv** - Environment configuration

## Project Structure
```
enterprise-rag/
  app.py                 # Streamlit frontend (to be implemented)
  requirements.txt       # Python dependencies
  .env.example           # Environment variables template
  .gitignore            # Git ignore rules
  README.md             # This file
  src/                  # Source code modules
    __init__.py
    config.py           # Configuration management
    ingestion/          # PDF ingestion logic (to be implemented)
      __init__.py
    retrieval/          # Vector search logic (to be implemented)
      __init__.py
    generation/         # LLM generation logic (to be implemented)
      __init__.py
  data/                 # Data storage
    documents/          # Input PDF documents
  tests/                # Unit tests (to be implemented)
```

## Setup Instructions

### 1. Create Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
```bash
# Copy example env file
cp .env.example .env

# Edit .env with your actual values
# Required: GROQ_API_KEY
```

### 4. Add Documents
Place your PDF documents in `data/documents/` directory.

### 5. Run the Application (to be implemented)
```bash
streamlit run app.py
```

## Development Status
This is the initial setup. Ingestion, retrieval, and generation modules are yet to be implemented.
