"""
EnterpriseRAG Streamlit Frontend
Clean chat interface for RAG pipeline
"""
import os
import streamlit as st
from pathlib import Path
from datetime import datetime

from src.rag_pipeline import RAGPipeline
from src.config import config
from src.validation import validate_filename, validate_question, ValidationError


# Page configuration
st.set_page_config(
    page_title="EnterpriseRAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "pipeline" not in st.session_state:
    try:
        st.session_state.pipeline = RAGPipeline()
    except Exception as e:
        st.session_state.pipeline = None
        st.session_state.init_error = str(e)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "indexed" not in st.session_state:
    st.session_state.indexed = False


def main():
    """Main application function."""
    st.title("📚 EnterpriseRAG - Document-based AI Assistant")
    st.markdown("Ask questions about your PDF documents using AI")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Document Management")
        
        # Check for initialization errors
        if st.session_state.pipeline is None:
            st.error("Failed to initialize pipeline")
            if hasattr(st.session_state, 'init_error'):
                st.caption(st.session_state.init_error)
            st.info("Make sure you have set GROQ_API_KEY in your .env file")
            st.stop()
        
        # Upload section
        st.subheader("Upload PDFs")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF documents"
        )
        
        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} file(s)")
            for file in uploaded_files:
                st.caption(f"• {file.name}")
            
            if st.button("💾 Save Uploaded Files", use_container_width=True):
                save_uploaded_files(uploaded_files)
        
        st.divider()
        
        # Indexing section
        st.subheader("Index Documents")
        
        # Show document count in directory
        pdf_dir = Path(config.PDF_INPUT_DIR)
        pdf_files = list(pdf_dir.glob("*.pdf"))
        st.write(f"📄 PDFs in directory: {len(pdf_files)}")
        
        if pdf_files:
            with st.expander("View files"):
                for pdf in pdf_files:
                    st.caption(f"• {pdf.name}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Index/Process", use_container_width=True):
                index_documents()
        with col2:
            if st.button("🗑️ Clear Index", use_container_width=True):
                clear_index()
        
        # Status
        st.divider()
        st.subheader("📊 Status")
        try:
            status = st.session_state.pipeline.get_status()
            st.metric("Documents Indexed", status["documents_in_store"])
            st.caption(f"LLM: {status['llm_model']}")
            st.caption(f"Embeddings: {status['embedding_model']}")
            st.caption(f"Dim: {status['embedding_dim']}")
            if status["documents_in_store"] > 0:
                st.session_state.indexed = True
            else:
                st.session_state.indexed = False
        except Exception as e:
            st.error(f"Error getting status: {e}")
    
    # Main chat area
    st.header("💬 Chat")
    
    if not st.session_state.indexed and st.session_state.pipeline.get_status()["documents_in_store"] == 0:
        st.warning("⚠️ No documents indexed yet. Upload and index PDFs using the sidebar to start chatting.")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])
                    if message.get("citations"):
                        with st.expander("📚 Sources"):
                            for i, citation in enumerate(message["citations"], 1):
                                st.caption(f"{i}. {citation['source']} (Page {citation['page_number']})")
    
    # Question input
    if question := st.chat_input("Ask a question about your documents...", disabled=st.session_state.pipeline.get_status()["documents_in_store"] == 0):
        handle_question(question)


def save_uploaded_files(uploaded_files):
    """Save uploaded files to configured document directory."""
    try:
        pdf_dir = Path(config.PDF_INPUT_DIR)
        pdf_dir.mkdir(parents=True, exist_ok=True)
        
        saved_count = 0
        for file in uploaded_files:
            try:
                # Validate filename
                safe_name = validate_filename(file.name)
                file_path = pdf_dir / safe_name
                # Save file
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                saved_count += 1
                st.toast(f"Saved {safe_name}", icon="✅")
            except ValidationError as e:
                st.error(f"Skipped {file.name}: {e}")
            except Exception as e:
                st.error(f"Error saving {file.name}: {e}")
        
        if saved_count > 0:
            st.success(f"✅ Saved {saved_count} file(s) to {pdf_dir}")
        st.rerun()
    except Exception as e:
        st.error(f"Error saving files: {e}")


def index_documents():
    """Index/process documents in the document directory."""
    try:
        with st.spinner("🔄 Indexing documents... This may take a while..."):
            count = st.session_state.pipeline.ingest_documents()
            if count > 0:
                st.success(f"✅ Successfully indexed {count} chunks from documents")
                st.session_state.indexed = True
            else:
                st.warning("⚠️ No documents found or no text extracted. Check if PDFs are in the documents folder.")
        st.rerun()
    except Exception as e:
        st.error(f"Error indexing documents: {e}")
        st.caption("Check logs for details")


def clear_index():
    """Clear the vector store index."""
    try:
        st.session_state.pipeline.clear()
        st.session_state.indexed = False
        st.success("✅ Index cleared")
        st.rerun()
    except Exception as e:
        st.error(f"Error clearing index: {e}")


def handle_question(question):
    """Handle user question and generate answer."""
    # Validate question in UI
    try:
        validated_question = validate_question(question)
    except ValidationError as e:
        st.error(f"Invalid question: {e}")
        return
    
    # Add user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": validated_question,
        "timestamp": datetime.now().isoformat()
    })
    
    # Display user message
    with st.chat_message("user"):
        st.write(validated_question)
    
    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                result = st.session_state.pipeline.query(validated_question)
                answer = result["answer"]
                citations = result.get("citations", [])
                
                st.write(answer)
                
                if citations:
                    with st.expander("📚 Sources"):
                        for i, citation in enumerate(citations, 1):
                            st.caption(f"{i}. {citation['source']} (Page {citation['page_number']})")
                
                # Add to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                error_msg = "An error occurred while processing your question."
                st.error(error_msg)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_msg,
                    "citations": [],
                    "timestamp": datetime.now().isoformat()
                })


if __name__ == "__main__":
    main()
