
import streamlit as st
from pathlib import Path
from datetime import datetime

from src.rag_pipeline import RAGPipeline
from src.config import config
from src.validation import validate_filename, validate_question, ValidationError


# Page configuration
st.set_page_config(
    page_title="EnterpriseRAG",
    page_icon="✳",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

:root {
    --ink: #e9eff5;
    --muted: #9aa9b7;
    --line: #293743;
    --sidebar: #111a23;
    --accent: #a9d8f2;
    --green: #74c995;
}

html, body, [class*="css"] { font-family: 'DM Sans', 'Segoe UI', sans-serif; color: var(--ink); }
[data-testid="stAppViewContainer"] {
    position: relative;
    isolation: isolate;
    background-color: #0c1219;
    background-image:
        linear-gradient(125deg, #0c1219 0%, #142432 40%, #101a22 68%, #0c1219 100%);
    background-size: 190% 190%;
}
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: 0;
    z-index: -1;
    pointer-events: none;
    background-image:
        linear-gradient(90deg, transparent 0 46px, rgba(87,190,218,.12) 46px 47px, transparent 47px 118px),
        linear-gradient(0deg, transparent 0 70px, rgba(87,190,218,.10) 70px 71px, transparent 71px 118px),
        radial-gradient(circle at 47px 71px, rgba(113,220,239,.58) 0 1.5px, transparent 2.5px);
    background-size: 118px 118px;
    opacity: .48;
    animation: circuit-scroll 30s linear infinite, circuit-pulse 4.8s ease-in-out infinite alternate;
}
[data-testid="stAppViewContainer"]::after {
    content: "";
    position: fixed;
    top: -24vh;
    left: 0;
    width: 100%;
    height: 24vh;
    z-index: -1;
    pointer-events: none;
    background: linear-gradient(180deg, transparent, rgba(71,190,222,.075), transparent);
    animation: scan-sweep 11s linear infinite;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMain"] .block-container {
    max-width: 1050px;
    padding: 1.25rem 2.5rem 6rem;
}
[data-testid="stSidebar"] { background: rgba(17,26,35,.97); border-right: 1px solid #273540; }
[data-testid="stSidebar"] > div:first-child { padding: 1.1rem .9rem; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color: var(--ink); }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: var(--muted); }
[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
    background: rgba(255,255,255,.035);
    border: 1px dashed #3b4a57;
    border-radius: 8px;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section * { color: #c9d5df; }
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
    background: rgba(255,255,255,.045);
    border: 1px solid #30404d;
    color: #dce5ed;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
    background: #23465a;
    border: 1px solid #315d75;
    color: #e2f5ff;
}
.brand {
    display: flex;
    align-items: center;
    gap: .55rem;
    margin: .2rem .25rem 1.2rem;
    color: #e9eff5;
    font-size: 15px;
    font-weight: 700;
}
.brand-mark { color: #9bd5f2; font-size: 21px; }
.sidebar-label {
    margin: 1rem .25rem .4rem;
    color: #8293a1;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
}
@keyframes rise-in {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes mark-pop {
    0% { opacity: 0; transform: scale(.82) rotate(-12deg); }
    70% { opacity: 1; transform: scale(1.06) rotate(3deg); }
    100% { opacity: 1; transform: scale(1) rotate(0); }
}
@keyframes circuit-scroll {
    from { background-position: 0 0; }
    to { background-position: 118px 118px; }
}
@keyframes circuit-pulse {
    from { opacity: .28; }
    to { opacity: .58; }
}
@keyframes scan-sweep {
    from { transform: translateY(0); }
    to { transform: translateY(150vh); }
}
.welcome {
    max-width: 680px;
    margin: 11vh auto 1.5rem;
    text-align: center;
    animation: rise-in .65s cubic-bezier(.2,.7,.25,1) both;
}
.welcome-mark {
    display: inline-grid;
    place-items: center;
    width: 42px;
    height: 42px;
    margin-bottom: 1.1rem;
    border-radius: 50%;
    color: #a9d8f2;
    background: #172936;
    font-size: 22px;
    animation: mark-pop .6s .12s cubic-bezier(.2,.7,.25,1) both;
    transition: transform .2s ease, box-shadow .2s ease;
}
.welcome-mark:hover { transform: translateY(-2px) rotate(5deg); box-shadow: 0 6px 18px rgba(90,165,204,.18); }
.welcome h1 {
    margin: 0;
    color: #edf2f7;
    font-size: 36px;
    font-weight: 500;
    letter-spacing: 0;
}
.welcome p { margin: .7rem auto 0; color: #a0adba; font-size: 14px; }
[data-testid="stBaseButton-secondary"] {
    border-radius: 8px;
    color: #dce5ed;
    background: rgba(22,32,42,.86);
    border: 1px solid #30404d;
    text-align: left;
    min-height: 54px;
    white-space: normal;
    transition: transform .18s ease, background-color .18s ease, border-color .18s ease, box-shadow .18s ease;
}
[data-testid="stBaseButton-secondary"]:hover {
    background: #1d2b37;
    border-color: #48677c;
    color: #f0f6fb;
    transform: translateY(-2px);
    box-shadow: 0 5px 18px rgba(0,0,0,.24);
}
[data-testid="stChatMessage"] {
    border: 0;
    background: transparent;
    align-items: flex-start;
    gap: .8rem;
    max-width: 900px;
    margin: 1rem auto;
    padding: .2rem 0;
    animation: rise-in .38s ease-out both;
}
[data-testid="stChatMessageContent"] {
    max-width: 100%;
    padding: .9rem 1.1rem;
    border: 1px solid rgba(126,154,176,.13);
    border-left: 2px solid rgba(151,194,221,.55);
    border-radius: 4px 14px 14px 14px;
    background: rgba(17,26,35,.76);
    color: #e3ebf2;
    line-height: 1.72;
}
[data-testid="stChatMessageContent"] p { margin: .25rem 0 .65rem; }
[data-testid="stChatMessageContent"] p:last-child { margin-bottom: .2rem; }
[data-testid="stChatMessageContent"] h1,
[data-testid="stChatMessageContent"] h2,
[data-testid="stChatMessageContent"] h3 { color: #f0f5f9; }
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) { justify-content: flex-end; }
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
    max-width: min(78%, 680px);
    border: 1px solid rgba(119,174,207,.24);
    border-radius: 14px 4px 14px 14px;
    background: linear-gradient(145deg, rgba(35,65,84,.92), rgba(27,48,64,.92));
    color: #e8f3fa;
}
[data-testid="chatAvatarIcon-assistant"] {
    border: 1px solid rgba(151,194,221,.22);
    border-radius: 50%;
    background: #172936;
    color: #a9d8f2;
}
[data-testid="chatAvatarIcon-user"] {
    border: 1px solid rgba(119,174,207,.28);
    border-radius: 50%;
    background: #20394a;
    color: #d7edfa;
}
[data-testid="stExpander"] {
    margin-top: .8rem;
    border: 1px solid #344858;
    border-radius: 8px;
    background: rgba(10,17,24,.42);
}
[data-testid="stExpander"] summary { color: #b9d9eb; font-size: 13px; }
[data-testid="stBottom"], [data-testid="stBottomBlockContainer"] {
    background: #0c1219 !important;
    background-image: none !important;
}
[data-testid="stBottom"] > div {
    background: #0c1219 !important;
    background-image: none !important;
    box-shadow: none !important;
}
[data-testid="stBottomBlockContainer"] { padding-bottom: 1rem; }
[data-testid="stChatInput"] {
    background: #151f29;
    border: 1px solid #354550;
    border-radius: 24px;
    box-shadow: 0 8px 26px rgba(0,0,0,.24);
    transition: border-color .2s ease, box-shadow .2s ease;
}
[data-testid="stChatInput"] > div:first-child { background: #151f29 !important; color: var(--ink) !important; }
[data-testid="stChatInputSubmitButton"] { background: #233848 !important; color: #b7dff4 !important; }
[data-testid="stChatInput"] textarea { color: var(--ink); background: #151f29; }
[data-testid="stChatInput"] textarea::placeholder { color: #92a1ae; opacity: 1; }
[data-testid="stChatInput"]:focus-within { border-color: #6592ac; box-shadow: 0 0 0 3px rgba(112,164,193,.12), 0 8px 26px rgba(0,0,0,.24); }
[data-testid="stAlert"] { background: #17232d; border-color: #344756; color: var(--ink); }
[data-testid="stExpander"] { background: rgba(21,31,41,.72); border-color: #30404d; }
@media (max-width: 760px) {
    [data-testid="stMain"] .block-container { padding: 1rem 1rem 5.5rem; }
    .welcome { margin-top: 8vh; }
    .welcome h1 { font-size: 29px; }
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: .01ms !important;
        animation-iteration-count: 1 !important;
        scroll-behavior: auto !important;
        transition-duration: .01ms !important;
    }
}
</style>
""", unsafe_allow_html=True)

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
if "pending_delete" not in st.session_state:
    st.session_state.pending_delete = None


def main():
    """Main application function."""
    if st.session_state.pipeline is None:
        st.error("The assistant could not start.")
        if hasattr(st.session_state, "init_error"):
            st.caption(st.session_state.init_error)
        st.info("Check that GROQ_API_KEY is configured, then restart the app.")
        st.stop()

    pdf_dir = Path(config.PDF_INPUT_DIR)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    status = st.session_state.pipeline.get_status()
    chunk_count = status["documents_in_store"]
    st.session_state.indexed = chunk_count > 0

    with st.sidebar:
        st.markdown('<div class="brand"><span class="brand-mark">✳</span>EnterpriseRAG</div>', unsafe_allow_html=True)
        if st.button("＋  New chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

        st.markdown('<div class="sidebar-label">Your documents</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Add PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF documents"
        )

        if uploaded_files:
            st.caption(f"{len(uploaded_files)} file(s) selected")
            for file in uploaded_files:
                st.caption(file.name)
            if st.button("Save files", type="primary", use_container_width=True):
                save_uploaded_files(uploaded_files)

        st.divider()
        st.markdown('<div class="sidebar-label">Library status</div>', unsafe_allow_html=True)
        st.caption(f"{len(pdf_files)} PDF file(s) · {chunk_count} indexed passages")
        if pdf_files:
            with st.expander("Manage documents"):
                for index, pdf in enumerate(pdf_files):
                    file_col, delete_col = st.columns([4, 1])
                    with file_col:
                        st.caption(pdf.name)
                    with delete_col:
                        if st.button("Delete", key=f"delete_pdf_{index}", help=f"Delete {pdf.name}"):
                            st.session_state.pending_delete = pdf.name

                if st.session_state.pending_delete:
                    confirm_col, cancel_col = st.columns(2)
                    with confirm_col:
                        if st.button("Confirm delete", type="primary", use_container_width=True):
                            delete_uploaded_file(st.session_state.pending_delete)
                    with cancel_col:
                        if st.button("Cancel", use_container_width=True):
                            st.session_state.pending_delete = None
                            st.rerun()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Index PDFs", use_container_width=True, disabled=not pdf_files):
                index_documents()
        with col2:
            if st.button("Clear", use_container_width=True, disabled=chunk_count == 0):
                clear_index()

        st.divider()
        st.caption(f"Model · {status['llm_model']}")

    if not st.session_state.chat_history:
        st.markdown(
            '<div class="welcome"><div class="welcome-mark">✳</div>'
            '<h1>Hello, what can I help you find?</h1>'
            '<p>Ask a question about your PDFs and get answers with page-level sources.</p></div>',
            unsafe_allow_html=True
        )
        if not chunk_count:
            st.info("Upload PDFs in the sidebar, then select **Index PDFs** to start asking questions.")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant" and message.get("citations"):
                with st.expander("Sources"):
                    for citation in message["citations"]:
                        st.caption(f"{citation['source']} (Page {citation['page_number']})")

    question = st.chat_input("Ask about your documents", disabled=chunk_count == 0)
    if question:
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


def delete_uploaded_file(filename):
    """Delete a PDF and remove its chunks from the vector index."""
    try:
        safe_name = validate_filename(filename)
        pdf_dir = Path(config.PDF_INPUT_DIR).resolve()
        file_path = (pdf_dir / safe_name).resolve()
        file_path.relative_to(pdf_dir)

        if not file_path.is_file():
            st.error(f"Could not find {safe_name} in the documents folder.")
            st.session_state.pending_delete = None
            return

        deleted_chunks = st.session_state.pipeline.delete_document(safe_name)
        file_path.unlink()
        st.session_state.pending_delete = None
        st.toast(f"Deleted {safe_name} and {deleted_chunks} indexed passage(s)", icon="🗑️")
        st.rerun()
    except Exception as e:
        st.error(f"Could not delete {filename}: {e}")


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
                        for citation in citations:
                            st.caption(f"{citation['source']} (Page {citation['page_number']})")
                
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
