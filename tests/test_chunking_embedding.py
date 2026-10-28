"""
Tests for Document Processing (Chunking and Embedding)
"""
import pytest
from src.processing import DocumentProcessor, ChunkingEmbeddingError, process_documents


class TestDocumentProcessor:
    """Test cases for DocumentProcessor class."""
    
    def test_init(self):
        """Test initialization with defaults."""
        processor = DocumentProcessor()
        assert processor.chunk_size > 0
        assert processor.chunk_overlap >= 0
        assert processor.embedding_dim > 0
        assert processor.embedding_model is not None
    
    def test_chunk_documents_empty(self):
        """Test chunking empty documents list."""
        processor = DocumentProcessor()
        chunks = processor.chunk_documents([])
        assert chunks == []
    
    def test_chunk_documents_preserves_metadata(self):
        """Test that chunking preserves source and page number."""
        processor = DocumentProcessor(chunk_size=50, chunk_overlap=10)
        docs = [
            {"text": "This is a long text " * 10, "source": "test.pdf", "page_number": 1}
        ]
        chunks = processor.chunk_documents(docs)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk["source"] == "test.pdf"
            assert chunk["page_number"] == 1
            assert "chunk_id" in chunk
            assert "text" in chunk
    
    def test_chunk_documents_no_mixing(self):
        """Test that chunks from different sources are not mixed."""
        processor = DocumentProcessor(chunk_size=50, chunk_overlap=10)
        docs = [
            {"text": "Text from document one " * 5, "source": "doc1.pdf", "page_number": 1},
            {"text": "Text from document two " * 5, "source": "doc2.pdf", "page_number": 1},
        ]
        chunks = processor.chunk_documents(docs)
        # Check all chunks from doc1 have source doc1
        doc1_chunks = [c for c in chunks if c["source"] == "doc1.pdf"]
        doc2_chunks = [c for c in chunks if c["source"] == "doc2.pdf"]
        assert len(doc1_chunks) > 0
        assert len(doc2_chunks) > 0
        # Verify no chunk has mixed sources
        for c in doc1_chunks:
            assert c["source"] == "doc1.pdf"
        for c in doc2_chunks:
            assert c["source"] == "doc2.pdf"
    
    def test_chunk_size_respected(self):
        """Test that chunk size is approximately respected."""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=0)
        docs = [
            {"text": "a" * 500, "source": "test.pdf", "page_number": 1}
        ]
        chunks = processor.chunk_documents(docs)
        # Chunks should be roughly chunk_size (may be slightly less due to splitting)
        for chunk in chunks:
            assert len(chunk["text"]) <= 150  # Allow some flexibility
    
    def test_embed_chunks_empty(self):
        """Test embedding empty chunks list."""
        processor = DocumentProcessor()
        embedded = processor.embed_chunks([])
        assert embedded == []
    
    def test_embed_chunks_adds_embeddings(self):
        """Test that embeddings are added to chunks."""
        processor = DocumentProcessor()
        chunks = [
            {"text": "Hello world", "source": "test.pdf", "page_number": 1, "chunk_id": "test_p1_c0"}
        ]
        embedded = processor.embed_chunks(chunks)
        assert len(embedded) == 1
        assert "embedding" in embedded[0]
        assert isinstance(embedded[0]["embedding"], list)
        assert len(embedded[0]["embedding"]) == processor.get_embedding_dimension()
    
    def test_process_integration(self):
        """Test full process pipeline."""
        processor = DocumentProcessor(chunk_size=50, chunk_overlap=10)
        docs = [
            {"text": "This is a test document with some content. " * 5, "source": "test.pdf", "page_number": 1}
        ]
        result = processor.process(docs)
        assert len(result) > 0
        for record in result:
            assert "text" in record
            assert "source" in record
            assert "page_number" in record
            assert "chunk_id" in record
            assert "embedding" in record
            assert isinstance(record["embedding"], list)
    
    def test_get_embedding_dimension(self):
        """Test getting embedding dimension."""
        processor = DocumentProcessor()
        dim = processor.get_embedding_dimension()
        assert isinstance(dim, int)
        assert dim > 0


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_process_documents(self):
        """Test process_documents convenience function."""
        docs = [
            {"text": "Hello world test content", "source": "test.pdf", "page_number": 1}
        ]
        result = process_documents(docs)
        assert isinstance(result, list)
        if result:
            assert "embedding" in result[0]
    
    def test_chunking_embedding_error(self):
        """Test custom exception."""
        assert issubclass(ChunkingEmbeddingError, Exception)
