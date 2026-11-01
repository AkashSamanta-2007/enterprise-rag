"""
Tests for RAG Pipeline and Generation
Using mocks to avoid real API calls
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.generation.llm_client import GroqClient, LLMError
from src.rag_pipeline import RAGPipeline


class TestGroqClient:
    """Test Groq client with mocks."""
    
    @patch('src.generation.llm_client.Groq')
    def test_init_with_api_key(self, mock_groq):
        """Test initialization with API key."""
        client = GroqClient(api_key="test_key", model="test-model")
        assert client.api_key == "test_key"
        assert client.model == "test-model"
        mock_groq.assert_called_once_with(api_key="test_key")
    
    @patch('src.generation.llm_client.config')
    def test_init_without_api_key_raises_error(self, mock_config):
        """Test that missing API key raises error."""
        mock_config.GROQ_API_KEY = ""
        mock_config.GROQ_MODEL_NAME = "test"
        with pytest.raises(LLMError) as exc:
            GroqClient()
        assert "API key" in str(exc.value)
    
    @patch('src.generation.llm_client.Groq')
    def test_chat_completion(self, mock_groq):
        """Test chat completion."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test answer"
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_groq.return_value = mock_client
        
        client = GroqClient(api_key="test_key", model="test-model")
        result = client.chat_completion([{"role": "user", "content": "hello"}])
        
        assert result == "Test answer"
        mock_client.chat.completions.create.assert_called_once()
    
    @patch('src.generation.llm_client.Groq')
    def test_chat_completion_auth_error(self, mock_groq):
        """Test authentication error handling."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("Invalid API key")
        mock_groq.return_value = mock_client
        
        client = GroqClient(api_key="bad_key", model="test-model")
        with pytest.raises(LLMError) as exc:
            client.chat_completion([{"role": "user", "content": "hello"}])
        assert "Authentication" in str(exc.value)
    
    @patch('src.generation.llm_client.Groq')
    def test_chat_completion_rate_limit_error(self, mock_groq):
        """Test rate limit error handling."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("rate limit exceeded 429")
        mock_groq.return_value = mock_client
        
        client = GroqClient(api_key="test_key", model="test-model")
        with pytest.raises(LLMError) as exc:
            client.chat_completion([{"role": "user", "content": "hello"}])
        assert "Rate limit" in str(exc.value)


class TestRAGPipeline:
    """Test RAG pipeline with mocks."""
    
    def test_query_empty_question(self):
        """Test query with empty question."""
        with patch('src.rag_pipeline.VectorStore'), \
             patch('src.rag_pipeline.GroqClient'), \
             patch('src.rag_pipeline.DocumentProcessor'), \
             patch('src.rag_pipeline.PDFExtractor'):
            pipeline = RAGPipeline()
            result = pipeline.query("")
            assert "Invalid question" in result["answer"]
            assert result["citations"] == []
    
    def test_query_no_documents_in_store(self):
        """Test query when no documents in store."""
        with patch('src.rag_pipeline.VectorStore') as mock_vs, \
             patch('src.rag_pipeline.GroqClient'), \
             patch('src.rag_pipeline.DocumentProcessor'), \
             patch('src.rag_pipeline.PDFExtractor'):
            mock_vs.return_value.count.return_value = 0
            pipeline = RAGPipeline()
            result = pipeline.query("What is this about?")
            assert "No documents have been ingested" in result["answer"]
            assert result["citations"] == []
    
    @patch('src.rag_pipeline.VectorStore')
    @patch('src.rag_pipeline.GroqClient')
    @patch('src.rag_pipeline.DocumentProcessor')
    @patch('src.rag_pipeline.PDFExtractor')
    def test_query_with_results(self, mock_pdf, mock_dp, mock_llm, mock_vs):
        """Test query with retrieved results."""
        import numpy as np
        mock_vs.return_value.count.return_value = 5
        mock_vs.return_value.similarity_search.return_value = [
            {"text": "Test content", "source": "doc.pdf", "page_number": 1}
        ]
        mock_dp.return_value.embedding_model.encode.return_value = np.array([[0.1, 0.2]])
        mock_llm.return_value.generate_answer.return_value = "This is the answer. [doc.pdf, page 1]"
        
        pipeline = RAGPipeline()
        result = pipeline.query("Test question")
        
        assert result["answer"] == "This is the answer. [doc.pdf, page 1]"
        assert len(result["citations"]) == 1
        assert result["citations"][0]["source"] == "doc.pdf"
        assert result["citations"][0]["page_number"] == 1
    
    @patch('src.rag_pipeline.VectorStore')
    @patch('src.rag_pipeline.GroqClient')
    @patch('src.rag_pipeline.DocumentProcessor')
    @patch('src.rag_pipeline.PDFExtractor')
    def test_query_insufficient_info(self, mock_pdf, mock_dp, mock_llm, mock_vs):
        """Test query when documents don't have answer."""
        import numpy as np
        mock_vs.return_value.count.return_value = 5
        mock_vs.return_value.similarity_search.return_value = [
            {"text": "Unrelated content", "source": "doc.pdf", "page_number": 1}
        ]
        mock_dp.return_value.embedding_model.encode.return_value = np.array([[0.1, 0.2]])
        mock_llm.return_value.generate_answer.return_value = "The available documents do not provide enough information to answer this question."
        
        pipeline = RAGPipeline()
        result = pipeline.query("Unknown question")
        
        assert "do not provide enough information" in result["answer"].lower()
        assert result["citations"] == []
    
    @patch('src.rag_pipeline.VectorStore')
    @patch('src.rag_pipeline.GroqClient')
    @patch('src.rag_pipeline.DocumentProcessor')
    @patch('src.rag_pipeline.PDFExtractor')
    def test_query_llm_error(self, mock_pdf, mock_dp, mock_llm, mock_vs):
        """Test query with LLM error."""
        import numpy as np
        mock_vs.return_value.count.return_value = 5
        mock_vs.return_value.similarity_search.return_value = [
            {"text": "Test", "source": "doc.pdf", "page_number": 1}
        ]
        mock_dp.return_value.embedding_model.encode.return_value = np.array([[0.1, 0.2]])
        mock_llm.return_value.generate_answer.side_effect = LLMError("API error")
        
        pipeline = RAGPipeline()
        result = pipeline.query("Test")
        
        assert "Error generating answer" in result["answer"]
        assert result["citations"] == []
    
    @patch('src.rag_pipeline.VectorStore')
    @patch('src.rag_pipeline.GroqClient')
    @patch('src.rag_pipeline.DocumentProcessor')
    @patch('src.rag_pipeline.PDFExtractor')
    def test_citations_from_actual_retrieved_sources(self, mock_pdf, mock_dp, mock_llm, mock_vs):
        """Test that citations use actual retrieved source and page."""
        import numpy as np
        mock_vs.return_value.count.return_value = 5
        mock_vs.return_value.similarity_search.return_value = [
            {"text": "Content 1", "source": "manual.pdf", "page_number": 5},
            {"text": "Content 2", "source": "guide.pdf", "page_number": 10},
        ]
        mock_dp.return_value.embedding_model.encode.return_value = np.array([[0.1, 0.2]])
        mock_llm.return_value.generate_answer.return_value = "Answer [guide.pdf, page 10]"
        
        pipeline = RAGPipeline()
        result = pipeline.query("Test")
        
        assert result["citations"] == [
            {"source": "guide.pdf", "page_number": 10}
        ]
    
    def test_get_status(self):
        """Test getting pipeline status."""
        with patch('src.rag_pipeline.VectorStore') as mock_vs, \
             patch('src.rag_pipeline.GroqClient') as mock_llm, \
             patch('src.rag_pipeline.DocumentProcessor') as mock_dp, \
             patch('src.rag_pipeline.PDFExtractor'):
            mock_vs.return_value.count.return_value = 10
            mock_llm.return_value.model = "llama-3.1-70b-versatile"
            mock_dp.return_value.embedding_model_name = "all-MiniLM-L6-v2"
            mock_dp.return_value.get_embedding_dimension.return_value = 384
            
            pipeline = RAGPipeline()
            status = pipeline.get_status()
            
            assert status["documents_in_store"] == 10
            assert status["llm_model"] == "llama-3.1-70b-versatile"
            assert status["embedding_dim"] == 384

    def test_delete_document_removes_source_from_vector_store(self):
        """Test that deleting a document delegates to source-specific index deletion."""
        vector_store = Mock()
        vector_store.delete_by_source.return_value = 3

        with patch('src.rag_pipeline.PDFExtractor'):
            pipeline = RAGPipeline(
                vector_store=vector_store,
                llm_client=Mock(),
                document_processor=Mock()
            )

        assert pipeline.delete_document("policy.pdf") == 3
        vector_store.delete_by_source.assert_called_once_with("policy.pdf")
