"""
Text Chunking and Embedding Module
Handles splitting documents into chunks and generating embeddings using sentence-transformers.
"""

import logging
from typing import List, Dict, Any, Optional

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    RecursiveCharacterTextSplitter = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from src.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChunkingEmbeddingError(Exception):
    """Custom exception for chunking and embedding errors."""
    pass


class DocumentProcessor:
    """
    Processes documents by chunking text and generating embeddings.
    """
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        embedding_model: Optional[str] = None
    ):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Size of each chunk. Defaults to config.CHUNK_SIZE.
            chunk_overlap: Overlap between chunks. Defaults to config.CHUNK_OVERLAP.
            embedding_model: Embedding model name. Defaults to config.EMBEDDING_MODEL_NAME.
        """
        if RecursiveCharacterTextSplitter is None:
            raise ChunkingEmbeddingError(
                "langchain-text-splitters is not installed. "
                "Please install it with: pip install langchain-text-splitters"
            )
        
        if SentenceTransformer is None:
            raise ChunkingEmbeddingError(
                "sentence-transformers is not installed. "
                "Please install it with: pip install sentence-transformers"
            )
        
        self.chunk_size = chunk_size if chunk_size is not None else config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP
        self.embedding_model_name = embedding_model if embedding_model is not None else config.EMBEDDING_MODEL_NAME
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize embedding model
        try:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            logger.info(f"Embedding model loaded. Dimension: {self.embedding_dim}")
        except Exception as e:
            raise ChunkingEmbeddingError(f"Failed to load embedding model: {e}")
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split documents into chunks while preserving metadata.
        Does not mix chunks from different source documents.
        
        Args:
            documents: List of document records from ingestion.
                      Each should have 'text', 'source', 'page_number'.
        
        Returns:
            List of chunked records with text, source, page_number, and chunk_id.
        """
        chunked_records = []
        
        if not documents:
            logger.warning("No documents provided for chunking")
            return chunked_records
        
        logger.info(f"Chunking {len(documents)} document pages")
        
        for doc in documents:
            try:
                text = doc.get("text", "")
                source = doc.get("source", "unknown")
                page_number = doc.get("page_number", 0)
                
                if not text or not text.strip():
                    logger.warning(f"Skipping empty text from {source} page {page_number}")
                    continue
                
                # Split text into chunks
                chunks = self.text_splitter.split_text(text)
                
                for idx, chunk in enumerate(chunks):
                    if not chunk.strip():
                        continue
                    
                    chunk_record = {
                        "text": chunk.strip(),
                        "source": source,
                        "page_number": page_number,
                        "chunk_id": f"{source}_p{page_number}_c{idx}"
                    }
                    chunked_records.append(chunk_record)
                
                logger.debug(f"Split {source} page {page_number} into {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Error chunking document {doc}: {e}")
                continue
        
        logger.info(f"Created {len(chunked_records)} chunks from {len(documents)} pages")
        return chunked_records
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for each chunk.
        
        Args:
            chunks: List of chunked records.
        
        Returns:
            List of records with embeddings added.
        """
        embedded_records = []
        
        if not chunks:
            logger.warning("No chunks provided for embedding")
            return embedded_records
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        try:
            # Extract texts for batch embedding
            texts = [chunk["text"] for chunk in chunks]
            
            # Generate embeddings in batch
            embeddings = self.embedding_model.encode(
                texts,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            for chunk, embedding in zip(chunks, embeddings):
                embedded_record = chunk.copy()
                embedded_record["embedding"] = embedding.tolist()
                embedded_records.append(embedded_record)
            
            logger.info(f"Generated {len(embedded_records)} embeddings (dim={self.embedding_dim})")
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise ChunkingEmbeddingError(f"Failed to generate embeddings: {e}")
        
        return embedded_records
    
    def process(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Complete pipeline: chunk documents and generate embeddings.
        
        Args:
            documents: List of document records from ingestion.
        
        Returns:
            List of processed records with chunks and embeddings.
        """
        chunks = self.chunk_documents(documents)
        embedded_chunks = self.embed_chunks(chunks)
        return embedded_chunks
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension of the loaded model."""
        return self.embedding_dim


def process_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convenience function to process documents (chunk + embed).
    
    Args:
        documents: List of document records from ingestion.
        
    Returns:
        List of processed records with chunks and embeddings.
    """
    processor = DocumentProcessor()
    return processor.process(documents)
