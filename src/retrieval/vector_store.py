"""
Vector Storage and Retrieval Module
Handles ChromaDB vector storage and similarity search.
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None
    Settings = None

from src.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RetrievalError(Exception):
    """Custom exception for retrieval errors."""
    pass


class VectorStore:
    """
    ChromaDB vector store for document chunks and similarity search.
    """
    
    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: str = "enterprise_rag"
    ):
        """
        Initialize vector store.
        
        Args:
            persist_dir: Directory to persist ChromaDB. Defaults to config.CHROMA_PERSIST_DIR.
            collection_name: Name of the collection.
        """
        if chromadb is None:
            raise RetrievalError(
                "chromadb is not installed. Please install it with: pip install chromadb"
            )
        
        if persist_dir is None:
            persist_dir = config.CHROMA_PERSIST_DIR
        
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        
        try:
            logger.info(f"Initializing ChromaDB at {self.persist_dir}")
            self.client = chromadb.PersistentClient(path=str(self.persist_dir))
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Collection '{self.collection_name}' ready. Count: {self.collection.count()}")
        except Exception as e:
            raise RetrievalError(f"Failed to initialize ChromaDB: {e}")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents with embeddings to the vector store.
        
        Args:
            documents: List of processed documents with text, source, page_number, chunk_id, embedding.
        """
        if not documents:
            logger.warning("No documents provided to add to vector store")
            return
        
        try:
            ids = []
            texts = []
            embeddings = []
            metadatas = []
            
            for doc in documents:
                if "embedding" not in doc:
                    logger.warning(f"Skipping document without embedding: {doc.get('chunk_id')}")
                    continue
                
                chunk_id = doc.get("chunk_id", f"chunk_{len(ids)}")
                ids.append(chunk_id)
                texts.append(doc["text"])
                embeddings.append(doc["embedding"])
                metadatas.append({
                    "source": doc.get("source", "unknown"),
                    "page_number": doc.get("page_number", 0)
                })
            
            if ids:
                self.collection.add(
                    ids=ids,
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
                logger.info(f"Added {len(ids)} documents to vector store")
            else:
                logger.warning("No valid documents with embeddings to add")
                
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise RetrievalError(f"Failed to add documents: {e}")
    
    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Embedding vector of the query.
            top_k: Number of results to return. Defaults to config.RETRIEVAL_TOP_K.
        
        Returns:
            List of results with text, metadata, and distance.
        """
        if top_k is None:
            top_k = config.RETRIEVAL_TOP_K
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            retrieved = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    retrieved.append({
                        "text": results['documents'][0][i],
                        "source": results['metadatas'][0][i].get("source", "unknown"),
                        "page_number": results['metadatas'][0][i].get("page_number", 0),
                        "distance": results['distances'][0][i] if results['distances'] else None
                    })
            
            logger.info(f"Retrieved {len(retrieved)} documents")
            return retrieved
            
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            raise RetrievalError(f"Failed to search: {e}")
    
    def count(self) -> int:
        """Get number of documents in collection."""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Error getting count: {e}")
            return 0

    def delete_by_source(self, source: str) -> int:
        """Delete all indexed chunks belonging to one source file."""
        try:
            results = self.collection.get(
                where={"source": source},
                include=["metadatas"]
            )
            ids = results.get("ids", [])
            if ids:
                self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} chunks for source {source}")
            return len(ids)
        except Exception as e:
            logger.error(f"Error deleting source {source} from vector store: {e}")
            raise RetrievalError(f"Failed to delete source: {e}")
    
    def clear(self) -> None:
        """Clear all documents from collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Collection cleared")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise RetrievalError(f"Failed to clear collection: {e}")
