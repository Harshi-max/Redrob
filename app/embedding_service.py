"""
Embedding Service Module - Handles semantic embeddings for candidates and JDs.

Uses sentence-transformers/all-MiniLM-L6-v2 for efficient CPU-based embeddings.
Implements caching and batching for memory efficiency.
"""

import numpy as np
from typing import List, Dict, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating and caching semantic embeddings.
    
    Uses sentence-transformers model for efficient CPU-based embeddings.
    Supports batch processing and local caching.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: Optional[str] = None):
        """
        Initialize embedding service.
        
        Args:
            model_name: Hugging Face model identifier
            cache_dir: Directory for caching embeddings
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.model = None
        self.embedding_cache = {}
        
    def load_model(self):
        """Lazy load the embedding model."""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading {self.model_name}...")
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
            except ImportError:
                logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
                raise
    
    def embed_text(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Embed a single text string.
        
        Args:
            text: Text to embed
            use_cache: Whether to use/store in cache
            
        Returns:
            Embedding vector
        """
        if use_cache and text in self.embedding_cache:
            return self.embedding_cache[text]
        
        self.load_model()
        embedding = self.model.encode([text], convert_to_numpy=True)[0]
        
        if use_cache:
            self.embedding_cache[text] = embedding
        
        return embedding
    
    def embed_batch(self, texts: List[str], batch_size: int = 32, 
                   use_cache: bool = True) -> np.ndarray:
        """
        Embed multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            use_cache: Whether to use/store in cache
            
        Returns:
            Array of embeddings (n_texts, embedding_dim)
        """
        self.load_model()
        
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Separate cached and uncached texts
        if use_cache:
            for i, text in enumerate(texts):
                if text in self.embedding_cache:
                    embeddings.append(self.embedding_cache[text])
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        # Process uncached texts in batches
        if uncached_texts:
            logger.info(f"Embedding {len(uncached_texts)}/{len(texts)} uncached texts...")
            batch_embeddings = self.model.encode(uncached_texts, batch_size=batch_size, 
                                               convert_to_numpy=True, show_progress_bar=True)
            
            if use_cache:
                for text, emb in zip(uncached_texts, batch_embeddings):
                    self.embedding_cache[text] = emb
        
        # Reconstruct full embeddings array in original order
        if use_cache and uncached_indices:
            full_embeddings = [None] * len(texts)
            emb_idx = 0
            cache_idx = 0
            
            for i, text in enumerate(texts):
                if i in uncached_indices:
                    full_embeddings[i] = batch_embeddings[emb_idx]
                    emb_idx += 1
                else:
                    full_embeddings[i] = self.embedding_cache[text]
            
            return np.array(full_embeddings)
        elif not use_cache:
            return batch_embeddings
        else:
            return np.array([self.embedding_cache[text] for text in texts])
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Normalize embeddings
        e1 = embedding1 / (np.linalg.norm(embedding1) + 1e-10)
        e2 = embedding2 / (np.linalg.norm(embedding2) + 1e-10)
        
        similarity = np.dot(e1, e2)
        
        # Ensure in [0, 1]
        return max(0, min(1, (similarity + 1) / 2))
    
    def batch_similarity(self, query_embedding: np.ndarray, 
                        document_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute similarities between query and multiple documents.
        
        Args:
            query_embedding: Query embedding
            document_embeddings: Array of document embeddings
            
        Returns:
            Array of similarity scores
        """
        # Normalize
        q = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        docs = document_embeddings / (np.linalg.norm(document_embeddings, axis=1, keepdims=True) + 1e-10)
        
        # Batch similarity
        similarities = np.dot(docs, q)
        
        # Normalize to [0, 1]
        return (similarities + 1) / 2
    
    def save_embeddings(self, embeddings: np.ndarray, path: str):
        """Save embeddings to disk."""
        np.save(path, embeddings)
        logger.info(f"Saved embeddings to {path}")
    
    def load_embeddings(self, path: str) -> np.ndarray:
        """Load embeddings from disk."""
        embeddings = np.load(path)
        logger.info(f"Loaded embeddings from {path}: shape {embeddings.shape}")
        return embeddings
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self.embedding_cache.clear()
        logger.info("Cleared embedding cache")
    
    def get_embedding_dimension(self) -> int:
        """Get the dimensionality of embeddings."""
        self.load_model()
        return self.model.get_sentence_embedding_dimension()
