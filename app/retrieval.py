"""
Retrieval Module - Hybrid retrieval combining semantic and lexical matching.

Implements FAISS for efficient approximate nearest neighbor search,
BM25 for keyword matching, and fusion strategies.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class HybridRetrieval:
    """Hybrid retrieval combining semantic and lexical matching."""
    
    def __init__(self, embedding_service=None, use_faiss: bool = True):
        """
        Initialize retrieval system.
        
        Args:
            embedding_service: Service for computing embeddings
            use_faiss: Whether to use FAISS for approximate search
        """
        self.embedding_service = embedding_service
        self.use_faiss = use_faiss
        self.faiss_index = None
        self.bm25_index = None
        self.documents = None
        self.embeddings = None
        
    def build_index(self, documents: List[str], embeddings: Optional[np.ndarray] = None):
        """
        Build retrieval indices.
        
        Args:
            documents: List of documents to index
            embeddings: Pre-computed embeddings (optional)
        """
        self.documents = documents
        
        # Build BM25 index
        logger.info("Building BM25 index...")
        self._build_bm25_index(documents)
        
        # Build semantic index
        logger.info("Building semantic index...")
        if embeddings is None:
            self.embeddings = self.embedding_service.embed_batch(documents)
        else:
            self.embeddings = embeddings
        
        if self.use_faiss:
            self._build_faiss_index(self.embeddings)
    
    def _build_bm25_index(self, documents: List[str]):
        """Build BM25 index."""
        try:
            from rank_bm25 import BM25Okapi
            
            def tokenize(text):
                import re
                return re.findall(r'\b\w+\b', text.lower())
            
            tokenized_docs = [tokenize(doc) for doc in documents]
            self.bm25_index = BM25Okapi(tokenized_docs)
            logger.info(f"BM25 index built for {len(documents)} documents")
        except ImportError:
            logger.warning("rank_bm25 not installed. BM25 matching disabled.")
    
    def _build_faiss_index(self, embeddings: np.ndarray):
        """Build FAISS index."""
        try:
            import faiss
            
            # Convert to float32 if needed
            embeddings = embeddings.astype(np.float32)
            
            # Create index
            dim = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatL2(dim)
            self.faiss_index.add(embeddings)
            
            logger.info(f"FAISS index built with {embeddings.shape[0]} vectors of dim {dim}")
        except ImportError:
            logger.warning("FAISS not installed. Approximate search disabled.")
            self.faiss_index = None
    
    def retrieve(self, query: str, top_k: int = 100, 
                weights: Dict[str, float] = None) -> List[Tuple[int, float]]:
        """
        Retrieve top-k documents for a query.
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve
            weights: Weights for combining scores (semantic, bm25)
            
        Returns:
            List of (document_index, score) tuples
        """
        if weights is None:
            weights = {'semantic': 0.6, 'bm25': 0.4}
        
        scores = np.zeros(len(self.documents))
        
        # Semantic search
        query_embedding = self.embedding_service.embed_text(query)
        semantic_scores = self.embedding_service.batch_similarity(
            query_embedding, self.embeddings
        )
        scores += weights['semantic'] * semantic_scores
        
        # BM25 search
        if self.bm25_index:
            import re
            query_tokens = re.findall(r'\b\w+\b', query.lower())
            bm25_scores = np.array(self.bm25_index.get_scores(query_tokens))
            
            # Normalize BM25 scores
            if bm25_scores.max() > 0:
                bm25_scores = bm25_scores / bm25_scores.max()
            
            scores += weights['bm25'] * bm25_scores
        
        # Get top-k
        top_indices = np.argsort(scores)[-top_k:][::-1]
        results = [(idx, scores[idx]) for idx in top_indices]
        
        return results
    
    def retrieve_batch(self, queries: List[str], top_k: int = 100,
                      weights: Dict[str, float] = None) -> Dict[int, List[Tuple[int, float]]]:
        """
        Retrieve top-k documents for multiple queries.
        
        Args:
            queries: List of queries
            top_k: Number of documents per query
            weights: Weights for combining scores
            
        Returns:
            Dict mapping query index to list of (doc_index, score) tuples
        """
        results = {}
        for i, query in enumerate(queries):
            results[i] = self.retrieve(query, top_k, weights)
        return results
