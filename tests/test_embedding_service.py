"""
Unit Tests for Embedding Service Module

Tests embedding generation, caching, and similarity computation.
"""

import pytest
import numpy as np
from app.embedding_service import EmbeddingService


@pytest.fixture
def embedding_service():
    """Create an embedding service instance."""
    return EmbeddingService()


@pytest.fixture
def sample_texts():
    """Sample texts for embedding."""
    return [
        "Machine learning engineer with 5 years of experience",
        "Senior Python developer specializing in data processing",
        "Product manager in AI startup with technical background",
        "Full-stack developer with expertise in cloud infrastructure",
    ]


class TestEmbeddingService:
    """Test EmbeddingService functionality."""
    
    def test_initialization(self, embedding_service):
        """Test embedding service initialization."""
        assert embedding_service is not None
        assert embedding_service.model_name == "all-MiniLM-L6-v2"
        assert embedding_service.embedding_cache == {}
    
    def test_model_lazy_loading(self, embedding_service):
        """Test lazy loading of model."""
        assert embedding_service.model is None
        
        # Model loads on first embedding call
        embedding_service.load_model()
        assert embedding_service.model is not None
    
    def test_embed_text_shape(self, embedding_service, sample_texts):
        """Test single text embedding shape."""
        text = sample_texts[0]
        embedding = embedding_service.embed_text(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.ndim == 1
        assert embedding.shape[0] == 384  # all-MiniLM-L6-v2 dimension
    
    def test_embed_text_cache(self, embedding_service, sample_texts):
        """Test caching of embeddings."""
        text = sample_texts[0]
        
        # First call computes embedding
        embedding1 = embedding_service.embed_text(text, use_cache=True)
        assert text in embedding_service.embedding_cache
        
        # Second call uses cache
        embedding2 = embedding_service.embed_text(text, use_cache=True)
        assert np.allclose(embedding1, embedding2)
    
    def test_no_cache_option(self, embedding_service, sample_texts):
        """Test embedding without caching."""
        text = sample_texts[0]
        
        embedding = embedding_service.embed_text(text, use_cache=False)
        assert text not in embedding_service.embedding_cache
        assert embedding is not None
    
    def test_embed_batch(self, embedding_service, sample_texts):
        """Test batch embedding."""
        embeddings = embedding_service.embed_batch(sample_texts, batch_size=2)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (len(sample_texts), 384)
    
    def test_embed_batch_cache(self, embedding_service, sample_texts):
        """Test batch embedding with caching."""
        embeddings1 = embedding_service.embed_batch(sample_texts, use_cache=True)
        
        # All texts should be cached
        for text in sample_texts:
            assert text in embedding_service.embedding_cache
        
        # Second call should use cache
        embeddings2 = embedding_service.embed_batch(sample_texts, use_cache=True)
        assert np.allclose(embeddings1, embeddings2)
    
    def test_cosine_similarity_computation(self, embedding_service, sample_texts):
        """Test cosine similarity computation."""
        emb1 = embedding_service.embed_text(sample_texts[0])
        emb2 = embedding_service.embed_text(sample_texts[1])
        
        similarity = embedding_service.compute_similarity(emb1, emb2)
        
        assert 0 <= similarity <= 1
        assert isinstance(similarity, (float, np.floating))
    
    def test_self_similarity_is_one(self, embedding_service, sample_texts):
        """Test that similarity with self is 1.0."""
        embedding = embedding_service.embed_text(sample_texts[0])
        
        similarity = embedding_service.compute_similarity(embedding, embedding)
        assert similarity >= 0.99  # Allow small floating point error
    
    def test_batch_similarity_computation(self, embedding_service, sample_texts):
        """Test batch similarity computation."""
        query_embedding = embedding_service.embed_text(sample_texts[0])
        doc_embeddings = embedding_service.embed_batch(sample_texts[1:], use_cache=False)
        
        similarities = embedding_service.batch_similarity(query_embedding, doc_embeddings)
        
        assert isinstance(similarities, np.ndarray)
        assert len(similarities) == len(sample_texts) - 1
        assert all(0 <= sim <= 1 for sim in similarities)
    
    def test_embedding_dimension(self, embedding_service):
        """Test getting embedding dimension."""
        dim = embedding_service.get_embedding_dimension()
        assert dim == 384
    
    def test_clear_cache(self, embedding_service, sample_texts):
        """Test clearing embedding cache."""
        embedding_service.embed_batch(sample_texts, use_cache=True)
        assert len(embedding_service.embedding_cache) > 0
        
        embedding_service.clear_cache()
        assert len(embedding_service.embedding_cache) == 0


class TestSimilaritySemantics:
    """Test semantic similarity behavior."""
    
    def test_similar_texts_high_similarity(self, embedding_service):
        """Test that similar texts have high similarity."""
        text1 = "Machine learning engineer with Python expertise"
        text2 = "Software engineer specializing in machine learning with Python"
        
        emb1 = embedding_service.embed_text(text1)
        emb2 = embedding_service.embed_text(text2)
        
        similarity = embedding_service.compute_similarity(emb1, emb2)
        assert similarity > 0.7  # Should be similar
    
    def test_dissimilar_texts_low_similarity(self, embedding_service):
        """Test that dissimilar texts have low similarity."""
        text1 = "Machine learning engineer"
        text2 = "Veterinary surgeon specializing in exotic animals"
        
        emb1 = embedding_service.embed_text(text1)
        emb2 = embedding_service.embed_text(text2)
        
        similarity = embedding_service.compute_similarity(emb1, emb2)
        assert similarity < 0.65  # Should be dissimilar (normalized similarity)


class TestBatchProcessing:
    """Test batch processing logic."""
    
    def test_batch_size_handling(self, embedding_service):
        """Test handling of various batch sizes."""
        texts = [f"Text {i}" for i in range(100)]
        
        for batch_size in [1, 16, 32, 64]:
            embeddings = embedding_service.embed_batch(texts, batch_size=batch_size)
            assert embeddings.shape[0] == len(texts)
    
    def test_empty_batch(self, embedding_service):
        """Test handling of empty batch."""
        embeddings = embedding_service.embed_batch([], batch_size=32)
        assert embeddings.shape[0] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
