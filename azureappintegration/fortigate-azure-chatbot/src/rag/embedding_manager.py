"""
Embedding Management for RAG Agent
Handles text embedding generation and management
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time

# LangChain imports
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from .config import get_rag_config

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manages text embedding generation and caching"""
    
    def __init__(self):
        self.config = get_rag_config()
        self.embeddings_model = None
        self.embedding_cache = {}
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize the embeddings model"""
        try:
            if self.config.embedding_provider == "openai":
                self.embeddings_model = OpenAIEmbeddings(
                    model=self.config.embedding_model,-ur
                    openai_api_key=self.config.openai_api_key
                )
                logger.info(f"Initialized OpenAI embeddings with model: {self.config.embedding_model}")
                
            elif self.config.embedding_provider == "huggingface":
                model_name = self.config.huggingface_model or "sentence-transformers/all-MiniLM-L6-v2"
                self.embeddings_model = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={'device': 'cpu'}  # Use CPU for compatibility
                )
                logger.info(f"Initialized HuggingFace embeddings with model: {model_name}")
                
            else:
                raise ValueError(f"Unsupported embedding provider: {self.config.embedding_provider}")
                
        except Exception as e:
            logger.error(f"Failed to initialize embeddings model: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        try:
            if self.config.embedding_provider == "openai":
                # OpenAI text-embedding-ada-002 has 1536 dimensions
                # text-embedding-3-small has 1536 dimensions
                # text-embedding-3-large has 3072 dimensions
                if "3-large" in self.config.embedding_model:
                    return 3072
                else:
                    return 1536
            elif self.config.embedding_provider == "huggingface":
                # Most sentence-transformers models have 384 or 768 dimensions
                # We can test with a dummy text to get the actual dimension
                test_embedding = self.embeddings_model.embed_query("test")
                return len(test_embedding)
            else:
                return 1536  # Default fallback
                
        except Exception as e:
            logger.warning(f"Failed to get embedding dimension: {e}, using default 1536")
            return 1536
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        try:
            # Check cache first
            text_hash = hash(text)
            if text_hash in self.embedding_cache:
                return self.embedding_cache[text_hash]
            
            # Generate embedding
            embedding = self.embeddings_model.embed_query(text)
            
            # Cache the result
            self.embedding_cache[text_hash] = embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {e}")
            raise
    
    def embed_texts(self, texts: List[str], show_progress: bool = True) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        try:
            if not texts:
                return []
            
            # Check which texts are already cached
            cached_embeddings = {}
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                text_hash = hash(text)
                if text_hash in self.embedding_cache:
                    cached_embeddings[i] = self.embedding_cache[text_hash]
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            logger.info(f"Found {len(cached_embeddings)} cached embeddings, generating {len(uncached_texts)} new ones")
            
            # Generate embeddings for uncached texts
            new_embeddings = []
            if uncached_texts:
                if show_progress:
                    import streamlit as st
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                
                # Process in batches to avoid rate limits
                batch_size = self.config.embedding_batch_size
                for i in range(0, len(uncached_texts), batch_size):
                    batch_texts = uncached_texts[i:i + batch_size]
                    
                    if show_progress:
                        progress = (i + len(batch_texts)) / len(uncached_texts)
                        progress_bar.progress(progress)
                        status_text.text(f"Generating embeddings: {i + len(batch_texts)}/{len(uncached_texts)}")
                    
                    # Generate embeddings for batch
                    try:
                        batch_embeddings = self.embeddings_model.embed_documents(batch_texts)
                        new_embeddings.extend(batch_embeddings)
                        
                        # Cache the new embeddings
                        for text, embedding in zip(batch_texts, batch_embeddings):
                            text_hash = hash(text)
                            self.embedding_cache[text_hash] = embedding
                        
                        # Rate limiting
                        if i + batch_size < len(uncached_texts):
                            time.sleep(0.1)  # Small delay between batches
                            
                    except Exception as e:
                        logger.error(f"Failed to generate embeddings for batch {i//batch_size + 1}: {e}")
                        # Generate embeddings one by one as fallback
                        for text in batch_texts:
                            try:
                                embedding = self.embed_text(text)
                                new_embeddings.append(embedding)
                            except Exception as fallback_error:
                                logger.error(f"Failed to generate embedding even with fallback: {fallback_error}")
                                # Use zero vector as last resort
                                dimension = self.get_embedding_dimension()
                                new_embeddings.append([0.0] * dimension)
                
                if show_progress:
                    progress_bar.progress(1.0)
                    status_text.text("Embeddings generation complete!")
            
            # Combine cached and new embeddings in correct order
            all_embeddings = [None] * len(texts)
            
            # Fill in cached embeddings
            for index, embedding in cached_embeddings.items():
                all_embeddings[index] = embedding
            
            # Fill in new embeddings
            for uncached_idx, embedding in zip(uncached_indices, new_embeddings):
                all_embeddings[uncached_idx] = embedding
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings for texts: {e}")
            raise
    
    async def embed_texts_async(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings asynchronously"""
        try:
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=4) as executor:
                embeddings = await loop.run_in_executor(
                    executor, 
                    self.embed_texts, 
                    texts, 
                    False  # Don't show progress in async mode
                )
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings asynchronously: {e}")
            raise
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for document chunks"""
        try:
            # Extract texts from chunks
            texts = [chunk["content"] for chunk in chunks]
            
            # Generate embeddings
            embeddings = self.embed_texts(texts)
            
            # Add embeddings to chunks
            enriched_chunks = []
            for chunk, embedding in zip(chunks, embeddings):
                enriched_chunk = chunk.copy()
                enriched_chunk["embedding"] = embedding
                enriched_chunk["metadata"]["embedding_model"] = self.config.embedding_provider
                enriched_chunk["metadata"]["embedding_dimension"] = len(embedding)
                enriched_chunks.append(enriched_chunk)
            
            logger.info(f"Generated embeddings for {len(enriched_chunks)} chunks")
            return enriched_chunks
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings for chunks: {e}")
            raise
    
    def similarity_search(self, query_embedding: List[float], chunk_embeddings: List[List[float]], top_k: int = 5) -> List[int]:
        """Find most similar chunks using cosine similarity"""
        try:
            if not chunk_embeddings:
                return []
            
            # Convert to numpy arrays for efficient computation
            query_vec = np.array(query_embedding)
            chunk_vecs = np.array(chunk_embeddings)
            
            # Compute cosine similarities
            similarities = []
            for chunk_vec in chunk_vecs:
                # Cosine similarity = dot product / (norm1 * norm2)
                similarity = np.dot(query_vec, chunk_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec))
                similarities.append(similarity)
            
            # Get top-k most similar indices
            similar_indices = np.argsort(similarities)[::-1][:top_k]
            
            return similar_indices.tolist()
            
        except Exception as e:
            logger.error(f"Failed to perform similarity search: {e}")
            return []
    
    def clear_cache(self):
        """Clear the embedding cache"""
        self.embedding_cache.clear()
        logger.info("Embedding cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": len(self.embedding_cache),
            "embedding_provider": self.config.embedding_provider,
            "embedding_model": (
                self.config.embedding_model 
                if self.config.embedding_provider == "openai" 
                else self.config.huggingface_model
            ),
            "embedding_dimension": self.get_embedding_dimension()
        }

# Global instance
_embedding_manager = None

def get_embedding_manager() -> EmbeddingManager:
    """Get the global embedding manager instance"""
    global _embedding_manager
    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager()
    return _embedding_manager
