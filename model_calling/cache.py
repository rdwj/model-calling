"""
Simple in-memory cache for embeddings to improve performance.
"""
import hashlib
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from collections import OrderedDict

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Simple in-memory LRU cache for embedding results"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """Initialize the cache with a maximum size and time-to-live."""
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()  # {key: (timestamp, value)}
        self.hits = 0
        self.misses = 0
        logger.info(f"Initialized embedding cache with max_size={max_size}, ttl={ttl_seconds}s")
    
    def _generate_key(self, model: str, input_text: str) -> str:
        """Generate a cache key from model and input text."""
        # Normalize the input to ensure consistent caching
        model_norm = model.lower().strip()
        input_norm = input_text.strip()
        
        # Create a hash of the input to use as the key
        key_str = f"{model_norm}:{input_norm}"
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()
    
    def get(self, model: str, input_text: str) -> Optional[List[float]]:
        """Get an embedding from the cache if it exists and is not expired."""
        key = self._generate_key(model, input_text)
        
        if key in self.cache:
            timestamp, embedding = self.cache[key]
            
            # Check if the entry has expired
            if time.time() - timestamp > self.ttl_seconds:
                # Remove expired entry
                self.cache.pop(key)
                logger.debug(f"Cache entry expired for key: {key}")
                self.misses += 1
                return None
            
            # Move to end to mark as recently used (for LRU eviction)
            self.cache.move_to_end(key)
            self.hits += 1
            logger.debug(f"Cache hit for model={model}, text length={len(input_text)}, hits={self.hits}, misses={self.misses}")
            return embedding
        
        self.misses += 1
        logger.debug(f"Cache miss for model={model}, text length={len(input_text)}, hits={self.hits}, misses={self.misses}")
        return None
    
    def set(self, model: str, input_text: str, embedding: List[float]) -> None:
        """Store an embedding in the cache with the current timestamp."""
        key = self._generate_key(model, input_text)
        
        # Evict least recently used item if we're at capacity
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)  # Remove from the front (least recently used)
            logger.debug("Cache at capacity, evicted oldest entry")
        
        # Add new entry
        self.cache[key] = (time.time(), embedding)
        logger.debug(f"Cached embedding for model={model}, text length={len(input_text)}")
    
    def clear(self) -> None:
        """Clear all entries from the cache."""
        self.cache.clear()
        logger.info("Embedding cache cleared")
    
    def size(self) -> int:
        """Return the current number of entries in the cache."""
        return len(self.cache)
    
    def remove_expired(self) -> int:
        """Remove all expired entries from the cache and return the count removed."""
        current_time = time.time()
        initial_size = len(self.cache)
        
        # Identify expired keys
        expired_keys = [
            key for key, (timestamp, _) in self.cache.items() 
            if current_time - timestamp > self.ttl_seconds
        ]
        
        # Remove expired keys
        for key in expired_keys:
            self.cache.pop(key)
        
        removed_count = initial_size - len(self.cache)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} expired entries from embedding cache")
        
        return removed_count


# Global singleton instance
embedding_cache = EmbeddingCache()
