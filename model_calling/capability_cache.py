"""
Cache system for model capabilities to avoid redundant testing.
"""
import os
import json
import logging
import time
from typing import Dict, Any, Optional, List, Set

logger = logging.getLogger(__name__)

class CapabilityCache:
    """Cache for model capabilities to avoid repeated testing"""
    
    def __init__(self, cache_file: str = None):
        """Initialize the capability cache
        
        Args:
            cache_file (str, optional): Path to the cache file. Defaults to ~/.model_calling/capability_cache.json.
        """
        if cache_file is None:
            # Default cache location
            cache_dir = os.path.expanduser("~/.model_calling")
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_file = os.path.join(cache_dir, "capability_cache.json")
        else:
            self.cache_file = os.path.expanduser(cache_file)
        
        self.cache = self._load_cache()
        logger.debug(f"Initialized capability cache at {self.cache_file}")
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load cache from disk or create empty cache"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    cache = json.load(f)
                logger.debug(f"Loaded capability cache with {len(cache)} entries")
                return cache
        except Exception as e:
            logger.warning(f"Failed to load capability cache: {e}")
        
        # Return empty cache if loading fails or file doesn't exist
        return {"models": {}, "last_updated": int(time.time())}
    
    def save_cache(self):
        """Save cache to disk"""
        try:
            cache_dir = os.path.dirname(self.cache_file)
            os.makedirs(cache_dir, exist_ok=True)
            
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
            
            logger.debug(f"Saved capability cache with {len(self.cache.get('models', {}))} entries")
        except Exception as e:
            logger.warning(f"Failed to save capability cache: {e}")
    
    def get_capability(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get capability from cache
        
        Args:
            model_id (str): The model ID to look up
            
        Returns:
            Optional[Dict[str, Any]]: The cached capability data or None if not found
        """
        models = self.cache.get("models", {})
        return models.get(model_id)
    
    def set_capability(self, model_id: str, capability: Dict[str, Any]):
        """Set capability in cache and save
        
        Args:
            model_id (str): The model ID to store
            capability (Dict[str, Any]): The capability data to cache
        """
        # Make sure models dict exists
        if "models" not in self.cache:
            self.cache["models"] = {}
        
        # Add timestamp to capability data
        capability["last_tested"] = int(time.time())
        
        # Store in cache
        self.cache["models"][model_id] = capability
        self.cache["last_updated"] = int(time.time())
        
        # Save to disk
        self.save_cache()
        logger.info(f"Cached capability data for {model_id}")
    
    def is_cache_valid(self, model_id: str, max_age_seconds: int = 86400) -> bool:
        """Check if the cached capability is still valid (not too old)
        
        Args:
            model_id (str): The model ID to check
            max_age_seconds (int, optional): Maximum age in seconds. Defaults to 86400 (1 day).
            
        Returns:
            bool: True if the cache is valid, False otherwise
        """
        capability = self.get_capability(model_id)
        if not capability:
            return False
        
        last_tested = capability.get("last_tested", 0)
        current_time = int(time.time())
        
        return (current_time - last_tested) < max_age_seconds
    
    def get_or_test_capability(self, model_id: str, test_func, max_age_seconds: int = 86400):
        """Get capability from cache or test if not available/expired
        
        Args:
            model_id (str): The model ID to check
            test_func (callable): Function to call to test capability if not cached
            max_age_seconds (int, optional): Maximum age in seconds. Defaults to 86400 (1 day).
            
        Returns:
            Dict[str, Any]: The capability data (either from cache or freshly tested)
        """
        # Check if we have a valid cached entry
        if self.is_cache_valid(model_id, max_age_seconds):
            logger.debug(f"Using cached capability data for {model_id}")
            return self.get_capability(model_id)
        
        # Otherwise test the capability
        logger.info(f"Testing capability for {model_id} (cache missing or expired)")
        capability = test_func(model_id)
        
        # Cache the result
        self.set_capability(model_id, capability)
        
        return capability
    
    def get_models_with_capability(self, capability_name: str, capability_value: Any = True) -> List[str]:
        """Get all models that have a specific capability
        
        Args:
            capability_name (str): The name of the capability to check
            capability_value (Any, optional): The value of the capability. Defaults to True.
            
        Returns:
            List[str]: List of model IDs with the specified capability
        """
        models = self.cache.get("models", {})
        result = []
        
        for model_id, capability in models.items():
            if capability.get(capability_name) == capability_value:
                result.append(model_id)
        
        return result
    
    def clear_cache(self):
        """Clear the entire cache"""
        self.cache = {"models": {}, "last_updated": int(time.time())}
        self.save_cache()
        logger.info("Cleared capability cache")

# Create singleton instance
capability_cache = CapabilityCache()
