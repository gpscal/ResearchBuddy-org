"""Enhanced embedding utilities for the research assistant (CPU-only)."""

from __future__ import annotations

import os
import logging
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - use lightweight model for low-resource systems
_DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# Device detection - CPU-only
_DEVICE = "cpu"
_GPU_INFO = {}
logger.info("Using CPU mode for embeddings")

# Batch size optimization for low-resource systems
def get_optimal_batch_size() -> int:
    """Determine optimal batch size based on available system RAM"""
    import psutil
    
    # Check system RAM - very conservative for low-resource systems
    system_ram_gb = psutil.virtual_memory().total / (1024**3)
    logger.info(f"System RAM: {system_ram_gb:.2f}GB")
    
    # Very conservative batch sizes for low-resource systems
    if system_ram_gb < 1:  # Extremely limited RAM (< 1GB)
        logger.warning("Extremely low system RAM detected (< 1GB), using batch size 1")
        return 1
    elif system_ram_gb < 2:  # Very limited RAM (1-2GB)
        logger.warning("Very low system RAM detected (1-2GB), using minimal batch size")
        return 2
    elif system_ram_gb < 4:  # Limited RAM (2-4GB)
        logger.warning("Low system RAM detected, using small batch size")
        return 4
    elif system_ram_gb < 8:  # Limited RAM (4-8GB)
        logger.warning("Low system RAM detected, using small batch size")
        return 8
    elif system_ram_gb < 16:  # Moderate RAM (8-16GB)
        logger.info("Moderate system RAM detected, using conservative batch size")
        return 8
    else:  # 16GB+
        return min(16, max(8, int(system_ram_gb / 4)))  # Still conservative

# Optimal batch size for current device
_OPTIMAL_BATCH_SIZE = get_optimal_batch_size()

@lru_cache(maxsize=1)
def get_embedding_model(model_name: str = _DEFAULT_MODEL) -> SentenceTransformer:
    """Load and cache the sentence transformer model with device optimization."""
    try:
        logger.info(f"Loading embedding model '{model_name}' on {_DEVICE}")
        model = SentenceTransformer(model_name, device=_DEVICE)
        
        # Print model size information
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
        logger.info(f"Model loaded, size: {model_size_mb:.2f}MB")
        
        return model
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        # Fallback to CPU if GPU loading fails
        if _DEVICE == "cuda":
            logger.info("Falling back to CPU for embedding model")
            return SentenceTransformer(model_name, device="cpu")
        raise

def embedding_device() -> str:
    """Return the current embedding device"""
    return _DEVICE

def get_gpu_info() -> Dict:
    """Return information about the GPU if available"""
    return _GPU_INFO

def encode_texts(texts: Iterable[str], batch_size: Optional[int] = None):
    """Encode multiple texts with optimized batch processing
    
    Args:
        texts: Iterable of texts to encode
        batch_size: Override default batch size (otherwise uses optimal size for device)
        
    Returns:
        Tensor of embeddings
    """
    model = get_embedding_model()
    # Convert to list to support generators used multiple times
    text_list: List[str] = list(texts)
    if not text_list:
        # Return numpy array instead of torch tensor for compatibility
        import numpy as np
        return np.empty((0, model.get_sentence_embedding_dimension()))
    
    # Use optimal batch size for device if not specified
    actual_batch_size = batch_size or _OPTIMAL_BATCH_SIZE
    
    # Log processing information for large batches
    if len(text_list) > 100:
        logger.info(f"Encoding {len(text_list)} texts with batch size {actual_batch_size}")
    
    try:
        # Encode on CPU
        embeddings = model.encode(
            text_list,
            batch_size=actual_batch_size,
            convert_to_tensor=False,  # Return numpy array for CPU
            device=_DEVICE,
            show_progress_bar=len(text_list) > 100,
            normalize_embeddings=True,
        )
        return embeddings
    except RuntimeError as e:
        # Handle out of memory errors gracefully (CPU memory)
        if "out of memory" in str(e).lower() and batch_size is None:
            # Try again with smaller batch size
            reduced_batch = max(1, _OPTIMAL_BATCH_SIZE // 2)
            logger.warning(f"Out of memory error, retrying with reduced batch size {reduced_batch}")
            import gc
            gc.collect()  # Clear CPU memory
            return encode_texts(text_list, batch_size=reduced_batch)
        else:
            # If explicit batch size was provided or it's not an OOM error, raise it
            raise

def encode_text(text: str):
    """Encode a single text
    
    Args:
        text: Text string to encode
        
    Returns:
        Tensor embedding for the text
    """
    return encode_texts([text]).squeeze(0)

def batch_encode_texts(texts: List[str], max_batch_size: Optional[int] = None) -> np.ndarray:
    """Encode texts with automatic batching and memory management
    
    Useful for very large document collections that might not fit in memory at once.
    
    Args:
        texts: List of texts to encode
        max_batch_size: Maximum batch size (defaults to optimal size for device)
        
    Returns:
        NumPy array of embeddings
    """
    if not texts:
        return np.array([])
        
    model = get_embedding_model()
    embedding_dim = model.get_sentence_embedding_dimension()
    batch_size = max_batch_size or _OPTIMAL_BATCH_SIZE
    
    # Pre-allocate results array
    all_embeddings = np.zeros((len(texts), embedding_dim), dtype=np.float32)
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        if i % (batch_size * 5) == 0 and i > 0:
            logger.info(f"Processed {i}/{len(texts)} embeddings")
            
        # Generate embeddings for this batch (CPU-only)
        batch_embeddings = model.encode(
            batch,
            batch_size=batch_size,
            convert_to_tensor=False,
            device=_DEVICE,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        
        # Store in results array
        all_embeddings[i:i+len(batch)] = batch_embeddings
        
        # Clear memory
        del batch_embeddings
        import gc
        gc.collect()
    
    return all_embeddings
