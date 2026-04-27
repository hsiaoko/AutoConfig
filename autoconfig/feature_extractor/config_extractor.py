"""
Configuration Feature Extractor
Extracts features from system configuration (Conf).
"""

import numpy as np
from typing import Dict, Any, List

# I/O / processing batch sizes used across configs and merged features.
CONFIG_BATCH_SIZE_CHOICES: tuple[int, ...] = (64, 128, 256, 512)
DEFAULT_CONFIG_BATCH_SIZE: int = 256


class ConfigFeatureExtractor:
    """
    Extracts features from system configuration.
    
    Configuration features include:
    - Memory settings
    - Parallelism settings
    - Cache settings
    - Other system parameters
    """
    
    def __init__(self, config_schema: Dict[str, Any] = None):
        """
        Initialize with optional config schema.
        
        Args:
            config_schema: Dictionary defining config parameter types and ranges
        """
        self.config_schema = config_schema or {}
        self.feature_names = [
            'conf_memory_limit',
            'conf_num_threads',
            'conf_cache_size',
            'conf_batch_size',
            'conf_io_buffer_size',
            'conf_num_workers',
            'conf_timeout',
            'conf_enable_index',
            'conf_index_type',
            'conf_compression_enabled',
        ]
    
    def extract(self, config: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from a configuration dictionary.
        
        Args:
            config: Dictionary containing configuration parameters
            
        Returns:
            numpy array of configuration features
        """
        features = []
        
        # Memory settings (in MB)
        memory_limit = float(config.get('memory_limit', 8192))
        features.append(memory_limit)
        
        # Parallelism settings
        num_threads = float(config.get('num_threads', 4))
        features.append(num_threads)
        
        # Cache settings (in MB)
        cache_size = float(config.get('cache_size', 1024))
        features.append(cache_size)
        
        # Batch size
        batch_size = float(
            config.get("batch_size", DEFAULT_CONFIG_BATCH_SIZE)
        )
        features.append(batch_size)
        
        # I/O buffer size (in KB)
        io_buffer_size = float(config.get('io_buffer_size', 64))
        features.append(io_buffer_size)
        
        # Number of workers
        num_workers = float(config.get('num_workers', 2))
        features.append(num_workers)
        
        # Timeout (in seconds)
        timeout = float(config.get('timeout', 300))
        features.append(timeout)
        
        # Index enabled (binary)
        enable_index = 1.0 if config.get('enable_index', True) else 0.0
        features.append(enable_index)
        
        # Index type (encoded)
        index_type_map = {'none': 0, 'btree': 1, 'hash': 2, 'bitmap': 3}
        index_type = config.get('index_type', 'btree')
        features.append(float(index_type_map.get(index_type, 1)))
        
        # Compression enabled (binary)
        compression_enabled = 1.0 if config.get('compression_enabled', False) else 0.0
        features.append(compression_enabled)
        
        return np.array(features, dtype=np.float64)
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.feature_names.copy()
    
    def set_schema(self, schema: Dict[str, Any]):
        """Set configuration schema for validation."""
        self.config_schema = schema
