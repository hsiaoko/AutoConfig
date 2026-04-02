"""
Model Registry

Manages trained model versions and metadata.
"""

import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List


class ModelRegistry:
    """
    Registry for trained models.
    
    Tracks:
    - Model versions
    - Training metrics
    - Model paths
    - Metadata
    """
    
    def __init__(self, registry_path: str = 'data/models/registry.yaml'):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load registry from file or create empty."""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                return yaml.safe_load(f) or {'models': []}
        return {'models': []}
    
    def _save_registry(self):
        """Save registry to file."""
        with open(self.registry_path, 'w') as f:
            yaml.dump(self.registry, f, default_flow_style=None)
    
    def register_model(
        self,
        model_name: str,
        model_path: str,
        metrics: Dict[str, Any],
        description: str = ''
    ) -> str:
        """
        Register a trained model.
        
        Args:
            model_name: Name of the model (e.g., 'time_model', 'cost_model')
            model_path: Path to model file
            metrics: Training metrics
            description: Optional description
            
        Returns:
            Model version ID
        """
        version = f"v{len([m for m in self.registry['models'] if m['name'] == model_name]) + 1}"
        
        entry = {
            'name': model_name,
            'version': version,
            'path': str(model_path),
            'metrics': metrics,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'status': 'active',
        }
        
        self.registry['models'].append(entry)
        self._save_registry()
        
        return version
    
    def get_model(self, model_name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get model info by name and optionally version.
        
        Args:
            model_name: Name of the model
            version: Specific version or None for latest
            
        Returns:
            Model entry dictionary or None
        """
        models = [m for m in self.registry['models'] 
                  if m['name'] == model_name and m['status'] == 'active']
        
        if not models:
            return None
        
        if version is None:
            # Return latest version
            return models[-1]
        
        for m in models:
            if m['version'] == version:
                return m
        
        return None
    
    def list_models(self, status: str = 'active') -> List[Dict[str, Any]]:
        """List all registered models."""
        return [m for m in self.registry['models'] if m.get('status', 'active') == status]
    
    def deactivate_model(self, model_name: str, version: str):
        """Deactivate a model version."""
        for m in self.registry['models']:
            if m['name'] == model_name and m['version'] == version:
                m['status'] = 'deprecated'
                break
        self._save_registry()
    
    def get_latest_models(self) -> Dict[str, Dict[str, Any]]:
        """Get latest version of each model type."""
        latest = {}
        for m in self.registry['models']:
            if m['status'] == 'active':
                name = m['name']
                if name not in latest or m['version'] > latest[name]['version']:
                    latest[name] = m
        return latest
