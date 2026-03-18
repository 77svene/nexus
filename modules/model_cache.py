import gc
import sys
import weakref
import hashlib
import threading
from collections import OrderedDict
from typing import Dict, Optional, Any, Tuple, Union
import torch
import torch.nn as nn
from modules import shared, devices, errors

class ModelCache:
    """Unified model cache with LRU eviction and smart VRAM management."""
    
    def __init__(self, max_models: int = 3, max_vram_gb: float = 4.0, enable_mmap: bool = True):
        self.max_models = max_models
        self.max_vram_bytes = int(max_vram_gb * 1024**3)
        self.enable_mmap = enable_mmap
        
        # Main cache storage: OrderedDict for LRU behavior
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        
        # Weak references to models for automatic cleanup
        self._model_refs: Dict[str, weakref.ref] = {}
        
        # Memory tracking
        self._memory_usage: Dict[str, int] = {}  # model_key -> bytes
        self._total_vram_used: int = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_saved': 0
        }
        
        # Register for memory pressure callbacks
        self._register_memory_callbacks()
    
    def _register_memory_callbacks(self):
        """Register callbacks for memory pressure events."""
        try:
            # Hook into PyTorch's memory management
            if hasattr(torch.cuda, 'memory'):
                torch.cuda.memory._record_memory_history(enabled='all')
        except Exception:
            pass
    
    def _calculate_model_hash(self, model_path: str, model_type: str) -> str:
        """Create a unique hash for model identification."""
        key_str = f"{model_path}:{model_type}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _estimate_model_size(self, model: nn.Module) -> int:
        """Estimate model size in bytes."""
        total_params = 0
        for param in model.parameters():
            total_params += param.nelement() * param.element_size()
        for buffer in model.buffers():
            total_params += buffer.nelement() * buffer.element_size()
        return total_params
    
    def _get_available_vram(self) -> int:
        """Get available VRAM in bytes."""
        try:
            if torch.cuda.is_available():
                return torch.cuda.mem_get_info()[0]
        except Exception:
            pass
        return self.max_vram_bytes
    
    def _check_memory_pressure(self) -> bool:
        """Check if we're under memory pressure."""
        available = self._get_available_vram()
        return available < (self.max_vram_bytes * 0.2)  # Less than 20% available
    
    def _evict_lru_model(self):
        """Evict least recently used model."""
        with self._lock:
            if not self._cache:
                return
            
            # Find LRU model
            lru_key = next(iter(self._cache))
            self._evict_model(lru_key)
    
    def _evict_model(self, model_key: str):
        """Evict a specific model from cache."""
        with self._lock:
            if model_key in self._cache:
                model_data = self._cache[model_key]
                
                # Clear GPU memory
                if 'model' in model_data:
                    model = model_data['model']
                    if hasattr(model, 'to'):
                        try:
                            model.to('cpu')
                        except Exception:
                            pass
                
                # Update memory tracking
                if model_key in self._memory_usage:
                    self._total_vram_used -= self._memory_usage[model_key]
                    del self._memory_usage[model_key]
                
                # Remove from cache
                del self._cache[model_key]
                self._stats['evictions'] += 1
                
                # Force garbage collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    def _cleanup_weak_refs(self):
        """Clean up dead weak references."""
        with self._lock:
            dead_keys = []
            for key, ref in self._model_refs.items():
                if ref() is None:
                    dead_keys.append(key)
            
            for key in dead_keys:
                if key in self._cache:
                    self._evict_model(key)
                del self._model_refs[key]
    
    def get(self, model_path: str, model_type: str, device: Optional[torch.device] = None) -> Optional[nn.Module]:
        """Get a model from cache or load it."""
        model_key = self._calculate_model_hash(model_path, model_type)
        
        with self._lock:
            # Check if model is in cache
            if model_key in self._cache:
                model_data = self._cache[model_key]
                
                # Move to end (most recently used)
                self._cache.move_to_end(model_key)
                
                # Check if weak reference is still valid
                if model_key in self._model_refs:
                    model = self._model_refs[model_key]()
                    if model is not None:
                        self._stats['hits'] += 1
                        
                        # Move to requested device if specified
                        if device is not None and hasattr(model, 'to'):
                            try:
                                model.to(device)
                            except Exception:
                                pass
                        
                        return model
                
                # Reference is dead, remove from cache
                del self._cache[model_key]
            
            self._stats['misses'] += 1
            return None
    
    def put(self, model_path: str, model_type: str, model: nn.Module, 
            device: Optional[torch.device] = None, metadata: Optional[Dict] = None):
        """Add a model to cache."""
        model_key = self._calculate_model_hash(model_path, model_type)
        
        with self._lock:
            # Check memory pressure before adding
            model_size = self._estimate_model_size(model)
            
            # Evict models if necessary
            while (len(self._cache) >= self.max_models or 
                   self._total_vram_used + model_size > self.max_vram_bytes):
                if not self._cache:
                    break
                self._evict_lru_model()
            
            # Move model to device if specified
            if device is not None and hasattr(model, 'to'):
                try:
                    model.to(device)
                except Exception:
                    pass
            
            # Store model with weak reference
            model_data = {
                'model': model,
                'path': model_path,
                'type': model_type,
                'size': model_size,
                'metadata': metadata or {}
            }
            
            self._cache[model_key] = model_data
            self._model_refs[model_key] = weakref.ref(model)
            self._memory_usage[model_key] = model_size
            self._total_vram_used += model_size
            
            # Move to end (most recently used)
            self._cache.move_to_end(model_key)
    
    def load_with_mmap(self, model_path: str, model_class: type, device: torch.device) -> nn.Module:
        """Load a model using memory-mapped files to reduce peak VRAM."""
        if not self.enable_mmap:
            # Standard loading
            model = model_class()
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            return model
        
        try:
            # Memory-mapped loading
            state_dict = torch.load(model_path, map_location='cpu', mmap=True)
            model = model_class()
            
            # Load state dict with memory mapping
            for name, param in model.named_parameters():
                if name in state_dict:
                    param.data = state_dict[name]
            
            for name, buffer in model.named_buffers():
                if name in state_dict:
                    buffer.data = state_dict[name]
            
            # Move to device
            if device.type == 'cuda':
                model.to(device)
            
            self._stats['memory_saved'] += self._estimate_model_size(model)
            return model
            
        except Exception as e:
            print(f"Memory-mapped loading failed, falling back to standard loading: {e}")
            model = model_class()
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            return model
    
    def clear(self):
        """Clear the entire cache."""
        with self._lock:
            keys = list(self._cache.keys())
            for key in keys:
                self._evict_model(key)
            
            self._cache.clear()
            self._model_refs.clear()
            self._memory_usage.clear()
            self._total_vram_used = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                **self._stats,
                'cached_models': len(self._cache),
                'total_vram_used_gb': self._total_vram_used / (1024**3),
                'max_vram_gb': self.max_vram_bytes / (1024**3)
            }
    
    def optimize_memory(self):
        """Optimize memory usage by moving unused models to CPU."""
        with self._lock:
            for model_key in list(self._cache.keys()):
                if model_key in self._model_refs:
                    model = self._model_refs[model_key]()
                    if model is not None:
                        # Check if model hasn't been used recently
                        # Move to CPU to free VRAM
                        try:
                            model.to('cpu')
                            if model_key in self._memory_usage:
                                self._total_vram_used -= self._memory_usage[model_key]
                                self._memory_usage[model_key] = 0
                        except Exception:
                            pass
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

# Global model cache instance
model_cache = ModelCache(
    max_models=getattr(shared.opts, 'model_cache_max_models', 3),
    max_vram_gb=getattr(shared.opts, 'model_cache_max_vram_gb', 4.0),
    enable_mmap=getattr(shared.opts, 'model_cache_enable_mmap', True)
)

def setup_model_cache():
    """Setup model cache with settings from shared.opts."""
    global model_cache
    
    if hasattr(shared, 'opts'):
        model_cache.max_models = getattr(shared.opts, 'model_cache_max_models', 3)
        model_cache.max_vram_bytes = int(getattr(shared.opts, 'model_cache_max_vram_gb', 4.0) * 1024**3)
        model_cache.enable_mmap = getattr(shared.opts, 'model_cache_enable_mmap', True)

def register_settings():
    """Register model cache settings in the UI."""
    try:
        from modules import shared
        from modules.shared import OptionInfo
        
        if not hasattr(shared, 'opts'):
            return
        
        section = ('model_cache', 'Model Cache')
        
        shared.opts.add_option("model_cache_max_models", shared.OptionInfo(
            3, "Maximum models to keep in VRAM", section=section))
        
        shared.opts.add_option("model_cache_max_vram_gb", shared.OptionInfo(
            4.0, "Maximum VRAM for cached models (GB)", section=section))
        
        shared.opts.add_option("model_cache_enable_mmap", shared.OptionInfo(
            True, "Enable memory-mapped model loading", section=section))
        
        shared.opts.add_option("model_cache_auto_optimize", shared.OptionInfo(
            True, "Auto-optimize memory on pressure", section=section))
        
    except Exception as e:
        print(f"Failed to register model cache settings: {e}")

# Initialize settings when module loads
register_settings()

# Hook into shared.opts.onchange if available
try:
    if hasattr(shared, 'opts') and hasattr(shared.opts, 'onchange'):
        shared.opts.onchange("model_cache_max_models", setup_model_cache)
        shared.opts.onchange("model_cache_max_vram_gb", setup_model_cache)
        shared.opts.onchange("model_cache_enable_mmap", setup_model_cache)
except Exception:
    pass

# Integration with existing modules
def patch_ldsr_model():
    """Patch LDSR model to use unified cache."""
    try:
        from extensions_built_in.LDSR import ldsr_model
        
        original_ldsr_model = ldsr_model.LDSR
        
        class CachedLDSRModel(original_ldsr_model):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._model_key = None
            
            def load_model_from_config(self, model_path, *args, **kwargs):
                # Try to get from cache first
                cached_model = model_cache.get(model_path, 'ldsr')
                if cached_model is not None:
                    return cached_model
                
                # Load with memory mapping if enabled
                if model_cache.enable_mmap:
                    model = model_cache.load_with_mmap(
                        model_path, 
                        lambda: super().load_model_from_config(model_path, *args, **kwargs),
                        devices.device
                    )
                else:
                    model = super().load_model_from_config(model_path, *args, **kwargs)
                
                # Cache the model
                model_cache.put(model_path, 'ldsr', model, devices.device)
                return model
        
        ldsr_model.LDSR = CachedLDSRModel
        
    except ImportError:
        pass
    except Exception as e:
        print(f"Failed to patch LDSR model: {e}")

# Apply patches when module loads
patch_ldsr_model()

# Memory pressure handler
def handle_memory_pressure():
    """Handle memory pressure by optimizing cache."""
    if model_cache._check_memory_pressure():
        if getattr(shared.opts, 'model_cache_auto_optimize', True):
            model_cache.optimize_memory()

# Register memory pressure callback
try:
    if torch.cuda.is_available():
        torch.cuda.memory._set_allocator_settings('expandable_segments:True')
except Exception:
    pass

# Export main components
__all__ = ['ModelCache', 'model_cache', 'setup_model_cache', 'handle_memory_pressure']