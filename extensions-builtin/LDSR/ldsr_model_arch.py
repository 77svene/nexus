import os
import gc
import time
import weakref
import threading
import queue
import hashlib
import shutil
import tempfile
from collections import OrderedDict
from typing import Optional, Dict, Tuple, Any, Union, List
from pathlib import Path
import json

import numpy as np
import torch
import torchvision
from PIL import Image
from einops import rearrange, repeat
from omegaconf import OmegaConf
import safetensors.torch

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config, ismap
from modules import shared, sd_hijack, devices, paths


class FormatDetector:
    """Detects model format and provides appropriate loading strategies."""
    
    SUPPORTED_FORMATS = {
        '.ckpt': 'pytorch',
        '.pt': 'pytorch', 
        '.pth': 'pytorch',
        '.safetensors': 'safetensors',
        '.bin': 'pytorch',
        '.onnx': 'onnx',
    }
    
    DIFFUSERS_INDICATORS = [
        'model_index.json',
        'unet/diffusion_pytorch_model.bin',
        'unet/diffusion_pytorch_model.safetensors',
        'text_encoder/pytorch_model.bin',
        'vae/diffusion_pytorch_model.bin',
    ]
    
    @classmethod
    def detect_format(cls, model_path: str) -> str:
        """Detect model format from path or directory structure."""
        model_path = Path(model_path)
        
        # Check if it's a directory (diffusers format)
        if model_path.is_dir():
            for indicator in cls.DIFFUSERS_INDICATORS:
                if (model_path / indicator).exists():
                    return 'diffusers'
            # Check for any safetensors/ckpt files in directory
            for file in model_path.glob('**/*'):
                if file.suffix in cls.SUPPORTED_FORMATS:
                    return cls.SUPPORTED_FORMATS[file.suffix]
            return 'diffusers'
        
        # Check file extension
        suffix = model_path.suffix.lower()
        if suffix in cls.SUPPORTED_FORMATS:
            return cls.SUPPORTED_FORMATS[suffix]
        
        # Try to detect by file content
        try:
            with open(model_path, 'rb') as f:
                header = f.read(8)
                if header[:8] == b'PK\x03\x04':  # ZIP file (could be diffusers or pytorch)
                    return 'pytorch_zip'
                elif header[:8] == b'version':  # SafeTensors header
                    return 'safetensors'
        except:
            pass
        
        return 'unknown'
    
    @classmethod
    def get_loader_class(cls, format_type: str):
        """Get appropriate loader class for format."""
        from modules.sd_models import CheckpointLoader, SafetensorsLoader, DiffusersLoader, ONNXLoader
        
        loaders = {
            'pytorch': CheckpointLoader,
            'safetensors': SafetensorsLoader,
            'diffusers': DiffusersLoader,
            'onnx': ONNXLoader,
            'pytorch_zip': CheckpointLoader,
        }
        return loaders.get(format_type)


class ModelConverter:
    """Handles conversion between different model formats."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or os.path.join(paths.models_path, "LDSR", "converted_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.conversion_lock = threading.RLock()
        self.conversion_progress = {}
        
    def get_cache_key(self, model_path: str, target_format: str = 'safetensors') -> str:
        """Generate unique cache key for model conversion."""
        path_stat = os.stat(model_path)
        key_data = f"{model_path}:{path_stat.st_mtime}:{path_stat.st_size}:{target_format}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_cached_model(self, cache_key: str) -> Optional[str]:
        """Check if converted model exists in cache."""
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.safetensors")
        if os.path.exists(cache_path):
            # Verify cache is valid
            meta_path = os.path.join(self.cache_dir, f"{cache_key}.json")
            if os.path.exists(meta_path):
                return cache_path
        return None
    
    def save_to_cache(self, model_path: str, state_dict: Dict, cache_key: str) -> str:
        """Save converted model to cache."""
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.safetensors")
        meta_path = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        # Save model
        safetensors.torch.save_file(state_dict, cache_path)
        
        # Save metadata
        metadata = {
            'original_path': model_path,
            'conversion_time': time.time(),
            'format': 'safetensors',
            'keys_count': len(state_dict),
        }
        with open(meta_path, 'w') as f:
            json.dump(metadata, f)
        
        return cache_path
    
    def convert_pytorch_to_safetensors(self, model_path: str, progress_callback=None) -> Dict:
        """Convert PyTorch checkpoint to SafeTensors format."""
        if progress_callback:
            progress_callback(0.1, "Loading PyTorch checkpoint...")
        
        # Load PyTorch checkpoint
        state_dict = torch.load(model_path, map_location="cpu", mmap=True)
        
        if isinstance(state_dict, dict):
            # Handle nested state dicts
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']
        
        if progress_callback:
            progress_callback(0.5, "Converting to SafeTensors format...")
        
        # Clean up state dict
        cleaned_dict = {}
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                cleaned_dict[key] = value.cpu().contiguous()
        
        if progress_callback:
            progress_callback(1.0, "Conversion complete")
        
        return cleaned_dict
    
    def convert_diffusers_to_safetensors(self, model_dir: str, progress_callback=None) -> Dict:
        """Convert diffusers model to SafeTensors format."""
        if progress_callback:
            progress_callback(0.1, "Scanning diffusers model...")
        
        model_dir = Path(model_dir)
        state_dict = {}
        
        # Load each component
        components = ['unet', 'text_encoder', 'vae']
        total_components = len(components)
        
        for i, component in enumerate(components):
            component_dir = model_dir / component
            if not component_dir.exists():
                continue
            
            if progress_callback:
                progress_callback(0.1 + (i / total_components) * 0.7, 
                                f"Loading {component}...")
            
            # Try safetensors first, then pytorch
            safetensors_file = component_dir / "diffusion_pytorch_model.safetensors"
            pytorch_file = component_dir / "diffusion_pytorch_model.bin"
            
            if safetensors_file.exists():
                component_dict = safetensors.torch.load_file(str(safetensors_file), device="cpu")
            elif pytorch_file.exists():
                component_dict = torch.load(pytorch_file, map_location="cpu")
            else:
                continue
            
            # Add component prefix to keys
            for key, value in component_dict.items():
                state_dict[f"{component}.{key}"] = value
        
        if progress_callback:
            progress_callback(1.0, "Diffusers conversion complete")
        
        return state_dict
    
    def convert_onnx_to_safetensors(self, onnx_path: str, progress_callback=None) -> Dict:
        """Convert ONNX model to SafeTensors format (simplified)."""
        if progress_callback:
            progress_callback(0.1, "Loading ONNX model...")
        
        try:
            import onnx
            from onnx import numpy_helper
            
            onnx_model = onnx.load(onnx_path)
            state_dict = {}
            
            # Extract initializers as tensors
            for i, initializer in enumerate(onnx_model.graph.initializer):
                if progress_callback:
                    progress_callback(0.1 + (i / len(onnx_model.graph.initializer)) * 0.8,
                                    f"Converting tensor {i+1}/{len(onnx_model.graph.initializer)}...")
                
                np_array = numpy_helper.to_array(initializer)
                tensor = torch.from_numpy(np_array)
                state_dict[initializer.name] = tensor
            
            if progress_callback:
                progress_callback(1.0, "ONNX conversion complete")
            
            return state_dict
            
        except ImportError:
            raise ImportError("ONNX package required for ONNX conversion. Install with: pip install onnx")
    
    def convert_model(self, model_path: str, target_format: str = 'safetensors',
                     progress_callback=None) -> Tuple[str, Dict]:
        """Convert model to target format with caching."""
        cache_key = self.get_cache_key(model_path, target_format)
        
        # Check cache first
        cached_path = self.get_cached_model(cache_key)
        if cached_path:
            if progress_callback:
                progress_callback(1.0, "Using cached conversion")
            state_dict = safetensors.torch.load_file(cached_path, device="cpu")
            return cached_path, state_dict
        
        with self.conversion_lock:
            # Double-check after acquiring lock
            cached_path = self.get_cached_model(cache_key)
            if cached_path:
                state_dict = safetensors.torch.load_file(cached_path, device="cpu")
                return cached_path, state_dict
            
            # Perform conversion
            format_type = FormatDetector.detect_format(model_path)
            
            if format_type == 'pytorch' or format_type == 'pytorch_zip':
                state_dict = self.convert_pytorch_to_safetensors(model_path, progress_callback)
            elif format_type == 'diffusers':
                state_dict = self.convert_diffusers_to_safetensors(model_path, progress_callback)
            elif format_type == 'onnx':
                state_dict = self.convert_onnx_to_safetensors(model_path, progress_callback)
            elif format_type == 'safetensors':
                # Already in target format, just load
                state_dict = safetensors.torch.load_file(model_path, device="cpu")
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            
            # Save to cache
            cached_path = self.save_to_cache(model_path, state_dict, cache_key)
            
            return cached_path, state_dict


class ProgressiveLoader:
    """Manages progressive model loading with quality ramping."""
    
    def __init__(self):
        self.loading_threads = {}
        self.loading_progress = {}
        self.partial_models = {}
        self.quality_levels = {}
        self.load_lock = threading.RLock()
        self.background_queue = queue.Queue()
        self.converter = ModelConverter()
        
    def start_progressive_load(self, model_path: str, config_path: str, 
                              half_attention: bool, priority_layers: list = None):
        """Start progressive loading of a model."""
        load_id = f"{model_path}_{half_attention}"
        
        with self.load_lock:
            if load_id in self.loading_threads:
                return load_id  # Already loading
            
            self.loading_progress[load_id] = 0.0
            self.quality_levels[load_id] = 0  # Start with lowest quality
            
            # Start loading thread
            thread = threading.Thread(
                target=self._progressive_load_worker,
                args=(load_id, model_path, config_path, half_attention, priority_layers),
                daemon=True
            )
            self.loading_threads[load_id] = thread
            thread.start()
            
            return load_id
    
    def _progressive_load_worker(self, load_id: str, model_path: str, 
                                config_path: str, half_attention: bool,
                                priority_layers: list = None):
        """Worker thread for progressive loading."""
        try:
            # Load config first
            config = OmegaConf.load(config_path)
            model = instantiate_from_config(config.model)
            
            def progress_callback(progress, message):
                with self.load_lock:
                    self.loading_progress[load_id] = progress * 0.2  # First 20% for conversion
                print(f"Progressive loading: {message} ({progress*100:.1f}%)")
            
            # Convert model to SafeTensors format if needed
            try:
                converted_path, state_dict = self.converter.convert_model(
                    model_path, 
                    target_format='safetensors',
                    progress_callback=progress_callback
                )
            except Exception as e:
                print(f"Conversion failed, trying direct load: {e}")
                # Fallback to direct loading
                _, extension = os.path.splitext(model_path)
                if extension.lower() == ".safetensors":
                    state_dict = safetensors.torch.load_file(model_path, device="cpu")
                else:
                    state_dict = torch.load(model_path, map_location="cpu", mmap=True)
                    if "state_dict" in state_dict:
                        state_dict = state_dict["state_dict"]
            
            with self.load_lock:
                self.loading_progress[load_id] = 0.2
            
            # Get model keys sorted by priority
            all_keys = list(state_dict.keys())
            priority_keys = self._get_priority_keys(all_keys, priority_layers)
            
            # Load in phases
            total_keys = len(all_keys)
            
            # Phase 1: Load priority layers (first 30% of weights)
            print(f"Progressive loading: Phase 1 - Loading priority layers for {load_id}")
            partial_state = {}
            
            for i, key in enumerate(priority_keys):
                if key in state_dict:
                    partial_state[key] = state_dict[key]
                    progress = 0.2 + (i + 1) / len(priority_keys) * 0.3
                    with self.load_lock:
                        self.loading_progress[load_id] = progress
                        self.quality_levels[load_id] = 1  # Basic quality
                
            # Load partial state into model
            model.load_state_dict(partial_state, strict=False)
            
            # Store partial model
            with self.load_lock:
                self.partial_models[load_id] = model
                self.quality_levels[load_id] = 1
            
            # Phase 2: Load remaining layers in background
            remaining_keys = [k for k in all_keys if k not in priority_keys]
            batch_size = max(1, len(remaining_keys) // 5)  # Load in 5 batches
            
            for batch_idx in range(0, len(remaining_keys), batch_size):
                batch_keys = remaining_keys[batch_idx:batch_idx + batch_size]
                
                # Load batch
                batch_state = {}
                for key in batch_keys:
                    if key in state_dict:
                        batch_state[key] = state_dict[key]
                
                # Update model
                model.load_state_dict(batch_state, strict=False)
                
                # Update progress
                progress = 0.5 + (batch_idx + batch_size) / len(remaining_keys) * 0.5
                with self.load_lock:
                    self.loading_progress[load_id] = min(progress, 1.0)
                    
                    # Update quality level based on progress
                    if progress > 0.8:
                        self.quality_levels[load_id] = 3  # High quality
                    elif progress > 0.5:
                        self.quality_levels[load_id] = 2  # Medium quality
                
                # Small delay to not overwhelm system
                time.sleep(0.01)
            
            # Finalize
            with self.load_lock:
                self.loading_progress[load_id] = 1.0
                self.quality_levels[load_id] = 3  # Full quality
                self.partial_models[load_id] = model
            
            print(f"Progressive loading complete for {load_id}")
            
        except Exception as e:
            print(f"Error in progressive loading: {e}")
            import traceback
            traceback.print_exc()
            with self.load_lock:
                self.loading_progress[load_id] = -1  # Error state
    
    def _get_priority_keys(self, all_keys: list, priority_layers: list = None) -> list:
        """Get priority keys for early loading."""
        if priority_layers is None:
            # Default priority: early layers and attention blocks
            priority_keywords = [
                'input_blocks', 'encoder', 'conv_in', 'time_embed',
                'middle_block', 'attention', 'attn', 'transformer'
            ]
            
            priority_keys = []
            for key in all_keys:
                if any(keyword in key.lower() for keyword in priority_keywords):
                    priority_keys.append(key)
            
            # If no priority keys found, use first 30% of keys
            if not priority_keys:
                priority_keys = all_keys[:int(len(all_keys) * 0.3)]
            
            return priority_keys
        else:
            # Use provided priority layers
            return [k for k in all_keys if any(pl in k for pl in priority_layers)]
    
    def get_model(self, load_id: str) -> Tuple[Optional[Any], float, int]:
        """Get current model state, progress, and quality level."""
        with self.load_lock:
            model = self.partial_models.get(load_id)
            progress = self.loading_progress.get(load_id, 0.0)
            quality = self.quality_levels.get(load_id, 0)
            return model, progress, quality
    
    def is_loading(self, load_id: str) -> bool:
        """Check if model is still loading."""
        with self.load_lock:
            progress = self.loading_progress.get(load_id, 0.0)
            return 0 < progress < 1.0
    
    def wait_for_quality(self, load_id: str, min_quality: int = 1, timeout: float = 30.0) -> bool:
        """Wait until model reaches minimum quality level."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.load_lock:
                quality = self.quality_levels.get(load_id, 0)
                if quality >= min_quality:
                    return True
                if self.loading_progress.get(load_id, 0.0) < 0:
                    return False  # Error state
            time.sleep(0.1)
        return False


class ModelCache:
    """Unified LRU cache for upscaler models with smart memory management."""
    
    def __init__(self, max_size_gb: float = 4.0):
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.current_size = 0
        self.cache = OrderedDict()  # model_path -> (model, size, last_access)
        self.lock = threading.RLock()
        self.weak_refs = weakref.WeakValueDictionary()
        
    def get(self, model_path: str) -> Optional[Any]:
        """Get model from cache."""
        with self.lock:
            if model_path in self.cache:
                model, size, _ = self.cache[model_path]
                # Update access time
                self.cache[model_path] = (model, size, time.time())
                # Move to end (most recently used)
                self.cache.move_to_end(model_path)
                return model
            return None
    
    def put(self, model_path: str, model: Any, size_mb: Optional[float] = None):
        """Add model to cache."""
        with self.lock:
            # Estimate size if not provided
            if size_mb is None:
                size_mb = self._estimate_model_size(model)
            
            size_bytes = int(size_mb * 1024 * 1024)
            
            # Remove if already exists
            if model_path in self.cache:
                self.current_size -= self.cache[model_path][1]
            
            # Evict least recently used models if needed
            while self.current_size + size_bytes > self.max_size_bytes and self.cache:
                evicted_path, (evicted_model, evicted_size, _) = self.cache.popitem(last=False)
                self.current_size -= evicted_size
                print(f"ModelCache: Evicted {evicted_path} ({evicted_size/1024/1024:.1f}MB)")
                del evicted_model
                gc.collect()
            
            # Add new model
            self.cache[model_path] = (model, size_bytes, time.time())
            self.current_size += size_bytes
            
            # Store weak reference for external access
            self.weak_refs[model_path] = model
            
            print(f"ModelCache: Added {model_path} ({size_bytes/1024/1024:.1f}MB), "
                  f"Total: {self.current_size/1024/1024:.1f}MB/{self.max_size_bytes/1024/1024:.1f}MB")
    
    def _estimate_model_size(self, model: Any) -> float:
        """Estimate model size in MB."""
        total_params = 0
        if hasattr(model, 'parameters'):
            for param in model.parameters():
                total_params += param.numel()
        elif isinstance(model, dict):
            for tensor in model.values():
                if isinstance(tensor, torch.Tensor):
                    total_params += tensor.numel()
        
        # Estimate: 4 bytes per parameter (float32)
        return total_params * 4 / (1024 * 1024)
    
    def clear(self):
        """Clear all cached models."""
        with self.lock:
            self.cache.clear()
            self.current_size = 0
            gc.collect()
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with self.lock:
            return {
                'total_models': len(self.cache),
                'total_size_mb': self.current_size / (1024 * 1024),
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'utilization': self.current_size / self.max_size_bytes * 100,
            }


class ModelFormatBridge:
    """Main bridge for seamless model format conversion and loading."""
    
    def __init__(self):
        self.progressive_loader = ProgressiveLoader()
        self.model_cache = ModelCache(max_size_gb=shared.opts.data.get('ldsr_cache_size_gb', 4.0))
        self.converter = ModelConverter()
        self.format_detector = FormatDetector()
        
    def load_model(self, model_path: str, config_path: str, 
                  half_attention: bool = False, 
                  priority_layers: Optional[List[str]] = None,
                  use_cache: bool = True) -> Tuple[Any, str]:
        """
        Load model with automatic format detection and conversion.
        
        Returns:
            Tuple of (model, load_id) where load_id can be used to check loading progress
        """
        # Check cache first
        if use_cache:
            cached_model = self.model_cache.get(model_path)
            if cached_model is not None:
                print(f"Model loaded from cache: {model_path}")
                return cached_model, "cached"
        
        # Detect format
        format_type = self.format_detector.detect_format(model_path)
        print(f"Detected format: {format_type} for {model_path}")
        
        # Start progressive loading
        load_id = self.progressive_loader.start_progressive_load(
            model_path, config_path, half_attention, priority_layers
        )
        
        # Wait for at least basic quality
        if not self.progressive_loader.wait_for_quality(load_id, min_quality=1, timeout=60.0):
            raise RuntimeError(f"Failed to load model: {model_path}")
        
        # Get model
        model, progress, quality = self.progressive_loader.get_model(load_id)
        
        if model is None:
            raise RuntimeError(f"Model loading failed: {model_path}")
        
        # Cache the model if fully loaded
        if quality >= 3 and use_cache:
            self.model_cache.put(model_path, model)
        
        return model, load_id
    
    def get_loading_progress(self, load_id: str) -> Tuple[float, int]:
        """Get loading progress and quality level for a model."""
        if load_id == "cached":
            return 1.0, 3  # Cached models are fully loaded
        
        _, progress, quality = self.progressive_loader.get_model(load_id)
        return progress, quality
    
    def preload_model(self, model_path: str, config_path: str, 
                     half_attention: bool = False):
        """Preload model in background without waiting."""
        load_id = self.progressive_loader.start_progressive_load(
            model_path, config_path, half_attention
        )
        return load_id
    
    def clear_cache(self):
        """Clear all caches."""
        self.model_cache.clear()
        self.converter = ModelConverter()  # Reset converter cache
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return self.model_cache.get_stats()


# Global instance
format_bridge = ModelFormatBridge()


def load_ldsr_model(model_path: str, config_path: str, 
                   half_attention: bool = False,
                   progress_callback=None) -> Any:
    """
    Main function to load LDSR model with automatic format conversion.
    
    This replaces the original load_ldsr_model function with format bridge support.
    """
    try:
        model, load_id = format_bridge.load_model(
            model_path, config_path, half_attention
        )
        
        # If using progressive loading, report progress
        if load_id != "cached" and progress_callback:
            def check_progress():
                while True:
                    progress, quality = format_bridge.get_loading_progress(load_id)
                    if progress >= 1.0 or progress < 0:
                        break
                    progress_callback(progress, f"Loading... Quality: {quality}/3")
                    time.sleep(0.1)
            
            # Start progress reporting in background
            progress_thread = threading.Thread(target=check_progress, daemon=True)
            progress_thread.start()
        
        return model
        
    except Exception as e:
        print(f"Error loading LDSR model with format bridge: {e}")
        # Fallback to original loading method
        from ldsr_model_arch import LDSR
        return LDSR.load_model_from_config(model_path, config_path)


# Original classes and functions preserved below
class LDSR:
    """Original LDSR class - preserved for compatibility."""
    
    @staticmethod
    def load_model_from_config(model_path, config_path):
        """Original model loading method - kept as fallback."""
        print(f"Loading LDSR model from {model_path}")
        config = OmegaConf.load(config_path)
        model = instantiate_from_config(config.model)
        
        _, extension = os.path.splitext(model_path)
        if extension.lower() == ".safetensors":
            state_dict = safetensors.torch.load_file(model_path, device="cpu")
        else:
            state_dict = torch.load(model_path, map_location="cpu", mmap=True)
        
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        model = model.to(devices.device_ldsr)
        
        return model


# Additional utility functions for format bridge
def get_supported_formats() -> List[str]:
    """Get list of supported model formats."""
    return list(FormatDetector.SUPPORTED_FORMATS.keys()) + ['diffusers']


def convert_model_format(input_path: str, output_path: str, 
                        target_format: str = 'safetensors',
                        progress_callback=None) -> bool:
    """
    Convert model to specified format.
    
    Args:
        input_path: Path to input model
        output_path: Path to save converted model
        target_format: Target format (safetensors, pytorch, etc.)
        progress_callback: Optional callback for progress updates
    
    Returns:
        True if conversion successful
    """
    try:
        converter = ModelConverter()
        
        def internal_progress(progress, message):
            if progress_callback:
                progress_callback(progress, message)
        
        _, state_dict = converter.convert_model(
            input_path, target_format, internal_progress
        )
        
        # Save to output path
        if target_format == 'safetensors':
            safetensors.torch.save_file(state_dict, output_path)
        else:
            torch.save(state_dict, output_path)
        
        return True
        
    except Exception as e:
        print(f"Model conversion failed: {e}")
        return False


# Integration with shared options
def update_cache_settings():
    """Update cache settings from shared options."""
    if hasattr(shared.opts, 'data'):
        cache_size = shared.opts.data.get('ldsr_cache_size_gb', 4.0)
        format_bridge.model_cache.max_size_bytes = int(cache_size * 1024 * 1024 * 1024)


# Register settings if available
try:
    from modules import shared
    if hasattr(shared, 'opts'):
        shared.opts.add_option("ldsr_cache_size_gb", shared.OptionInfo(
            4.0, "LDSR model cache size (GB)", section=("ldsr", "LDSR")
        ))
except:
    pass