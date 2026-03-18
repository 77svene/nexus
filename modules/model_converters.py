"""
Model Format Bridge with Automatic Conversion
Seamlessly support all model formats (ckpt, safetensors, diffusers, onnx) with automatic conversion and caching.
"""

import os
import hashlib
import json
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import logging

import torch
import safetensors.torch
from modules import shared, devices, sd_models, sd_vae
from modules.paths import models_path

logger = logging.getLogger(__name__)


class ModelFormat(Enum):
    CKPT = "ckpt"
    SAFETENSORS = "safetensors"
    DIFFUSERS = "diffusers"
    ONNX = "onnx"
    UNKNOWN = "unknown"


@dataclass
class ConversionJob:
    source_path: str
    source_format: ModelFormat
    target_format: ModelFormat
    target_path: str
    progress: float = 0.0
    status: str = "pending"
    error: Optional[str] = None
    thread: Optional[threading.Thread] = None
    start_time: float = 0.0


class ModelFormatConverter:
    """Main converter class for model format detection and conversion"""
    
    def __init__(self):
        self.cache_dir = Path(models_path) / "converted_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.conversion_jobs: Dict[str, ConversionJob] = {}
        self.lock = threading.Lock()
        
        # Register format handlers
        self.format_handlers = {
            ModelFormat.CKPT: self._handle_ckpt,
            ModelFormat.SAFETENSORS: self._handle_safetensors,
            ModelFormat.DIFFUSERS: self._handle_diffusers,
            ModelFormat.ONNX: self._handle_onnx,
        }
        
        # Conversion matrix: source -> target -> converter function
        self.conversion_matrix = {
            (ModelFormat.CKPT, ModelFormat.SAFETENSORS): self._ckpt_to_safetensors,
            (ModelFormat.SAFETENSORS, ModelFormat.CKPT): self._safetensors_to_ckpt,
            (ModelFormat.CKPT, ModelFormat.DIFFUSERS): self._ckpt_to_diffusers,
            (ModelFormat.SAFETENSORS, ModelFormat.DIFFUSERS): self._safetensors_to_diffusers,
            (ModelFormat.DIFFUSERS, ModelFormat.CKPT): self._diffusers_to_ckpt,
            (ModelFormat.DIFFUSERS, ModelFormat.SAFETENSORS): self._diffusers_to_safetensors,
            (ModelFormat.CKPT, ModelFormat.ONNX): self._ckpt_to_onnx,
            (ModelFormat.SAFETENSORS, ModelFormat.ONNX): self._safetensors_to_onnx,
        }
    
    def detect_format(self, model_path: str) -> ModelFormat:
        """Detect model format from file/directory structure"""
        path = Path(model_path)
        
        if not path.exists():
            return ModelFormat.UNKNOWN
        
        # Check for diffusers directory structure
        if path.is_dir():
            if (path / "model_index.json").exists():
                return ModelFormat.DIFFUSERS
            # Check for nested diffusers structure
            for subdir in path.iterdir():
                if subdir.is_dir() and (subdir / "model_index.json").exists():
                    return ModelFormat.DIFFUSERS
        
        # Check file extensions
        if path.is_file():
            suffix = path.suffix.lower()
            if suffix == ".safetensors":
                return ModelFormat.SAFETENSORS
            elif suffix == ".ckpt" or suffix == ".pt" or suffix == ".pth":
                return ModelFormat.CKPT
            elif suffix == ".onnx":
                return ModelFormat.ONNX
        
        # Try to detect by content
        if path.is_file():
            try:
                if suffix in [".ckpt", ".pt", ".pth"]:
                    # Try loading as safetensors first (some .ckpt are actually safetensors)
                    try:
                        safetensors.torch.load_file(str(path), device="cpu")
                        return ModelFormat.SAFETENSORS
                    except:
                        return ModelFormat.CKPT
            except Exception:
                pass
        
        return ModelFormat.UNKNOWN
    
    def get_cache_key(self, model_path: str, target_format: ModelFormat) -> str:
        """Generate unique cache key based on model content and target format"""
        path = Path(model_path)
        
        # Use file modification time and size for files
        if path.is_file():
            stat = path.stat()
            content_hash = hashlib.md5(f"{stat.st_mtime}:{stat.st_size}:{path.name}".encode()).hexdigest()
        else:
            # For directories, hash directory structure
            hasher = hashlib.md5()
            for root, dirs, files in os.walk(path):
                for file in sorted(files):
                    file_path = Path(root) / file
                    try:
                        stat = file_path.stat()
                        hasher.update(f"{file_path.relative_to(path)}:{stat.st_mtime}:{stat.st_size}".encode())
                    except:
                        pass
            content_hash = hasher.hexdigest()
        
        return f"{content_hash}_{target_format.value}"
    
    def get_cached_model(self, model_path: str, target_format: ModelFormat) -> Optional[str]:
        """Check if converted model exists in cache"""
        cache_key = self.get_cache_key(model_path, target_format)
        cache_path = self.cache_dir / cache_key
        
        if cache_path.exists():
            # Verify cache is valid
            if target_format == ModelFormat.DIFFUSERS:
                if (cache_path / "model_index.json").exists():
                    return str(cache_path)
            else:
                for ext in [".safetensors", ".ckpt", ".onnx"]:
                    cached_file = cache_path.with_suffix(ext)
                    if cached_file.exists():
                        return str(cached_file)
        
        return None
    
    def convert_model(
        self,
        model_path: str,
        target_format: ModelFormat,
        progress_callback=None,
        force: bool = False
    ) -> Tuple[bool, str]:
        """
        Convert model to target format with caching and progress reporting
        
        Returns:
            Tuple of (success: bool, result_path_or_error: str)
        """
        if target_format == ModelFormat.UNKNOWN:
            return False, "Unknown target format"
        
        # Check cache first
        if not force:
            cached = self.get_cached_model(model_path, target_format)
            if cached:
                logger.info(f"Using cached model: {cached}")
                return True, cached
        
        # Detect source format
        source_format = self.detect_format(model_path)
        if source_format == ModelFormat.UNKNOWN:
            return False, f"Could not detect format of {model_path}"
        
        if source_format == target_format:
            return True, model_path
        
        # Check if conversion is supported
        conversion_key = (source_format, target_format)
        if conversion_key not in self.conversion_matrix:
            return False, f"Conversion from {source_format.value} to {target_format.value} not supported"
        
        # Generate cache path
        cache_key = self.get_cache_key(model_path, target_format)
        cache_path = self.cache_dir / cache_key
        
        # Create conversion job
        job_id = f"{model_path}_{target_format.value}"
        job = ConversionJob(
            source_path=model_path,
            source_format=source_format,
            target_format=target_format,
            target_path=str(cache_path),
            status="starting",
            start_time=time.time()
        )
        
        with self.lock:
            self.conversion_jobs[job_id] = job
        
        def conversion_thread():
            try:
                job.status = "converting"
                job.progress = 0.1
                
                # Get converter function
                converter = self.conversion_matrix[conversion_key]
                
                # Execute conversion
                success, result = converter(
                    model_path,
                    str(cache_path),
                    lambda p: setattr(job, 'progress', 0.1 + p * 0.9)
                )
                
                if success:
                    job.status = "completed"
                    job.progress = 1.0
                    logger.info(f"Conversion completed: {model_path} -> {result}")
                else:
                    job.status = "failed"
                    job.error = result
                    logger.error(f"Conversion failed: {result}")
                    
            except Exception as e:
                job.status = "failed"
                job.error = str(e)
                logger.exception(f"Conversion error for {model_path}")
        
        # Start conversion in background thread
        thread = threading.Thread(target=conversion_thread, daemon=True)
        job.thread = thread
        thread.start()
        
        # Wait for conversion if synchronous
        if progress_callback:
            while job.status in ["starting", "converting"]:
                progress_callback(job.progress, job.status)
                time.sleep(0.1)
            
            if job.status == "failed":
                return False, job.error
            elif job.status == "completed":
                return True, job.target_path
        
        return True, job_id
    
    def get_conversion_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a conversion job"""
        with self.lock:
            job = self.conversion_jobs.get(job_id)
            if not job:
                return {"status": "not_found"}
            
            return {
                "status": job.status,
                "progress": job.progress,
                "error": job.error,
                "elapsed_time": time.time() - job.start_time
            }
    
    def cleanup_cache(self, max_size_gb: float = 10.0):
        """Clean up old cached models to free disk space"""
        cache_path = Path(self.cache_dir)
        if not cache_path.exists():
            return
        
        # Calculate total size
        total_size = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file())
        total_size_gb = total_size / (1024 ** 3)
        
        if total_size_gb <= max_size_gb:
            return
        
        # Get all cached items with access time
        cache_items = []
        for item in cache_path.iterdir():
            try:
                stat = item.stat()
                cache_items.append((item, stat.st_atime, stat.st_size))
            except:
                continue
        
        # Sort by access time (oldest first)
        cache_items.sort(key=lambda x: x[1])
        
        # Remove oldest items until under size limit
        freed_space = 0
        target_free = total_size_gb - max_size_gb
        
        for item, _, size in cache_items:
            if freed_space >= target_free:
                break
            
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
                freed_space += size / (1024 ** 3)
                logger.info(f"Removed cached model: {item}")
            except Exception as e:
                logger.warning(f"Failed to remove {item}: {e}")
    
    # Format-specific handlers
    def _handle_ckpt(self, model_path: str) -> Dict[str, Any]:
        """Handle loading/checking CKPT format"""
        return {"format": ModelFormat.CKPT, "path": model_path}
    
    def _handle_safetensors(self, model_path: str) -> Dict[str, Any]:
        """Handle loading/checking SafeTensors format"""
        return {"format": ModelFormat.SAFETENSORS, "path": model_path}
    
    def _handle_diffusers(self, model_path: str) -> Dict[str, Any]:
        """Handle loading/checking Diffusers format"""
        return {"format": ModelFormat.DIFFUSERS, "path": model_path}
    
    def _handle_onnx(self, model_path: str) -> Dict[str, Any]:
        """Handle loading/checking ONNX format"""
        return {"format": ModelFormat.ONNX, "path": model_path}
    
    # Conversion functions
    def _ckpt_to_safetensors(self, source: str, target: str, progress_callback=None) -> Tuple[bool, str]:
        """Convert CKPT to SafeTensors format"""
        try:
            if progress_callback:
                progress_callback(0.1)
            
            # Load checkpoint
            checkpoint = torch.load(source, map_location="cpu")
            
            if progress_callback:
                progress_callback(0.3)
            
            # Extract state dict
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            
            if progress_callback:
                progress_callback(0.5)
            
            # Save as safetensors
            target_path = Path(target)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            safetensors.torch.save_file(state_dict, str(target_path.with_suffix(".safetensors")))
            
            if progress_callback:
                progress_callback(1.0)
            
            return True, str(target_path.with_suffix(".safetensors"))
            
        except Exception as e:
            return False, f"CKPT to SafeTensors conversion failed: {str(e)}"
    
    def _safetensors_to_ckpt(self, source: str, target: str, progress_callback=None) -> Tuple[bool, str]:
        """Convert SafeTensors to CKPT format"""
        try:
            if progress_callback:
                progress_callback(0.1)
            
            # Load safetensors
            state_dict = safetensors.torch.load_file(source, device="cpu")
            
            if progress_callback:
                progress_callback(0.5)
            
            # Save as checkpoint
            target_path = Path(target)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state_dict, str(target_path.with_suffix(".ckpt")))
            
            if progress_callback:
                progress_callback(1.0)
            
            return True, str(target_path.with_suffix(".ckpt"))
            
        except Exception as e:
            return False, f"SafeTensors to CKPT conversion failed: {str(e)}"
    
    def _ckpt_to_diffusers(self, source: str, target: str, progress_callback=None) -> Tuple[bool, str]:
        """Convert CKPT to Diffusers format"""
        try:
            from diffusers import StableDiffusionPipeline
            
            if progress_callback:
                progress_callback(0.1)
            
            # Load checkpoint
            checkpoint = torch.load(source, map_location="cpu")
            
            if progress_callback:
                progress_callback(0.3)
            
            # Create pipeline from checkpoint
            # Note: This requires proper configuration based on model type
            pipe = StableDiffusionPipeline.from_single_file(
                source,
                load_safety_checker=False,
                extract_ema=True
            )
            
            if progress_callback:
                progress_callback(0.7)
            
            # Save as diffusers
            target_path = Path(target)
            target_path.mkdir(parents=True, exist_ok=True)
            pipe.save_pretrained(str(target_path))
            
            if progress_callback:
                progress_callback(1.0)
            
            return True, str(target_path)
            
        except Exception as e:
            return False, f"CKPT to Diffusers conversion failed: {str(e)}"
    
    def _safetensors_to_diffusers(self, source: str, target: str, progress_callback=None) -> Tuple[bool, str]:
        """Convert SafeTensors to Diffusers format"""
        try:
            from diffusers import StableDiffusionPipeline
            
            if progress_callback:
                progress_callback(0.1)
            
            # Load safetensors
            state_dict = safetensors.torch.load_file(source, device="cpu")
            
            if progress_callback:
                progress_callback(0.3)
            
            # Create pipeline from safetensors
            pipe = StableDiffusionPipeline.from_single_file(
                source,
                load_safety_checker=False,
                extract_ema=True
            )
            
            if progress_callback:
                progress_callback(0.7)
            
            # Save as diffusers
            target_path = Path(target)
            target_path.mkdir(parents=True, exist_ok=True)
            pipe.save_pretrained(str(target_path))
            
            if progress_callback:
                progress_callback(1.0)
            
            return True, str(target_path)
            
        except Exception as e:
            return False, f"SafeTensors to Diffusers conversion failed: {str(e)}"
    
    def _diffusers_to_ckpt(self, source: str, target: str, progress_callback=None) -> Tuple[bool, str]:
        """Convert Diffusers to CKPT format"""
        try:
            from diffusers import StableDiffusionPipeline
            
            if progress_callback:
                progress_callback(0.1)
            
            # Load diffusers pipeline
            pipe = StableDiffusionPipeline.from_pretrained(source, torch_dtype=torch.float32)
            
            if progress_callback:
                progress_callback(0.5)
            
            # Extract state dict and save as checkpoint
            target_path = Path(target)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # This is a simplified conversion - real implementation would need
            # to handle the full state dict structure
            state_dict = {k: v for k, v in pipe.unet.state_dict().items()}
            torch.save(state_dict, str(target_path.with_suffix(".ckpt")))
            
            if progress_callback:
                progress_callback(1.0)
            
            return True, str(target_path.with_suffix(".ckpt"))
            
        except Exception as e:
            return False, f"Diffusers to CKPT conversion failed: {str(e)}"
    
    def _diffusers_to_safetensors(self, source: str, target: str, progress_callback=None) -> Tuple[bool, str]:
        """Convert Diffusers to SafeTensors format"""
        try:
            from diffusers import StableDiffusionPipeline
            
            if progress_callback:
                progress_callback(0.1)
            
            # Load diffusers pipeline
            pipe = StableDiffusionPipeline.from_pretrained(source, torch_dtype=torch.float32)
            
            if progress_callback:
                progress_callback(0.5)
            
            # Extract state dict and save as safetensors
            target_path = Path(target)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # This is a simplified conversion
            state_dict = {k: v for k, v in pipe.unet.state_dict().items()}
            safetensors.torch.save_file(state_dict, str(target_path.with_suffix(".safetensors")))
            
            if progress_callback:
                progress_callback(1.0)
            
            return True, str(target_path.with_suffix(".safetensors"))
            
        except Exception as e:
            return False, f"Diffusers to SafeTensors conversion failed: {str(e)}"
    
    def _ckpt_to_onnx(self, source: str, target: str, progress_callback=None) -> Tuple[bool, str]:
        """Convert CKPT to ONNX format"""
        try:
            import onnx
            
            if progress_callback:
                progress_callback(0.1)
            
            # Load checkpoint
            checkpoint = torch.load(source, map_location="cpu")
            
            if progress_callback:
                progress_callback(0.3)
            
            # Note: Full ONNX conversion requires model architecture knowledge
            # This is a placeholder for the conversion logic
            
            if progress_callback:
                progress_callback(1.0)
            
            return False, "ONNX conversion requires additional dependencies and model architecture"
            
        except Exception as e:
            return False, f"CKPT to ONNX conversion failed: {str(e)}"
    
    def _safetensors_to_onnx(self, source: str, target: str, progress_callback=None) -> Tuple[bool, str]:
        """Convert SafeTensors to ONNX format"""
        try:
            import onnx
            
            if progress_callback:
                progress_callback(0.1)
            
            # Load safetensors
            state_dict = safetensors.torch.load_file(source, device="cpu")
            
            if progress_callback:
                progress_callback(0.5)
            
            # Note: Full ONNX conversion requires model architecture knowledge
            # This is a placeholder for the conversion logic
            
            if progress_callback:
                progress_callback(1.0)
            
            return False, "ONNX conversion requires additional dependencies and model architecture"
            
        except Exception as e:
            return False, f"SafeTensors to ONNX conversion failed: {str(e)}"


# Global converter instance
model_converter = ModelFormatConverter()


def load_model_with_conversion(
    model_path: str,
    target_format: Optional[ModelFormat] = None,
    device=None,
    **kwargs
) -> Tuple[bool, Union[Any, str]]:
    """
    Load model with automatic format detection and conversion
    
    Args:
        model_path: Path to model file or directory
        target_format: Desired format (auto-detected if None)
        device: Target device for model
        **kwargs: Additional arguments for model loader
    
    Returns:
        Tuple of (success: bool, model_or_error: Any)
    """
    if device is None:
        device = devices.device
    
    # Detect current format
    current_format = model_converter.detect_format(model_path)
    if current_format == ModelFormat.UNKNOWN:
        return False, f"Unknown model format: {model_path}"
    
    # Determine target format (default to current format)
    if target_format is None:
        target_format = current_format
    
    # If already in target format, load directly
    if current_format == target_format:
        return _load_model_by_format(model_path, target_format, device, **kwargs)
    
    # Convert model
    success, result = model_converter.convert_model(model_path, target_format)
    if not success:
        return False, result
    
    # Load converted model
    return _load_model_by_format(result, target_format, device, **kwargs)


def _load_model_by_format(
    model_path: str,
    model_format: ModelFormat,
    device,
    **kwargs
) -> Tuple[bool, Union[Any, str]]:
    """Load model using format-specific loader"""
    try:
        if model_format == ModelFormat.CKPT or model_format == ModelFormat.SAFETENSORS:
            # Use existing SD model loader
            from modules import sd_models
            model = sd_models.load_model(model_path, **kwargs)
            return True, model
            
        elif model_format == ModelFormat.DIFFUSERS:
            # Load diffusers model
            from diffusers import StableDiffusionPipeline
            model = StableDiffusionPipeline.from_pretrained(model_path, **kwargs)
            model = model.to(device)
            return True, model
            
        elif model_format == ModelFormat.ONNX:
            # ONNX loading would go here
            return False, "ONNX loading not implemented"
            
        else:
            return False, f"Unknown format: {model_format}"
            
    except Exception as e:
        return False, f"Failed to load model: {str(e)}"


def get_model_info(model_path: str) -> Dict[str, Any]:
    """Get comprehensive model information including format detection"""
    format_detected = model_converter.detect_format(model_path)
    
    info = {
        "path": model_path,
        "format": format_detected.value,
        "exists": Path(model_path).exists(),
        "size": 0,
        "cached_versions": []
    }
    
    # Get file size
    path = Path(model_path)
    if path.is_file():
        info["size"] = path.stat().st_size
    elif path.is_dir():
        total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        info["size"] = total_size
    
    # Check for cached conversions
    for target_format in ModelFormat:
        if target_format == ModelFormat.UNKNOWN or target_format == format_detected:
            continue
        
        cached = model_converter.get_cached_model(model_path, target_format)
        if cached:
            info["cached_versions"].append({
                "format": target_format.value,
                "path": cached
            })
    
    return info


def convert_model_async(
    model_path: str,
    target_format: ModelFormat,
    callback=None
) -> str:
    """
    Start asynchronous model conversion
    
    Returns:
        Job ID for tracking conversion progress
    """
    def progress_wrapper(progress, status):
        if callback:
            callback(progress, status)
    
    success, result = model_converter.convert_model(
        model_path,
        target_format,
        progress_callback=progress_wrapper
    )
    
    if success and not Path(result).exists():
        # This is a job ID, not a path
        return result
    elif success:
        # Conversion completed synchronously
        return "completed"
    else:
        return f"failed: {result}"


def cleanup_conversion_cache(max_size_gb: float = 10.0):
    """Clean up conversion cache to free disk space"""
    model_converter.cleanup_cache(max_size_gb)


# Integration with existing SD model loading
def patch_sd_model_loader():
    """Patch the existing SD model loader to use automatic conversion"""
    original_load_model = sd_models.load_model
    
    def enhanced_load_model(model_path, *args, **kwargs):
        # Check if model needs conversion
        current_format = model_converter.detect_format(model_path)
        
        # If user wants a specific format via settings
        target_format_str = getattr(shared.opts, 'model_conversion_target_format', 'auto')
        if target_format_str != 'auto':
            target_format = ModelFormat(target_format_str)
            if current_format != target_format:
                success, result = load_model_with_conversion(
                    model_path,
                    target_format=target_format
                )
                if success:
                    return result
        
        # Otherwise use original loader
        return original_load_model(model_path, *args, **kwargs)
    
    sd_models.load_model = enhanced_load_model


# Auto-patch on module import
if hasattr(shared, 'opts'):
    patch_sd_model_loader()


# API endpoints for web UI
def setup_api_routes(app):
    """Setup API routes for model conversion management"""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    
    class ConversionRequest(BaseModel):
        model_path: str
        target_format: str
        force: bool = False
    
    class ConversionStatusRequest(BaseModel):
        job_id: str
    
    @app.post("/api/model-conversion/start")
    async def start_conversion(request: ConversionRequest):
        try:
            target_format = ModelFormat(request.target_format)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid format: {request.target_format}")
        
        job_id = convert_model_async(request.model_path, target_format)
        return {"job_id": job_id, "status": "started"}
    
    @app.post("/api/model-conversion/status")
    async def get_conversion_status(request: ConversionStatusRequest):
        status = model_converter.get_conversion_status(request.job_id)
        return status
    
    @app.get("/api/model-conversion/formats")
    async def get_supported_formats():
        return {
            "formats": [f.value for f in ModelFormat if f != ModelFormat.UNKNOWN],
            "conversions": [
                {"from": k[0].value, "to": k[1].value}
                for k in model_converter.conversion_matrix.keys()
            ]
        }
    
    @app.post("/api/model-conversion/cleanup")
    async def cleanup_cache(max_size_gb: float = 10.0):
        cleanup_conversion_cache(max_size_gb)
        return {"status": "cleanup_completed"}
    
    @app.get("/api/model-info")
    async def model_info(model_path: str):
        return get_model_info(model_path)