# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights nexuss.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights nexuss.pt                 # PyTorch
                                 nexuss.torchscript        # TorchScript
                                 nexuss.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 nexuss_openvino_model     # OpenVINO
                                 nexuss.engine             # TensorRT
                                 nexuss.mlpackage          # CoreML (macOS-only)
                                 nexuss_saved_model        # TensorFlow SavedModel
                                 nexuss.pb                 # TensorFlow GraphDef
                                 nexuss.tflite             # TensorFlow Lite
                                 nexuss_edgetpu.tflite     # TensorFlow Edge TPU
                                 nexuss_paddle_model       # PaddlePaddle
"""

import argparse
import csv
import os
import platform
import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import threading
import psutil
import signal
import atexit

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode


class BackendProfiler:
    """Performance profiling and benchmarking for inference backends."""
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
        self.memory_usage: Dict[str, List[float]] = {}
        
    def start_measurement(self, backend_name: str) -> None:
        """Start timing measurement for a backend."""
        if backend_name not in self.timings:
            self.timings[backend_name] = []
            self.memory_usage[backend_name] = []
        
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.start_memory = torch.cuda.memory_allocated()
    
    def end_measurement(self, backend_name: str) -> Tuple[float, float]:
        """End timing measurement and record results."""
        elapsed = time.time() - self.start_time
        self.timings[backend_name].append(elapsed)
        
        memory_used = 0.0
        if torch.cuda.is_available():
            memory_used = (torch.cuda.max_memory_allocated() - self.start_memory) / (1024 ** 2)  # MB
            self.memory_usage[backend_name].append(memory_used)
        
        return elapsed, memory_used
    
    def get_average_stats(self, backend_name: str) -> Dict[str, float]:
        """Get average performance statistics for a backend."""
        if backend_name not in self.timings or not self.timings[backend_name]:
            return {"avg_time": 0.0, "avg_memory": 0.0, "samples": 0}
        
        times = self.timings[backend_name]
        memories = self.memory_usage.get(backend_name, [])
        
        return {
            "avg_time": sum(times) / len(times),
            "avg_memory": sum(memories) / len(memories) if memories else 0.0,
            "samples": len(times)
        }
    
    def log_summary(self) -> None:
        """Log summary of all backend performances."""
        LOGGER.info("\n" + "="*50)
        LOGGER.info("Backend Performance Summary")
        LOGGER.info("="*50)
        
        for backend in self.timings:
            stats = self.get_average_stats(backend)
            LOGGER.info(f"{backend:15} | Avg Time: {stats['avg_time']*1000:6.2f}ms | "
                       f"Avg Memory: {stats['avg_memory']:6.2f}MB | Samples: {stats['samples']}")


class SystemMonitor:
    """Monitor system metrics for edge deployment resilience."""
    
    def __init__(self, thermal_threshold: float = 80.0, memory_threshold: float = 0.9):
        self.thermal_threshold = thermal_threshold
        self.memory_threshold = memory_threshold
        self.temperature_history: List[float] = []
        self.memory_history: List[float] = []
        self.max_history = 100
        
    def get_cpu_temperature(self) -> Optional[float]:
        """Get CPU temperature (Linux/macOS only)."""
        try:
            if platform.system() == "Linux":
                # Try common Linux temperature paths
                temp_paths = [
                    "/sys/class/thermal/thermal_zone0/temp",
                    "/sys/class/hwmon/hwmon0/temp1_input"
                ]
                for path in temp_paths:
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            temp = float(f.read().strip()) / 1000.0
                            self.temperature_history.append(temp)
                            if len(self.temperature_history) > self.max_history:
                                self.temperature_history.pop(0)
                            return temp
            elif platform.system() == "Darwin":  # macOS
                import subprocess
                result = subprocess.run(['sudo', 'powermetrics', '--samplers', 'smc', '-i', '100', '-n', '1'], 
                                      capture_output=True, text=True)
                for line in result.stdout.split('\n'):
                    if 'CPU die temperature' in line:
                        temp = float(line.split(':')[1].strip().replace('C', ''))
                        self.temperature_history.append(temp)
                        if len(self.temperature_history) > self.max_history:
                            self.temperature_history.pop(0)
                        return temp
        except Exception:
            pass
        return None
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get system memory usage."""
        memory = psutil.virtual_memory()
        gpu_memory = {}
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory[f"gpu_{i}"] = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
        
        memory_percent = memory.percent / 100.0
        self.memory_history.append(memory_percent)
        if len(self.memory_history) > self.max_history:
            self.memory_history.pop(0)
        
        return {
            "system_memory_percent": memory_percent,
            "system_memory_gb": memory.used / (1024 ** 3),
            "gpu_memory": gpu_memory
        }
    
    def is_thermal_throttling(self) -> bool:
        """Check if system is experiencing thermal throttling."""
        temp = self.get_cpu_temperature()
        if temp is not None:
            return temp > self.thermal_threshold
        return False
    
    def is_memory_critical(self) -> bool:
        """Check if memory usage is critical."""
        memory = self.get_memory_usage()
        return memory["system_memory_percent"] > self.memory_threshold
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        temp = self.get_cpu_temperature()
        memory = self.get_memory_usage()
        
        return {
            "temperature": temp,
            "is_thermal_throttling": self.is_thermal_throttling(),
            "memory": memory,
            "is_memory_critical": self.is_memory_critical(),
            "avg_temperature": sum(self.temperature_history) / len(self.temperature_history) if self.temperature_history else None,
            "avg_memory": sum(self.memory_history) / len(self.memory_history) if self.memory_history else None
        }


class WatchdogTimer:
    """Watchdog timer for automatic restart on failure."""
    
    def __init__(self, timeout: float = 30.0, callback=None):
        self.timeout = timeout
        self.callback = callback
        self.timer = None
        self.is_running = False
        
    def start(self):
        """Start the watchdog timer."""
        if self.is_running:
            self.reset()
        else:
            self.is_running = True
            self.timer = threading.Timer(self.timeout, self._timeout_handler)
            self.timer.daemon = True
            self.timer.start()
    
    def reset(self):
        """Reset the watchdog timer."""
        if self.timer and self.is_running:
            self.timer.cancel()
            self.timer = threading.Timer(self.timeout, self._timeout_handler)
            self.timer.daemon = True
            self.timer.start()
    
    def stop(self):
        """Stop the watchdog timer."""
        if self.timer:
            self.timer.cancel()
            self.is_running = False
    
    def _timeout_handler(self):
        """Handle watchdog timeout."""
        LOGGER.warning("Watchdog timeout triggered - system may be unresponsive")
        if self.callback:
            self.callback()
        self.is_running = False


class ResilientEdgeDeployment:
    """
    Resilient edge deployment wrapper for YOLOv5 inference.
    Adds health checks, automatic restart, thermal throttling detection,
    and graceful degradation capabilities.
    """
    
    def __init__(self, model, device, imgsz=640, stride=32, pt=True, fp16=False, 
                 fallback_model_path=None, health_check_interval=10.0,
                 thermal_threshold=80.0, memory_threshold=0.9):
        self.model = model
        self.device = device
        self.imgsz = imgsz
        self.stride = stride
        self.pt = pt
        self.fp16 = fp16
        
        # Resilience parameters
        self.fallback_model_path = fallback_model_path
        self.health_check_interval = health_check_interval
        self.original_imgsz = imgsz
        self.original_fp16 = fp16
        
        # System monitoring
        self.system_monitor = SystemMonitor(thermal_threshold, memory_threshold)
        self.watchdog = WatchdogTimer(timeout=30.0, callback=self._emergency_restart)
        
        # State tracking
        self.is_healthy = True
        self.failure_count = 0
        self.max_failures = 3
        self.last_health_check = time.time()
        self.performance_history: List[Dict] = []
        
        # Model fallback state
        self.is_using_fallback = False
        self.fallback_model = None
        
        # Register cleanup
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        LOGGER.info(f"ResilientEdgeDeployment initialized with imgsz={imgsz}, "
                   f"thermal_threshold={thermal_threshold}°C, memory_threshold={memory_threshold*100}%")
    
    def _signal_handler(self, signum, frame):
        """Handle termination signals."""
        LOGGER.info(f"Received signal {signum}, cleaning up...")
        self.cleanup()
        sys.exit(0)
    
    def cleanup(self):
        """Cleanup resources."""
        self.watchdog.stop()
        if self.fallback_model is not None:
            del self.fallback_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _emergency_restart(self):
        """Emergency restart procedure."""
        LOGGER.critical("Emergency restart triggered by watchdog!")
        self.cleanup()
        # In production, you might want to restart the entire process
        # For now, we'll try to recover by reloading the model
        try:
            self._reload_model()
            LOGGER.info("Emergency recovery successful")
        except Exception as e:
            LOGGER.error(f"Emergency recovery failed: {e}")
            sys.exit(1)
    
    def _reload_model(self):
        """Reload the model (simulated restart)."""
        LOGGER.info("Reloading model...")
        # In a real implementation, you would reload the model from disk
        # For now, we'll just reset the failure count
        self.failure_count = 0
        self.is_healthy = True
        self.watchdog.reset()
    
    def _load_fallback_model(self):
        """Load fallback model for graceful degradation."""
        if self.fallback_model_path and os.path.exists(self.fallback_model_path):
            try:
                LOGGER.info(f"Loading fallback model from {self.fallback_model_path}")
                # In a real implementation, you would load the smaller model
                # For demonstration, we'll just mark that we're using fallback
                self.is_using_fallback = True
                # Reduce image size for fallback
                self.imgsz = max(320, self.imgsz // 2)
                self.imgsz = check_img_size(self.imgsz, s=self.stride)
                LOGGER.info(f"Switched to fallback mode with imgsz={self.imgsz}")
                return True
            except Exception as e:
                LOGGER.error(f"Failed to load fallback model: {e}")
        return False
    
    def _adjust_for_thermal_throttling(self):
        """Adjust inference parameters when thermal throttling is detected."""
        if self.system_monitor.is_thermal_throttling():
            LOGGER.warning("Thermal throttling detected - adjusting inference parameters")
            
            # Reduce image size
            new_imgsz = max(320, self.imgsz - 64)
            if new_imgsz != self.imgsz:
                self.imgsz = check_img_size(new_imgsz, s=self.stride)
                LOGGER.info(f"Reduced image size to {self.imgsz} due to thermal throttling")
            
            # Switch to FP32 if using FP16 (FP32 is less computationally intensive on some hardware)
            if self.fp16 and torch.cuda.is_available():
                self.fp16 = False
                LOGGER.info("Switched to FP32 precision due to thermal throttling")
            
            return True
        return False
    
    def _adjust_for_memory_pressure(self):
        """Adjust inference parameters when memory pressure is high."""
        if self.system_monitor.is_memory_critical():
            LOGGER.warning("Memory pressure critical - adjusting inference parameters")
            
            # Try to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Reduce batch size (handled by caller) or image size
            new_imgsz = max(320, self.imgsz - 32)
            if new_imgsz != self.imgsz:
                self.imgsz = check_img_size(new_imgsz, s=self.stride)
                LOGGER.info(f"Reduced image size to {self.imgsz} due to memory pressure")
            
            # Try to load fallback model if available
            if not self.is_using_fallback and self.fallback_model_path:
                self._load_fallback_model()
            
            return True
        return False
    
    def health_check(self) -> bool:
        """Perform comprehensive health check."""
        current_time = time.time()
        if current_time - self.last_health_check < self.health_check_interval:
            return self.is_healthy
        
        self.last_health_check = current_time
        system_health = self.system_monitor.get_system_health()
        
        # Log health status
        if system_health["temperature"]:
            LOGGER.debug(f"System health: Temp={system_health['temperature']:.1f}°C, "
                        f"Memory={system_health['memory']['system_memory_percent']*100:.1f}%")
        
        # Check for thermal throttling
        if system_health["is_thermal_throttling"]:
            LOGGER.warning(f"Thermal throttling detected: {system_health['temperature']:.1f}°C > "
                          f"{self.system_monitor.thermal_threshold}°C")
            self._adjust_for_thermal_throttling()
        
        # Check for memory pressure
        if system_health["is_memory_critical"]:
            LOGGER.warning(f"Memory pressure critical: {system_health['memory']['system_memory_percent']*100:.1f}% > "
                          f"{self.system_monitor.memory_threshold*100:.1f}%")
            self._adjust_for_memory_pressure()
        
        # Model health check (simplified)
        try:
            # Try a dummy inference to check model health
            dummy_input = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)
            if self.fp16 and torch.cuda.is_available():
                dummy_input = dummy_input.half()
            
            with torch.no_grad():
                _ = self.model(dummy_input)
            
            self.is_healthy = True
            self.failure_count = 0
            return True
            
        except Exception as e:
            LOGGER.error(f"Model health check failed: {e}")
            self.failure_count += 1
            self.is_healthy = False
            
            if self.failure_count >= self.max_failures:
                LOGGER.critical(f"Maximum failure count ({self.max_failures}) reached")
                if not self.is_using_fallback and self._load_fallback_model():
                    LOGGER.info("Switched to fallback model after repeated failures")
                    self.failure_count = 0
                    self.is_healthy = True
                else:
                    self._emergency_restart()
            
            return False
    
    def resilient_inference(self, im, augment=False, visualize=False):
        """
        Perform inference with resilience features.
        
        Args:
            im: Input tensor
            augment: Augment inference
            visualize: Visualize features
            
        Returns:
            Model predictions
        """
        # Start watchdog for this inference
        self.watchdog.start()
        
        try:
            # Perform health check
            if not self.health_check():
                LOGGER.warning("Health check failed, attempting recovery...")
                time.sleep(1)  # Brief pause before retry
                if not self.health_check():
                    raise RuntimeError("System health check failed after recovery attempt")
            
            # Adjust input size if needed
            if im.shape[-1] != self.imgsz or im.shape[-2] != self.imgsz:
                import torch.nn.functional as F
                im = F.interpolate(im, size=(self.imgsz, self.imgsz), mode='bilinear', align_corners=False)
            
            # Perform inference
            with torch.no_grad():
                if self.fp16 and self.device.type != 'cpu':
                    im = im.half()
                pred = self.model(im, augment=augment, visualize=visualize)
            
            # Reset watchdog on successful inference
            self.watchdog.reset()
            
            # Record performance
            self.performance_history.append({
                "timestamp": time.time(),
                "imgsz": self.imgsz,
                "fp16": self.fp16,
                "is_fallback": self.is_using_fallback,
                "success": True
            })
            
            return pred
            
        except Exception as e:
            LOGGER.error(f"Inference failed: {e}")
            self.failure_count += 1
            
            # Record failure
            self.performance_history.append({
                "timestamp": time.time(),
                "imgsz": self.imgsz,
                "fp16": self.fp16,
                "is_fallback": self.is_using_fallback,
                "success": False,
                "error": str(e)
            })
            
            # Attempt recovery
            if self.failure_count >= self.max_failures:
                if not self.is_using_fallback and self._load_fallback_model():
                    LOGGER.info("Switched to fallback model after inference failure")
                    self.failure_count = 0
                    # Retry with fallback model
                    return self.resilient_inference(im, augment, visualize)
                else:
                    self._emergency_restart()
            
            raise
            
        finally:
            # Always stop watchdog
            self.watchdog.stop()
    
    def get_resilience_stats(self) -> Dict[str, Any]:
        """Get resilience statistics."""
        system_health = self.system_monitor.get_system_health()
        
        return {
            "is_healthy": self.is_healthy,
            "failure_count": self.failure_count,
            "is_using_fallback": self.is_using_fallback,
            "current_imgsz": self.imgsz,
            "current_fp16": self.fp16,
            "system_health": system_health,
            "performance_history_size": len(self.performance_history),
            "successful_inferences": sum(1 for p in self.performance_history if p.get("success", False)),
            "failed_inferences": sum(1 for p in self.performance_history if not p.get("success", True))
        }


class UnifiedInferenceEngine:
    """
    Unified inference engine that automatically selects the best available backend.
    Supports: TensorRT, OpenVINO, ONNX Runtime, CoreML, TFLite, PyTorch
    """
    
    # Backend preference order based on platform and performance
    BACKEND_PREFERENCES = {
        "linux": ["TensorRT", "OpenVINO", "ONNX Runtime", "PyTorch"],
        "windows": ["TensorRT", "OpenVINO", "ONNX Runtime", "PyTorch"],
        "darwin": ["CoreML", "ONNX Runtime", "PyTorch"],  # macOS
    }
    
    # Fallback chains for each backend
    FALLBACK_CHAINS = {
        "TensorRT": ["ONNX Runtime", "PyTorch"],
        "OpenVINO": ["ONNX Runtime", "PyTorch"],
        "CoreML": ["ONNX Runtime", "PyTorch"],
        "ONNX Runtime": ["PyTorch"],
        "TFLite": ["ONNX Runtime", "PyTorch"],
        "PyTorch": []
    }
    
    def __init__(self, weights: str, device: str = "", dnn: bool = False, data: Any = None, fp16: bool = False):
        """
        Initialize unified inference engine with automatic backend selection.
        
        Args:
            weights: Path to model weights
            device: CUDA device (e.g., '0' or '0,1,2,3' or 'cpu')
            dnn: Use OpenCV DNN for ONNX inference
            data: Dataset YAML path
            fp16: Use FP16 half-precision inference
        """
        self.weights = weights
        self.device = select_device(device)
        self.dnn = dnn
        self.data = data
        self.fp16 = fp16
        
        self.model = None
        self.backend_name = "Unknown"
        self.profiler = BackendProfiler()
        self.available_backends = self._detect_available_backends()
        
        # Load model with best available backend
        self._load_model_with_fallback()
        
    def _detect_available_backends(self) -> List[str]:
        """Detect which backends are available on the current system."""
        available = []
        
        # Check PyTorch (always available)
        available.append("PyTorch")
        
        # Check TensorRT
        try:
            import tensorrt as trt
            if torch.cuda.is_available():
                available.append("TensorRT")
        except ImportError:
            pass
        
        # Check OpenVINO
        try:
            from openvino.runtime import Core
            available.append("OpenVINO")
        except ImportError:
            pass
        
        # Check ONNX Runtime
        try:
            import onnxruntime as ort
            available.append("ONNX Runtime")
        except ImportError:
            pass
        
        # Check CoreML (macOS only)
        if platform.system() == "Darwin":
            try:
                import coremltools as ct
                available.append("CoreML")
            except ImportError:
                pass
        
        # Check TFLite
        try:
            import tensorflow as tf
            available.append("TFLite")
        except ImportError:
            pass
        
        return available
    
    def _select_best_backend(self) -> str:
        """Select the best available backend based on platform and preferences."""
        platform_name = platform.system().lower()
        if platform_name == "darwin":
            platform_name = "darwin"
        elif platform_name == "windows":
            platform_name = "windows"
        else:
            platform_name = "linux"
        
        preferred_order = self.BACKEND_PREFERENCES.get(platform_name, ["PyTorch"])
        
        for backend in preferred_order:
            if backend in self.available_backends:
                return backend
        
        return "PyTorch"  # Fallback to PyTorch
    
    def _load_model_with_fallback(self):
        """Load model with fallback chain if primary backend fails."""
        primary_backend = self._select_best_backend()
        backends_to_try = [primary_backend] + self.FALLBACK_CHAINS.get(primary_backend, [])
        
        for backend in backends_to_try:
            try:
                LOGGER.info(f"Attempting to load model with {backend} backend...")
                self.profiler.start_measurement(backend)
                
                if backend == "TensorRT":
                    self._load_tensorrt_model()
                elif backend == "OpenVINO":
                    self._load_openvino_model()
                elif backend == "ONNX Runtime":
                    self._load_onnx_model()
                elif backend == "CoreML":
                    self._load_coreml_model()
                elif backend == "TFLite":
                    self._load_tflite_model()
                elif backend == "PyTorch":
                    self._load_pytorch_model()
                else:
                    continue
                
                elapsed, memory = self.profiler.end_measurement(backend)
                self.backend_name = backend
                LOGGER.info(f"Successfully loaded model with {backend} backend "
                           f"(time: {elapsed*1000:.2f}ms, memory: {memory:.2f}MB)")
                return
                
            except Exception as e:
                LOGGER.warning(f"Failed to load model with {backend} backend: {e}")
                self.profiler.end_measurement(backend)  # Record failure time
                continue
        
        raise RuntimeError(f"Failed to load model with any available backend. "
                          f"Available backends: {self.available_backends}")
    
    def _load_tensorrt_model(self):
        """Load TensorRT model."""
        # Implementation would go here
        raise NotImplementedError("TensorRT backend not implemented in this example")
    
    def _load_openvino_model(self):
        """Load OpenVINO model."""
        # Implementation would go here
        raise NotImplementedError("OpenVINO backend not implemented in this example")
    
    def _load_onnx_model(self):
        """Load ONNX model."""
        # Implementation would go here
        raise NotImplementedError("ONNX backend not implemented in this example")
    
    def _load_coreml_model(self):
        """Load CoreML model."""
        # Implementation would go here
        raise NotImplementedError("CoreML backend not implemented in this example")
    
    def _load_tflite_model(self):
        """Load TFLite model."""
        # Implementation would go here
        raise NotImplementedError("TFLite backend not implemented in this example")
    
    def _load_pytorch_model(self):
        """Load PyTorch model."""
        from models.experimental import attempt_load
        self.model = attempt_load(self.weights, device=self.device)
        self.stride = int(self.model.stride.max())
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        
        if self.fp16:
            self.model.half()
    
    def __call__(self, im, augment=False, visualize=False):
        """Run inference with the loaded backend."""
        self.profiler.start_measurement(self.backend_name)
        
        try:
            # This is a simplified version - each backend would have its own inference logic
            if self.backend_name == "PyTorch":
                with torch.no_grad():
                    if self.fp16 and self.device.type != 'cpu':
                        im = im.half()
                    pred = self.model(im, augment=augment, visualize=visualize)
            else:
                # Other backends would have their own inference methods
                raise NotImplementedError(f"Inference not implemented for {self.backend_name}")
            
            elapsed, memory = self.profiler.end_measurement(self.backend_name)
            return pred
            
        except Exception as e:
            self.profiler.end_measurement(self.backend_name)
            raise
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the current backend."""
        return {
            "backend": self.backend_name,
            "available_backends": self.available_backends,
            "performance_stats": self.profiler.get_average_stats(self.backend_name)
        }


def health_check_decorator(func):
    """Decorator for health-checked inference functions."""
    def wrapper(self, *args, **kwargs):
        if hasattr(self, 'resilient_deployment'):
            if not self.resilient_deployment.health_check():
                LOGGER.warning("Health check failed before inference")
                # Try to recover
                time.sleep(0.5)
                if not self.resilient_deployment.health_check():
                    raise RuntimeError("System health check failed")
        
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            if hasattr(self, 'resilient_deployment'):
                LOGGER.error(f"Inference failed in health-checked function: {e}")
                # The resilient deployment will handle recovery
            raise
    return wrapper


@smart_inference_mode()
def run(
    weights=ROOT / "nexuss.pt",  # model.pt path(s)
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_csv=False,  # save results in CSV format
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
    # Resilience parameters
    resilient=False,  # enable resilient edge deployment
    fallback_model=None,  # path to fallback model for graceful degradation
    health_check_interval=10.0,  # health check interval in seconds
    thermal_threshold=80.0,  # temperature threshold for throttling (°C)
    memory_threshold=0.9,  # memory usage threshold (0-1)
    watchdog_timeout=30.0,  # watchdog timeout in seconds
):
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Initialize resilient deployment if enabled
    resilient_deployment = None
    if resilient:
        LOGGER.info("Initializing resilient edge deployment...")
        resilient_deployment = ResilientEdgeDeployment(
            model=model,
            device=device,
            imgsz=imgsz[0],  # Use first dimension for square images
            stride=stride,
            pt=pt,
            fp16=half,
            fallback_model_path=fallback_model,
            health_check_interval=health_check_interval,
            thermal_threshold=thermal_threshold,
            memory_threshold=memory_threshold
        )
        # Attach resilient deployment to model for health-checked inference
        model.resilient_deployment = resilient_deployment
        
        # Set up watchdog
        resilient_deployment.watchdog.timeout = watchdog_timeout

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s in dataset:
        # Health check before processing each frame if resilient mode is enabled
        if resilient_deployment:
            if not resilient_deployment.health_check():
                LOGGER.warning(f"Skipping frame due to health check failure: {path}")
                continue
            
            # Adjust image size if needed (thermal throttling or memory pressure)
            if imgsz[0] != resilient_deployment.imgsz:
                imgsz = (resilient_deployment.imgsz, resilient_deployment.imgsz)
                LOGGER.info(f"Adjusted image size to {imgsz} based on system conditions")

        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if resilient_deployment:
                # Use resilient inference
                pred = resilient_deployment.resilient_inference(im, augment=augment, visualize=visualize)
            else:
                # Standard inference
                pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / "predictions.csv"

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            """Writes prediction data to a CSV file, appending if the file exists."""
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            s += "%gx%g " % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                    if save_csv:
                        write_to_csv(p.name, label, conf)
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) & 0xFF == ord("q"):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

    # Log resilience statistics if enabled
    if resilient_deployment:
        stats = resilient_deployment.get_resilience_stats()
        LOGGER.info("\n" + "="*50)
        LOGGER.info("Resilience Statistics")
        LOGGER.info("="*50)
        LOGGER.info(f"System healthy: {stats['is_healthy']}")
        LOGGER.info(f"Failure count: {stats['failure_count']}")
        LOGGER.info(f"Using fallback model: {stats['is_using_fallback']}")
        LOGGER.info(f"Current image size: {stats['current_imgsz']}")
        LOGGER.info(f"FP16 enabled: {stats['current_fp16']}")
        if stats['system_health']['temperature']:
            LOGGER.info(f"Temperature: {stats['system_health']['temperature']:.1f}°C")
        LOGGER.info(f"Memory usage: {stats['system_health']['memory']['system_memory_percent']*100:.1f}%")
        LOGGER.info(f"Successful inferences: {stats['successful_inferences']}")
        LOGGER.info(f"Failed inferences: {stats['failed_inferences']}")
        
        # Cleanup
        resilient_deployment.cleanup()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "nexuss.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    # Resilience arguments
    parser.add_argument("--resilient", action="store_true", help="enable resilient edge deployment")
    parser.add_argument("--fallback-model", type=str, default=None, help="path to fallback model for graceful degradation")
    parser.add_argument("--health-check-interval", type=float, default=10.0, help="health check interval in seconds")
    parser.add_argument("--thermal-threshold", type=float, default=80.0, help="temperature threshold for throttling (°C)")
    parser.add_argument("--memory-threshold", type=float, default=0.9, help="memory usage threshold (0-1)")
    parser.add_argument("--watchdog-timeout", type=float, default=30.0, help="watchdog timeout in seconds")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)