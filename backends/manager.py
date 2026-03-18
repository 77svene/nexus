# backends/manager.py
"""
Unified Multi-Backend Inference Engine for YOLOv5
Automatically selects optimal backend based on platform and model format
"""

import os
import time
import platform
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod
import torch
import torch.nn as nn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try importing all possible backends
BACKENDS_AVAILABLE = {}

# TensorRT
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    BACKENDS_AVAILABLE['tensorrt'] = True
except ImportError:
    BACKENDS_AVAILABLE['tensorrt'] = False

# OpenVINO
try:
    from openvino.runtime import Core
    BACKENDS_AVAILABLE['openvino'] = True
except ImportError:
    BACKENDS_AVAILABLE['openvino'] = False

# ONNX Runtime
try:
    import onnxruntime as ort
    BACKENDS_AVAILABLE['onnxruntime'] = True
except ImportError:
    BACKENDS_AVAILABLE['onnxruntime'] = False

# CoreML
try:
    import coremltools as ct
    BACKENDS_AVAILABLE['coreml'] = True
except ImportError:
    BACKENDS_AVAILABLE['coreml'] = False

# TFLite
try:
    import tensorflow as tf
    BACKENDS_AVAILABLE['tflite'] = True
except ImportError:
    BACKENDS_AVAILABLE['tflite'] = False

# PyTorch (always available)
BACKENDS_AVAILABLE['pytorch'] = True


class BackendBase(ABC):
    """Abstract base class for all inference backends"""
    
    def __init__(self, model_path: str, device: str = 'cpu', **kwargs):
        self.model_path = Path(model_path)
        self.device = device
        self.model = None
        self.input_shape = None
        self.output_shape = None
        self.backend_name = self.__class__.__name__
        
    @abstractmethod
    def load(self) -> None:
        """Load model into memory"""
        pass
    
    @abstractmethod
    def predict(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Run inference on input image"""
        pass
    
    @abstractmethod
    def warmup(self, iterations: int = 10) -> float:
        """Warm up model and return average inference time"""
        pass
    
    def get_input_shape(self) -> Tuple:
        """Return expected input shape"""
        return self.input_shape
    
    def get_output_shape(self) -> Tuple:
        """Return output shape"""
        return self.output_shape
    
    def is_available(self) -> bool:
        """Check if backend is available"""
        return True
    
    def __repr__(self) -> str:
        return f"{self.backend_name}(model={self.model_path}, device={self.device})"


class PyTorchBackend(BackendBase):
    """PyTorch backend using torch.hub or direct loading"""
    
    def __init__(self, model_path: str, device: str = 'cpu', **kwargs):
        super().__init__(model_path, device, **kwargs)
        self.model_type = kwargs.get('model_type', 'nexus')
        
    def load(self) -> None:
        """Load PyTorch model"""
        try:
            if self.model_path.suffix == '.pt':
                # Load PyTorch model
                self.model = torch.load(str(self.model_path), map_location=self.device)
                if isinstance(self.model, dict):
                    self.model = self.model['model']  # Handle YOLOv5 checkpoint format
                self.model.to(self.device).eval()
            else:
                # Try loading via torch.hub for YOLOv5 models
                import hubconf
                self.model = hubconf.custom(path_or_model=str(self.model_path))
                self.model.to(self.device).eval()
            
            # Get input shape from model
            if hasattr(self.model, 'stride'):
                self.input_shape = (1, 3, 640, 640)  # Default YOLOv5 shape
            else:
                self.input_shape = tuple(next(self.model.parameters()).shape)
                
            logger.info(f"PyTorch model loaded: {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            raise
    
    def predict(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Run PyTorch inference"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Convert numpy to torch tensor
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).to(self.device)
        
        # Add batch dimension if needed
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            output = self.model(image, **kwargs)
        
        # Convert to numpy
        if isinstance(output, torch.Tensor):
            output = output.cpu().numpy()
        elif isinstance(output, (list, tuple)):
            output = [o.cpu().numpy() if isinstance(o, torch.Tensor) else o for o in output]
        
        return output
    
    def warmup(self, iterations: int = 10) -> float:
        """Warmup PyTorch model"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Create dummy input
        dummy_input = torch.randn(self.input_shape).to(self.device)
        
        # Warmup iterations
        for _ in range(iterations):
            _ = self.model(dummy_input)
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.time()
            _ = self.model(dummy_input)
            torch.cuda.synchronize() if self.device == 'cuda' else None
            times.append(time.time() - start)
        
        avg_time = np.mean(times)
        logger.info(f"PyTorch warmup complete. Avg inference: {avg_time*1000:.2f}ms")
        return avg_time


class ONNXRuntimeBackend(BackendBase):
    """ONNX Runtime backend"""
    
    def __init__(self, model_path: str, device: str = 'cpu', **kwargs):
        super().__init__(model_path, device, **kwargs)
        self.session = None
        self.providers = self._get_providers()
        
    def _get_providers(self) -> List[str]:
        """Get available ONNX Runtime providers"""
        providers = []
        if self.device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        return providers
    
    def load(self) -> None:
        """Load ONNX model"""
        try:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options,
                providers=self.providers
            )
            
            # Get input/output shapes
            input_meta = self.session.get_inputs()[0]
            self.input_shape = input_meta.shape
            self.output_shape = self.session.get_outputs()[0].shape
            
            logger.info(f"ONNX Runtime model loaded: {self.model_path}")
            logger.info(f"Using providers: {self.session.get_providers()}")
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise
    
    def predict(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Run ONNX Runtime inference"""
        if self.session is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Get input name
        input_name = self.session.get_inputs()[0].name
        
        # Prepare input
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        # Ensure correct dtype
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        # Run inference
        outputs = self.session.run(None, {input_name: image})
        
        return outputs[0] if len(outputs) == 1 else outputs
    
    def warmup(self, iterations: int = 10) -> float:
        """Warmup ONNX Runtime model"""
        if self.session is None:
            raise RuntimeError("Model not loaded")
        
        # Create dummy input
        dummy_input = np.random.randn(*self.input_shape).astype(np.float32)
        input_name = self.session.get_inputs()[0].name
        
        # Warmup
        for _ in range(iterations):
            _ = self.session.run(None, {input_name: dummy_input})
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.time()
            _ = self.session.run(None, {input_name: dummy_input})
            times.append(time.time() - start)
        
        avg_time = np.mean(times)
        logger.info(f"ONNX Runtime warmup complete. Avg inference: {avg_time*1000:.2f}ms")
        return avg_time


class TensorRTBackend(BackendBase):
    """TensorRT backend for NVIDIA GPUs"""
    
    def __init__(self, model_path: str, device: str = 'cuda', **kwargs):
        super().__init__(model_path, device, **kwargs)
        self.engine = None
        self.context = None
        self.stream = None
        self.bindings = []
        self.batch_size = kwargs.get('batch_size', 1)
        
    def load(self) -> None:
        """Load TensorRT engine"""
        if not BACKENDS_AVAILABLE['tensorrt']:
            raise ImportError("TensorRT not available")
        
        try:
            # Load TensorRT engine
            logger.info(f"Loading TensorRT engine: {self.model_path}")
            
            with open(str(self.model_path), 'rb') as f:
                runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
                self.engine = runtime.deserialize_cuda_engine(f.read())
            
            self.context = self.engine.create_execution_context()
            self.stream = cuda.Stream()
            
            # Allocate buffers
            self._allocate_buffers()
            
            logger.info(f"TensorRT engine loaded: {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load TensorRT engine: {e}")
            raise
    
    def _allocate_buffers(self) -> None:
        """Allocate device buffers for TensorRT"""
        for binding in self.engine:
            binding_idx = self.engine.get_binding_index(binding)
            shape = self.engine.get_binding_shape(binding_idx)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding_idx))
            
            # Calculate size
            size = trt.volume(shape) * self.batch_size
            if size < 0:  # Dynamic shape
                size = trt.volume(self.engine.get_profile_shape(0, binding_idx)[0]) * self.batch_size
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append({
                'host': host_mem,
                'device': device_mem,
                'shape': shape,
                'dtype': dtype,
                'name': binding,
                'is_input': self.engine.binding_is_input(binding_idx)
            })
    
    def predict(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Run TensorRT inference"""
        if self.context is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Find input binding
        input_binding = None
        for binding in self.bindings:
            if binding['is_input']:
                input_binding = binding
                break
        
        if input_binding is None:
            raise RuntimeError("No input binding found")
        
        # Copy input to device
        np.copyto(input_binding['host'], image.ravel())
        cuda.memcpy_htod_async(input_binding['device'], input_binding['host'], self.stream)
        
        # Run inference
        self.context.execute_async_v2(
            bindings=[b['device'] for b in self.bindings],
            stream_handle=self.stream.handle
        )
        
        # Copy output back
        output_bindings = [b for b in self.bindings if not b['is_input']]
        outputs = []
        for output in output_bindings:
            cuda.memcpy_dtoh_async(output['host'], output['device'], self.stream)
            outputs.append(output['host'].reshape(output['shape']))
        
        self.stream.synchronize()
        
        return outputs[0] if len(outputs) == 1 else outputs
    
    def warmup(self, iterations: int = 10) -> float:
        """Warmup TensorRT model"""
        if self.context is None:
            raise RuntimeError("Model not loaded")
        
        # Create dummy input
        input_binding = next(b for b in self.bindings if b['is_input'])
        dummy_input = np.random.randn(*input_binding['shape']).astype(input_binding['dtype'])
        
        # Warmup
        for _ in range(iterations):
            self.predict(dummy_input)
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.time()
            self.predict(dummy_input)
            times.append(time.time() - start)
        
        avg_time = np.mean(times)
        logger.info(f"TensorRT warmup complete. Avg inference: {avg_time*1000:.2f}ms")
        return avg_time


class OpenVINOBackend(BackendBase):
    """OpenVINO backend for Intel hardware"""
    
    def __init__(self, model_path: str, device: str = 'CPU', **kwargs):
        super().__init__(model_path, device, **kwargs)
        self.core = None
        self.compiled_model = None
        self.infer_request = None
        self.device = device.upper()
        
    def load(self) -> None:
        """Load OpenVINO model"""
        if not BACKENDS_AVAILABLE['openvino']:
            raise ImportError("OpenVINO not available")
        
        try:
            self.core = Core()
            
            # Read model
            model = self.core.read_model(str(self.model_path))
            
            # Compile model for target device
            self.compiled_model = self.core.compile_model(model, self.device)
            self.infer_request = self.compiled_model.create_infer_request()
            
            # Get input/output shapes
            input_layer = self.compiled_model.input(0)
            output_layer = self.compiled_model.output(0)
            
            self.input_shape = tuple(input_layer.shape)
            self.output_shape = tuple(output_layer.shape)
            
            logger.info(f"OpenVINO model loaded: {self.model_path}")
            logger.info(f"Target device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load OpenVINO model: {e}")
            raise
    
    def predict(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Run OpenVINO inference"""
        if self.infer_request is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Prepare input
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        # Ensure correct layout (NCHW)
        if len(image.shape) == 3:
            image = np.expand_dims(image, 0)
        
        # Run inference
        input_tensor = self.infer_request.get_input_tensor()
        input_tensor.data[:] = image
        
        self.infer_request.infer()
        
        # Get output
        output_tensor = self.infer_request.get_output_tensor()
        return output_tensor.data
    
    def warmup(self, iterations: int = 10) -> float:
        """Warmup OpenVINO model"""
        if self.infer_request is None:
            raise RuntimeError("Model not loaded")
        
        # Create dummy input
        dummy_input = np.random.randn(*self.input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(iterations):
            self.predict(dummy_input)
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.time()
            self.predict(dummy_input)
            times.append(time.time() - start)
        
        avg_time = np.mean(times)
        logger.info(f"OpenVINO warmup complete. Avg inference: {avg_time*1000:.2f}ms")
        return avg_time


class CoreMLBackend(BackendBase):
    """CoreML backend for Apple devices"""
    
    def __init__(self, model_path: str, device: str = 'cpu', **kwargs):
        super().__init__(model_path, device, **kwargs)
        self.model = None
        
    def load(self) -> None:
        """Load CoreML model"""
        if not BACKENDS_AVAILABLE['coreml']:
            raise ImportError("CoreML tools not available")
        
        try:
            self.model = ct.models.MLModel(str(self.model_path))
            
            # Get input shape from model spec
            input_desc = self.model.get_spec().description.input[0]
            if input_desc.type.WhichOneof('Type') == 'imageType':
                shape = (
                    1,
                    input_desc.type.imageType.colorSpace,
                    input_desc.type.imageType.height,
                    input_desc.type.imageType.width
                )
                self.input_shape = shape
            
            logger.info(f"CoreML model loaded: {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load CoreML model: {e}")
            raise
    
    def predict(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Run CoreML inference"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Convert to CoreML format
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        # CoreML expects channel-last format
        if len(image.shape) == 4:  # Batch dimension
            image = np.transpose(image, (0, 2, 3, 1))
        elif len(image.shape) == 3:
            image = np.transpose(image, (1, 2, 0))
        
        # Run prediction
        prediction = self.model.predict({'input': image})
        
        # Get output
        output_key = list(prediction.keys())[0]
        return prediction[output_key]
    
    def warmup(self, iterations: int = 10) -> float:
        """Warmup CoreML model"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Create dummy input (CoreML expects channel-last)
        dummy_input = np.random.randn(*self.input_shape).astype(np.float32)
        if len(dummy_input.shape) == 4:
            dummy_input = np.transpose(dummy_input, (0, 2, 3, 1))
        else:
            dummy_input = np.transpose(dummy_input, (1, 2, 0))
        
        # Warmup
        for _ in range(iterations):
            _ = self.model.predict({'input': dummy_input})
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.time()
            _ = self.model.predict({'input': dummy_input})
            times.append(time.time() - start)
        
        avg_time = np.mean(times)
        logger.info(f"CoreML warmup complete. Avg inference: {avg_time*1000:.2f}ms")
        return avg_time


class TFLiteBackend(BackendBase):
    """TensorFlow Lite backend"""
    
    def __init__(self, model_path: str, device: str = 'cpu', **kwargs):
        super().__init__(model_path, device, **kwargs)
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        
    def load(self) -> None:
        """Load TFLite model"""
        if not BACKENDS_AVAILABLE['tflite']:
            raise ImportError("TensorFlow not available")
        
        try:
            # Load TFLite model
            self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
            self.interpreter.allocate_tensors()
            
            # Get input/output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Set input shape
            self.input_shape = self.input_details[0]['shape']
            self.output_shape = self.output_details[0]['shape']
            
            logger.info(f"TFLite model loaded: {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load TFLite model: {e}")
            raise
    
    def predict(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Run TFLite inference"""
        if self.interpreter is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Prepare input
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        # Ensure correct dtype
        input_dtype = self.input_details[0]['dtype']
        if image.dtype != input_dtype:
            image = image.astype(input_dtype)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], image)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output
    
    def warmup(self, iterations: int = 10) -> float:
        """Warmup TFLite model"""
        if self.interpreter is None:
            raise RuntimeError("Model not loaded")
        
        # Create dummy input
        dummy_input = np.random.randn(*self.input_shape).astype(
            self.input_details[0]['dtype']
        )
        
        # Warmup
        for _ in range(iterations):
            self.predict(dummy_input)
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.time()
            self.predict(dummy_input)
            times.append(time.time() - start)
        
        avg_time = np.mean(times)
        logger.info(f"TFLite warmup complete. Avg inference: {avg_time*1000:.2f}ms")
        return avg_time


class BackendManager:
    """
    Unified Multi-Backend Inference Engine
    Automatically selects and manages inference backends
    """
    
    # Backend priority by platform and performance
    BACKEND_PRIORITY = {
        'linux': ['tensorrt', 'openvino', 'onnxruntime', 'pytorch'],
        'windows': ['tensorrt', 'onnxruntime', 'openvino', 'pytorch'],
        'darwin': ['coreml', 'onnxruntime', 'pytorch'],
        'default': ['onnxruntime', 'pytorch']
    }
    
    # File extension to backend mapping
    EXTENSION_MAP = {
        '.engine': 'tensorrt',
        '.trt': 'tensorrt',
        '.onnx': 'onnxruntime',
        '.xml': 'openvino',
        '.bin': 'openvino',
        '.mlmodel': 'coreml',
        '.mlpackage': 'coreml',
        '.tflite': 'tflite',
        '.pt': 'pytorch',
        '.pth': 'pytorch',
        '.torchscript': 'pytorch'
    }
    
    def __init__(self, 
                 model_path: str, 
                 device: str = None,
                 backend_priority: List[str] = None,
                 enable_fallback: bool = True,
                 warmup_iterations: int = 10,
                 **kwargs):
        """
        Initialize Backend Manager
        
        Args:
            model_path: Path to model file
            device: Device to run on ('cpu', 'cuda', etc.)
            backend_priority: Custom backend priority list
            enable_fallback: Enable fallback to next available backend
            warmup_iterations: Number of warmup iterations for benchmarking
            **kwargs: Additional backend-specific arguments
        """
        self.model_path = Path(model_path)
        self.enable_fallback = enable_fallback
        self.warmup_iterations = warmup_iterations
        self.kwargs = kwargs
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Set backend priority
        if backend_priority is None:
            system = platform.system().lower()
            self.backend_priority = self.BACKEND_PRIORITY.get(
                system, self.BACKEND_PRIORITY['default']
            )
        else:
            self.backend_priority = backend_priority
        
        # Initialize state
        self.current_backend = None
        self.available_backends = self._detect_available_backends()
        self.backend_performance = {}
        
        # Load model
        self._load_model()
    
    def _detect_available_backends(self) -> Dict[str, bool]:
        """Detect available backends on current system"""
        available = {}
        for backend_name, is_available in BACKENDS_AVAILABLE.items():
            if is_available:
                try:
                    # Additional check for GPU availability
                    if backend_name == 'tensorrt' and self.device == 'cuda':
                        if not torch.cuda.is_available():
                            available[backend_name] = False
                            continue
                    available[backend_name] = True
                except:
                    available[backend_name] = False
            else:
                available[backend_name] = False
        
        logger.info(f"Available backends: {[k for k, v in available.items() if v]}")
        return available
    
    def _get_backend_for_model(self, model_path: Path) -> List[str]:
        """Determine which backends can load this model format"""
        ext = model_path.suffix.lower()
        primary_backend = self.EXTENSION_MAP.get(ext)
        
        if primary_backend:
            return [primary_backend] + [b for b in self.backend_priority if b != primary_backend]
        else:
            return self.backend_priority
    
    def _create_backend(self, backend_name: str) -> BackendBase:
        """Create backend instance"""
        backend_classes = {
            'pytorch': PyTorchBackend,
            'onnxruntime': ONNXRuntimeBackend,
            'tensorrt': TensorRTBackend,
            'openvino': OpenVINOBackend,
            'coreml': CoreMLBackend,
            'tflite': TFLiteBackend
        }
        
        if backend_name not in backend_classes:
            raise ValueError(f"Unknown backend: {backend_name}")
        
        # Adjust device for specific backends
        device = self.device
        if backend_name == 'openvino':
            device = 'GPU' if self.device == 'cuda' else 'CPU'
        
        return backend_classes[backend_name](
            model_path=str(self.model_path),
            device=device,
            **self.kwargs
        )
    
    def _load_model(self) -> None:
        """Load model with fallback chain"""
        model_backends = self._get_backend_for_model(self.model_path)
        
        for backend_name in model_backends:
            if not self.available_backends.get(backend_name, False):
                logger.debug(f"Backend {backend_name} not available, skipping")
                continue
            
            try:
                logger.info(f"Attempting to load model with {backend_name} backend")
                backend = self._create_backend(backend_name)
                backend.load()
                
                # Warmup and benchmark
                if self.warmup_iterations > 0:
                    avg_time = backend.warmup(self.warmup_iterations)
                    self.backend_performance[backend_name] = avg_time
                
                self.current_backend = backend
                logger.info(f"Successfully loaded model with {backend_name} backend")
                return
                
            except Exception as e:
                logger.warning(f"Failed to load with {backend_name}: {e}")
                if not self.enable_fallback:
                    raise RuntimeError(f"Failed to load model with {backend_name}: {e}")
                continue
        
        raise RuntimeError(
            f"Failed to load model with any available backend. "
            f"Available: {[k for k, v in self.available_backends.items() if v]}"
        )
    
    def predict(self, image: Union[np.ndarray, torch.Tensor], **kwargs) -> np.ndarray:
        """
        Run inference using current backend
        
        Args:
            image: Input image (numpy array or torch tensor)
            **kwargs: Additional backend-specific arguments
            
        Returns:
            Model output as numpy array
        """
        if self.current_backend is None:
            raise RuntimeError("No backend loaded")
        
        # Convert to numpy if needed
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        # Ensure batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, 0)
        
        return self.current_backend.predict(image, **kwargs)
    
    def benchmark(self, iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark all available backends
        
        Args:
            iterations: Number of benchmark iterations
            
        Returns:
            Dictionary of backend names to average inference times
        """
        results = {}
        
        for backend_name in self.backend_priority:
            if not self.available_backends.get(backend_name, False):
                continue
            
            try:
                backend = self._create_backend(backend_name)
                backend.load()
                avg_time = backend.warmup(iterations)
                results[backend_name] = avg_time
                logger.info(f"{backend_name}: {avg_time*1000:.2f}ms")
            except Exception as e:
                logger.warning(f"Failed to benchmark {backend_name}: {e}")
                continue
        
        self.backend_performance = results
        return results
    
    def get_best_backend(self) -> str:
        """Get name of best performing backend"""
        if not self.backend_performance:
            self.benchmark()
        
        if not self.backend_performance:
            return None
        
        return min(self.backend_performance, key=self.backend_performance.get)
    
    def switch_backend(self, backend_name: str) -> None:
        """
        Switch to a different backend
        
        Args:
            backend_name: Name of backend to switch to
        """
        if backend_name not in self.available_backends:
            raise ValueError(f"Backend {backend_name} not available")
        
        if not self.available_backends[backend_name]:
            raise RuntimeError(f"Backend {backend_name} is not functional")
        
        try:
            new_backend = self._create_backend(backend_name)
            new_backend.load()
            
            if self.warmup_iterations > 0:
                new_backend.warmup(self.warmup_iterations)
            
            self.current_backend = new_backend
            logger.info(f"Switched to {backend_name} backend")
            
        except Exception as e:
            raise RuntimeError(f"Failed to switch to {backend_name}: {e}")
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about current backend"""
        if self.current_backend is None:
            return {}
        
        return {
            'backend': self.current_backend.backend_name,
            'device': self.current_backend.device,
            'input_shape': self.current_backend.get_input_shape(),
            'output_shape': self.current_backend.get_output_shape(),
            'model_path': str(self.model_path),
            'performance': self.backend_performance.get(
                self.current_backend.backend_name.lower(), None
            )
        }
    
    def get_available_backends(self) -> List[str]:
        """Get list of available backends"""
        return [k for k, v in self.available_backends.items() if v]
    
    def get_performance_report(self) -> str:
        """Generate performance report"""
        if not self.backend_performance:
            self.benchmark()
        
        report = ["Backend Performance Report", "=" * 40]
        
        # Sort by performance
        sorted_backends = sorted(
            self.backend_performance.items(),
            key=lambda x: x[1]
        )
        
        for backend, time_ms in sorted_backends:
            report.append(f"{backend:15} {time_ms*1000:8.2f} ms")
        
        if self.current_backend:
            report.append("")
            report.append(f"Current backend: {self.current_backend.backend_name}")
            report.append(f"Device: {self.current_backend.device}")
        
        return "\n".join(report)


def load_model(model_path: str, 
               device: str = None,
               backend: str = None,
               **kwargs) -> BackendManager:
    """
    Convenience function to load model with automatic backend selection
    
    Args:
        model_path: Path to model file
        device: Device to run on
        backend: Specific backend to use (optional)
        **kwargs: Additional arguments
        
    Returns:
        BackendManager instance
    """
    if backend:
        # Use specific backend
        manager = BackendManager(
            model_path=model_path,
            device=device,
            backend_priority=[backend],
            enable_fallback=False,
            **kwargs
        )
    else:
        # Auto-select best backend
        manager = BackendManager(
            model_path=model_path,
            device=device,
            **kwargs
        )
    
    return manager


# Integration with existing YOLOv5 codebase
def create_inference_engine(model_path: str, 
                           device: str = None,
                           **kwargs) -> BackendManager:
    """
    Create inference engine compatible with YOLOv5
    
    Args:
        model_path: Path to model
        device: Device for inference
        **kwargs: Additional arguments
        
    Returns:
        BackendManager instance
    """
    return load_model(model_path, device, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv5 Multi-Backend Inference')
    parser.add_argument('--model', type=str, required=True, help='Model path')
    parser.add_argument('--device', type=str, default=None, help='Device (cpu/cuda)')
    parser.add_argument('--backend', type=str, default=None, help='Specific backend')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup iterations')
    
    args = parser.parse_args()
    
    try:
        # Load model
        engine = load_model(
            model_path=args.model,
            device=args.device,
            backend=args.backend,
            warmup_iterations=args.warmup
        )
        
        # Print info
        print(f"Loaded model: {args.model}")
        print(f"Backend: {engine.current_backend.backend_name}")
        print(f"Device: {engine.current_backend.device}")
        print(f"Input shape: {engine.current_backend.get_input_shape()}")
        
        # Run benchmark if requested
        if args.benchmark:
            print("\nRunning benchmark...")
            results = engine.benchmark(iterations=50)
            print("\n" + engine.get_performance_report())
        
        # Example inference
        if engine.current_backend.get_input_shape():
            dummy_input = np.random.randn(
                *engine.current_backend.get_input_shape()
            ).astype(np.float32)
            
            output = engine.predict(dummy_input)
            print(f"\nOutput shape: {output.shape if isinstance(output, np.ndarray) else [o.shape for o in output]}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()