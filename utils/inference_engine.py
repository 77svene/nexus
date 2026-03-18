# utils/inference_engine.py

import os
import time
import platform
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union, Any
import numpy as np
import torch

# Configure logging
logger = logging.getLogger(__name__)

# Backend availability flags
_TENSORRT_AVAILABLE = False
_OPENVINO_AVAILABLE = False
_ONNXRUNTIME_AVAILABLE = False
_COREML_AVAILABLE = False
_TFLITE_AVAILABLE = False

# Try importing backends
try:
    import tensorrt as trt
    _TENSORRT_AVAILABLE = True
except ImportError:
    pass

try:
    from openvino.inference_engine import IECore
    _OPENVINO_AVAILABLE = True
except ImportError:
    pass

try:
    import onnxruntime as ort
    _ONNXRUNTIME_AVAILABLE = True
except ImportError:
    pass

try:
    import coremltools as ct
    _COREML_AVAILABLE = True
except ImportError:
    pass

try:
    import tflite_runtime.interpreter as tflite
    _TFLITE_AVAILABLE = True
except ImportError:
    try:
        import tensorflow as tf
        _TFLITE_AVAILABLE = True
    except ImportError:
        pass


class InferenceBackend:
    """Base class for inference backends."""
    
    def __init__(self, model_path: str, device: str = 'cpu', fp16: bool = False):
        self.model_path = Path(model_path)
        self.device = device
        self.fp16 = fp16
        self.model = None
        self.input_name = None
        self.output_names = None
        self.input_shape = None
        
    def load(self) -> None:
        """Load the model into memory."""
        raise NotImplementedError
        
    def infer(self, img: np.ndarray) -> np.ndarray:
        """Run inference on input image."""
        raise NotImplementedError
        
    def warmup(self, iterations: int = 10) -> float:
        """Warm up the model and return average inference time."""
        if self.input_shape is None:
            raise ValueError("Model not loaded or input shape unknown")
            
        # Create dummy input
        dummy_input = np.random.randn(*self.input_shape).astype(np.float32)
        
        # Warmup runs
        for _ in range(iterations):
            self.infer(dummy_input)
            
        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.time()
            self.infer(dummy_input)
            times.append(time.time() - start)
            
        return np.mean(times)
    
    def get_backend_name(self) -> str:
        """Return the name of the backend."""
        return self.__class__.__name__.replace('Backend', '').lower()


class PyTorchBackend(InferenceBackend):
    """PyTorch backend (fallback)."""
    
    def __init__(self, model_path: str, device: str = 'cpu', fp16: bool = False):
        super().__init__(model_path, device, fp16)
        self.model = None
        
    def load(self) -> None:
        """Load PyTorch model."""
        try:
            from models.experimental import attempt_load
            self.model = attempt_load(self.model_path, device=self.device)
            self.model.eval()
            
            # Get input shape from model
            if hasattr(self.model, 'stride'):
                stride = int(self.model.stride.max())
                self.input_shape = (1, 3, 640, 640)  # Default YOLOv5 input
            else:
                self.input_shape = (1, 3, 640, 640)
                
            # Convert to FP16 if requested
            if self.fp16 and self.device != 'cpu':
                self.model = self.model.half()
                
            logger.info(f"Loaded PyTorch model from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            raise
            
    def infer(self, img: np.ndarray) -> np.ndarray:
        """Run PyTorch inference."""
        import torch
        
        # Convert to tensor
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).to(self.device)
            
        # Add batch dimension if needed
        if img.dim() == 3:
            img = img.unsqueeze(0)
            
        # Convert to FP16 if model is FP16
        if self.fp16 and self.device != 'cpu':
            img = img.half()
            
        # Run inference
        with torch.no_grad():
            pred = self.model(img)[0]
            
        return pred.cpu().numpy()


class TensorRTBackend(InferenceBackend):
    """TensorRT backend for NVIDIA GPUs."""
    
    def __init__(self, model_path: str, device: str = 'cuda', fp16: bool = False):
        super().__init__(model_path, device, fp16)
        self.engine = None
        self.context = None
        self.stream = None
        self.bindings = []
        
    def load(self) -> None:
        """Load TensorRT engine."""
        if not _TENSORRT_AVAILABLE:
            raise ImportError("TensorRT not available")
            
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            # Load engine
            logger.info(f"Loading TensorRT engine from {self.model_path}")
            with open(self.model_path, 'rb') as f:
                runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
                self.engine = runtime.deserialize_cuda_engine(f.read())
                
            self.context = self.engine.create_execution_context()
            self.stream = cuda.Stream()
            
            # Get input/output info
            for binding in self.engine:
                binding_idx = self.engine.get_binding_index(binding)
                shape = self.engine.get_binding_shape(binding_idx)
                dtype = trt.nptype(self.engine.get_binding_dtype(binding_idx))
                
                if self.engine.binding_is_input(binding_idx):
                    self.input_name = binding
                    self.input_shape = shape
                else:
                    if self.output_names is None:
                        self.output_names = []
                    self.output_names.append(binding)
                    
            logger.info(f"TensorRT engine loaded with input shape {self.input_shape}")
            
        except Exception as e:
            logger.error(f"Failed to load TensorRT engine: {e}")
            raise
            
    def infer(self, img: np.ndarray) -> np.ndarray:
        """Run TensorRT inference."""
        import pycuda.driver as cuda
        
        # Allocate device memory
        d_input = cuda.mem_alloc(img.nbytes)
        outputs = []
        d_outputs = []
        
        for output_name in self.output_names:
            binding_idx = self.engine.get_binding_index(output_name)
            shape = self.engine.get_binding_shape(binding_idx)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding_idx))
            output = np.empty(shape, dtype)
            d_output = cuda.mem_alloc(output.nbytes)
            outputs.append(output)
            d_outputs.append(d_output)
            
        # Transfer input to device
        cuda.memcpy_htod_async(d_input, img, self.stream)
        
        # Bindings
        bindings = [int(d_input)] + [int(d_output) for d_output in d_outputs]
        
        # Run inference
        self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)
        
        # Transfer predictions back
        for output, d_output in zip(outputs, d_outputs):
            cuda.memcpy_dtoh_async(output, d_output, self.stream)
            
        # Synchronize
        self.stream.synchronize()
        
        # Free device memory
        d_input.free()
        for d_output in d_outputs:
            d_output.free()
            
        return outputs[0] if len(outputs) == 1 else outputs


class OpenVINOBackend(InferenceBackend):
    """OpenVINO backend for Intel hardware."""
    
    def __init__(self, model_path: str, device: str = 'CPU', fp16: bool = False):
        super().__init__(model_path, device, fp16)
        self.ie = None
        self.net = None
        self.exec_net = None
        
    def load(self) -> None:
        """Load OpenVINO model."""
        if not _OPENVINO_AVAILABLE:
            raise ImportError("OpenVINO not available")
            
        try:
            # Check for IR format (.xml and .bin)
            model_xml = str(self.model_path)
            model_bin = str(self.model_path.with_suffix('.bin'))
            
            if not os.path.exists(model_bin):
                raise FileNotFoundError(f"OpenVINO weights file not found: {model_bin}")
                
            self.ie = IECore()
            self.net = self.ie.read_network(model=model_xml, weights=model_bin)
            
            # Adjust for FP16 if requested
            if self.fp16 and 'GPU' in self.device:
                from openvino.inference_engine import IENetwork
                self.net = IENetwork(model=model_xml, weights=model_bin)
                self.net = self.ie.read_network(
                    model=self.net.serialize(),
                    weights=model_bin,
                    init_from_buffer=True
                )
                
            # Load network
            self.exec_net = self.ie.load_network(network=self.net, device_name=self.device)
            
            # Get input/output info
            input_blob = next(iter(self.net.input_info))
            self.input_name = input_blob
            self.input_shape = self.net.input_info[input_blob].input_data.shape
            
            output_blob = next(iter(self.net.outputs))
            self.output_names = [output_blob]
            
            logger.info(f"OpenVINO model loaded on {self.device} with input shape {self.input_shape}")
            
        except Exception as e:
            logger.error(f"Failed to load OpenVINO model: {e}")
            raise
            
    def infer(self, img: np.ndarray) -> np.ndarray:
        """Run OpenVINO inference."""
        # Prepare input
        input_data = {self.input_name: img}
        
        # Run inference
        result = self.exec_net.infer(inputs=input_data)
        
        # Get output
        output_name = self.output_names[0]
        return result[output_name]


class ONNXRuntimeBackend(InferenceBackend):
    """ONNX Runtime backend."""
    
    def __init__(self, model_path: str, device: str = 'cpu', fp16: bool = False):
        super().__init__(model_path, device, fp16)
        self.session = None
        
    def load(self) -> None:
        """Load ONNX model."""
        if not _ONNXRUNTIME_AVAILABLE:
            raise ImportError("ONNX Runtime not available")
            
        try:
            # Set providers based on device
            providers = ['CPUExecutionProvider']
            if self.device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                
            # Create session
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=providers
            )
            
            # Get input/output info
            input_meta = self.session.get_inputs()[0]
            self.input_name = input_meta.name
            self.input_shape = input_meta.shape
            
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            logger.info(f"ONNX Runtime model loaded with providers {providers}")
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise
            
    def infer(self, img: np.ndarray) -> np.ndarray:
        """Run ONNX Runtime inference."""
        # Prepare input
        input_data = {self.input_name: img}
        
        # Run inference
        result = self.session.run(self.output_names, input_data)
        
        return result[0] if len(result) == 1 else result


class CoreMLBackend(InferenceBackend):
    """CoreML backend for Apple devices."""
    
    def __init__(self, model_path: str, device: str = 'cpu', fp16: bool = False):
        super().__init__(model_path, device, fp16)
        self.model = None
        
    def load(self) -> None:
        """Load CoreML model."""
        if not _COREML_AVAILABLE:
            raise ImportError("CoreML tools not available")
            
        try:
            # Load model
            self.model = ct.models.MLModel(str(self.model_path))
            
            # Get input description
            input_desc = self.model.get_spec().description.input[0]
            self.input_name = input_desc.name
            
            # Parse input shape from CoreML spec
            if input_desc.type.WhichOneof('Type') == 'imageType':
                image_type = input_desc.type.imageType
                self.input_shape = (1, 3, image_type.height, image_type.width)
            else:
                # Default shape for YOLOv5
                self.input_shape = (1, 3, 640, 640)
                
            self.output_names = [output.name for output in self.model.get_spec().description.output]
            
            logger.info(f"CoreML model loaded with input shape {self.input_shape}")
            
        except Exception as e:
            logger.error(f"Failed to load CoreML model: {e}")
            raise
            
    def infer(self, img: np.ndarray) -> np.ndarray:
        """Run CoreML inference."""
        import coremltools as ct
        
        # Convert input to CoreML format
        if isinstance(img, np.ndarray):
            # CoreML expects channel-first format
            if img.shape[1] == 3:  # Already CHW
                input_dict = {self.input_name: img}
            else:  # Convert from HWC to CHW
                input_dict = {self.input_name: np.transpose(img, (0, 3, 1, 2))}
        else:
            input_dict = {self.input_name: img}
            
        # Run prediction
        prediction = self.model.predict(input_dict)
        
        # Get output
        output_name = self.output_names[0]
        return prediction[output_name]


class TFLiteBackend(InferenceBackend):
    """TensorFlow Lite backend."""
    
    def __init__(self, model_path: str, device: str = 'cpu', fp16: bool = False):
        super().__init__(model_path, device, fp16)
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        
    def load(self) -> None:
        """Load TFLite model."""
        if not _TFLITE_AVAILABLE:
            raise ImportError("TensorFlow Lite not available")
            
        try:
            # Try tflite_runtime first, then tensorflow
            try:
                self.interpreter = tflite.Interpreter(model_path=str(self.model_path))
            except:
                import tensorflow as tf
                self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
                
            self.interpreter.allocate_tensors()
            
            # Get input/output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Set input info
            input_detail = self.input_details[0]
            self.input_name = input_detail['name']
            self.input_shape = input_detail['shape']
            
            self.output_names = [detail['name'] for detail in self.output_details]
            
            # Use GPU delegate if available and requested
            if self.device == 'cuda':
                try:
                    import tensorflow as tf
                    delegate = tf.lite.experimental.load_delegate('libtensorflowlite_gpu_delegate.so')
                    self.interpreter = tf.lite.Interpreter(
                        model_path=str(self.model_path),
                        experimental_delegates=[delegate]
                    )
                    self.interpreter.allocate_tensors()
                except:
                    logger.warning("GPU delegate not available, using CPU")
                    
            logger.info(f"TFLite model loaded with input shape {self.input_shape}")
            
        except Exception as e:
            logger.error(f"Failed to load TFLite model: {e}")
            raise
            
    def infer(self, img: np.ndarray) -> np.ndarray:
        """Run TFLite inference."""
        # Set input tensor
        input_detail = self.input_details[0]
        self.interpreter.set_tensor(input_detail['index'], img)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output tensor
        output_detail = self.output_details[0]
        return self.interpreter.get_tensor(output_detail['index'])


class InferenceEngine:
    """Unified inference engine with automatic backend selection."""
    
    # Backend priority (higher = better)
    BACKEND_PRIORITY = {
        'tensorrt': 100,
        'openvino': 90,
        'coreml': 80,
        'onnxruntime': 70,
        'tflite': 60,
        'pytorch': 10
    }
    
    # Model format to backend mapping
    FORMAT_BACKEND_MAP = {
        '.engine': 'tensorrt',
        '.trt': 'tensorrt',
        '.xml': 'openvino',
        '.onnx': 'onnxruntime',
        '.mlmodel': 'coreml',
        '.mlpackage': 'coreml',
        '.tflite': 'tflite',
        '.pt': 'pytorch',
        '.pth': 'pytorch',
        '.torchscript': 'pytorch'
    }
    
    def __init__(
        self,
        model_path: str,
        device: str = '',
        fp16: bool = False,
        backend: Optional[str] = None,
        fallback: bool = True,
        benchmark: bool = False
    ):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to model file
            device: Device to use ('cpu', 'cuda', 'auto')
            fp16: Use FP16 precision if available
            backend: Force specific backend (None for auto-selection)
            fallback: Enable fallback to other backends if primary fails
            benchmark: Run performance benchmark on initialization
        """
        self.model_path = Path(model_path)
        self.fp16 = fp16
        self.fallback = fallback
        self.benchmark_results = {}
        
        # Auto-detect device
        if device == 'auto' or device == '':
            self.device = self._auto_detect_device()
        else:
            self.device = device
            
        # Determine backend
        if backend:
            self.backend_name = backend.lower()
        else:
            self.backend_name = self._select_backend()
            
        # Initialize backend
        self.backend = self._create_backend(self.backend_name)
        
        # Load model
        self._load_model()
        
        # Benchmark if requested
        if benchmark:
            self._run_benchmark()
            
    def _auto_detect_device(self) -> str:
        """Auto-detect the best available device."""
        if torch.cuda.is_available():
            return 'cuda'
        elif platform.system() == 'Darwin' and platform.machine() == 'arm64':
            # Apple Silicon
            return 'cpu'  # CoreML will handle device selection
        else:
            return 'cpu'
            
    def _select_backend(self) -> str:
        """Select the best available backend based on model format and platform."""
        model_ext = self.model_path.suffix.lower()
        
        # Check if we have a specific format mapping
        if model_ext in self.FORMAT_BACKEND_MAP:
            backend_name = self.FORMAT_BACKEND_MAP[model_ext]
            if self._is_backend_available(backend_name):
                return backend_name
                
        # Auto-detect based on platform and available backends
        available_backends = self._get_available_backends()
        
        # Platform-specific preferences
        system = platform.system()
        machine = platform.machine()
        
        if system == 'Darwin':  # macOS
            if machine == 'arm64' and 'coreml' in available_backends:
                return 'coreml'
            elif 'onnxruntime' in available_backends:
                return 'onnxruntime'
                
        elif system == 'Linux':
            if 'tensorrt' in available_backends and self.device == 'cuda':
                return 'tensorrt'
            elif 'openvino' in available_backends and 'intel' in platform.processor().lower():
                return 'openvino'
            elif 'onnxruntime' in available_backends:
                return 'onnxruntime'
                
        elif system == 'Windows':
            if 'tensorrt' in available_backends and self.device == 'cuda':
                return 'tensorrt'
            elif 'onnxruntime' in available_backends:
                return 'onnxruntime'
                
        # Default fallback
        if 'pytorch' in available_backends:
            return 'pytorch'
            
        raise RuntimeError("No suitable inference backend available")
        
    def _get_available_backends(self) -> List[str]:
        """Get list of available backends."""
        backends = []
        
        if _TENSORRT_AVAILABLE:
            backends.append('tensorrt')
        if _OPENVINO_AVAILABLE:
            backends.append('openvino')
        if _ONNXRUNTIME_AVAILABLE:
            backends.append('onnxruntime')
        if _COREML_AVAILABLE:
            backends.append('coreml')
        if _TFLITE_AVAILABLE:
            backends.append('tflite')
            
        # PyTorch is always available
        backends.append('pytorch')
        
        return backends
        
    def _is_backend_available(self, backend_name: str) -> bool:
        """Check if a specific backend is available."""
        backend_map = {
            'tensorrt': _TENSORRT_AVAILABLE,
            'openvino': _OPENVINO_AVAILABLE,
            'onnxruntime': _ONNXRUNTIME_AVAILABLE,
            'coreml': _COREML_AVAILABLE,
            'tflite': _TFLITE_AVAILABLE,
            'pytorch': True
        }
        return backend_map.get(backend_name, False)
        
    def _create_backend(self, backend_name: str) -> InferenceBackend:
        """Create backend instance."""
        backend_classes = {
            'tensorrt': TensorRTBackend,
            'openvino': OpenVINOBackend,
            'onnxruntime': ONNXRuntimeBackend,
            'coreml': CoreMLBackend,
            'tflite': TFLiteBackend,
            'pytorch': PyTorchBackend
        }
        
        if backend_name not in backend_classes:
            raise ValueError(f"Unknown backend: {backend_name}")
            
        backend_class = backend_classes[backend_name]
        
        # Adjust device for specific backends
        device = self.device
        if backend_name == 'openvino':
            if device == 'cuda':
                device = 'GPU'
            else:
                device = 'CPU'
                
        return backend_class(str(self.model_path), device, self.fp16)
        
    def _load_model(self) -> None:
        """Load model with fallback support."""
        backends_to_try = [self.backend_name]
        
        # Add fallback backends if enabled
        if self.fallback:
            available = self._get_available_backends()
            # Sort by priority
            available.sort(key=lambda x: self.BACKEND_PRIORITY.get(x, 0), reverse=True)
            # Remove primary backend
            if self.backend_name in available:
                available.remove(self.backend_name)
            backends_to_try.extend(available)
            
        last_error = None
        for backend_name in backends_to_try:
            try:
                logger.info(f"Attempting to load model with {backend_name} backend")
                self.backend = self._create_backend(backend_name)
                self.backend.load()
                self.backend_name = backend_name
                logger.info(f"Successfully loaded model with {backend_name} backend")
                return
            except Exception as e:
                last_error = e
                logger.warning(f"Failed to load with {backend_name}: {e}")
                continue
                
        raise RuntimeError(f"Failed to load model with any backend. Last error: {last_error}")
        
    def _run_benchmark(self, iterations: int = 50, warmup: int = 10) -> Dict[str, float]:
        """Run performance benchmark."""
        logger.info("Running performance benchmark...")
        
        # Warmup
        logger.info(f"Warming up for {warmup} iterations...")
        warmup_time = self.backend.warmup(warmup)
        
        # Benchmark
        logger.info(f"Benchmarking for {iterations} iterations...")
        times = []
        dummy_input = np.random.randn(*self.backend.input_shape).astype(np.float32)
        
        for i in range(iterations):
            start = time.time()
            self.backend.infer(dummy_input)
            elapsed = time.time() - start
            times.append(elapsed)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Completed {i + 1}/{iterations} iterations")
                
        # Calculate statistics
        self.benchmark_results = {
            'backend': self.backend_name,
            'device': self.device,
            'fp16': self.fp16,
            'iterations': iterations,
            'warmup_time_ms': warmup_time * 1000,
            'mean_ms': np.mean(times) * 1000,
            'std_ms': np.std(times) * 1000,
            'min_ms': np.min(times) * 1000,
            'max_ms': np.max(times) * 1000,
            'fps': 1.0 / np.mean(times)
        }
        
        logger.info(f"Benchmark results: {self.benchmark_results}")
        return self.benchmark_results
        
    def infer(self, img: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Run inference on input image.
        
        Args:
            img: Input image as numpy array or torch tensor
            
        Returns:
            Model predictions as numpy array
        """
        # Convert torch tensor to numpy if needed
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
            
        # Ensure correct shape
        if len(img.shape) == 3:
            img = np.expand_dims(img, 0)
            
        # Ensure correct dtype
        if img.dtype != np.float32:
            img = img.astype(np.float32)
            
        return self.backend.infer(img)
        
    def benchmark(self, iterations: int = 50, warmup: int = 10) -> Dict[str, float]:
        """Run and return benchmark results."""
        return self._run_benchmark(iterations, warmup)
        
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the current backend."""
        return {
            'backend': self.backend_name,
            'device': self.device,
            'fp16': self.fp16,
            'input_shape': self.backend.input_shape,
            'input_name': self.backend.input_name,
            'output_names': self.backend.output_names,
            'model_path': str(self.model_path)
        }
        
    def switch_backend(self, backend_name: str) -> None:
        """Switch to a different backend."""
        if not self._is_backend_available(backend_name):
            raise RuntimeError(f"Backend {backend_name} not available")
            
        self.backend_name = backend_name
        self.backend = self._create_backend(backend_name)
        self.backend.load()
        logger.info(f"Switched to {backend_name} backend")
        
    @staticmethod
    def list_available_backends() -> Dict[str, bool]:
        """List all available backends and their status."""
        return {
            'tensorrt': _TENSORRT_AVAILABLE,
            'openvino': _OPENVINO_AVAILABLE,
            'onnxruntime': _ONNXRUNTIME_AVAILABLE,
            'coreml': _COREML_AVAILABLE,
            'tflite': _TFLITE_AVAILABLE,
            'pytorch': True
        }


def create_inference_engine(
    model_path: str,
    device: str = 'auto',
    fp16: bool = False,
    backend: Optional[str] = None,
    fallback: bool = True,
    benchmark: bool = False
) -> InferenceEngine:
    """
    Factory function to create inference engine.
    
    Args:
        model_path: Path to model file
        device: Device to use ('cpu', 'cuda', 'auto')
        fp16: Use FP16 precision if available
        backend: Force specific backend (None for auto-selection)
        fallback: Enable fallback to other backends if primary fails
        benchmark: Run performance benchmark on initialization
        
    Returns:
        Configured InferenceEngine instance
    """
    return InferenceEngine(
        model_path=model_path,
        device=device,
        fp16=fp16,
        backend=backend,
        fallback=fallback,
        benchmark=benchmark
    )


def export_to_backend(
    model: torch.nn.Module,
    backend: str,
    output_path: str,
    img_size: Tuple[int, int] = (640, 640),
    batch_size: int = 1,
    fp16: bool = False,
    device: str = 'cpu',
    **kwargs
) -> str:
    """
    Export PyTorch model to specific backend format.
    
    Args:
        model: PyTorch model
        backend: Target backend ('tensorrt', 'onnx', 'openvino', 'coreml', 'tflite')
        output_path: Output file path
        img_size: Input image size (height, width)
        batch_size: Batch size for export
        fp16: Use FP16 precision
        device: Device for export
        
    Returns:
        Path to exported model
    """
    from utils.export import export_onnx  # Import existing export functionality
    
    output_path = Path(output_path)
    backend = backend.lower()
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, 3, *img_size).to(device)
    
    if backend == 'onnx' or backend == 'onnxruntime':
        # Export to ONNX
        onnx_path = output_path.with_suffix('.onnx')
        export_onnx(model, dummy_input, onnx_path, fp16=fp16, **kwargs)
        return str(onnx_path)
        
    elif backend == 'tensorrt':
        # First export to ONNX, then convert to TensorRT
        onnx_path = output_path.with_suffix('.onnx')
        export_onnx(model, dummy_input, onnx_path, fp16=fp16, **kwargs)
        
        # Convert to TensorRT (requires trtexec or similar)
        trt_path = output_path.with_suffix('.engine')
        try:
            import subprocess
            cmd = [
                'trtexec',
                f'--onnx={onnx_path}',
                f'--saveEngine={trt_path}',
                '--fp16' if fp16 else '',
                '--workspace=4096'
            ]
            subprocess.run(cmd, check=True)
            return str(trt_path)
        except:
            logger.warning("TensorRT export failed. Please convert manually using trtexec.")
            return str(onnx_path)
            
    elif backend == 'openvino':
        # Export to ONNX first, then convert to OpenVINO
        onnx_path = output_path.with_suffix('.onnx')
        export_onnx(model, dummy_input, onnx_path, fp16=fp16, **kwargs)
        
        # Convert to OpenVINO IR
        try:
            import subprocess
            mo_cmd = [
                'mo',
                '--input_model', str(onnx_path),
                '--output_dir', str(output_path.parent),
                '--data_type', 'FP16' if fp16 else 'FP32'
            ]
            subprocess.run(mo_cmd, check=True)
            
            # Return path to .xml file
            xml_path = output_path.with_suffix('.xml')
            return str(xml_path)
        except:
            logger.warning("OpenVINO export failed. Please convert manually using Model Optimizer.")
            return str(onnx_path)
            
    elif backend == 'coreml':
        # Export to CoreML
        try:
            import coremltools as ct
            
            # Trace model
            traced_model = torch.jit.trace(model, dummy_input)
            
            # Convert to CoreML
            mlmodel = ct.convert(
                traced_model,
                inputs=[ct.TensorType(name="input", shape=dummy_input.shape)],
                convert_to="mlprogram" if fp16 else "neuralnetwork",
                compute_precision=ct.precision.FLOAT16 if fp16 else ct.precision.FLOAT32
            )
            
            # Save model
            ml_path = output_path.with_suffix('.mlmodel')
            mlmodel.save(str(ml_path))
            return str(ml_path)
            
        except ImportError:
            raise ImportError("CoreML tools required for CoreML export")
            
    elif backend == 'tflite':
        # Export to TFLite
        try:
            import tensorflow as tf
            
            # Export to SavedModel first
            saved_model_dir = output_path.parent / 'saved_model'
            torch.jit.save(torch.jit.trace(model, dummy_input), str(saved_model_dir / 'model.pt'))
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
            
            if fp16:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
                
            tflite_model = converter.convert()
            
            # Save model
            tflite_path = output_path.with_suffix('.tflite')
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
                
            return str(tflite_path)
            
        except ImportError:
            raise ImportError("TensorFlow required for TFLite export")
            
    else:
        raise ValueError(f"Unsupported backend: {backend}")


# Integration with existing YOLOv5 code
def load_model(
    weights: str,
    device: str = '',
    fp16: bool = False,
    backend: Optional[str] = None,
    **kwargs
) -> InferenceEngine:
    """
    Load model with unified inference engine.
    
    This function can be used as a drop-in replacement for attempt_load.
    
    Args:
        weights: Path to model weights
        device: Device to use
        fp16: Use FP16 precision
        backend: Force specific backend
        **kwargs: Additional arguments for InferenceEngine
        
    Returns:
        InferenceEngine instance
    """
    # Handle device
    if not device:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
    # Create inference engine
    engine = create_inference_engine(
        model_path=weights,
        device=device,
        fp16=fp16,
        backend=backend,
        **kwargs
    )
    
    return engine


# Add to existing modules
def patch_nexus_for_inference_engine():
    """
    Patch YOLOv5 modules to use the unified inference engine.
    
    This function modifies the existing modules to use InferenceEngine
    instead of the default PyTorch loading.
    """
    import sys
    
    # Store original attempt_load
    from models import experimental
    original_attempt_load = experimental.attempt_load
    
    def patched_attempt_load(weights, device=None, inplace=False, fuse=True):
        """Patched attempt_load that uses InferenceEngine."""
        # For single weight file
        if isinstance(weights, (str, Path)):
            engine = load_model(str(weights), device=device or 'auto')
            return engine.backend.model  # Return the underlying model for compatibility
            
        # For multiple weights (ensemble)
        models = []
        for w in weights if isinstance(weights, list) else [weights]:
            engine = load_model(str(w), device=device or 'auto')
            models.append(engine.backend.model)
            
        # Return first model for compatibility (ensembles not fully supported)
        return models[0] if len(models) == 1 else models
        
    # Apply patch
    experimental.attempt_load = patched_attempt_load
    logger.info("Patched YOLOv5 to use unified inference engine")


# Auto-patch if this module is imported
if os.environ.get('YOLOV5_USE_INFERENCE_ENGINE', '').lower() in ('1', 'true', 'yes'):
    try:
        patch_nexus_for_inference_engine()
    except Exception as e:
        logger.warning(f"Failed to patch YOLOv5: {e}")