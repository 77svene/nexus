# YOLOv5 Resilient Edge Deployment Toolkit
# Implements health checks, automatic restart, thermal throttling detection,
# and graceful degradation for robust edge deployments

import os
import sys
import time
import threading
import logging
import traceback
import functools
import signal
import atexit
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Union, List
from enum import Enum
from dataclasses import dataclass, field
from contextlib import contextmanager

import torch
import torch.nn as nn
import numpy as np

# Add parent directory to path for imports
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from utils.torch_utils import select_device
from utils.general import check_yaml, check_font, colorstr, increment_path
from utils.dataloaders import LoadImages, LoadStreams
from utils.plots import Annotator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('nexus_resilience.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """System health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"


class DegradationMode(Enum):
    """Graceful degradation strategies"""
    NONE = "none"
    REDUCE_PRECISION = "reduce_precision"
    REDUCE_INPUT_SIZE = "reduce_input_size"
    SWITCH_MODEL = "switch_model"
    BATCH_PROCESSING = "batch_processing"


@dataclass
class SystemMetrics:
    """System metrics for monitoring"""
    temperature: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    gpu_usage: float = 0.0
    inference_time: float = 0.0
    failure_count: int = 0
    last_failure_time: float = 0.0
    consecutive_failures: int = 0
    
    def update_temperature(self, temp: float):
        """Update temperature with validation"""
        if temp < -50 or temp > 150:  # Sanity check
            logger.warning(f"Invalid temperature reading: {temp}°C")
            return
        self.temperature = temp
    
    def update_memory(self, usage: float):
        """Update memory usage percentage"""
        self.memory_usage = min(max(usage, 0.0), 100.0)
    
    def record_failure(self):
        """Record a failure event"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.consecutive_failures += 1
    
    def reset_consecutive_failures(self):
        """Reset consecutive failure counter on success"""
        self.consecutive_failures = 0


@dataclass
class ResilienceConfig:
    """Configuration for resilience mechanisms"""
    # Health check settings
    enable_health_checks: bool = True
    health_check_interval: float = 1.0  # seconds
    max_consecutive_failures: int = 3
    
    # Thermal management
    enable_thermal_throttling: bool = True
    temperature_threshold_warning: float = 70.0  # °C
    temperature_threshold_critical: float = 85.0  # °C
    temperature_check_interval: float = 5.0  # seconds
    
    # Memory management
    enable_memory_monitoring: bool = True
    memory_threshold_warning: float = 80.0  # % usage
    memory_threshold_critical: float = 90.0  # % usage
    
    # Automatic restart
    enable_auto_restart: bool = True
    restart_delay: float = 2.0  # seconds
    max_restart_attempts: int = 3
    
    # Watchdog timer
    enable_watchdog: bool = True
    watchdog_timeout: float = 30.0  # seconds
    
    # Graceful degradation
    enable_graceful_degradation: bool = True
    degradation_modes: List[DegradationMode] = field(default_factory=lambda: [
        DegradationMode.REDUCE_PRECISION,
        DegradationMode.REDUCE_INPUT_SIZE,
        DegradationMode.SWITCH_MODEL
    ])
    
    # Model fallback settings
    fallback_model_path: Optional[str] = None
    default_input_size: int = 640
    min_input_size: int = 320
    input_size_step: int = 32
    
    # Precision settings
    available_precisions: List[str] = field(default_factory=lambda: ['fp32', 'fp16', 'int8'])
    default_precision: str = 'fp32'
    
    # Logging and monitoring
    log_metrics: bool = True
    metrics_log_interval: float = 60.0  # seconds


class SystemMonitor:
    """Monitors system resources and health metrics"""
    
    def __init__(self, config: ResilienceConfig):
        self.config = config
        self.metrics = SystemMetrics()
        self._monitoring = False
        self._monitor_thread = None
        self._callbacks = []
        
        # Try to import platform-specific monitoring
        self._has_psutil = False
        self._has_jetson = False
        self._has_raspberrypi = False
        
        try:
            import psutil
            self._has_psutil = True
            self.psutil = psutil
        except ImportError:
            logger.warning("psutil not available. System monitoring will be limited.")
        
        # Check for Jetson platform
        try:
            from jtop import jtop
            self._has_jetson = True
            self.jtop = jtop
        except ImportError:
            pass
        
        # Check for Raspberry Pi
        try:
            import gpiozero
            self._has_raspberrypi = True
        except ImportError:
            pass
    
    def start_monitoring(self):
        """Start background monitoring thread"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="SystemMonitor"
        )
        self._monitor_thread.start()
        logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            logger.info("System monitoring stopped")
    
    def register_callback(self, callback: Callable[[SystemMetrics], None]):
        """Register a callback for metrics updates"""
        self._callbacks.append(callback)
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        last_temp_check = 0
        last_memory_check = 0
        last_metrics_log = 0
        
        while self._monitoring:
            current_time = time.time()
            
            # Check temperature
            if (self.config.enable_thermal_throttling and 
                current_time - last_temp_check >= self.config.temperature_check_interval):
                self._check_temperature()
                last_temp_check = current_time
            
            # Check memory
            if (self.config.enable_memory_monitoring and 
                current_time - last_memory_check >= self.config.health_check_interval):
                self._check_memory()
                last_memory_check = current_time
            
            # Log metrics periodically
            if (self.config.log_metrics and 
                current_time - last_metrics_log >= self.config.metrics_log_interval):
                self._log_metrics()
                last_metrics_log = current_time
            
            # Notify callbacks
            for callback in self._callbacks:
                try:
                    callback(self.metrics)
                except Exception as e:
                    logger.error(f"Error in metrics callback: {e}")
            
            time.sleep(0.1)  # Small sleep to prevent CPU spinning
    
    def _check_temperature(self):
        """Check system temperature"""
        temp = None
        
        if self._has_jetson:
            try:
                with self.jtop() as jetson:
                    if jetson.ok():
                        temp = jetson.temperature
            except Exception:
                pass
        
        if temp is None and self._has_psutil:
            try:
                temps = self.psutil.sensors_temperatures()
                if temps:
                    # Get the highest temperature from any sensor
                    temp = max([temp.current for sensors in temps.values() 
                               for temp in sensors])
            except Exception:
                pass
        
        if temp is not None:
            self.metrics.update_temperature(temp)
            
            # Log warnings for high temperatures
            if temp >= self.config.temperature_threshold_critical:
                logger.critical(f"Critical temperature detected: {temp}°C")
            elif temp >= self.config.temperature_threshold_warning:
                logger.warning(f"High temperature detected: {temp}°C")
    
    def _check_memory(self):
        """Check system memory usage"""
        memory_percent = None
        
        if self._has_psutil:
            try:
                memory_percent = self.psutil.virtual_memory().percent
            except Exception:
                pass
        
        if memory_percent is not None:
            self.metrics.update_memory(memory_percent)
            
            # Log warnings for high memory usage
            if memory_percent >= self.config.memory_threshold_critical:
                logger.critical(f"Critical memory usage: {memory_percent}%")
            elif memory_percent >= self.config.memory_threshold_warning:
                logger.warning(f"High memory usage: {memory_percent}%")
    
    def _log_metrics(self):
        """Log current metrics"""
        logger.info(
            f"System Metrics - "
            f"Temp: {self.metrics.temperature:.1f}°C, "
            f"Memory: {self.metrics.memory_usage:.1f}%, "
            f"Failures: {self.metrics.failure_count}"
        )
    
    def get_health_status(self) -> HealthStatus:
        """Determine overall health status"""
        if self.metrics.consecutive_failures >= self.config.max_consecutive_failures:
            return HealthStatus.FAILED
        
        if (self.metrics.temperature >= self.config.temperature_threshold_critical or
            self.metrics.memory_usage >= self.config.memory_threshold_critical):
            return HealthStatus.CRITICAL
        
        if (self.metrics.temperature >= self.config.temperature_threshold_warning or
            self.metrics.memory_usage >= self.config.memory_threshold_warning or
            self.metrics.consecutive_failures > 0):
            return HealthStatus.DEGRADED
        
        return HealthStatus.HEALTHY


class ModelManager:
    """Manages model loading, fallback, and precision adjustments"""
    
    def __init__(self, config: ResilienceConfig, device: str = ''):
        self.config = config
        self.device = select_device(device)
        self.models = {}  # Cache for loaded models
        self.current_model_name = None
        self.current_input_size = config.default_input_size
        self.current_precision = config.default_precision
        self._model_lock = threading.RLock()
        
        # Load fallback model if specified
        if config.fallback_model_path:
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load fallback model for graceful degradation"""
        try:
            # This would be implemented based on your model loading logic
            # For now, we'll create a placeholder
            logger.info(f"Loading fallback model from {self.config.fallback_model_path}")
            # In a real implementation, you would load a smaller/faster model here
        except Exception as e:
            logger.error(f"Failed to load fallback model: {e}")
    
    def load_model(self, model_path: str, model_name: str = "primary") -> nn.Module:
        """Load a model with error handling"""
        with self._model_lock:
            if model_name in self.models:
                logger.info(f"Model '{model_name}' already loaded")
                return self.models[model_name]
            
            try:
                # This is a placeholder - replace with your actual model loading logic
                # For YOLOv5, you might use torch.hub.load or custom loading
                logger.info(f"Loading model '{model_name}' from {model_path}")
                
                # Simulate model loading (replace with actual loading)
                model = self._simulate_model_loading(model_path)
                model = model.to(self.device)
                
                # Apply current precision
                model = self._apply_precision(model, self.current_precision)
                
                self.models[model_name] = model
                self.current_model_name = model_name
                
                logger.info(f"Successfully loaded model '{model_name}'")
                return model
                
            except Exception as e:
                logger.error(f"Failed to load model '{model_name}': {e}")
                raise
    
    def _simulate_model_loading(self, model_path: str) -> nn.Module:
        """Simulate model loading (replace with actual implementation)"""
        # In a real implementation, this would load your YOLOv5 model
        # For example:
        # model = torch.hub.load('ultralytics/nexus', 'custom', path=model_path)
        
        # For demonstration, create a simple model
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 10)
        )
        return model
    
    def _apply_precision(self, model: nn.Module, precision: str) -> nn.Module:
        """Apply precision conversion to model"""
        if precision == 'fp16':
            return model.half()
        elif precision == 'int8':
            # Placeholder for int8 quantization
            # In practice, you would use torch.quantization
            logger.warning("INT8 quantization not fully implemented in this example")
            return model
        else:  # fp32
            return model.float()
    
    def get_model(self, model_name: str = None) -> Optional[nn.Module]:
        """Get a loaded model by name"""
        with self._model_lock:
            if model_name is None:
                model_name = self.current_model_name
            return self.models.get(model_name)
    
    def switch_model(self, model_name: str) -> bool:
        """Switch to a different model"""
        with self._model_lock:
            if model_name not in self.models:
                logger.error(f"Model '{model_name}' not loaded")
                return False
            
            self.current_model_name = model_name
            logger.info(f"Switched to model '{model_name}'")
            return True
    
    def adjust_precision(self, precision: str) -> bool:
        """Adjust model precision"""
        if precision not in self.config.available_precisions:
            logger.error(f"Unsupported precision: {precision}")
            return False
        
        with self._model_lock:
            model = self.get_model()
            if model is None:
                logger.error("No model loaded to adjust precision")
                return False
            
            try:
                # Re-apply precision to current model
                model = self._apply_precision(model, precision)
                self.models[self.current_model_name] = model
                self.current_precision = precision
                logger.info(f"Adjusted precision to {precision}")
                return True
            except Exception as e:
                logger.error(f"Failed to adjust precision: {e}")
                return False
    
    def adjust_input_size(self, new_size: int) -> bool:
        """Adjust input size for inference"""
        if new_size < self.config.min_input_size:
            logger.warning(f"Input size {new_size} below minimum {self.config.min_input_size}")
            return False
        
        # Round to nearest step
        new_size = max(self.config.min_input_size, 
                      (new_size // self.config.input_size_step) * self.config.input_size_step)
        
        self.current_input_size = new_size
        logger.info(f"Adjusted input size to {new_size}")
        return True
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get current model configuration"""
        return {
            'model_name': self.current_model_name,
            'input_size': self.current_input_size,
            'precision': self.current_precision,
            'device': self.device
        }


class WatchdogTimer:
    """Watchdog timer for detecting hung inferences"""
    
    def __init__(self, timeout: float, callback: Callable[[], None]):
        self.timeout = timeout
        self.callback = callback
        self._timer = None
        self._active = False
    
    def start(self):
        """Start the watchdog timer"""
        if self._active:
            self.cancel()
        
        self._active = True
        self._timer = threading.Timer(self.timeout, self._timeout_handler)
        self._timer.daemon = True
        self._timer.start()
    
    def cancel(self):
        """Cancel the watchdog timer"""
        if self._timer:
            self._timer.cancel()
            self._timer = None
        self._active = False
    
    def reset(self):
        """Reset the watchdog timer"""
        self.cancel()
        self.start()
    
    def _timeout_handler(self):
        """Handle watchdog timeout"""
        self._active = False
        logger.error(f"Watchdog timeout after {self.timeout} seconds")
        self.callback()


class ResilientInference:
    """Main class for resilient inference with health checks and recovery"""
    
    def __init__(self, config: ResilienceConfig = None, device: str = ''):
        self.config = config or ResilienceConfig()
        self.monitor = SystemMonitor(self.config)
        self.model_manager = ModelManager(self.config, device)
        self._inference_lock = threading.RLock()
        self._restart_count = 0
        self._degradation_level = 0
        self._original_settings = {}
        
        # Initialize watchdog if enabled
        self.watchdog = None
        if self.config.enable_watchdog:
            self.watchdog = WatchdogTimer(
                self.config.watchdog_timeout,
                self._handle_watchdog_timeout
            )
        
        # Register shutdown handler
        atexit.register(self.shutdown)
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        logger.info("Resilient inference system initialized")
    
    def health_check_decorator(self, func: Callable) -> Callable:
        """Decorator for adding health checks to inference functions"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self._execute_with_resilience(func, *args, **kwargs)
        return wrapper
    
    def _execute_with_resilience(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with resilience mechanisms"""
        if not self.config.enable_health_checks:
            return func(*args, **kwargs)
        
        # Check system health before inference
        health_status = self.monitor.get_health_status()
        
        if health_status == HealthStatus.FAILED:
            logger.error("System health FAILED - attempting recovery")
            self._attempt_recovery()
            # Retry after recovery attempt
            health_status = self.monitor.get_health_status()
            if health_status == HealthStatus.FAILED:
                raise RuntimeError("System health failed and recovery unsuccessful")
        
        elif health_status == HealthStatus.CRITICAL:
            logger.warning("System health CRITICAL - applying degradation")
            self._apply_degradation()
        
        # Start watchdog if enabled
        if self.watchdog:
            self.watchdog.start()
        
        try:
            with self._inference_lock:
                start_time = time.time()
                
                # Execute the inference function
                result = func(*args, **kwargs)
                
                # Record success
                inference_time = time.time() - start_time
                self.monitor.metrics.inference_time = inference_time
                self.monitor.metrics.reset_consecutive_failures()
                
                # Reset watchdog on success
                if self.watchdog:
                    self.watchdog.cancel()
                
                # Check if we can restore from degradation
                if (self._degradation_level > 0 and 
                    health_status == HealthStatus.HEALTHY):
                    self._consider_restoration()
                
                return result
                
        except Exception as e:
            # Record failure
            self.monitor.metrics.record_failure()
            logger.error(f"Inference failed: {e}")
            logger.debug(traceback.format_exc())
            
            # Cancel watchdog
            if self.watchdog:
                self.watchdog.cancel()
            
            # Attempt immediate recovery
            if self.config.enable_auto_restart:
                self._handle_inference_failure(e)
            
            # Re-raise the exception
            raise
    
    def _handle_inference_failure(self, error: Exception):
        """Handle inference failure with recovery attempts"""
        logger.warning(f"Handling inference failure: {error}")
        
        if (self.monitor.metrics.consecutive_failures >= 
            self.config.max_consecutive_failures):
            logger.error("Max consecutive failures reached - attempting full restart")
            self._attempt_recovery()
        else:
            # Try lighter recovery for single failures
            self._apply_degradation()
    
    def _attempt_recovery(self):
        """Attempt to recover from failure state"""
        if self._restart_count >= self.config.max_restart_attempts:
            logger.error("Max restart attempts reached - manual intervention required")
            return False
        
        logger.info(f"Attempting recovery (attempt {self._restart_count + 1})")
        self._restart_count += 1
        
        try:
            # Wait before restart
            time.sleep(self.config.restart_delay)
            
            # Reset degradation
            self._degradation_level = 0
            
            # Reinitialize model (simplified - in practice, reload your model)
            current_config = self.model_manager.get_current_config()
            logger.info(f"Reinitializing model with config: {current_config}")
            
            # Reset failure counters
            self.monitor.metrics.consecutive_failures = 0
            
            logger.info("Recovery successful")
            self._restart_count = 0  # Reset on successful recovery
            return True
            
        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            return False
    
    def _apply_degradation(self):
        """Apply graceful degradation strategies"""
        if not self.config.enable_graceful_degradation:
            return
        
        if self._degradation_level >= len(self.config.degradation_modes):
            logger.warning("All degradation strategies exhausted")
            return
        
        mode = self.config.degradation_modes[self._degradation_level]
        logger.info(f"Applying degradation level {self._degradation_level}: {mode.value}")
        
        try:
            if mode == DegradationMode.REDUCE_PRECISION:
                self._degrade_precision()
            elif mode == DegradationMode.REDUCE_INPUT_SIZE:
                self._degrade_input_size()
            elif mode == DegradationMode.SWITCH_MODEL:
                self._switch_to_fallback_model()
            elif mode == DegradationMode.BATCH_PROCESSING:
                self._enable_batch_processing()
            
            self._degradation_level += 1
            
        except Exception as e:
            logger.error(f"Failed to apply degradation: {e}")
    
    def _degrade_precision(self):
        """Degrade to lower precision"""
        current = self.model_manager.current_precision
        precisions = self.config.available_precisions
        
        if current in precisions:
            idx = precisions.index(current)
            if idx < len(precisions) - 1:
                new_precision = precisions[idx + 1]
                self.model_manager.adjust_precision(new_precision)
                logger.info(f"Degraded precision from {current} to {new_precision}")
    
    def _degrade_input_size(self):
        """Reduce input size for faster inference"""
        current_size = self.model_manager.current_input_size
        new_size = current_size - self.config.input_size_step
        
        if new_size >= self.config.min_input_size:
            self.model_manager.adjust_input_size(new_size)
            logger.info(f"Reduced input size from {current_size} to {new_size}")
    
    def _switch_to_fallback_model(self):
        """Switch to fallback model if available"""
        if "fallback" in self.model_manager.models:
            self.model_manager.switch_model("fallback")
            logger.info("Switched to fallback model")
        elif self.config.fallback_model_path:
            try:
                self.model_manager.load_model(
                    self.config.fallback_model_path, 
                    "fallback"
                )
                self.model_manager.switch_model("fallback")
                logger.info("Loaded and switched to fallback model")
            except Exception as e:
                logger.error(f"Failed to load fallback model: {e}")
    
    def _enable_batch_processing(self):
        """Enable batch processing mode (placeholder)"""
        # This would be implemented based on your specific use case
        logger.info("Batch processing mode enabled (implementation specific)")
    
    def _consider_restoration(self):
        """Consider restoring from degraded state"""
        if self._degradation_level == 0:
            return
        
        health_status = self.monitor.get_health_status()
        if health_status == HealthStatus.HEALTHY:
            logger.info("System healthy - considering restoration from degradation")
            # In a real implementation, you might gradually restore settings
            # For now, we'll just reset the degradation level
            self._degradation_level = max(0, self._degradation_level - 1)
    
    def _handle_watchdog_timeout(self):
        """Handle watchdog timeout"""
        logger.critical("Watchdog timeout detected - forcing restart")
        self._attempt_recovery()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'health': self.monitor.get_health_status().value,
            'metrics': {
                'temperature': self.monitor.metrics.temperature,
                'memory_usage': self.monitor.metrics.memory_usage,
                'inference_time': self.monitor.metrics.inference_time,
                'failure_count': self.monitor.metrics.failure_count,
                'consecutive_failures': self.monitor.metrics.consecutive_failures,
            },
            'model_config': self.model_manager.get_current_config(),
            'degradation_level': self._degradation_level,
            'restart_count': self._restart_count,
        }
    
    def shutdown(self):
        """Clean shutdown of resilience system"""
        logger.info("Shutting down resilience system")
        self.monitor.stop_monitoring()
        if self.watchdog:
            self.watchdog.cancel()


# Context manager for resilient inference
@contextmanager
def resilient_inference_context(config: ResilienceConfig = None, device: str = ''):
    """Context manager for resilient inference"""
    resilient = ResilientInference(config, device)
    try:
        yield resilient
    finally:
        resilient.shutdown()


# Example usage and integration with YOLOv5
def create_resilient_detector(
    model_path: str,
    config: ResilienceConfig = None,
    device: str = '',
    **kwargs
) -> Callable:
    """
    Create a resilient object detector
    
    Args:
        model_path: Path to YOLOv5 model
        config: Resilience configuration
        device: Device to run on
        **kwargs: Additional arguments for detection
    
    Returns:
        Resilient detection function
    """
    config = config or ResilienceConfig()
    resilient = ResilientInference(config, device)
    
    # Load the model
    model = resilient.model_manager.load_model(model_path, "primary")
    
    def resilient_detect(
        source: Union[str, int, List],
        img_size: int = 640,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        **detect_kwargs
    ):
        """Resilient detection function"""
        
        @resilient.health_check_decorator
        def _detect():
            # This is where you would integrate with actual YOLOv5 detection
            # For example:
            # results = model(source, size=img_size, conf_thres=conf_thres, iou_thres=iou_thres)
            
            # For demonstration, we'll simulate detection
            logger.info(f"Running detection with input size {img_size}")
            
            # Simulate inference time based on input size
            inference_time = (img_size / 640) * 0.1  # Simulated
            time.sleep(inference_time)
            
            # Simulate occasional failures for demonstration
            if np.random.random() < 0.05:  # 5% failure rate
                raise RuntimeError("Simulated inference failure")
            
            # Return simulated results
            return {
                'detections': [],
                'inference_time': inference_time,
                'input_size': img_size,
                'model_config': resilient.model_manager.get_current_config()
            }
        
        return _detect()
    
    return resilient_detect


# Utility functions for edge deployment
def check_edge_requirements():
    """Check if system meets edge deployment requirements"""
    requirements = {
        'python_version': sys.version_info >= (3, 7),
        'torch_available': 'torch' in sys.modules,
        'cuda_available': torch.cuda.is_available() if 'torch' in sys.modules else False,
    }
    
    # Check for recommended packages
    try:
        import psutil
        requirements['psutil'] = True
    except ImportError:
        requirements['psutil'] = False
    
    return requirements


def optimize_for_edge(model: nn.Module, config: ResilienceConfig = None) -> nn.Module:
    """
    Optimize model for edge deployment
    
    Args:
        model: PyTorch model
        config: Resilience configuration
    
    Returns:
        Optimized model
    """
    config = config or ResilienceConfig()
    
    # Apply optimizations based on configuration
    if config.default_precision == 'fp16':
        model = model.half()
    
    # Additional edge optimizations could include:
    # - Pruning
    # - Quantization
    # - Layer fusion
    # - Operator optimization
    
    logger.info(f"Model optimized for edge deployment with precision: {config.default_precision}")
    return model


# Example of integrating with existing YOLOv5 detect.py
def patch_detect_with_resilience():
    """
    Example function showing how to patch existing detect.py with resilience
    This would be called at the start of detect.py
    """
    # Import the original detect function
    # from detect import run as original_run
    
    # Create resilient wrapper
    config = ResilienceConfig(
        enable_health_checks=True,
        enable_thermal_throttling=True,
        enable_graceful_degradation=True
    )
    
    resilient = ResilientInference(config)
    
    def resilient_run(*args, **kwargs):
        """Resilient version of detect.run()"""
        @resilient.health_check_decorator
        def _run():
            # Call original function
            # return original_run(*args, **kwargs)
            pass
        return _run()
    
    # Replace the original function
    # detect.run = resilient_run
    
    return resilient


if __name__ == "__main__":
    # Example usage
    print("YOLOv5 Resilient Edge Deployment Toolkit")
    print("=" * 50)
    
    # Check system requirements
    requirements = check_edge_requirements()
    print("\nSystem Requirements:")
    for req, status in requirements.items():
        print(f"  {req}: {'✓' if status else '✗'}")
    
    # Create resilient detector
    config = ResilienceConfig(
        enable_health_checks=True,
        enable_watchdog=True,
        enable_graceful_degradation=True,
        temperature_threshold_warning=65.0,
        temperature_threshold_critical=75.0
    )
    
    # Example: Create a resilient detector (commented out as it requires actual model)
    # detector = create_resilient_detector(
    #     model_path='nexuss.pt',
    #     config=config,
    #     device='0'  # GPU 0
    # )
    
    # Example: Run detection with resilience
    # results = detector(
    #     source='data/images',
    #     img_size=640,
    #     conf_thres=0.25
    # )
    
    print("\nResilience toolkit ready for integration.")
    print("See documentation for integration with YOLOv5 detect.py")