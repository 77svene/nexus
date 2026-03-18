"""
Edge Deployment Monitor for YOLOv5
Resilient edge deployment toolkit with health checks, automatic restart,
thermal throttling detection, and graceful degradation.
"""

import os
import sys
import time
import signal
import threading
import logging
import traceback
import json
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from queue import Queue, Empty
from collections import deque
import psutil
import torch
import torch.nn as nn

# Add parent directory to path for imports
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.experimental import attempt_load
from utils.torch_utils import select_device, time_sync
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.datasets import LoadImages, LoadStreams


class DeploymentState(Enum):
    """Deployment state enumeration"""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    RECOVERING = "recovering"
    STOPPED = "stopped"


class DegradationLevel(Enum):
    """Graceful degradation levels"""
    NONE = 0  # Full performance
    LEVEL_1 = 1  # Reduced input size
    LEVEL_2 = 2  # Reduced precision (FP16)
    LEVEL_3 = 3  # Smaller model
    LEVEL_4 = 4  # Minimal inference


@dataclass
class SystemMetrics:
    """System health metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    temperature_cpu: Optional[float] = None
    temperature_gpu: Optional[float] = None
    inference_fps: Optional[float] = None
    inference_latency_ms: Optional[float] = None
    model_load_time: Optional[float] = None
    consecutive_failures: int = 0
    state: DeploymentState = DeploymentState.INITIALIZING
    degradation_level: DegradationLevel = DegradationLevel.NONE


@dataclass
class HealthCheckResult:
    """Health check result"""
    timestamp: float
    success: bool
    latency_ms: float
    error_message: Optional[str] = None
    output_valid: bool = True
    memory_leak_detected: bool = False


class ThermalMonitor:
    """Monitor system thermal state"""
    
    def __init__(self, 
                 cpu_threshold: float = 80.0,
                 gpu_threshold: float = 85.0,
                 check_interval: float = 5.0):
        self.cpu_threshold = cpu_threshold
        self.gpu_threshold = gpu_threshold
        self.check_interval = check_interval
        self.thermal_history = deque(maxlen=100)
        self.throttling_detected = False
        self._monitor_thread = None
        self._stop_event = threading.Event()
        
    def start(self):
        """Start thermal monitoring"""
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
    def stop(self):
        """Stop thermal monitoring"""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            
    def _get_cpu_temperature(self) -> Optional[float]:
        """Get CPU temperature (platform specific)"""
        try:
            if sys.platform == "linux":
                with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                    temp = float(f.read().strip()) / 1000.0
                return temp
            elif sys.platform == "darwin":
                # macOS - would need additional libraries for actual reading
                return None
            elif sys.platform == "win32":
                # Windows - would need WMI or similar
                return None
        except:
            return None
            
    def _get_gpu_temperature(self) -> Optional[float]:
        """Get GPU temperature if available"""
        try:
            if torch.cuda.is_available():
                # Note: PyTorch doesn't directly expose temperature
                # Would need nvidia-smi or pynvml for actual reading
                return None
        except:
            return None
            
    def _monitor_loop(self):
        """Main monitoring loop"""
        while not self._stop_event.is_set():
            try:
                cpu_temp = self._get_cpu_temperature()
                gpu_temp = self._get_gpu_temperature()
                
                metrics = {
                    "timestamp": time.time(),
                    "cpu_temp": cpu_temp,
                    "gpu_temp": gpu_temp
                }
                self.thermal_history.append(metrics)
                
                # Check for throttling
                if cpu_temp and cpu_temp > self.cpu_threshold:
                    self.throttling_detected = True
                    logging.warning(f"CPU thermal throttling detected: {cpu_temp}°C")
                elif gpu_temp and gpu_temp > self.gpu_threshold:
                    self.throttling_detected = True
                    logging.warning(f"GPU thermal throttling detected: {gpu_temp}°C")
                else:
                    # Check if recovering
                    if self.throttling_detected:
                        recent_temps = [m["cpu_temp"] for m in list(self.thermal_history)[-5:] 
                                      if m["cpu_temp"] is not None]
                        if recent_temps and max(recent_temps) < self.cpu_threshold - 5:
                            self.throttling_detected = False
                            logging.info("Thermal throttling recovery detected")
                            
            except Exception as e:
                logging.error(f"Error in thermal monitoring: {e}")
                
            time.sleep(self.check_interval)
            
    def get_thermal_state(self) -> Dict[str, Any]:
        """Get current thermal state"""
        recent = list(self.thermal_history)[-10:] if self.thermal_history else []
        cpu_temps = [m["cpu_temp"] for m in recent if m["cpu_temp"] is not None]
        gpu_temps = [m["gpu_temp"] for m in recent if m["gpu_temp"] is not None]
        
        return {
            "throttling_detected": self.throttling_detected,
            "cpu_temp_current": cpu_temps[-1] if cpu_temps else None,
            "cpu_temp_avg": np.mean(cpu_temps) if cpu_temps else None,
            "cpu_temp_max": max(cpu_temps) if cpu_temps else None,
            "gpu_temp_current": gpu_temps[-1] if gpu_temps else None,
            "gpu_temp_avg": np.mean(gpu_temps) if gpu_temps else None,
        }


class WatchdogTimer:
    """Watchdog timer for detecting hangs"""
    
    def __init__(self, timeout: float = 30.0, callback: Optional[Callable] = None):
        self.timeout = timeout
        self.callback = callback
        self._timer = None
        self._last_reset = time.time()
        self._lock = threading.Lock()
        
    def start(self):
        """Start watchdog timer"""
        self.reset()
        
    def reset(self):
        """Reset the watchdog timer"""
        with self._lock:
            self._last_reset = time.time()
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(self.timeout, self._timeout_handler)
            self._timer.daemon = True
            self._timer.start()
            
    def stop(self):
        """Stop watchdog timer"""
        with self._lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None
                
    def _timeout_handler(self):
        """Handle watchdog timeout"""
        elapsed = time.time() - self._last_reset
        logging.error(f"Watchdog timeout after {elapsed:.1f}s")
        if self.callback:
            self.callback()
            
    def is_alive(self) -> bool:
        """Check if watchdog is alive (no timeout)"""
        with self._lock:
            return self._timer is not None and self._timer.is_alive()


class ModelHealthChecker:
    """Model health checking and validation"""
    
    def __init__(self, model_path: str, device: str = ''):
        self.model_path = model_path
        self.device = select_device(device)
        self.model = None
        self.input_shapes = {}
        self.health_check_input = None
        self.last_health_check = None
        self.health_history = deque(maxlen=100)
        
    def load_model(self, half: bool = False) -> bool:
        """Load model with health validation"""
        try:
            start_time = time_sync()
            self.model = attempt_load(self.model_path, device=self.device)
            self.model.half() if half else self.model.float()
            
            # Generate health check input
            img_size = check_img_size(max(self.model.stride), s=self.model.stride.max())
            self.health_check_input = torch.zeros((1, 3, img_size, img_size), 
                                                 device=self.device)
            if half and self.device.type != 'cpu':
                self.health_check_input = self.health_check_input.half()
                
            load_time = time_sync() - start_time
            logging.info(f"Model loaded in {load_time:.2f}s")
            
            # Run initial health check
            return self.run_health_check()
            
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            return False
            
    def run_health_check(self) -> HealthCheckResult:
        """Run comprehensive health check"""
        if self.model is None:
            return HealthCheckResult(
                timestamp=time.time(),
                success=False,
                latency_ms=0,
                error_message="Model not loaded"
            )
            
        start_time = time_sync()
        try:
            # Run inference with health check input
            with torch.no_grad():
                output = self.model(self.health_check_input)
                
            # Validate output
            if isinstance(output, tuple):
                output = output[0]
                
            output_valid = (
                output is not None and 
                isinstance(output, torch.Tensor) and 
                not torch.isnan(output).any() and
                not torch.isinf(output).any()
            )
            
            # Check for memory leaks
            memory_leak = False
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_allocated = torch.cuda.memory_allocated(self.device)
                memory_reserved = torch.cuda.memory_reserved(self.device)
                # Simple heuristic: if reserved > 2x allocated, might be a leak
                if memory_reserved > 2 * memory_allocated and memory_allocated > 100 * 1024 * 1024:
                    memory_leak = True
                    
            latency_ms = (time_sync() - start_time) * 1000
            
            result = HealthCheckResult(
                timestamp=time.time(),
                success=True,
                latency_ms=latency_ms,
                output_valid=output_valid,
                memory_leak_detected=memory_leak
            )
            
        except Exception as e:
            result = HealthCheckResult(
                timestamp=time.time(),
                success=False,
                latency_ms=(time_sync() - start_time) * 1000,
                error_message=str(e)
            )
            
        self.last_health_check = result
        self.health_history.append(result)
        return result
        
    def get_health_stats(self) -> Dict[str, Any]:
        """Get health statistics"""
        if not self.health_history:
            return {}
            
        successful = [h for h in self.health_history if h.success]
        failed = [h for h in self.health_history if not h.success]
        
        latencies = [h.latency_ms for h in successful]
        
        return {
            "total_checks": len(self.health_history),
            "successful_checks": len(successful),
            "failed_checks": len(failed),
            "success_rate": len(successful) / len(self.health_history) if self.health_history else 0,
            "avg_latency_ms": np.mean(latencies) if latencies else 0,
            "max_latency_ms": max(latencies) if latencies else 0,
            "min_latency_ms": min(latencies) if latencies else 0,
            "last_check": asdict(self.last_health_check) if self.last_health_check else None
        }


class EdgeMonitor:
    """Main edge deployment monitor with resilience features"""
    
    def __init__(self, 
                 model_path: str,
                 device: str = '',
                 config: Optional[Dict] = None):
        self.model_path = model_path
        self.device = select_device(device)
        self.config = config or self._default_config()
        
        # Initialize components
        self.thermal_monitor = ThermalMonitor(
            cpu_threshold=self.config.get('cpu_temp_threshold', 80.0),
            gpu_threshold=self.config.get('gpu_temp_threshold', 85.0),
            check_interval=self.config.get('thermal_check_interval', 5.0)
        )
        
        self.watchdog = WatchdogTimer(
            timeout=self.config.get('watchdog_timeout', 30.0),
            callback=self._watchdog_timeout_handler
        )
        
        self.health_checker = ModelHealthChecker(model_path, device)
        
        # State management
        self.state = DeploymentState.INITIALIZING
        self.degradation_level = DegradationLevel.NONE
        self.metrics_history = deque(maxlen=1000)
        self.failure_count = 0
        self.last_restart_time = 0
        self.restart_count = 0
        
        # Model variants for degradation
        self.model_variants = {
            DegradationLevel.NONE: {'path': model_path, 'half': False, 'img_size': 640},
            DegradationLevel.LEVEL_1: {'path': model_path, 'half': False, 'img_size': 416},
            DegradationLevel.LEVEL_2: {'path': model_path, 'half': True, 'img_size': 416},
            DegradationLevel.LEVEL_3: {'path': self._get_smaller_model_path(), 'half': True, 'img_size': 320},
            DegradationLevel.LEVEL_4: {'path': self._get_smallest_model_path(), 'half': True, 'img_size': 256}
        }
        
        # Current model configuration
        self.current_model_config = self.model_variants[DegradationLevel.NONE]
        self.model = None
        self.half = False
        self.img_size = 640
        
        # Threading
        self._monitor_thread = None
        self._stop_event = threading.Event()
        self._inference_lock = threading.Lock()
        self._metrics_queue = Queue()
        
        # Setup logging
        self._setup_logging()
        
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'health_check_interval': 60.0,  # seconds
            'thermal_check_interval': 5.0,  # seconds
            'watchdog_timeout': 30.0,  # seconds
            'max_consecutive_failures': 3,
            'cpu_temp_threshold': 80.0,  # Celsius
            'gpu_temp_threshold': 85.0,  # Celsius
            'memory_threshold': 90.0,  # percent
            'degradation_cooldown': 300.0,  # seconds between degradation changes
            'enable_auto_restart': True,
            'enable_graceful_degradation': True,
            'enable_thermal_throttling': True,
            'log_level': 'INFO',
            'metrics_export_interval': 60.0,  # seconds
        }
        
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config.get('log_level', 'INFO').upper())
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('edge_monitor.log')
            ]
        )
        self.logger = logging.getLogger('EdgeMonitor')
        
    def _get_smaller_model_path(self) -> str:
        """Get path to smaller model variant"""
        model_dir = Path(self.model_path).parent
        model_name = Path(self.model_path).stem
        
        # Try to find smaller variants
        for size in ['n', 's', 'm']:  # nano, small, medium
            smaller_path = model_dir / f"{model_name[:-1]}{size}.pt"  # e.g., nexuss.pt -> nexusn.pt
            if smaller_path.exists():
                return str(smaller_path)
                
        # Fallback to same model
        return self.model_path
        
    def _get_smallest_model_path(self) -> str:
        """Get path to smallest model variant"""
        model_dir = Path(self.model_path).parent
        model_name = Path(self.model_path).stem
        
        # Try nano first
        nano_path = model_dir / f"{model_name[:-1]}n.pt"
        if nano_path.exists():
            return str(nano_path)
            
        return self._get_smaller_model_path()
        
    def initialize(self) -> bool:
        """Initialize the edge monitor and load model"""
        self.logger.info("Initializing Edge Monitor...")
        
        try:
            # Start thermal monitoring
            if self.config.get('enable_thermal_throttling', True):
                self.thermal_monitor.start()
                
            # Load initial model
            if not self._load_model(self.current_model_config):
                self.logger.error("Failed to load initial model")
                return False
                
            # Start monitoring thread
            self._start_monitoring()
            
            # Start watchdog
            self.watchdog.start()
            
            self.state = DeploymentState.HEALTHY
            self.logger.info("Edge Monitor initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            self.state = DeploymentState.CRITICAL
            return False
            
    def _load_model(self, config: Dict) -> bool:
        """Load model with given configuration"""
        try:
            model_path = config['path']
            half = config['half']
            img_size = config['img_size']
            
            self.logger.info(f"Loading model: {model_path} (half={half}, img_size={img_size})")
            
            # Update health checker
            self.health_checker.model_path = model_path
            
            # Load model
            if not self.health_checker.load_model(half=half):
                return False
                
            self.model = self.health_checker.model
            self.half = half
            self.img_size = img_size
            self.current_model_config = config
            
            # Warmup inference
            self._warmup_inference()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
            
    def _warmup_inference(self):
        """Run warmup inference"""
        try:
            self.logger.debug("Running warmup inference...")
            dummy_input = self.health_checker.health_check_input
            with torch.no_grad():
                _ = self.model(dummy_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.logger.debug("Warmup inference completed")
        except Exception as e:
            self.logger.warning(f"Warmup inference failed: {e}")
            
    def _start_monitoring(self):
        """Start background monitoring thread"""
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        
    def _monitoring_loop(self):
        """Main monitoring loop"""
        last_health_check = 0
        last_metrics_export = 0
        
        while not self._stop_event.is_set():
            try:
                current_time = time.time()
                
                # Collect system metrics
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                self._metrics_queue.put(metrics)
                
                # Periodic health check
                if current_time - last_health_check > self.config.get('health_check_interval', 60.0):
                    self._perform_health_check()
                    last_health_check = current_time
                    
                # Export metrics periodically
                if current_time - last_metrics_export > self.config.get('metrics_export_interval', 60.0):
                    self._export_metrics()
                    last_metrics_export = current_time
                    
                # Check for state transitions
                self._evaluate_state()
                
                # Check for recovery opportunities
                self._check_recovery()
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                
            time.sleep(1.0)  # Check every second
            
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # GPU metrics if available
        gpu_memory_used = None
        gpu_memory_total = None
        if torch.cuda.is_available():
            try:
                gpu_memory_used = torch.cuda.memory_allocated(self.device) / (1024 ** 2)  # MB
                gpu_memory_total = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 2)  # MB
            except:
                pass
                
        # Thermal state
        thermal_state = self.thermal_monitor.get_thermal_state()
        
        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 ** 2),
            gpu_memory_used_mb=gpu_memory_used,
            gpu_memory_total_mb=gpu_memory_total,
            temperature_cpu=thermal_state.get('cpu_temp_current'),
            temperature_gpu=thermal_state.get('gpu_temp_current'),
            consecutive_failures=self.failure_count,
            state=self.state,
            degradation_level=self.degradation_level
        )
        
    def _perform_health_check(self):
        """Perform comprehensive health check"""
        self.logger.debug("Performing health check...")
        
        result = self.health_checker.run_health_check()
        
        if result.success:
            self.failure_count = 0
            if self.state == DeploymentState.CRITICAL:
                self.state = DeploymentState.RECOVERING
            self.logger.debug(f"Health check passed (latency: {result.latency_ms:.1f}ms)")
        else:
            self.failure_count += 1
            self.logger.warning(f"Health check failed: {result.error_message}")
            
            # Check if we need to take action
            if self.failure_count >= self.config.get('max_consecutive_failures', 3):
                self._handle_failure()
                
    def _handle_failure(self):
        """Handle consecutive failures"""
        self.logger.error(f"Handling failure (count: {self.failure_count})")
        
        if self.config.get('enable_auto_restart', True):
            self._restart_model()
            
        if self.config.get('enable_graceful_degradation', True):
            self._consider_degradation()
            
    def _restart_model(self):
        """Restart/reload the model"""
        # Cooldown check
        if time.time() - self.last_restart_time < 60.0:  # Minimum 60s between restarts
            self.logger.warning("Restart cooldown active, skipping restart")
            return
            
        self.logger.info("Attempting model restart...")
        self.state = DeploymentState.RECOVERING
        
        try:
            # Unload current model
            if self.model is not None:
                del self.model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            # Reload model
            if self._load_model(self.current_model_config):
                self.logger.info("Model restart successful")
                self.failure_count = 0
                self.restart_count += 1
                self.last_restart_time = time.time()
                self.state = DeploymentState.HEALTHY
            else:
                self.logger.error("Model restart failed")
                self.state = DeploymentState.CRITICAL
                
        except Exception as e:
            self.logger.error(f"Error during model restart: {e}")
            self.state = DeploymentState.CRITICAL
            
    def _consider_degradation(self):
        """Consider graceful degradation based on system state"""
        if not self.config.get('enable_graceful_degradation', True):
            return
            
        # Check cooldown
        if hasattr(self, '_last_degradation_change'):
            if time.time() - self._last_degradation_change < self.config.get('degradation_cooldown', 300.0):
                return
                
        current_metrics = self._collect_system_metrics()
        
        # Check conditions for degradation
        should_degrade = False
        should_upgrade = False
        
        # Thermal throttling
        if self.thermal_monitor.throttling_detected:
            should_degrade = True
            
        # High memory usage
        if current_metrics.memory_percent > self.config.get('memory_threshold', 90.0):
            should_degrade = True
            
        # High failure rate
        if self.failure_count > self.config.get('max_consecutive_failures', 3):
            should_degrade = True
            
        # Check if we can upgrade (recover)
        if (self.state == DeploymentState.HEALTHY and 
            self.degradation_level != DegradationLevel.NONE and
            not self.thermal_monitor.throttling_detected and
            current_metrics.memory_percent < 70.0 and
            self.failure_count == 0):
            should_upgrade = True
            
        # Apply degradation/upgrade
        if should_degrade and self.degradation_level != DegradationLevel.LEVEL_4:
            new_level = DegradationLevel(min(self.degradation_level.value + 1, 4))
            self._apply_degradation(new_level)
        elif should_upgrade and self.degradation_level != DegradationLevel.NONE:
            new_level = DegradationLevel(max(self.degradation_level.value - 1, 0))
            self._apply_degradation(new_level)
            
    def _apply_degradation(self, level: DegradationLevel):
        """Apply specific degradation level"""
        if level == self.degradation_level:
            return
            
        self.logger.info(f"Applying degradation level: {level.name}")
        
        config = self.model_variants[level]
        if self._load_model(config):
            self.degradation_level = level
            self._last_degradation_change = time.time()
            self.logger.info(f"Degradation applied successfully: {level.name}")
        else:
            self.logger.error(f"Failed to apply degradation level: {level.name}")
            
    def _evaluate_state(self):
        """Evaluate current deployment state"""
        if not self.metrics_history:
            return
            
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 metrics
        
        # Check for critical conditions
        critical_conditions = []
        
        for metrics in recent_metrics:
            if metrics.temperature_cpu and metrics.temperature_cpu > self.config.get('cpu_temp_threshold', 80.0) + 10:
                critical_conditions.append("CPU temperature critical")
            if metrics.memory_percent > 95.0:
                critical_conditions.append("Memory usage critical")
            if metrics.consecutive_failures > self.config.get('max_consecutive_failures', 3) * 2:
                critical_conditions.append("Excessive failures")
                
        if critical_conditions:
            self.state = DeploymentState.CRITICAL
            self.logger.error(f"Critical state detected: {', '.join(critical_conditions)}")
        elif self.failure_count > 0:
            self.state = DeploymentState.DEGRADED
        elif self.state == DeploymentState.RECOVERING:
            # Check if recovered
            if self.failure_count == 0 and not self.thermal_monitor.throttling_detected:
                self.state = DeploymentState.HEALTHY
                self.logger.info("System recovered to healthy state")
                
    def _check_recovery(self):
        """Check if we can recover from degraded state"""
        if self.state != DeploymentState.DEGRADED and self.state != DeploymentState.CRITICAL:
            return
            
        # Check if conditions have improved
        recent_metrics = list(self.metrics_history)[-5:] if self.metrics_history else []
        
        if not recent_metrics:
            return
            
        avg_memory = np.mean([m.memory_percent for m in recent_metrics])
        max_temp = max([m.temperature_cpu for m in recent_metrics if m.temperature_cpu] or [0])
        
        if (avg_memory < 80.0 and 
            max_temp < self.config.get('cpu_temp_threshold', 80.0) - 5 and
            self.failure_count == 0):
            
            self.logger.info("Conditions improved, attempting recovery...")
            if self.config.get('enable_graceful_degradation', True):
                self._consider_degradation()  # This might upgrade
                
    def _watchdog_timeout_handler(self):
        """Handle watchdog timeout"""
        self.logger.error("Watchdog timeout detected!")
        self.state = DeploymentState.CRITICAL
        
        if self.config.get('enable_auto_restart', True):
            self._restart_model()
            
    def _export_metrics(self):
        """Export metrics to file"""
        try:
            metrics_dict = {
                'timestamp': datetime.now().isoformat(),
                'state': self.state.value,
                'degradation_level': self.degradation_level.value,
                'failure_count': self.failure_count,
                'restart_count': self.restart_count,
                'health_stats': self.health_checker.get_health_stats(),
                'thermal_state': self.thermal_monitor.get_thermal_state(),
                'system_metrics': asdict(self._collect_system_metrics()) if self.metrics_history else None,
                'config': self.config
            }
            
            with open('edge_monitor_metrics.json', 'w') as f:
                json.dump(metrics_dict, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
            
    def run_inference(self, 
                     source: Union[str, int, List],
                     **kwargs) -> Tuple[bool, Any]:
        """
        Run inference with resilience features
        
        Args:
            source: Input source (image, video, stream, etc.)
            **kwargs: Additional arguments for inference
            
        Returns:
            Tuple of (success, results)
        """
        with self._inference_lock:
            # Reset watchdog before inference
            self.watchdog.reset()
            
            try:
                start_time = time_sync()
                
                # Check if model is loaded
                if self.model is None:
                    self.logger.error("Model not loaded")
                    return False, None
                    
                # Run inference based on source type
                if isinstance(source, (str, Path)) and Path(source).is_file():
                    results = self._run_inference_file(source, **kwargs)
                elif isinstance(source, int) or (isinstance(source, str) and source.startswith(('rtsp://', 'rtmp://', 'http://'))):
                    results = self._run_inference_stream(source, **kwargs)
                elif isinstance(source, (list, tuple)):
                    results = self._run_inference_batch(source, **kwargs)
                elif isinstance(source, torch.Tensor):
                    results = self._run_inference_tensor(source, **kwargs)
                else:
                    self.logger.error(f"Unsupported source type: {type(source)}")
                    return False, None
                    
                # Update metrics
                inference_time = time_sync() - start_time
                self._update_inference_metrics(inference_time, success=True)
                
                # Reset failure count on success
                self.failure_count = 0
                
                return True, results
                
            except Exception as e:
                self.logger.error(f"Inference failed: {e}")
                self.logger.debug(traceback.format_exc())
                
                # Update failure metrics
                self.failure_count += 1
                self._update_inference_metrics(0, success=False)
                
                # Handle failure if needed
                if self.failure_count >= self.config.get('max_consecutive_failures', 3):
                    self._handle_failure()
                    
                return False, None
                
            finally:
                # Always reset watchdog after inference attempt
                self.watchdog.reset()
                
    def _run_inference_file(self, source: str, **kwargs) -> Any:
        """Run inference on a single file"""
        # Implementation would use YOLOv5's detect.py logic
        # This is a simplified version
        from detect import run as detect_run
        
        # Prepare arguments
        opt = {
            'weights': self.current_model_config['path'],
            'source': source,
            'imgsz': [self.img_size],
            'conf_thres': kwargs.get('conf_thres', 0.25),
            'iou_thres': kwargs.get('iou_thres', 0.45),
            'device': self.device,
            'half': self.half,
            'save_txt': False,
            'save_conf': False,
            'nosave': True,
            'classes': kwargs.get('classes', None),
            'agnostic_nms': kwargs.get('agnostic_nms', False),
            'augment': kwargs.get('augment', False),
            'update': False,
            'project': 'runs/detect',
            'name': 'edge_inference',
            'exist_ok': True,
        }
        
        # Run detection
        results = detect_run(**opt)
        return results
        
    def _run_inference_stream(self, source: Union[int, str], **kwargs) -> Any:
        """Run inference on a stream"""
        # Similar to file but with stream handling
        from detect import run as detect_run
        
        opt = {
            'weights': self.current_model_config['path'],
            'source': source,
            'imgsz': [self.img_size],
            'conf_thres': kwargs.get('conf_thres', 0.25),
            'iou_thres': kwargs.get('iou_thres', 0.45),
            'device': self.device,
            'half': self.half,
            'save_txt': False,
            'save_conf': False,
            'nosave': True,
            'classes': kwargs.get('classes', None),
            'agnostic_nms': kwargs.get('agnostic_nms', False),
            'augment': kwargs.get('augment', False),
            'update': False,
            'project': 'runs/detect',
            'name': 'edge_stream',
            'exist_ok': True,
        }
        
        results = detect_run(**opt)
        return results
        
    def _run_inference_batch(self, sources: List, **kwargs) -> List:
        """Run inference on a batch of sources"""
        results = []
        for source in sources:
            success, result = self.run_inference(source, **kwargs)
            results.append((success, result))
        return results
        
    def _run_inference_tensor(self, tensor: torch.Tensor, **kwargs) -> Any:
        """Run inference on a tensor"""
        # Preprocess
        if self.half and self.device.type != 'cpu':
            tensor = tensor.half()
            
        # Inference
        with torch.no_grad():
            output = self.model(tensor)
            
        # Post-process
        if isinstance(output, tuple):
            output = output[0]
            
        # Apply NMS
        output = non_max_suppression(
            output,
            conf_thres=kwargs.get('conf_thres', 0.25),
            iou_thres=kwargs.get('iou_thres', 0.45),
            classes=kwargs.get('classes', None),
            agnostic=kwargs.get('agnostic_nms', False),
        )
        
        return output
        
    def _update_inference_metrics(self, inference_time: float, success: bool):
        """Update inference metrics"""
        if success and inference_time > 0:
            fps = 1.0 / inference_time if inference_time > 0 else 0
            latency_ms = inference_time * 1000
            
            # Update metrics in health checker
            if hasattr(self.health_checker, 'last_health_check'):
                self.health_checker.last_health_check.inference_fps = fps
                self.health_checker.last_health_check.inference_latency_ms = latency_ms
                
    def get_status(self) -> Dict[str, Any]:
        """Get current deployment status"""
        return {
            'state': self.state.value,
            'degradation_level': self.degradation_level.value,
            'failure_count': self.failure_count,
            'restart_count': self.restart_count,
            'current_model': self.current_model_config['path'],
            'img_size': self.img_size,
            'half_precision': self.half,
            'device': str(self.device),
            'thermal_state': self.thermal_monitor.get_thermal_state(),
            'health_stats': self.health_checker.get_health_stats(),
            'watchdog_alive': self.watchdog.is_alive(),
            'uptime_seconds': time.time() - (self.metrics_history[0].timestamp if self.metrics_history else time.time()),
        }
        
    def shutdown(self):
        """Shutdown the edge monitor"""
        self.logger.info("Shutting down Edge Monitor...")
        
        # Stop monitoring
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            
        # Stop thermal monitor
        self.thermal_monitor.stop()
        
        # Stop watchdog
        self.watchdog.stop()
        
        # Cleanup model
        if self.model is not None:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        # Final metrics export
        self._export_metrics()
        
        self.state = DeploymentState.STOPPED
        self.logger.info("Edge Monitor shutdown complete")
        
    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()


def health_check_decorator(monitor: EdgeMonitor):
    """Decorator for adding health checks to inference functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Run health check before inference
            health_result = monitor.health_checker.run_health_check()
            
            if not health_result.success:
                monitor.logger.warning(f"Pre-inference health check failed: {health_result.error_message}")
                monitor.failure_count += 1
                
                if monitor.failure_count >= monitor.config.get('max_consecutive_failures', 3):
                    monitor._handle_failure()
                    return False, None
                    
            # Reset watchdog
            monitor.watchdog.reset()
            
            try:
                # Run actual inference
                result = func(*args, **kwargs)
                
                # Reset failure count on success
                monitor.failure_count = 0
                
                return True, result
                
            except Exception as e:
                monitor.logger.error(f"Inference failed in decorator: {e}")
                monitor.failure_count += 1
                
                if monitor.failure_count >= monitor.config.get('max_consecutive_failures', 3):
                    monitor._handle_failure()
                    
                return False, None
                
            finally:
                # Always reset watchdog
                monitor.watchdog.reset()
                
        return wrapper
    return decorator


# Example usage and integration
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv5 Edge Deployment Monitor')
    parser.add_argument('--model', type=str, default='nexuss.pt', help='model path')
    parser.add_argument('--source', type=str, default='0', help='inference source')
    parser.add_argument('--device', type=str, default='', help='device (cpu or cuda device)')
    parser.add_argument('--config', type=str, default='', help='config file path')
    
    opt = parser.parse_args()
    
    # Load config if provided
    config = {}
    if opt.config and Path(opt.config).exists():
        with open(opt.config, 'r') as f:
            config = yaml.safe_load(f)
            
    # Initialize monitor
    with EdgeMonitor(opt.model, opt.device, config) as monitor:
        print(f"Initial status: {monitor.get_status()}")
        
        # Run inference with resilience
        if opt.source.isdigit():
            source = int(opt.source)
        else:
            source = opt.source
            
        # Example inference loop
        for i in range(10):  # Run 10 inferences as example
            print(f"\nInference {i+1}/10")
            success, results = monitor.run_inference(source)
            
            if success:
                print(f"  Success! State: {monitor.state.value}")
            else:
                print(f"  Failed! State: {monitor.state.value}")
                
            time.sleep(2)  # Simulate processing time
            
        print(f"\nFinal status: {monitor.get_status()}")
        
        # Keep running if source is a stream
        if isinstance(source, int) or (isinstance(source, str) and 
                                       source.startswith(('rtsp://', 'rtmp://', 'http://'))):
            print("\nStream mode - Press Ctrl+C to stop")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nShutting down...")
                
    print("Edge Monitor completed.")