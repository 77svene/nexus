"""
Real-time Performance Telemetry with Auto-tuning for nexus
Continuous performance monitoring with automatic optimization
"""

import time
import threading
import queue
import json
import os
import psutil
import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import logging
from pathlib import Path
import hashlib
import pickle

logger = logging.getLogger(__name__)

class OptimizationState(Enum):
    """State of the optimization process"""
    EXPLORING = "exploring"
    STABLE = "stable"
    DEGRADED = "degraded"
    RECOVERING = "recovering"

@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    timestamp: float
    inference_time: float
    memory_used: float
    memory_total: float
    gpu_utilization: float
    gpu_memory_used: float
    gpu_memory_total: float
    cpu_utilization: float
    batch_size: int
    resolution: Tuple[int, int]
    steps: int
    model_name: str
    device: str
    operation: str

@dataclass
class OptimizationAction:
    """Action taken by the optimization agent"""
    timestamp: float
    parameter: str
    old_value: Any
    new_value: Any
    confidence: float
    expected_improvement: float

class ReinforcementLearningAgent:
    """Q-learning agent for hardware optimization"""
    
    def __init__(self, state_bins: int = 10, learning_rate: float = 0.1, 
                 discount_factor: float = 0.9, exploration_rate: float = 0.2):
        self.state_bins = state_bins
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.state_history = deque(maxlen=1000)
        self.action_history = deque(maxlen=1000)
        self.reward_history = deque(maxlen=1000)
        
        # Optimization parameters and their ranges
        self.parameters = {
            'batch_size': {'min': 1, 'max': 8, 'type': 'int'},
            'resolution_limit': {'min': 512, 'max': 2048, 'type': 'int', 'step': 64},
            'steps_limit': {'min': 20, 'max': 150, 'type': 'int'},
            'tiling_threshold': {'min': 512, 'max': 2048, 'type': 'int', 'step': 128},
            'vae_precision': {'options': ['fp16', 'fp32'], 'type': 'choice'},
            'attention_implementation': {'options': ['default', 'xformers', 'flash'], 'type': 'choice'}
        }
        
        # Current optimization state
        self.current_state = {}
        self.state_hash = ""
        
    def discretize_state(self, metrics: PerformanceMetrics) -> str:
        """Convert continuous metrics to discrete state"""
        # Normalize metrics to bins
        mem_bin = int((metrics.memory_used / metrics.memory_total) * self.state_bins)
        gpu_bin = int(metrics.gpu_utilization / 10)
        time_bin = min(int(metrics.inference_time * 2), self.state_bins - 1)
        
        # Create state string
        state_str = f"mem{mem_bin}_gpu{gpu_bin}_time{time_bin}_batch{metrics.batch_size}_res{min(metrics.resolution)}"
        return hashlib.md5(state_str.encode()).hexdigest()[:8]
    
    def get_reward(self, metrics: PerformanceMetrics, prev_metrics: Optional[PerformanceMetrics] = None) -> float:
        """Calculate reward based on performance metrics"""
        reward = 0.0
        
        # Reward for faster inference (negative time)
        reward -= metrics.inference_time * 10
        
        # Penalty for high memory usage
        memory_ratio = metrics.memory_used / metrics.memory_total
        if memory_ratio > 0.9:
            reward -= 50  # High penalty for near-OOM
        elif memory_ratio > 0.7:
            reward -= 10
        
        # Reward for GPU utilization (but not over-utilization)
        if 70 <= metrics.gpu_utilization <= 95:
            reward += 5
        elif metrics.gpu_utilization < 30:
            reward -= 5
        
        # Bonus for stability if we have previous metrics
        if prev_metrics:
            time_change = abs(metrics.inference_time - prev_metrics.inference_time)
            if time_change < 0.1:  # Stable performance
                reward += 3
        
        return reward
    
    def choose_action(self, state: str) -> Optional[Tuple[str, Any]]:
        """Choose action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            # Exploration: random action
            param = np.random.choice(list(self.parameters.keys()))
            return self._random_action_for_param(param)
        else:
            # Exploitation: best known action
            if state in self.q_table and self.q_table[state]:
                best_action = max(self.q_table[state].items(), key=lambda x: x[1])[0]
                param, value = best_action.split(':', 1)
                return param, self._parse_action_value(param, value)
        
        return None
    
    def _random_action_for_param(self, param: str) -> Tuple[str, Any]:
        """Generate random action for a parameter"""
        config = self.parameters[param]
        
        if config['type'] == 'int':
            value = np.random.randint(config['min'], config['max'] + 1)
            if 'step' in config:
                value = (value // config['step']) * config['step']
            return param, int(value)
        elif config['type'] == 'choice':
            return param, np.random.choice(config['options'])
        
        return param, None
    
    def _parse_action_value(self, param: str, value_str: str) -> Any:
        """Parse action value from string"""
        config = self.parameters[param]
        
        if config['type'] == 'int':
            return int(value_str)
        elif config['type'] == 'choice':
            return value_str
        
        return None
    
    def update(self, state: str, action: Tuple[str, Any], reward: float, next_state: str):
        """Update Q-table with new experience"""
        action_str = f"{action[0]}:{action[1]}"
        
        # Get current Q-value
        current_q = self.q_table[state][action_str]
        
        # Get max Q-value for next state
        next_max_q = 0
        if next_state in self.q_table and self.q_table[next_state]:
            next_max_q = max(self.q_table[next_state].values())
        
        # Q-learning update
        new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state][action_str] = new_q
        
        # Store history
        self.state_history.append(state)
        self.action_history.append(action)
        self.reward_history.append(reward)

class Telemetry:
    """Main telemetry and auto-tuning system"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Telemetry, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.enabled = True
        self.auto_tune = True
        self.metrics_queue = queue.Queue(maxlen=1000)
        self.metrics_history = deque(maxlen=5000)
        self.optimization_history = deque(maxlen=1000)
        
        # Performance tracking
        self.operation_timers = {}
        self.operation_counts = defaultdict(int)
        self.operation_totals = defaultdict(float)
        
        # Optimization state
        self.optimization_state = OptimizationState.STABLE
        self.last_optimization_time = 0
        self.optimization_cooldown = 30  # seconds
        
        # Initialize RL agent
        self.agent = ReinforcementLearningAgent()
        
        # Current settings
        self.current_settings = {
            'batch_size': 1,
            'resolution_limit': 1024,
            'steps_limit': 50,
            'tiling_threshold': 1024,
            'vae_precision': 'fp16',
            'attention_implementation': 'default'
        }
        
        # Hardware info
        self.hardware_info = self._get_hardware_info()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        # Load previous optimization data
        self._load_optimization_data()
        
        logger.info("Telemetry system initialized")
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information"""
        info = {
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'platform': os.name,
        }
        
        try:
            import torch
            if torch.cuda.is_available():
                info['cuda_available'] = True
                info['cuda_device_count'] = torch.cuda.device_count()
                info['cuda_device_name'] = torch.cuda.get_device_name(0)
                info['cuda_memory_total'] = torch.cuda.get_device_properties(0).total_memory
            else:
                info['cuda_available'] = False
        except ImportError:
            info['cuda_available'] = False
        
        return info
    
    def _load_optimization_data(self):
        """Load previous optimization data from disk"""
        try:
            data_dir = Path("data/telemetry")
            data_dir.mkdir(parents=True, exist_ok=True)
            
            q_table_file = data_dir / "q_table.pkl"
            settings_file = data_dir / "best_settings.json"
            
            if q_table_file.exists():
                with open(q_table_file, 'rb') as f:
                    self.agent.q_table = pickle.load(f)
                logger.info("Loaded Q-table from disk")
            
            if settings_file.exists():
                with open(settings_file, 'r') as f:
                    saved_settings = json.load(f)
                    self.current_settings.update(saved_settings)
                logger.info("Loaded best settings from disk")
                
        except Exception as e:
            logger.warning(f"Could not load optimization data: {e}")
    
    def _save_optimization_data(self):
        """Save optimization data to disk"""
        try:
            data_dir = Path("data/telemetry")
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Save Q-table
            with open(data_dir / "q_table.pkl", 'wb') as f:
                pickle.dump(dict(self.agent.q_table), f)
            
            # Save best settings
            with open(data_dir / "best_settings.json", 'w') as f:
                json.dump(self.current_settings, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Could not save optimization data: {e}")
    
    def _monitoring_loop(self):
        """Background thread for monitoring and optimization"""
        while True:
            try:
                # Process metrics from queue
                self._process_metrics()
                
                # Check if we should run optimization
                if (self.auto_tune and 
                    time.time() - self.last_optimization_time > self.optimization_cooldown):
                    self._run_optimization()
                
                # Save data periodically
                if len(self.metrics_history) % 100 == 0:
                    self._save_optimization_data()
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)
    
    def _process_metrics(self):
        """Process metrics from the queue"""
        while not self.metrics_queue.empty():
            try:
                metrics = self.metrics_queue.get_nowait()
                self.metrics_history.append(metrics)
                
                # Update operation statistics
                op = metrics.operation
                self.operation_timers[op] = metrics.inference_time
                self.operation_counts[op] += 1
                self.operation_totals[op] += metrics.inference_time
                
            except queue.Empty:
                break
    
    def _run_optimization(self):
        """Run optimization step"""
        if len(self.metrics_history) < 10:
            return  # Not enough data
        
        # Get recent metrics
        recent_metrics = list(self.metrics_history)[-10:]
        avg_metrics = self._calculate_average_metrics(recent_metrics)
        
        # Get current state
        current_state = self.agent.discretize_state(avg_metrics)
        
        # Check if performance is degraded
        if self._is_performance_degraded(recent_metrics):
            self.optimization_state = OptimizationState.DEGRADED
            logger.warning("Performance degradation detected")
        
        # Choose action
        action = self.agent.choose_action(current_state)
        
        if action:
            param, new_value = action
            
            # Apply action if it's safe
            if self._is_safe_action(param, new_value, avg_metrics):
                old_value = self.current_settings.get(param)
                
                # Apply the change
                self.current_settings[param] = new_value
                
                # Record optimization action
                opt_action = OptimizationAction(
                    timestamp=time.time(),
                    parameter=param,
                    old_value=old_value,
                    new_value=new_value,
                    confidence=0.8,  # Placeholder
                    expected_improvement=0.1  # Placeholder
                )
                self.optimization_history.append(opt_action)
                
                logger.info(f"Auto-tuning: Changed {param} from {old_value} to {new_value}")
                
                # Update optimization state
                self.optimization_state = OptimizationState.EXPLORING
                self.last_optimization_time = time.time()
                
                # Get next state (will be available after next metrics)
                # For now, use current state as next state
                next_state = current_state
                
                # Calculate reward based on expected improvement
                reward = self._calculate_expected_reward(param, old_value, new_value)
                
                # Update agent
                self.agent.update(current_state, action, reward, next_state)
    
    def _calculate_average_metrics(self, metrics_list: List[PerformanceMetrics]) -> PerformanceMetrics:
        """Calculate average metrics from a list"""
        if not metrics_list:
            return None
        
        avg = {}
        for key in metrics_list[0].__dataclass_fields__:
            values = [getattr(m, key) for m in metrics_list]
            if isinstance(values[0], (int, float)):
                avg[key] = np.mean(values)
            elif isinstance(values[0], tuple):
                avg[key] = values[-1]  # Use last value for tuples
            else:
                avg[key] = values[-1]  # Use last value for non-numeric
        
        return PerformanceMetrics(**avg)
    
    def _is_performance_degraded(self, metrics_list: List[PerformanceMetrics]) -> bool:
        """Check if performance has degraded"""
        if len(metrics_list) < 5:
            return False
        
        # Check for increasing inference times
        times = [m.inference_time for m in metrics_list]
        if times[-1] > times[0] * 1.5:  # 50% slower
            return True
        
        # Check for high memory usage
        for metrics in metrics_list[-3:]:
            if metrics.memory_used / metrics.memory_total > 0.9:
                return True
        
        return False
    
    def _is_safe_action(self, param: str, new_value: Any, metrics: PerformanceMetrics) -> bool:
        """Check if an action is safe to apply"""
        # Don't change parameters during high load
        if metrics.gpu_utilization > 90 or metrics.cpu_utilization > 90:
            return False
        
        # Don't increase batch size if memory is high
        if param == 'batch_size' and new_value > self.current_settings['batch_size']:
            memory_ratio = metrics.memory_used / metrics.memory_total
            if memory_ratio > 0.8:
                return False
        
        # Don't increase resolution if memory is high
        if param == 'resolution_limit' and new_value > self.current_settings['resolution_limit']:
            memory_ratio = metrics.memory_used / metrics.memory_total
            if memory_ratio > 0.7:
                return False
        
        return True
    
    def _calculate_expected_reward(self, param: str, old_value: Any, new_value: Any) -> float:
        """Calculate expected reward for an action"""
        # Simple heuristic: reward for reducing resource usage
        reward = 0.0
        
        if param == 'batch_size':
            if new_value < old_value:
                reward += 2.0  # Reward for reducing batch size (less memory)
            else:
                reward -= 1.0  # Small penalty for increasing batch size
        
        elif param == 'resolution_limit':
            if new_value < old_value:
                reward += 1.5  # Reward for reducing resolution
            else:
                reward -= 0.5
        
        elif param == 'vae_precision':
            if new_value == 'fp16' and old_value == 'fp32':
                reward += 3.0  # Good reward for using fp16
        
        return reward
    
    def start_operation(self, operation: str) -> str:
        """Start timing an operation"""
        if not self.enabled:
            return ""
        
        operation_id = f"{operation}_{time.time()}"
        self.operation_timers[operation_id] = time.time()
        return operation_id
    
    def end_operation(self, operation_id: str, **extra_metrics):
        """End timing an operation and record metrics"""
        if not self.enabled or not operation_id:
            return
        
        start_time = self.operation_timers.pop(operation_id, None)
        if start_time is None:
            return
        
        inference_time = time.time() - start_time
        
        # Get system metrics
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=None)
            
            gpu_util = 0.0
            gpu_mem_used = 0.0
            gpu_mem_total = 0.0
            
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_mem_used = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
                    gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                    # Note: GPU utilization requires nvidia-smi or similar
            except ImportError:
                pass
            
            # Create metrics object
            metrics = PerformanceMetrics(
                timestamp=time.time(),
                inference_time=inference_time,
                memory_used=memory.used / (1024 ** 3),  # GB
                memory_total=memory.total / (1024 ** 3),
                gpu_utilization=gpu_util,
                gpu_memory_used=gpu_mem_used,
                gpu_memory_total=gpu_mem_total,
                cpu_utilization=cpu_percent,
                batch_size=extra_metrics.get('batch_size', self.current_settings['batch_size']),
                resolution=extra_metrics.get('resolution', (512, 512)),
                steps=extra_metrics.get('steps', self.current_settings['steps_limit']),
                model_name=extra_metrics.get('model_name', 'unknown'),
                device=extra_metrics.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
                operation=operation_id.rsplit('_', 1)[0]
            )
            
            # Queue metrics for processing
            self.metrics_queue.put(metrics)
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
    
    def get_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """Get current optimization suggestions"""
        suggestions = []
        
        if len(self.metrics_history) < 5:
            return suggestions
        
        recent_metrics = list(self.metrics_history)[-5:]
        avg_metrics = self._calculate_average_metrics(recent_metrics)
        
        # Check for memory pressure
        memory_ratio = avg_metrics.memory_used / avg_metrics.memory_total
        if memory_ratio > 0.8:
            suggestions.append({
                'type': 'memory',
                'severity': 'high',
                'message': f'High memory usage ({memory_ratio:.1%}). Consider reducing batch size or resolution.',
                'suggested_action': 'reduce_batch_size'
            })
        
        # Check for slow inference
        if avg_metrics.inference_time > 10.0:  # More than 10 seconds
            suggestions.append({
                'type': 'performance',
                'severity': 'medium',
                'message': f'Slow inference ({avg_metrics.inference_time:.1f}s). Consider using fp16 or reducing steps.',
                'suggested_action': 'enable_fp16'
            })
        
        # Check for low GPU utilization (if using CUDA)
        if avg_metrics.device == 'cuda' and avg_metrics.gpu_utilization < 30:
            suggestions.append({
                'type': 'utilization',
                'severity': 'low',
                'message': f'Low GPU utilization ({avg_metrics.gpu_utilization:.1f}%). Consider increasing batch size.',
                'suggested_action': 'increase_batch_size'
            })
        
        return suggestions
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.metrics_history:
            return {}
        
        recent_metrics = list(self.metrics_history)[-100:] if len(self.metrics_history) >= 100 else list(self.metrics_history)
        
        summary = {
            'total_operations': len(self.metrics_history),
            'avg_inference_time': np.mean([m.inference_time for m in recent_metrics]),
            'min_inference_time': np.min([m.inference_time for m in recent_metrics]),
            'max_inference_time': np.max([m.inference_time for m in recent_metrics]),
            'avg_memory_used': np.mean([m.memory_used for m in recent_metrics]),
            'optimization_state': self.optimization_state.value,
            'current_settings': self.current_settings.copy(),
            'hardware_info': self.hardware_info
        }
        
        # Add operation-specific stats
        for op, count in self.operation_counts.items():
            if count > 0:
                avg_time = self.operation_totals[op] / count
                summary[f'avg_{op}_time'] = avg_time
                summary[f'{op}_count'] = count
        
        return summary
    
    def set_parameter(self, parameter: str, value: Any):
        """Manually set a parameter"""
        if parameter in self.current_settings:
            old_value = self.current_settings[parameter]
            self.current_settings[parameter] = value
            
            # Record manual override
            opt_action = OptimizationAction(
                timestamp=time.time(),
                parameter=parameter,
                old_value=old_value,
                new_value=value,
                confidence=1.0,
                expected_improvement=0.0
            )
            self.optimization_history.append(opt_action)
            
            logger.info(f"Manual override: Set {parameter} to {value}")
    
    def enable(self):
        """Enable telemetry"""
        self.enabled = True
        logger.info("Telemetry enabled")
    
    def disable(self):
        """Disable telemetry"""
        self.enabled = False
        logger.info("Telemetry disabled")
    
    def enable_auto_tune(self):
        """Enable auto-tuning"""
        self.auto_tune = True
        logger.info("Auto-tuning enabled")
    
    def disable_auto_tune(self):
        """Disable auto-tuning"""
        self.auto_tune = False
        logger.info("Auto-tuning disabled")
    
    def reset(self):
        """Reset telemetry data"""
        self.metrics_history.clear()
        self.optimization_history.clear()
        self.operation_counts.clear()
        self.operation_totals.clear()
        self.agent.q_table.clear()
        logger.info("Telemetry data reset")

# Global telemetry instance
telemetry = Telemetry()

# Context manager for instrumenting code
class instrument_operation:
    """Context manager for instrumenting operations"""
    
    def __init__(self, operation_name: str, **extra_metrics):
        self.operation_name = operation_name
        self.extra_metrics = extra_metrics
        self.operation_id = None
    
    def __enter__(self):
        self.operation_id = telemetry.start_operation(self.operation_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.operation_id:
            telemetry.end_operation(self.operation_id, **self.extra_metrics)
        return False

# Decorator for instrumenting functions
def instrument(operation_name: str = None):
    """Decorator for instrumenting functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            with instrument_operation(op_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

# Integration hooks for existing modules
def patch_lora_module():
    """Patch LoRA module with telemetry"""
    try:
        from modules import extra_networks_lora
        original_forward = extra_networks_lora.ExtraNetworkLora.forward
        
        @instrument("lora_forward")
        def instrumented_forward(self, *args, **kwargs):
            return original_forward(self, *args, **kwargs)
        
        extra_networks_lora.ExtraNetworkLora.forward = instrumented_forward
        logger.info("Patched LoRA module with telemetry")
    except ImportError:
        pass

def patch_ldsr_module():
    """Patch LDSR module with telemetry"""
    try:
        from extensions_built_in.LDSR import ldsr_model_arch
        original_forward = ldsr_model_arch.LDSR.forward
        
        @instrument("ldsr_forward")
        def instrumented_forward(self, *args, **kwargs):
            return original_forward(self, *args, **kwargs)
        
        ldsr_model_arch.LDSR.forward = instrumented_forward
        logger.info("Patched LDSR module with telemetry")
    except ImportError:
        pass

# Initialize patches
def initialize_telemetry():
    """Initialize telemetry and patch modules"""
    telemetry  # Initialize singleton
    patch_lora_module()
    patch_ldsr_module()
    logger.info("Telemetry system fully initialized")

# Auto-initialize on import
initialize_telemetry()

# Export public API
__all__ = [
    'telemetry',
    'instrument_operation',
    'instrument',
    'PerformanceMetrics',
    'OptimizationAction',
    'initialize_telemetry'
]