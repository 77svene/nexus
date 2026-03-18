import time
import logging
import json
import os
import hashlib
import threading
import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import gc

try:
    import torch
    import torch.cuda
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)

class OptimizationTarget(Enum):
    """What aspect of performance to optimize for"""
    LATENCY = "latency"  # Minimize time per iteration
    THROUGHPUT = "throughput"  # Maximize images per minute
    MEMORY = "memory"  # Minimize memory usage
    BALANCED = "balanced"  # Balance all factors

@dataclass
class HardwareProfile:
    """Detected hardware capabilities"""
    gpu_name: str = "unknown"
    gpu_memory_mb: int = 0
    cpu_cores: int = 1
    system_memory_mb: int = 0
    has_cuda: bool = False
    has_mps: bool = False
    compute_capability: Tuple[int, int] = (0, 0)
    
    @classmethod
    def detect(cls) -> 'HardwareProfile':
        """Detect current hardware capabilities"""
        profile = cls()
        
        if TORCH_AVAILABLE:
            profile.has_cuda = torch.cuda.is_available()
            profile.has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            
            if profile.has_cuda:
                try:
                    profile.gpu_name = torch.cuda.get_device_name(0)
                    profile.gpu_memory_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
                    profile.compute_capability = torch.cuda.get_device_capability(0)
                except:
                    pass
        
        if PSUTIL_AVAILABLE:
            profile.cpu_cores = psutil.cpu_count(logical=False) or 1
            profile.system_memory_mb = psutil.virtual_memory().total // (1024 * 1024)
        
        return profile

@dataclass
class PerformanceMetrics:
    """Collected performance metrics for a single operation"""
    operation_name: str
    duration_ms: float
    memory_used_mb: float
    memory_peak_mb: float
    gpu_utilization: float = 0.0
    cpu_utilization: float = 0.0
    batch_size: int = 1
    resolution: Tuple[int, int] = (512, 512)
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

@dataclass
class OptimizationState:
    """Current optimization settings"""
    use_half_precision: bool = True
    use_tf32: bool = True
    use_channels_last: bool = False
    use_sdp_attention: bool = True
    use_xformers: bool = False
    use_vae_tiling: bool = False
    vae_tile_size: int = 512
    batch_size: int = 1
    steps: int = 20
    cfg_scale: float = 7.0
    sampler: str = "Euler a"
    compile_model: bool = False
    attention_slice_size: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationState':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

class RLOptimizer:
    """Reinforcement Learning agent for optimization"""
    
    def __init__(self, state_dim: int = 8, action_dim: int = 12):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Q-table for discrete state-action space
        self.q_table = defaultdict(lambda: np.zeros(action_dim))
        self.visit_counts = defaultdict(lambda: np.zeros(action_dim))
        
        # Hyperparameters
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.2  # Exploration rate
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        
        # Experience replay buffer
        self.experience_buffer = deque(maxlen=1000)
        
        # Current state and action tracking
        self.current_state = None
        self.current_action = None
        self.last_reward = 0.0
    
    def _state_to_key(self, state: np.ndarray) -> str:
        """Convert state array to hashable key"""
        # Discretize continuous state for Q-table
        discretized = np.round(state * 10).astype(int)
        return hashlib.md5(discretized.tobytes()).hexdigest()[:16]
    
    def select_action(self, state: np.ndarray, valid_actions: List[int]) -> int:
        """Select action using epsilon-greedy policy"""
        self.current_state = state
        state_key = self._state_to_key(state)
        
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            action = np.random.choice(valid_actions)
        else:
            q_values = self.q_table[state_key]
            # Mask invalid actions with -inf
            masked_q = np.full_like(q_values, -np.inf)
            masked_q[valid_actions] = q_values[valid_actions]
            action = np.argmax(masked_q)
        
        self.current_action = action
        return action
    
    def update(self, reward: float, next_state: np.ndarray, done: bool = False):
        """Update Q-values based on reward"""
        if self.current_state is None or self.current_action is None:
            return
        
        state_key = self._state_to_key(self.current_state)
        next_state_key = self._state_to_key(next_state)
        
        # Store experience
        self.experience_buffer.append((
            self.current_state.copy(),
            self.current_action,
            reward,
            next_state.copy(),
            done
        ))
        
        # Q-learning update
        current_q = self.q_table[state_key][self.current_action]
        next_max_q = np.max(self.q_table[next_state_key]) if not done else 0
        
        # Update Q-value
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state_key][self.current_action] = new_q
        
        # Update visit count
        self.visit_counts[state_key][self.current_action] += 1
        
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        # Update tracking
        self.last_reward = reward
        self.current_state = next_state
    
    def get_best_action(self, state: np.ndarray, valid_actions: List[int]) -> int:
        """Get best action for a state (no exploration)"""
        state_key = self._state_to_key(state)
        q_values = self.q_table[state_key]
        masked_q = np.full_like(q_values, -np.inf)
        masked_q[valid_actions] = q_values[valid_actions]
        return np.argmax(masked_q)
    
    def save(self, filepath: str):
        """Save Q-table to file"""
        data = {
            'q_table': dict(self.q_table),
            'visit_counts': dict(self.visit_counts),
            'epsilon': self.epsilon
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    def load(self, filepath: str):
        """Load Q-table from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.q_table = defaultdict(lambda: np.zeros(self.action_dim))
            for k, v in data.get('q_table', {}).items():
                self.q_table[k] = np.array(v)
            
            self.visit_counts = defaultdict(lambda: np.zeros(self.action_dim))
            for k, v in data.get('visit_counts', {}).items():
                self.visit_counts[k] = np.array(v)
            
            self.epsilon = data.get('epsilon', self.epsilon)
        except Exception as e:
            logger.warning(f"Failed to load RL optimizer state: {e}")

class PerformanceMonitor:
    """Real-time performance monitoring with minimal overhead"""
    
    def __init__(self, max_history: int = 1000):
        self.metrics_history = deque(maxlen=max_history)
        self.operation_stack = []
        self.start_times = {}
        self.memory_snapshots = {}
        
        # Performance counters
        self.counters = defaultdict(int)
        self.timers = defaultdict(float)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # GPU monitoring (if available)
        self.gpu_available = TORCH_AVAILABLE and torch.cuda.is_available()
    
    def start_operation(self, operation_name: str, metadata: Dict[str, Any] = None):
        """Start timing an operation"""
        with self.lock:
            operation_id = f"{operation_name}_{time.time()}"
            self.start_times[operation_id] = time.time()
            
            # Take memory snapshot
            if TORCH_AVAILABLE and self.gpu_available:
                torch.cuda.reset_peak_memory_stats()
                self.memory_snapshots[operation_id] = {
                    'start_allocated': torch.cuda.memory_allocated() // (1024 * 1024),
                    'start_reserved': torch.cuda.memory_reserved() // (1024 * 1024)
                }
            elif PSUTIL_AVAILABLE:
                process = psutil.Process()
                self.memory_snapshots[operation_id] = {
                    'start_rss': process.memory_info().rss // (1024 * 1024)
                }
            
            self.operation_stack.append((operation_id, operation_name, metadata or {}))
            return operation_id
    
    def end_operation(self, operation_id: str = None):
        """End timing an operation and collect metrics"""
        with self.lock:
            if not self.operation_stack:
                return
            
            if operation_id is None:
                operation_id, operation_name, metadata = self.operation_stack.pop()
            else:
                # Find and remove the operation from stack
                for i, (op_id, op_name, meta) in enumerate(self.operation_stack):
                    if op_id == operation_id:
                        operation_name = op_name
                        metadata = meta
                        self.operation_stack.pop(i)
                        break
                else:
                    return
            
            # Calculate duration
            duration_ms = (time.time() - self.start_times.get(operation_id, time.time())) * 1000
            
            # Calculate memory usage
            memory_used_mb = 0
            memory_peak_mb = 0
            
            if TORCH_AVAILABLE and self.gpu_available and operation_id in self.memory_snapshots:
                snapshot = self.memory_snapshots[operation_id]
                memory_used_mb = (torch.cuda.memory_allocated() // (1024 * 1024)) - snapshot['start_allocated']
                memory_peak_mb = torch.cuda.max_memory_allocated() // (1024 * 1024)
            elif PSUTIL_AVAILABLE and operation_id in self.memory_snapshots:
                process = psutil.Process()
                current_rss = process.memory_info().rss // (1024 * 1024)
                memory_used_mb = current_rss - self.memory_snapshots[operation_id]['start_rss']
            
            # Create metrics object
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                duration_ms=duration_ms,
                memory_used_mb=memory_used_mb,
                memory_peak_mb=memory_peak_mb,
                timestamp=time.time()
            )
            
            # Add metadata
            for key, value in metadata.items():
                if hasattr(metrics, key):
                    setattr(metrics, key, value)
            
            # Store in history
            self.metrics_history.append(metrics)
            
            # Cleanup
            self.start_times.pop(operation_id, None)
            self.memory_snapshots.pop(operation_id, None)
            
            # Update counters
            self.counters[f"{operation_name}_count"] += 1
            self.timers[f"{operation_name}_total_ms"] += duration_ms
            
            return metrics
    
    def get_operation_stats(self, operation_name: str, window_seconds: float = 60.0) -> Dict[str, float]:
        """Get statistics for a specific operation"""
        with self.lock:
            cutoff_time = time.time() - window_seconds
            recent_metrics = [
                m for m in self.metrics_history
                if m.operation_name == operation_name and m.timestamp >= cutoff_time
            ]
            
            if not recent_metrics:
                return {}
            
            durations = [m.duration_ms for m in recent_metrics]
            memories = [m.memory_used_mb for m in recent_metrics]
            
            return {
                'count': len(recent_metrics),
                'avg_duration_ms': np.mean(durations),
                'min_duration_ms': np.min(durations),
                'max_duration_ms': np.max(durations),
                'std_duration_ms': np.std(durations),
                'avg_memory_mb': np.mean(memories),
                'total_duration_ms': np.sum(durations),
                'p95_duration_ms': np.percentile(durations, 95),
                'p99_duration_ms': np.percentile(durations, 99)
            }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system statistics"""
        stats = {
            'timestamp': time.time(),
            'operations_in_progress': len(self.operation_stack)
        }
        
        if TORCH_AVAILABLE and self.gpu_available:
            stats.update({
                'gpu_memory_allocated_mb': torch.cuda.memory_allocated() // (1024 * 1024),
                'gpu_memory_reserved_mb': torch.cuda.memory_reserved() // (1024 * 1024),
                'gpu_memory_total_mb': torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            })
        
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            memory_info = process.memory_info()
            stats.update({
                'cpu_percent': process.cpu_percent(),
                'memory_rss_mb': memory_info.rss // (1024 * 1024),
                'memory_vms_mb': memory_info.vms // (1024 * 1024),
                'system_memory_percent': psutil.virtual_memory().percent
            })
        
        return stats
    
    def clear_history(self):
        """Clear metrics history"""
        with self.lock:
            self.metrics_history.clear()
            self.counters.clear()
            self.timers.clear()

class AutoOptimizer:
    """Main auto-optimizer with real-time tuning"""
    
    def __init__(self, config_dir: str = None):
        self.config_dir = config_dir or os.path.join(os.path.dirname(__file__), "optimizer_data")
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Hardware detection
        self.hardware_profile = HardwareProfile.detect()
        logger.info(f"Detected hardware: {self.hardware_profile.gpu_name} "
                   f"({self.hardware_profile.gpu_memory_mb}MB), "
                   f"{self.hardware_profile.cpu_cores} cores")
        
        # Performance monitoring
        self.monitor = PerformanceMonitor()
        
        # RL optimizer
        self.rl_optimizer = RLOptimizer(state_dim=10, action_dim=15)
        self.load_rl_state()
        
        # Current optimization state
        self.current_state = OptimizationState()
        self.target = OptimizationTarget.BALANCED
        
        # Optimization constraints
        self.constraints = {
            'min_batch_size': 1,
            'max_batch_size': 8,
            'min_steps': 10,
            'max_steps': 50,
            'min_cfg_scale': 1.0,
            'max_cfg_scale': 20.0,
            'max_memory_mb': self.hardware_profile.gpu_memory_mb * 0.8 if self.hardware_profile.gpu_memory_mb > 0 else 4096
        }
        
        # Performance tracking
        self.performance_history = deque(maxlen=100)
        self.best_state = None
        self.best_reward = -float('inf')
        
        # Auto-tuning thread
        self.auto_tune_thread = None
        self.auto_tune_running = False
        self.auto_tune_interval = 30.0  # seconds
        
        # Integration hooks
        self.hooks_installed = False
        
        # Action space definition
        self.action_space = self._define_action_space()
        
        # State space normalization
        self.state_mean = np.array([500.0, 100.0, 50.0, 0.5, 0.5, 512.0, 512.0, 20.0, 7.0, 1.0])
        self.state_std = np.array([300.0, 80.0, 30.0, 0.3, 0.3, 256.0, 256.0, 10.0, 3.0, 0.5])
    
    def _define_action_space(self) -> List[Callable]:
        """Define available optimization actions"""
        actions = [
            # Precision adjustments
            lambda s: self._adjust_precision(half=True),
            lambda s: self._adjust_precision(half=False),
            
            # Attention optimizations
            lambda s: self._adjust_attention(use_sdp=True, use_xformers=False),
            lambda s: self._adjust_attention(use_sdp=False, use_xformers=True),
            lambda s: self._adjust_attention(use_sdp=False, use_xformers=False),
            
            # Memory optimizations
            lambda s: self._adjust_vae_tiling(tile_size=256),
            lambda s: self._adjust_vae_tiling(tile_size=512),
            lambda s: self._adjust_vae_tiling(tile_size=1024),
            lambda s: self._adjust_vae_tiling(enabled=False),
            
            # Batch size adjustments
            lambda s: self._adjust_batch_size(delta=1),
            lambda s: self._adjust_batch_size(delta=-1),
            
            # Step adjustments
            lambda s: self._adjust_steps(delta=5),
            lambda s: self._adjust_steps(delta=-5),
            
            # Channels last memory format
            lambda s: self._adjust_memory_format(channels_last=True),
            lambda s: self._adjust_memory_format(channels_last=False),
        ]
        return actions
    
    def _adjust_precision(self, half: bool):
        """Adjust precision settings"""
        self.current_state.use_half_precision = half
        if TORCH_AVAILABLE and torch.cuda.is_available():
            if half:
                torch.set_float32_matmul_precision('high')
            else:
                torch.set_float32_matmul_precision('highest')
    
    def _adjust_attention(self, use_sdp: bool, use_xformers: bool):
        """Adjust attention mechanism"""
        self.current_state.use_sdp_attention = use_sdp
        self.current_state.use_xformers = use_xformers
    
    def _adjust_vae_tiling(self, tile_size: int = 512, enabled: bool = True):
        """Adjust VAE tiling settings"""
        self.current_state.use_vae_tiling = enabled
        self.current_state.vae_tile_size = tile_size
    
    def _adjust_batch_size(self, delta: int):
        """Adjust batch size within constraints"""
        new_size = max(self.constraints['min_batch_size'],
                      min(self.constraints['max_batch_size'],
                          self.current_state.batch_size + delta))
        self.current_state.batch_size = new_size
    
    def _adjust_steps(self, delta: int):
        """Adjust sampling steps within constraints"""
        new_steps = max(self.constraints['min_steps'],
                       min(self.constraints['max_steps'],
                           self.current_state.steps + delta))
        self.current_state.steps = new_steps
    
    def _adjust_memory_format(self, channels_last: bool):
        """Adjust memory format"""
        self.current_state.use_channels_last = channels_last
    
    def _get_current_state_vector(self) -> np.ndarray:
        """Convert current state and performance to normalized state vector"""
        # Get recent performance metrics
        recent_metrics = list(self.monitor.metrics_history)[-10:] if self.monitor.metrics_history else []
        
        if recent_metrics:
            avg_duration = np.mean([m.duration_ms for m in recent_metrics])
            avg_memory = np.mean([m.memory_used_mb for m in recent_metrics])
            p95_duration = np.percentile([m.duration_ms for m in recent_metrics], 95)
        else:
            avg_duration = 500.0
            avg_memory = 100.0
            p95_duration = 600.0
        
        # Create state vector
        state = np.array([
            avg_duration,
            avg_memory,
            p95_duration,
            float(self.current_state.use_half_precision),
            float(self.current_state.use_sdp_attention),
            float(self.current_state.vae_tile_size),
            float(self.current_state.batch_size),
            float(self.current_state.steps),
            float(self.current_state.cfg_scale),
            float(self.hardware_profile.gpu_memory_mb / 1024)  # GB
        ])
        
        # Normalize
        normalized_state = (state - self.state_mean) / self.state_std
        return normalized_state
    
    def _calculate_reward(self, metrics: PerformanceMetrics) -> float:
        """Calculate reward based on performance metrics and target"""
        reward = 0.0
        
        # Base reward from performance (negative because we want to minimize)
        duration_reward = -metrics.duration_ms / 1000.0  # Convert to seconds
        memory_reward = -metrics.memory_used_mb / 1024.0  # Convert to GB
        
        # Weight based on optimization target
        if self.target == OptimizationTarget.LATENCY:
            reward = duration_reward * 2.0 + memory_reward * 0.5
        elif self.target == OptimizationTarget.THROUGHPUT:
            # Throughput is inversely related to duration
            reward = (1000.0 / max(metrics.duration_ms, 1.0)) + memory_reward * 0.3
        elif self.target == OptimizationTarget.MEMORY:
            reward = duration_reward * 0.5 + memory_reward * 2.0
        else:  # BALANCED
            reward = duration_reward + memory_reward
        
        # Penalty for exceeding memory constraints
        if metrics.memory_peak_mb > self.constraints['max_memory_mb']:
            reward -= 10.0
        
        # Bonus for stable performance
        if len(self.performance_history) > 1:
            last_metrics = self.performance_history[-1]
            duration_variance = abs(metrics.duration_ms - last_metrics.duration_ms) / last_metrics.duration_ms
            if duration_variance < 0.1:  # Less than 10% variance
                reward += 0.5
        
        return reward
    
    def _get_valid_actions(self) -> List[int]:
        """Get list of valid actions given current state and constraints"""
        valid_actions = list(range(len(self.action_space)))
        
        # Remove actions that would violate constraints
        if self.current_state.batch_size >= self.constraints['max_batch_size']:
            # Remove batch size increase action
            if 9 in valid_actions:
                valid_actions.remove(9)
        
        if self.current_state.batch_size <= self.constraints['min_batch_size']:
            # Remove batch size decrease action
            if 10 in valid_actions:
                valid_actions.remove(10)
        
        if self.current_state.steps >= self.constraints['max_steps']:
            if 11 in valid_actions:
                valid_actions.remove(11)
        
        if self.current_state.steps <= self.constraints['min_steps']:
            if 12 in valid_actions:
                valid_actions.remove(12)
        
        return valid_actions
    
    def optimize_step(self):
        """Perform one optimization step"""
        # Get current state
        state_vector = self._get_current_state_vector()
        
        # Get valid actions
        valid_actions = self._get_valid_actions()
        if not valid_actions:
            return
        
        # Select action using RL agent
        action_idx = self.rl_optimizer.select_action(state_vector, valid_actions)
        
        # Apply action
        self.action_space[action_idx](state_vector)
        
        # Log the optimization
        logger.debug(f"Optimization step: action={action_idx}, state={self.current_state.to_dict()}")
    
    def update_from_metrics(self, metrics: PerformanceMetrics):
        """Update optimizer with new performance metrics"""
        # Store in history
        self.performance_history.append(metrics)
        
        # Calculate reward
        reward = self._calculate_reward(metrics)
        
        # Get next state
        next_state_vector = self._get_current_state_vector()
        
        # Update RL agent
        self.rl_optimizer.update(reward, next_state_vector)
        
        # Track best state
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_state = self.current_state.to_dict()
            logger.info(f"New best state: reward={reward:.3f}, state={self.best_state}")
        
        # Log performance
        logger.debug(f"Performance update: duration={metrics.duration_ms:.1f}ms, "
                    f"memory={metrics.memory_used_mb:.1f}MB, reward={reward:.3f}")
    
    def start_auto_tuning(self):
        """Start automatic tuning in background thread"""
        if self.auto_tune_running:
            return
        
        self.auto_tune_running = True
        self.auto_tune_thread = threading.Thread(target=self._auto_tune_loop, daemon=True)
        self.auto_tune_thread.start()
        logger.info("Auto-tuning started")
    
    def stop_auto_tuning(self):
        """Stop automatic tuning"""
        self.auto_tune_running = False
        if self.auto_tune_thread:
            self.auto_tune_thread.join(timeout=5.0)
        logger.info("Auto-tuning stopped")
    
    def _auto_tune_loop(self):
        """Background loop for auto-tuning"""
        while self.auto_tune_running:
            try:
                # Only optimize if we have enough data
                if len(self.performance_history) >= 5:
                    self.optimize_step()
                
                # Save state periodically
                if len(self.performance_history) % 20 == 0:
                    self.save_state()
                
                time.sleep(self.auto_tune_interval)
            except Exception as e:
                logger.error(f"Error in auto-tune loop: {e}")
                time.sleep(5.0)
    
    def install_hooks(self):
        """Install hooks into the nexus codebase"""
        if self.hooks_installed:
            return
        
        try:
            # Import webui modules
            from modules import processing, sd_samplers, sd_models
            from modules.shared import opts, state
            
            # Hook into processing
            original_process_images = processing.process_images
            
            def hooked_process_images(p, *args, **kwargs):
                # Start monitoring
                op_id = self.monitor.start_operation("process_images", {
                    'batch_size': getattr(p, 'batch_size', 1),
                    'width': getattr(p, 'width', 512),
                    'height': getattr(p, 'height', 512)
                })
                
                try:
                    # Apply current optimization state
                    self._apply_optimizations(p)
                    
                    # Call original function
                    result = original_process_images(p, *args, **kwargs)
                    
                    # Update optimizer with metrics
                    metrics = self.monitor.end_operation(op_id)
                    if metrics:
                        self.update_from_metrics(metrics)
                    
                    return result
                except Exception as e:
                    self.monitor.end_operation(op_id)
                    raise
            
            processing.process_images = hooked_process_images
            
            # Hook into VAE decoding if available
            try:
                from modules import sd_vae_decoders
                original_decode = sd_vae_decoders.decode
                
                def hooked_decode(*args, **kwargs):
                    op_id = self.monitor.start_operation("vae_decode")
                    try:
                        result = original_decode(*args, **kwargs)
                        self.monitor.end_operation(op_id)
                        return result
                    except Exception as e:
                        self.monitor.end_operation(op_id)
                        raise
                
                sd_vae_decoders.decode = hooked_decode
            except ImportError:
                pass
            
            self.hooks_installed = True
            logger.info("Performance monitoring hooks installed")
            
        except ImportError as e:
            logger.warning(f"Could not install hooks: {e}")
    
    def _apply_optimizations(self, p):
        """Apply current optimization state to processing object"""
        # Set precision
        if hasattr(p, 'half') and self.current_state.use_half_precision:
            p.half = True
        
        # Set batch size if allowed
        if hasattr(p, 'batch_size'):
            p.batch_size = self.current_state.batch_size
        
        # Set steps
        if hasattr(p, 'steps'):
            p.steps = self.current_state.steps
        
        # Set CFG scale
        if hasattr(p, 'cfg_scale'):
            p.cfg_scale = self.current_state.cfg_scale
        
        # Apply memory format
        if TORCH_AVAILABLE and self.current_state.use_channels_last:
            # This would need to be applied to the model, not the processing object
            pass
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate a comprehensive optimization report"""
        report = {
            'hardware': asdict(self.hardware_profile),
            'current_state': self.current_state.to_dict(),
            'best_state': self.best_state,
            'best_reward': self.best_reward,
            'optimization_target': self.target.value,
            'performance_stats': {},
            'system_stats': self.monitor.get_system_stats(),
            'rl_stats': {
                'epsilon': self.rl_optimizer.epsilon,
                'experience_buffer_size': len(self.rl_optimizer.experience_buffer),
                'q_table_size': len(self.rl_optimizer.q_table)
            }
        }
        
        # Add performance statistics for key operations
        for op_name in ['process_images', 'vae_decode']:
            stats = self.monitor.get_operation_stats(op_name)
            if stats:
                report['performance_stats'][op_name] = stats
        
        return report
    
    def save_state(self):
        """Save optimizer state to disk"""
        try:
            # Save RL state
            rl_path = os.path.join(self.config_dir, "rl_optimizer_state.json")
            self.rl_optimizer.save(rl_path)
            
            # Save best state
            if self.best_state:
                best_path = os.path.join(self.config_dir, "best_state.json")
                with open(best_path, 'w') as f:
                    json.dump(self.best_state, f, indent=2)
            
            # Save hardware profile
            hw_path = os.path.join(self.config_dir, "hardware_profile.json")
            with open(hw_path, 'w') as f:
                json.dump(asdict(self.hardware_profile), f, indent=2)
            
            logger.debug("Optimizer state saved")
        except Exception as e:
            logger.error(f"Failed to save optimizer state: {e}")
    
    def load_rl_state(self):
        """Load RL optimizer state from disk"""
        rl_path = os.path.join(self.config_dir, "rl_optimizer_state.json")
        if os.path.exists(rl_path):
            self.rl_optimizer.load(rl_path)
            logger.info("Loaded RL optimizer state")
    
    def load_best_state(self):
        """Load best known optimization state"""
        best_path = os.path.join(self.config_dir, "best_state.json")
        if os.path.exists(best_path):
            try:
                with open(best_path, 'r') as f:
                    best_state = json.load(f)
                self.current_state = OptimizationState.from_dict(best_state)
                logger.info(f"Loaded best optimization state: {self.current_state.to_dict()}")
            except Exception as e:
                logger.warning(f"Failed to load best state: {e}")
    
    def set_optimization_target(self, target: OptimizationTarget):
        """Set the optimization target"""
        self.target = target
        logger.info(f"Optimization target set to: {target.value}")
    
    def reset(self):
        """Reset optimizer to initial state"""
        self.current_state = OptimizationState()
        self.performance_history.clear()
        self.best_state = None
        self.best_reward = -float('inf')
        self.monitor.clear_history()
        logger.info("Optimizer reset to initial state")

# Global optimizer instance
_optimizer_instance = None

def get_optimizer() -> AutoOptimizer:
    """Get or create the global optimizer instance"""
    global _optimizer_instance
    if _optimizer_instance is None:
        _optimizer_instance = AutoOptimizer()
    return _optimizer_instance

def initialize_optimizer(config_dir: str = None, auto_start: bool = True):
    """Initialize the auto-optimizer system"""
    optimizer = get_optimizer()
    
    if config_dir:
        optimizer.config_dir = config_dir
        os.makedirs(config_dir, exist_ok=True)
    
    # Load previous state
    optimizer.load_best_state()
    optimizer.load_rl_state()
    
    # Install hooks
    optimizer.install_hooks()
    
    # Start auto-tuning if requested
    if auto_start:
        optimizer.start_auto_tuning()
    
    logger.info("Auto-optimizer initialized")
    return optimizer

def optimize_now():
    """Manually trigger an optimization step"""
    optimizer = get_optimizer()
    optimizer.optimize_step()

def get_performance_report() -> Dict[str, Any]:
    """Get current performance report"""
    optimizer = get_optimizer()
    return optimizer.get_optimization_report()

def set_optimization_target(target: str):
    """Set optimization target by string"""
    optimizer = get_optimizer()
    try:
        target_enum = OptimizationTarget(target.lower())
        optimizer.set_optimization_target(target_enum)
    except ValueError:
        logger.warning(f"Unknown optimization target: {target}. Using BALANCED.")
        optimizer.set_optimization_target(OptimizationTarget.BALANCED)

# Context manager for performance monitoring
class monitor_performance:
    """Context manager for monitoring performance of code blocks"""
    
    def __init__(self, operation_name: str, metadata: Dict[str, Any] = None):
        self.operation_name = operation_name
        self.metadata = metadata or {}
        self.optimizer = get_optimizer()
        self.operation_id = None
    
    def __enter__(self):
        self.operation_id = self.optimizer.monitor.start_operation(
            self.operation_name, self.metadata
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.operation_id:
            metrics = self.optimizer.monitor.end_operation(self.operation_id)
            if metrics:
                self.optimizer.update_from_metrics(metrics)
        return False

# Decorator for performance monitoring
def monitor_operation(operation_name: str = None):
    """Decorator for monitoring function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            with monitor_performance(op_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize optimizer
    optimizer = initialize_optimizer(auto_start=False)
    
    # Print hardware profile
    print("Hardware Profile:")
    print(json.dumps(asdict(optimizer.hardware_profile), indent=2))
    
    # Example: Monitor a simulated operation
    with monitor_performance("test_operation", {"batch_size": 2, "resolution": (512, 512)}):
        time.sleep(0.1)  # Simulate work
    
    # Get performance report
    report = get_performance_report()
    print("\nPerformance Report:")
    print(json.dumps(report, indent=2, default=str))
    
    # Test optimization step
    optimize_now()
    
    print("\nCurrent optimization state:")
    print(json.dumps(optimizer.current_state.to_dict(), indent=2))