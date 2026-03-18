import sys
import contextlib
import time
import threading
import os
from functools import lru_cache
from collections import deque
import json
from typing import Dict, List, Optional, Tuple

import torch
from modules import errors, shared, npu_specific

if sys.platform == "darwin":
    from modules import mac_specific

if shared.cmd_opts.use_ipex:
    from modules import xpu_specific


# Performance Telemetry System
class PerformanceTelemetry:
    """Real-time performance monitoring with auto-tuning capabilities."""
    
    def __init__(self):
        self.enabled = os.getenv("SD_WEBUI_TELEMETRY", "1").lower() not in ["0", "false"]
        self.metrics: Dict[str, deque] = {
            "latency": deque(maxlen=1000),
            "memory": deque(maxlen=1000),
            "throughput": deque(maxlen=1000),
        }
        self.current_config = self._get_current_config()
        self.best_config = self.current_config.copy()
        self.best_score = float('-inf')
        self.agent = RLAgent()
        self.lock = threading.Lock()
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_gc_time = 0
        self._operation_stack: List[Tuple[str, float]] = []
        
        if self.enabled:
            self._start_monitoring()
    
    def _get_current_config(self) -> Dict:
        """Get current configuration state."""
        return {
            "dtype": str(dtype),
            "force_fp16": force_fp16,
            "unet_needs_upcast": unet_needs_upcast,
            "fp8": fp8,
            "tf32_enabled": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            "cudnn_benchmark": torch.backends.cudnn.benchmark if torch.cuda.is_available() else False,
            "batch_size": getattr(shared.opts, 'batch_size', 1),
            "precision": str(dtype_inference),
        }
    
    def _apply_config(self, config: Dict):
        """Apply configuration changes."""
        global dtype, force_fp16, unet_needs_upcast, fp8, dtype_inference
        
        # Update global variables
        if "dtype" in config:
            dtype = getattr(torch, config["dtype"].split(".")[-1], torch.float16)
        if "force_fp16" in config:
            force_fp16 = config["force_fp16"]
        if "unet_needs_upcast" in config:
            unet_needs_upcast = config["unet_needs_upcast"]
        if "fp8" in config:
            fp8 = config["fp8"]
        
        # Update torch settings
        if torch.cuda.is_available():
            if "tf32_enabled" in config:
                torch.backends.cuda.matmul.allow_tf32 = config["tf32_enabled"]
                torch.backends.cudnn.allow_tf32 = config["tf32_enabled"]
            if "cudnn_benchmark" in config:
                torch.backends.cudnn.benchmark = config["cudnn_benchmark"]
        
        # Update shared options if available
        if hasattr(shared, 'opts') and "batch_size" in config:
            try:
                shared.opts.batch_size = config["batch_size"]
            except:
                pass
        
        self.current_config = config
    
    def _calculate_score(self, metrics: Dict) -> float:
        """Calculate performance score (higher is better)."""
        if not metrics["latency"]:
            return float('-inf')
        
        avg_latency = sum(metrics["latency"]) / len(metrics["latency"])
        avg_memory = sum(metrics["memory"]) / len(metrics["memory"]) if metrics["memory"] else 0
        throughput = len(metrics["latency"]) / (time.time() - self._start_time) if hasattr(self, '_start_time') else 0
        
        # Weighted score: prioritize throughput, penalize high latency and memory
        score = throughput * 100 - avg_latency * 10 - avg_memory / (1024**3) * 5
        return score
    
    def _start_monitoring(self):
        """Start background monitoring thread."""
        self._start_time = time.time()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while not self._stop_event.is_set():
            time.sleep(5)  # Check every 5 seconds
            self._analyze_and_tune()
    
    def _analyze_and_tune(self):
        """Analyze metrics and apply tuning if beneficial."""
        with self.lock:
            if len(self.metrics["latency"]) < 10:  # Need minimum samples
                return
            
            current_score = self._calculate_score(self.metrics)
            
            # Update best config if current is better
            if current_score > self.best_score:
                self.best_score = current_score
                self.best_config = self.current_config.copy()
            
            # Let RL agent suggest new configuration
            new_config = self.agent.suggest_config(self.metrics, self.current_config)
            if new_config != self.current_config:
                self._apply_config(new_config)
                # Clear metrics for new configuration
                for key in self.metrics:
                    self.metrics[key].clear()
    
    def start_operation(self, name: str):
        """Start timing an operation."""
        if not self.enabled:
            return
        self._operation_stack.append((name, time.time()))
    
    def end_operation(self, name: str, memory_delta: float = 0):
        """End timing an operation and record metrics."""
        if not self.enabled or not self._operation_stack:
            return
        
        op_name, start_time = self._operation_stack.pop()
        if op_name != name:
            return  # Mismatched operation
        
        latency = time.time() - start_time
        
        with self.lock:
            self.metrics["latency"].append(latency)
            if memory_delta > 0:
                self.metrics["memory"].append(memory_delta)
    
    @contextlib.contextmanager
    def track_operation(self, name: str):
        """Context manager to track operation performance."""
        self.start_operation(name)
        memory_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        try:
            yield
        finally:
            memory_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            self.end_operation(name, memory_after - memory_before)
    
    def record_iteration(self, batch_size: int = 1):
        """Record a complete iteration (e.g., image generation)."""
        if not self.enabled:
            return
        
        with self.lock:
            self.metrics["throughput"].append(batch_size)
    
    def stop(self):
        """Stop monitoring."""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)
    
    def get_stats(self) -> Dict:
        """Get current performance statistics."""
        with self.lock:
            stats = {
                "current_config": self.current_config,
                "best_config": self.best_config,
                "best_score": self.best_score,
                "samples": {k: len(v) for k, v in self.metrics.items()},
            }
            
            if self.metrics["latency"]:
                stats["avg_latency"] = sum(self.metrics["latency"]) / len(self.metrics["latency"])
                stats["p95_latency"] = sorted(self.metrics["latency"])[int(len(self.metrics["latency"]) * 0.95)]
            
            if self.metrics["memory"]:
                stats["avg_memory_mb"] = sum(self.metrics["memory"]) / len(self.metrics["memory"]) / (1024**2)
            
            return stats


class RLAgent:
    """Reinforcement learning agent for auto-tuning."""
    
    def __init__(self):
        self.exploration_rate = 0.3
        self.config_history: List[Dict] = []
        self.reward_history: List[float] = []
        
        # Define optimization space
        self.param_space = {
            "dtype": ["float16", "bfloat16", "float32"],
            "force_fp16": [True, False],
            "unet_needs_upcast": [True, False],
            "fp8": [True, False],
            "tf32_enabled": [True, False],
            "cudnn_benchmark": [True, False],
            "batch_size": [1, 2, 4, 8],
        }
    
    def suggest_config(self, metrics: Dict, current_config: Dict) -> Dict:
        """Suggest next configuration to try."""
        import random
        
        # Exploration vs exploitation
        if random.random() < self.exploration_rate:
            return self._explore_config()
        else:
            return self._exploit_config(metrics, current_config)
    
    def _explore_config(self) -> Dict:
        """Explore random configuration."""
        import random
        config = {}
        for param, values in self.param_space.items():
            config[param] = random.choice(values)
        return config
    
    def _exploit_config(self, metrics: Dict, current_config: Dict) -> Dict:
        """Exploit best known configuration."""
        # Simple heuristic: if latency is high, try to reduce precision
        if metrics["latency"]:
            avg_latency = sum(metrics["latency"]) / len(metrics["latency"])
            if avg_latency > 0.5:  # High latency threshold
                config = current_config.copy()
                if config.get("dtype") == "float32":
                    config["dtype"] = "float16"
                if not config.get("force_fp16"):
                    config["force_fp16"] = True
                return config
        
        return current_config


# Global telemetry instance
telemetry = PerformanceTelemetry()


def has_xpu() -> bool:
    return shared.cmd_opts.use_ipex and xpu_specific.has_xpu


def has_mps() -> bool:
    if sys.platform != "darwin":
        return False
    else:
        return mac_specific.has_mps


def cuda_no_autocast(device_id=None) -> bool:
    if device_id is None:
        device_id = get_cuda_device_id()
    return (
        torch.cuda.get_device_capability(device_id) == (7, 5)
        and torch.cuda.get_device_name(device_id).startswith("NVIDIA GeForce GTX 16")
    )


def get_cuda_device_id():
    return (
        int(shared.cmd_opts.device_id)
        if shared.cmd_opts.device_id is not None and shared.cmd_opts.device_id.isdigit()
        else 0
    ) or torch.cuda.current_device()


def get_cuda_device_string():
    if shared.cmd_opts.device_id is not None:
        return f"cuda:{shared.cmd_opts.device_id}"

    return "cuda"


def get_optimal_device_name():
    if torch.cuda.is_available():
        return get_cuda_device_string()

    if has_mps():
        return "mps"

    if has_xpu():
        return xpu_specific.get_xpu_device_string()

    if npu_specific.has_npu:
        return npu_specific.get_npu_device_string()

    return "cpu"


def get_optimal_device():
    return torch.device(get_optimal_device_name())


def get_device_for(task):
    if task in shared.cmd_opts.use_cpu or "all" in shared.cmd_opts.use_cpu:
        return cpu

    return get_optimal_device()


def torch_gc():
    with telemetry.track_operation("gc"):
        if torch.cuda.is_available():
            with torch.cuda.device(get_cuda_device_string()):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        if has_mps():
            mac_specific.torch_mps_gc()

        if has_xpu():
            xpu_specific.torch_xpu_gc()

        if npu_specific.has_npu:
            torch_npu_set_device()
            npu_specific.torch_npu_gc()


def torch_npu_set_device():
    # Work around due to bug in torch_npu, revert me after fixed, @see https://gitee.com/ascend/pytorch/issues/I8KECW?from=project-issue
    if npu_specific.has_npu:
        torch.npu.set_device(0)


def enable_tf32():
    with telemetry.track_operation("enable_tf32"):
        if torch.cuda.is_available():

            # enabling benchmark option seems to enable a range of cards to do fp16 when they otherwise can't
            # see https://github.com/AUTOMATIC1111/nexus/pull/4407
            if cuda_no_autocast():
                torch.backends.cudnn.benchmark = True

            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


errors.run(enable_tf32, "Enabling TF32")

cpu: torch.device = torch.device("cpu")
fp8: bool = False
# Force fp16 for all models in inference. No casting during inference.
# This flag is controlled by "--precision half" command line arg.
force_fp16: bool = False
device: torch.device = None
device_interrogate: torch.device = None
device_gfpgan: torch.device = None
device_esrgan: torch.device = None
device_codeformer: torch.device = None
dtype: torch.dtype = torch.float16
dtype_vae: torch.dtype = torch.float16
dtype_unet: torch.dtype = torch.float16
dtype_inference: torch.dtype = torch.float16
unet_needs_upcast = False


def cond_cast_unet(input):
    with telemetry.track_operation("cond_cast_unet"):
        if force_fp16:
            return input.to(torch.float16)
        return input.to(dtype_unet) if unet_needs_upcast else input


def cond_cast_float(input):
    with telemetry.track_operation("cond_cast_float"):
        return input.float() if unet_needs_upcast else input


nv_rng = None
patch_module_list = [
    torch.nn.Linear,
    torch.nn.Conv2d,
    torch.nn.MultiheadAttention,
    torch.nn.GroupNorm,
    torch.nn.LayerNorm,
]


def manual_cast_forward(target_dtype):
    def forward_wrapper(self, *args, **kwargs):
        with telemetry.track_operation(f"manual_cast_{self.__class__.__name__}"):
            if any(
                isinstance(arg, torch.Tensor) and arg.dtype != target_dtype
                for arg in args
            ):
                args = [arg.to(target_dtype) if isinstance(arg, torch.Tensor) else arg for arg in args]
                kwargs = {k: v.to(target_dtype) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}

            org_dtype = target_dtype
            for param in self.parameters():
                if param.dtype != target_dtype:
                    org_dtype = param.dtype
                    break

            if org_dtype != target_dtype:
                self.to(target_dtype)
            result = self.org_forward(*args, **kwargs)
            if org_dtype != target_dtype:
                self.to(org_dtype)

            if target_dtype != dtype_inference:
                if isinstance(result, tuple):
                    result = tuple(
                        i.to(dtype_inference)
                        if isinstance(i, torch.Tensor)
                        else i
                        for i in result
                    )
                elif isinstance(result, torch.Tensor):
                    result = result.to(dtype_inference)
            return result
    return forward_wrapper


@contextlib.contextmanager
def manual_cast(target_dtype):
    with telemetry.track_operation("manual_cast_context"):
        applied = False
        for module_type in patch_module_list:
            if hasattr(module_type, "org_forward"):
                continue
            applied = True
            org_forward = module_type.forward
            if module_type == torch.nn.MultiheadAttention:
                module_type.forward = manual_cast_forward(torch.float32)
            else:
                module_type.forward = manual_cast_forward(target_dtype)
            module_type.org_forward = org_forward
        try:
            yield None
        finally:
            if applied:
                for module_type in patch_module_list:
                    if hasattr(module_type, "org_forward"):
                        module_type.forward = module_type.org_forward
                        delattr(module_type, "org_forward")


def autocast(disable=False):
    if disable:
        return contextlib.nullcontext()

    if force_fp16:
        # No casting during inference if force_fp16 is enabled.
        # All tensor dtype conversion happens before inference.
        return contextlib.nullcontext()

    if fp8 and device==cpu:
        return torch.autocast("cpu", dtype=torch.bfloat16, enabled=True)

    if fp8 and dtype_inference == torch.float32:
        return manual_cast(dtype)

    if dtype == torch.float32 or dtype_inference == torch.float32:
        return contextlib.nullcontext()

    if has_xpu() or has_mps() or cuda_no_autocast():
        return manual_cast(dtype)

    return torch.autocast("cuda")


def without_autocast(disable=False):
    return torch.autocast("cuda", enabled=False) if torch.is_autocast_enabled() and not disable else contextlib.nullcontext()


class NansException(Exception):
    pass


def test_for_nans(x, where):
    with telemetry.track_operation(f"test_for_nans_{where}"):
        if shared.cmd_opts.disable_nan_check:
            return

        if not torch.isnan(x[(0, ) * len(x.shape)]):
            return

        if where == "unet":
            message = "A tensor with NaNs was produced in Unet."

            if not shared.cmd_opts.no_half:
                message += " This could be either because there's not enough precision to represent the picture, or because your video card does not support half type. Try setting the \"Upcast cross attention layer to float32\" option in Settings > Stable Diffusion or using the --no-half commandline argument to fix this."

        elif where == "vae":
            message = "A tensor with NaNs was produced in VAE."

            if not shared.cmd_opts.no_half and not shared.cmd_opts.no_half_vae:
                message += " This could be because there's not enough precision to represent the picture. Try adding --no-half-vae commandline argument to fix this."
        else:
            message = "A tensor with NaNs was produced."

        message += " Use --disable-nan-check commandline argument to disable this check."

        raise NansException(message)


@lru_cache
def first_time_calculation():
    """
    just do any calculation with pytorch layers - the first time this is done it allocates about 700MB of memory and
    spends about 2.7 seconds doing tha
    """
    with telemetry.track_operation("first_time_calculation"):
        pass  # Original implementation continues below


# Telemetry API functions
def get_telemetry_stats() -> Dict:
    """Get current telemetry statistics."""
    return telemetry.get_stats()


def enable_telemetry():
    """Enable performance telemetry."""
    telemetry.enabled = True
    if not telemetry._monitor_thread or not telemetry._monitor_thread.is_alive():
        telemetry._start_monitoring()


def disable_telemetry():
    """Disable performance telemetry."""
    telemetry.enabled = False
    telemetry.stop()


def record_generation_iteration(batch_size: int = 1):
    """Record a generation iteration for performance tracking."""
    telemetry.record_iteration(batch_size)


# Auto-initialization
if telemetry.enabled:
    try:
        # Initialize with current hardware capabilities
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
    except:
        pass