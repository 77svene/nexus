"""
core/resilience/circuit_breaker.py
Adaptive Agent Resilience Framework — Circuit breakers, retry policies, fallback strategies,
and automatic degradation when dependencies fail.
"""

import asyncio
import time
import random
import logging
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
import inspect

# Import existing monitoring modules
try:
    from monitoring.tracing import TracingManager, Span
    from monitoring.metrics_collector import MetricsCollector
    from monitoring.cost_tracker import CostTracker
except ImportError:
    # Fallback stubs if monitoring modules aren't available
    class TracingManager:
        @staticmethod
        def get_tracer(name):
            return TracingManager()
        def start_span(self, name):
            return Span()
    
    class Span:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def set_attribute(self, key, value):
            pass
    
    class MetricsCollector:
        def __init__(self):
            self.counters = {}
            self.gauges = {}
            self.histograms = {}
        
        def increment(self, name, tags=None):
            pass
        
        def gauge(self, name, value, tags=None):
            pass
        
        def histogram(self, name, value, tags=None):
            pass
    
    class CostTracker:
        def track_operation(self, operation, cost=0.0):
            pass

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "CLOSED"      # Normal operation, requests pass through
    OPEN = "OPEN"          # Circuit is open, requests fail fast
    HALF_OPEN = "HALF_OPEN"  # Testing if service has recovered

@dataclass
class RetryConfig:
    """Configuration for retry policies"""
    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on_exceptions: List[Type[Exception]] = field(default_factory=lambda: [Exception])
    retry_on_status_codes: List[int] = field(default_factory=lambda: [500, 502, 503, 504])

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5  # Number of failures before opening circuit
    recovery_timeout: float = 30.0  # Seconds before attempting recovery
    half_open_max_calls: int = 3  # Max calls allowed in HALF_OPEN state
    success_threshold: int = 2  # Successes needed to close circuit from HALF_OPEN
    monitored_exceptions: List[Type[Exception]] = field(default_factory=lambda: [Exception])
    timeout: float = 10.0  # Request timeout in seconds
    enable_health_check: bool = True
    health_check_interval: float = 60.0  # Seconds between health checks

@dataclass
class FallbackConfig:
    """Configuration for fallback strategies"""
    fallback_function: Optional[Callable] = None
    default_return_value: Any = None
    degrade_gracefully: bool = True
    degradation_message: str = "Service temporarily degraded"

class CircuitBreakerMetrics:
    """Collects and reports metrics for circuit breaker"""
    
    def __init__(self, name: str, metrics_collector: Optional[MetricsCollector] = None):
        self.name = name
        self.metrics = metrics_collector or MetricsCollector()
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.rejected_calls = 0
        self.circuit_opened_count = 0
        self.circuit_closed_count = 0
        self.circuit_half_opened_count = 0
        self.total_latency = 0.0
        self.last_state_change = datetime.now()
    
    def record_call(self, success: bool, latency: float, rejected: bool = False):
        """Record a call attempt"""
        self.total_calls += 1
        self.total_latency += latency
        
        if rejected:
            self.rejected_calls += 1
        elif success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
        
        # Update metrics collector
        tags = {"circuit": self.name}
        self.metrics.increment("circuit_breaker.calls_total", tags)
        if success:
            self.metrics.increment("circuit_breaker.calls_success", tags)
        elif rejected:
            self.metrics.increment("circuit_breaker.calls_rejected", tags)
        else:
            self.metrics.increment("circuit_breaker.calls_failed", tags)
        
        self.metrics.histogram("circuit_breaker.call_latency", latency, tags)
    
    def record_state_change(self, new_state: CircuitState):
        """Record circuit state change"""
        tags = {"circuit": self.name, "state": new_state.value}
        self.metrics.increment("circuit_breaker.state_changes", tags)
        
        if new_state == CircuitState.OPEN:
            self.circuit_opened_count += 1
        elif new_state == CircuitState.CLOSED:
            self.circuit_closed_count += 1
        elif new_state == CircuitState.HALF_OPEN:
            self.circuit_half_opened_count += 1
        
        self.last_state_change = datetime.now()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        avg_latency = (self.total_latency / self.total_calls 
                      if self.total_calls > 0 else 0.0)
        
        return {
            "name": self.name,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "rejected_calls": self.rejected_calls,
            "success_rate": (self.successful_calls / self.total_calls 
                           if self.total_calls > 0 else 0.0),
            "average_latency": avg_latency,
            "circuit_opened_count": self.circuit_opened_count,
            "circuit_closed_count": self.circuit_closed_count,
            "circuit_half_opened_count": self.circuit_half_opened_count,
            "last_state_change": self.last_state_change.isoformat()
        }

class CircuitBreaker:
    """
    Adaptive circuit breaker with retry policies, fallback strategies,
    and automatic degradation capabilities.
    """
    
    def __init__(
        self,
        name: str,
        circuit_config: Optional[CircuitBreakerConfig] = None,
        retry_config: Optional[RetryConfig] = None,
        fallback_config: Optional[FallbackConfig] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        cost_tracker: Optional[CostTracker] = None
    ):
        self.name = name
        self.circuit_config = circuit_config or CircuitBreakerConfig()
        self.retry_config = retry_config or RetryConfig()
        self.fallback_config = fallback_config or FallbackConfig()
        
        # Initialize monitoring
        self.tracer = TracingManager.get_tracer(f"circuit_breaker.{name}")
        self.metrics = CircuitBreakerMetrics(name, metrics_collector)
        self.cost_tracker = cost_tracker or CostTracker()
        
        # Circuit state
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_calls = 0
        
        # Health check
        self.last_health_check: Optional[datetime] = None
        self.health_check_task: Optional[asyncio.Task] = None
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        # Start health check if enabled
        if self.circuit_config.enable_health_check:
            self._start_health_check()
    
    async def _start_health_check(self):
        """Start periodic health checks"""
        async def health_check_loop():
            while True:
                await asyncio.sleep(self.circuit_config.health_check_interval)
                await self._perform_health_check()
        
        self.health_check_task = asyncio.create_task(health_check_loop())
    
    async def _perform_health_check(self):
        """Perform health check and adjust circuit state if needed"""
        async with self._lock:
            if self.state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if (self.last_failure_time and 
                    datetime.now() - self.last_failure_time > 
                    timedelta(seconds=self.circuit_config.recovery_timeout)):
                    logger.info(f"Circuit {self.name}: Attempting recovery (health check)")
                    self._transition_to_half_open()
    
    def _transition_to_open(self):
        """Transition circuit to OPEN state"""
        if self.state != CircuitState.OPEN:
            self.state = CircuitState.OPEN
            self.last_failure_time = datetime.now()
            self.metrics.record_state_change(CircuitState.OPEN)
            logger.warning(f"Circuit {self.name}: OPENED after {self.failure_count} failures")
    
    def _transition_to_half_open(self):
        """Transition circuit to HALF_OPEN state"""
        self.state = CircuitState.HALF_OPEN
        self.half_open_calls = 0
        self.success_count = 0
        self.metrics.record_state_change(CircuitState.HALF_OPEN)
        logger.info(f"Circuit {self.name}: HALF_OPEN for recovery testing")
    
    def _transition_to_closed(self):
        """Transition circuit to CLOSED state"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
        self.metrics.record_state_change(CircuitState.CLOSED)
        logger.info(f"Circuit {self.name}: CLOSED - service recovered")
    
    def _should_allow_request(self) -> bool:
        """Check if request should be allowed based on circuit state"""
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if (self.last_failure_time and 
                datetime.now() - self.last_failure_time > 
                timedelta(seconds=self.circuit_config.recovery_timeout)):
                self._transition_to_half_open()
                return True
            return False
        elif self.state == CircuitState.HALF_OPEN:
            return self.half_open_calls < self.circuit_config.half_open_max_calls
        return False
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and optional jitter"""
        delay = min(
            self.retry_config.base_delay * (self.retry_config.exponential_base ** attempt),
            self.retry_config.max_delay
        )
        
        if self.retry_config.jitter:
            # Add jitter: random value between 0 and delay
            delay = random.random() * delay
        
        return delay
    
    async def _execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(self.retry_config.max_attempts):
            try:
                # Check timeout
                if self.circuit_config.timeout > 0:
                    result = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=self.circuit_config.timeout
                    )
                else:
                    result = await func(*args, **kwargs)
                
                return result
                
            except asyncio.TimeoutError as e:
                last_exception = e
                logger.warning(f"Circuit {self.name}: Timeout on attempt {attempt + 1}")
                
            except Exception as e:
                last_exception = e
                
                # Check if we should retry this exception
                should_retry = any(
                    isinstance(e, exc_type) 
                    for exc_type in self.retry_config.retry_on_exceptions
                )
                
                if not should_retry or attempt == self.retry_config.max_attempts - 1:
                    raise
                
                logger.warning(f"Circuit {self.name}: Retry attempt {attempt + 1} after {type(e).__name__}")
            
            # Wait before retry (except on last attempt)
            if attempt < self.retry_config.max_attempts - 1:
                delay = self._calculate_retry_delay(attempt)
                await asyncio.sleep(delay)
        
        # All retries exhausted
        raise last_exception
    
    async def _handle_success(self):
        """Handle successful call"""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.circuit_config.success_threshold:
                    self._transition_to_closed()
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0  # Reset failure count on success
    
    async def _handle_failure(self, exception: Exception):
        """Handle failed call"""
        async with self._lock:
            # Check if this exception should be monitored
            should_monitor = any(
                isinstance(exception, exc_type)
                for exc_type in self.circuit_config.monitored_exceptions
            )
            
            if not should_monitor:
                return
            
            if self.state == CircuitState.HALF_OPEN:
                # Failure in half-open state opens the circuit immediately
                self._transition_to_open()
            elif self.state == CircuitState.CLOSED:
                self.failure_count += 1
                if self.failure_count >= self.circuit_config.failure_threshold:
                    self._transition_to_open()
    
    async def _get_fallback_result(self, exception: Exception) -> Any:
        """Get fallback result when circuit is open or call fails"""
        if self.fallback_config.fallback_function:
            try:
                if inspect.iscoroutinefunction(self.fallback_config.fallback_function):
                    return await self.fallback_config.fallback_function(exception)
                else:
                    return self.fallback_config.fallback_function(exception)
            except Exception as e:
                logger.error(f"Circuit {self.name}: Fallback function failed: {e}")
        
        if self.fallback_config.degrade_gracefully:
            logger.warning(f"Circuit {self.name}: {self.fallback_config.degradation_message}")
            return self.fallback_config.default_return_value
        
        raise exception
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Async function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result or fallback value
            
        Raises:
            CircuitBreakerOpenError: If circuit is open and no fallback available
            Exception: Original exception if all retries fail and no fallback
        """
        start_time = time.time()
        
        with self.tracer.start_span(f"circuit_breaker.{self.name}.call") as span:
            span.set_attribute("circuit.name", self.name)
            span.set_attribute("circuit.state", self.state.value)
            
            # Check if request should be allowed
            if not self._should_allow_request():
                self.metrics.record_call(False, 0, rejected=True)
                span.set_attribute("circuit.rejected", True)
                
                # Try fallback
                try:
                    return await self._get_fallback_result(
                        CircuitBreakerOpenError(f"Circuit {self.name} is OPEN")
                    )
                except Exception as fallback_error:
                    raise CircuitBreakerOpenError(
                        f"Circuit {self.name} is OPEN and fallback failed: {fallback_error}"
                    )
            
            # Increment half-open calls counter
            if self.state == CircuitState.HALF_OPEN:
                async with self._lock:
                    self.half_open_calls += 1
            
            # Track cost
            self.cost_tracker.track_operation(f"circuit_breaker.{self.name}.call")
            
            try:
                # Execute with retry logic
                result = await self._execute_with_retry(func, *args, **kwargs)
                
                # Record success
                latency = time.time() - start_time
                self.metrics.record_call(True, latency)
                await self._handle_success()
                
                span.set_attribute("circuit.success", True)
                span.set_attribute("circuit.latency", latency)
                
                return result
                
            except Exception as e:
                # Record failure
                latency = time.time() - start_time
                self.metrics.record_call(False, latency)
                await self._handle_failure(e)
                
                span.set_attribute("circuit.success", False)
                span.set_attribute("circuit.latency", latency)
                span.set_attribute("circuit.error", str(e))
                
                # Try fallback
                try:
                    return await self._get_fallback_result(e)
                except Exception as fallback_error:
                    raise e  # Re-raise original exception if fallback fails
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "half_open_calls": self.half_open_calls,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "metrics": self.metrics.get_stats(),
            "config": {
                "failure_threshold": self.circuit_config.failure_threshold,
                "recovery_timeout": self.circuit_config.recovery_timeout,
                "half_open_max_calls": self.circuit_config.half_open_max_calls,
                "timeout": self.circuit_config.timeout
            }
        }
    
    def reset(self):
        """Reset circuit breaker to initial state"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
        self.last_failure_time = None
        self.metrics.reset()
        logger.info(f"Circuit {self.name}: Reset to CLOSED state")
    
    def __del__(self):
        """Cleanup health check task"""
        if self.health_check_task:
            self.health_check_task.cancel()

class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""
    pass

class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers"""
    
    _instance = None
    _circuit_breakers: Dict[str, CircuitBreaker] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_or_create(
        cls,
        name: str,
        circuit_config: Optional[CircuitBreakerConfig] = None,
        retry_config: Optional[RetryConfig] = None,
        fallback_config: Optional[FallbackConfig] = None,
        **kwargs
    ) -> CircuitBreaker:
        """Get existing or create new circuit breaker"""
        if name not in cls._circuit_breakers:
            cls._circuit_breakers[name] = CircuitBreaker(
                name=name,
                circuit_config=circuit_config,
                retry_config=retry_config,
                fallback_config=fallback_config,
                **kwargs
            )
        return cls._circuit_breakers[name]
    
    @classmethod
    def get_all_states(cls) -> Dict[str, Dict[str, Any]]:
        """Get states of all registered circuit breakers"""
        return {
            name: cb.get_state()
            for name, cb in cls._circuit_breakers.items()
        }
    
    @classmethod
    def reset_all(cls):
        """Reset all circuit breakers"""
        for cb in cls._circuit_breakers.values():
            cb.reset()
    
    @classmethod
    def remove(cls, name: str):
        """Remove circuit breaker from registry"""
        if name in cls._circuit_breakers:
            del cls._circuit_breakers[name]

def circuit_breaker(
    name: Optional[str] = None,
    circuit_config: Optional[CircuitBreakerConfig] = None,
    retry_config: Optional[RetryConfig] = None,
    fallback_config: Optional[FallbackConfig] = None
):
    """
    Decorator for applying circuit breaker pattern to async functions.
    
    Args:
        name: Circuit breaker name (defaults to function name)
        circuit_config: Circuit breaker configuration
        retry_config: Retry policy configuration
        fallback_config: Fallback strategy configuration
    
    Example:
        @circuit_breaker(
            name="api_call",
            circuit_config=CircuitBreakerConfig(failure_threshold=3),
            retry_config=RetryConfig(max_attempts=2),
            fallback_config=FallbackConfig(default_return_value={"status": "degraded"})
        )
        async def call_external_api():
            # ... make API call
    """
    def decorator(func: Callable):
        cb_name = name or f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cb = CircuitBreakerRegistry.get_or_create(
                name=cb_name,
                circuit_config=circuit_config,
                retry_config=retry_config,
                fallback_config=fallback_config
            )
            return await cb.call(func, *args, **kwargs)
        
        # Attach circuit breaker to function for inspection
        wrapper.circuit_breaker = lambda: CircuitBreakerRegistry.get_or_create(
            name=cb_name,
            circuit_config=circuit_config,
            retry_config=retry_config,
            fallback_config=fallback_config
        )
        
        return wrapper
    
    return decorator

class ResilienceManager:
    """
    High-level manager for resilience patterns across the agent system.
    Integrates with existing monitoring and distributed modules.
    """
    
    def __init__(self):
        self.circuit_breaker_registry = CircuitBreakerRegistry()
        self.default_circuit_config = CircuitBreakerConfig()
        self.default_retry_config = RetryConfig()
        self.default_fallback_config = FallbackConfig()
        
        # Integration with existing modules
        self.metrics_collector = MetricsCollector()
        self.cost_tracker = CostTracker()
    
    def create_circuit_breaker(
        self,
        name: str,
        **config_overrides
    ) -> CircuitBreaker:
        """Create a new circuit breaker with default configuration"""
        circuit_config = CircuitBreakerConfig(**{
            **self.default_circuit_config.__dict__,
            **config_overrides.get('circuit_config', {})
        })
        
        retry_config = RetryConfig(**{
            **self.default_retry_config.__dict__,
            **config_overrides.get('retry_config', {})
        })
        
        fallback_config = FallbackConfig(**{
            **self.default_fallback_config.__dict__,
            **config_overrides.get('fallback_config', {})
        })
        
        return self.circuit_breaker_registry.get_or_create(
            name=name,
            circuit_config=circuit_config,
            retry_config=retry_config,
            fallback_config=fallback_config,
            metrics_collector=self.metrics_collector,
            cost_tracker=self.cost_tracker
        )
    
    def protect_service_call(
        self,
        service_name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Protect a service call with circuit breaker pattern.
        
        Args:
            service_name: Name of the service being called
            func: Async function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
        
        Returns:
            Function result or fallback value
        """
        cb = self.create_circuit_breaker(service_name)
        return cb.call(func, *args, **kwargs)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        circuit_states = self.circuit_breaker_registry.get_all_states()
        
        # Calculate overall health
        total_circuits = len(circuit_states)
        open_circuits = sum(
            1 for state in circuit_states.values()
            if state['state'] == CircuitState.OPEN.value
        )
        
        health_score = (
            (total_circuits - open_circuits) / total_circuits
            if total_circuits > 0 else 1.0
        )
        
        return {
            "health_score": health_score,
            "total_circuits": total_circuits,
            "open_circuits": open_circuits,
            "circuits": circuit_states,
            "timestamp": datetime.now().isoformat()
        }
    
    def enable_graceful_degradation(self, service_name: str, message: str = None):
        """Enable graceful degradation for a service"""
        cb = self.create_circuit_breaker(service_name)
        cb.fallback_config.degrade_gracefully = True
        if message:
            cb.fallback_config.degradation_message = message
        logger.info(f"Enabled graceful degradation for {service_name}")
    
    def reset_all_circuits(self):
        """Reset all circuit breakers"""
        self.circuit_breaker_registry.reset_all()
        logger.info("Reset all circuit breakers")

# Global resilience manager instance
resilience_manager = ResilienceManager()

# Integration with existing distributed modules
def integrate_with_distributed_systems():
    """
    Integration point for existing distributed system modules.
    This should be called during system initialization.
    """
    try:
        # Import existing distributed modules
        from core.distributed.executor import DistributedExecutor
        from core.distributed.consensus import ConsensusManager
        from core.distributed.state_manager import StateManager
        
        # Patch distributed executor with resilience
        original_execute = DistributedExecutor.execute
        
        async def resilient_execute(self, task, *args, **kwargs):
            """Execute task with circuit breaker protection"""
            task_name = getattr(task, '__name__', str(task))
            cb = resilience_manager.create_circuit_breaker(
                f"distributed_executor.{task_name}"
            )
            return await cb.call(original_execute, self, task, *args, **kwargs)
        
        DistributedExecutor.execute = resilient_execute
        
        logger.info("Integrated resilience framework with distributed systems")
        
    except ImportError as e:
        logger.warning(f"Could not integrate with distributed systems: {e}")

# Auto-initialization
try:
    integrate_with_distributed_systems()
except Exception as e:
    logger.error(f"Failed to initialize resilience integration: {e}")

# Export public API
__all__ = [
    'CircuitBreaker',
    'CircuitBreakerConfig',
    'RetryConfig',
    'FallbackConfig',
    'CircuitState',
    'CircuitBreakerOpenError',
    'CircuitBreakerRegistry',
    'ResilienceManager',
    'circuit_breaker',
    'resilience_manager'
]