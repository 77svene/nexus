"""
Adaptive Agent Resilience Framework - Fallback Manager

This module implements comprehensive resilience patterns for SOVEREIGN nexus,
including circuit breakers, retry policies, fallback strategies, and graceful
degradation when dependencies fail.
"""

import asyncio
import time
import logging
import random
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import inspect

# Import existing resilience components
try:
    from .circuit_breaker import CircuitBreaker, CircuitBreakerState
    from .retry_policy import RetryPolicy, RetryStrategy
except ImportError:
    # Fallback implementations if imports fail
    class CircuitBreakerState(Enum):
        CLOSED = "closed"
        OPEN = "open"
        HALF_OPEN = "half_open"
    
    class CircuitBreaker:
        def __init__(self, failure_threshold=5, recovery_timeout=60, half_open_max_calls=3):
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.failure_threshold = failure_threshold
            self.recovery_timeout = recovery_timeout
            self.half_open_max_calls = half_open_max_calls
            self.last_failure_time = None
            self.half_open_calls = 0
        
        def can_execute(self):
            if self.state == CircuitBreakerState.CLOSED:
                return True
            elif self.state == CircuitBreakerState.OPEN:
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.half_open_calls = 0
                    return True
                return False
            else:  # HALF_OPEN
                return self.half_open_calls < self.half_open_max_calls
        
        def record_success(self):
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.half_open_calls += 1
                if self.half_open_calls >= self.half_open_max_calls:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
            elif self.state == CircuitBreakerState.CLOSED:
                self.failure_count = 0
        
        def record_failure(self):
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitBreakerState.CLOSED and self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
            elif self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
    
    class RetryStrategy(Enum):
        FIXED = "fixed"
        EXPONENTIAL = "exponential"
        FIBONACCI = "fibonacci"
        RANDOM = "random"
    
    class RetryPolicy:
        def __init__(self, max_retries=3, base_delay=1.0, max_delay=60.0, 
                     strategy=RetryStrategy.EXPONENTIAL, jitter=True):
            self.max_retries = max_retries
            self.base_delay = base_delay
            self.max_delay = max_delay
            self.strategy = strategy
            self.jitter = jitter
        
        def get_delay(self, attempt: int) -> float:
            if self.strategy == RetryStrategy.FIXED:
                delay = self.base_delay
            elif self.strategy == RetryStrategy.EXPONENTIAL:
                delay = self.base_delay * (2 ** (attempt - 1))
            elif self.strategy == RetryStrategy.FIBONACCI:
                if attempt <= 2:
                    delay = self.base_delay
                else:
                    a, b = self.base_delay, self.base_delay
                    for _ in range(2, attempt):
                        a, b = b, a + b
                    delay = b
            else:  # RANDOM
                delay = self.base_delay * random.uniform(0.5, 1.5)
            
            delay = min(delay, self.max_delay)
            
            if self.jitter:
                delay = delay * random.uniform(0.8, 1.2)
            
            return delay

# Import monitoring components
try:
    from monitoring.tracing import TracingContext, get_current_trace_id
    from monitoring.metrics_collector import MetricsCollector
    from monitoring.cost_tracker import CostTracker
except ImportError:
    # Fallback implementations
    class TracingContext:
        def __init__(self, operation_name: str):
            self.operation_name = operation_name
            self.trace_id = str(time.time())
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass
    
    def get_current_trace_id():
        return str(time.time())
    
    class MetricsCollector:
        def __init__(self):
            self.counters = {}
            self.histograms = {}
        
        def increment(self, name: str, tags: Dict = None):
            key = f"{name}:{tags}" if tags else name
            self.counters[key] = self.counters.get(key, 0) + 1
        
        def observe(self, name: str, value: float, tags: Dict = None):
            key = f"{name}:{tags}" if tags else name
            if key not in self.histograms:
                self.histograms[key] = []
            self.histograms[key].append(value)
    
    class CostTracker:
        def __init__(self):
            self.costs = {}
        
        def track(self, operation: str, cost: float):
            self.costs[operation] = self.costs.get(operation, 0) + cost

logger = logging.getLogger(__name__)


class DegradationLevel(Enum):
    """Levels of service degradation"""
    NONE = "none"  # Full functionality
    PARTIAL = "partial"  # Reduced functionality
    MINIMAL = "minimal"  # Basic functionality only
    CRITICAL = "critical"  # Emergency mode


@dataclass
class HealthStatus:
    """Health status of a component or dependency"""
    is_healthy: bool
    last_check: float
    consecutive_failures: int = 0
    error_message: Optional[str] = None
    degradation_level: DegradationLevel = DegradationLevel.NONE


@dataclass
class FallbackConfig:
    """Configuration for fallback behavior"""
    enable_circuit_breaker: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: float = 60.0
    
    enable_retry: bool = True
    max_retries: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    retry_jitter: bool = True
    
    enable_fallback: bool = True
    fallback_timeout: float = 5.0
    
    enable_health_checks: bool = True
    health_check_interval: float = 30.0
    health_check_timeout: float = 10.0
    
    enable_auto_restart: bool = False
    restart_threshold: int = 10
    restart_cooldown: float = 300.0
    
    degradation_strategy: DegradationLevel = DegradationLevel.PARTIAL


class FallbackManager:
    """
    Central manager for resilience patterns in SOVEREIGN nexus.
    
    Implements circuit breakers, retry policies, fallback strategies,
    and graceful degradation to ensure system stability under failure conditions.
    """
    
    def __init__(self, config: Optional[FallbackConfig] = None):
        self.config = config or FallbackConfig()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_policies: Dict[str, RetryPolicy] = {}
        self.health_statuses: Dict[str, HealthStatus] = {}
        self.fallback_handlers: Dict[str, Callable] = {}
        self.degradation_handlers: Dict[DegradationLevel, List[Callable]] = {
            level: [] for level in DegradationLevel
        }
        
        # Monitoring
        self.metrics = MetricsCollector()
        self.cost_tracker = CostTracker()
        
        # State tracking
        self.restart_counts: Dict[str, int] = {}
        self.last_restart_time: Dict[str, float] = {}
        
        # Background tasks
        self._health_check_task = None
        self._running = False
        
        logger.info("FallbackManager initialized with config: %s", self.config)
    
    async def start(self):
        """Start background health checks and monitoring"""
        if self._running:
            return
        
        self._running = True
        if self.config.enable_health_checks:
            self._health_check_task = asyncio.create_task(self._run_health_checks())
        logger.info("FallbackManager started")
    
    async def stop(self):
        """Stop background tasks"""
        self._running = False
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        logger.info("FallbackManager stopped")
    
    def get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Get or create a circuit breaker for a named operation"""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(
                failure_threshold=self.config.circuit_breaker_failure_threshold,
                recovery_timeout=self.config.circuit_breaker_recovery_timeout
            )
        return self.circuit_breakers[name]
    
    def get_retry_policy(self, name: str) -> RetryPolicy:
        """Get or create a retry policy for a named operation"""
        if name not in self.retry_policies:
            self.retry_policies[name] = RetryPolicy(
                max_retries=self.config.max_retries,
                base_delay=self.config.retry_base_delay,
                max_delay=self.config.retry_max_delay,
                strategy=self.config.retry_strategy,
                jitter=self.config.retry_jitter
            )
        return self.retry_policies[name]
    
    def register_fallback(self, operation_name: str, fallback_func: Callable):
        """Register a fallback function for an operation"""
        self.fallback_handlers[operation_name] = fallback_func
        logger.debug("Registered fallback for operation: %s", operation_name)
    
    def register_degradation_handler(self, level: DegradationLevel, handler: Callable):
        """Register a handler for a specific degradation level"""
        self.degradation_handlers[level].append(handler)
        logger.debug("Registered degradation handler for level: %s", level.value)
    
    def set_health_status(self, component: str, is_healthy: bool, 
                         error_message: Optional[str] = None):
        """Update health status for a component"""
        if component not in self.health_statuses:
            self.health_statuses[component] = HealthStatus(
                is_healthy=is_healthy,
                last_check=time.time(),
                error_message=error_message
            )
        else:
            status = self.health_statuses[component]
            status.is_healthy = is_healthy
            status.last_check = time.time()
            status.error_message = error_message
            
            if not is_healthy:
                status.consecutive_failures += 1
            else:
                status.consecutive_failures = 0
        
        # Update degradation level based on health
        self._update_degradation_level(component)
        
        logger.debug("Health status updated for %s: %s", component, is_healthy)
    
    def _update_degradation_level(self, component: str):
        """Update degradation level based on health status"""
        if component not in self.health_statuses:
            return
        
        status = self.health_statuses[component]
        
        # Determine degradation level based on consecutive failures
        if status.consecutive_failures == 0:
            new_level = DegradationLevel.NONE
        elif status.consecutive_failures < 3:
            new_level = DegradationLevel.PARTIAL
        elif status.consecutive_failures < 10:
            new_level = DegradationLevel.MINIMAL
        else:
            new_level = DegradationLevel.CRITICAL
        
        if new_level != status.degradation_level:
            status.degradation_level = new_level
            self._trigger_degradation_handlers(new_level, component)
    
    def _trigger_degradation_handlers(self, level: DegradationLevel, component: str):
        """Trigger handlers for a specific degradation level"""
        for handler in self.degradation_handlers.get(level, []):
            try:
                if inspect.iscoroutinefunction(handler):
                    asyncio.create_task(handler(component, level))
                else:
                    handler(component, level)
            except Exception as e:
                logger.error("Error in degradation handler: %s", e)
    
    async def execute_with_resilience(
        self,
        operation_name: str,
        func: Callable,
        *args,
        fallback_func: Optional[Callable] = None,
        health_check_component: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Execute a function with full resilience patterns.
        
        Args:
            operation_name: Name of the operation for metrics and circuit breaking
            func: The function to execute
            fallback_func: Optional fallback function if primary fails
            health_check_component: Component to check health before execution
            *args, **kwargs: Arguments to pass to the function
        
        Returns:
            Result from the function or fallback
        """
        trace_id = get_current_trace_id()
        
        with TracingContext(operation_name) as trace:
            # Check health if specified
            if health_check_component and self.config.enable_health_checks:
                if not self._check_health(health_check_component):
                    logger.warning(
                        "Health check failed for %s, using fallback",
                        health_check_component
                    )
                    return await self._execute_fallback(
                        operation_name, fallback_func, *args, **kwargs
                    )
            
            # Check circuit breaker
            if self.config.enable_circuit_breaker:
                circuit_breaker = self.get_circuit_breaker(operation_name)
                if not circuit_breaker.can_execute():
                    logger.warning(
                        "Circuit breaker open for %s, using fallback",
                        operation_name
                    )
                    self.metrics.increment(
                        "circuit_breaker.blocked",
                        {"operation": operation_name}
                    )
                    return await self._execute_fallback(
                        operation_name, fallback_func, *args, **kwargs
                    )
            
            # Execute with retry
            retry_policy = self.get_retry_policy(operation_name) if self.config.enable_retry else None
            
            last_exception = None
            attempt = 0
            
            while True:
                attempt += 1
                start_time = time.time()
                
                try:
                    # Execute the function
                    if inspect.iscoroutinefunction(func):
                        result = await asyncio.wait_for(
                            func(*args, **kwargs),
                            timeout=self.config.fallback_timeout
                        )
                    else:
                        result = func(*args, **kwargs)
                    
                    # Record success
                    execution_time = time.time() - start_time
                    self._record_success(operation_name, execution_time, attempt)
                    
                    return result
                    
                except asyncio.TimeoutError as e:
                    last_exception = e
                    logger.warning(
                        "Timeout in %s (attempt %d): %s",
                        operation_name, attempt, e
                    )
                    self._record_failure(operation_name, "timeout")
                    
                except Exception as e:
                    last_exception = e
                    logger.warning(
                        "Error in %s (attempt %d): %s",
                        operation_name, attempt, e
                    )
                    self._record_failure(operation_name, str(e))
                
                # Check if we should retry
                if retry_policy and attempt <= retry_policy.max_retries:
                    delay = retry_policy.get_delay(attempt)
                    logger.info(
                        "Retrying %s in %.2fs (attempt %d/%d)",
                        operation_name, delay, attempt, retry_policy.max_retries
                    )
                    await asyncio.sleep(delay)
                else:
                    break
            
            # All retries failed, use fallback
            logger.error(
                "All retries failed for %s: %s",
                operation_name, last_exception
            )
            
            return await self._execute_fallback(
                operation_name, fallback_func, *args, **kwargs
            )
    
    async def _execute_fallback(
        self,
        operation_name: str,
        fallback_func: Optional[Callable],
        *args,
        **kwargs
    ) -> Any:
        """Execute fallback strategy"""
        self.metrics.increment(
            "fallback.executed",
            {"operation": operation_name}
        )
        
        # Use provided fallback or registered one
        handler = fallback_func or self.fallback_handlers.get(operation_name)
        
        if handler:
            try:
                if inspect.iscoroutinefunction(handler):
                    return await handler(*args, **kwargs)
                else:
                    return handler(*args, **kwargs)
            except Exception as e:
                logger.error(
                    "Fallback failed for %s: %s",
                    operation_name, e
                )
                self.metrics.increment(
                    "fallback.failed",
                    {"operation": operation_name}
                )
                raise
        
        # No fallback available, check for auto-restart
        if self.config.enable_auto_restart:
            await self._check_auto_restart(operation_name)
        
        raise RuntimeError(
            f"Operation {operation_name} failed and no fallback available"
        )
    
    def _record_success(self, operation_name: str, execution_time: float, attempt: int):
        """Record successful execution"""
        if self.config.enable_circuit_breaker:
            circuit_breaker = self.get_circuit_breaker(operation_name)
            circuit_breaker.record_success()
        
        self.metrics.increment(
            "operation.success",
            {"operation": operation_name}
        )
        self.metrics.observe(
            "operation.duration",
            execution_time,
            {"operation": operation_name}
        )
        
        # Reset restart count on success
        if operation_name in self.restart_counts:
            self.restart_counts[operation_name] = 0
    
    def _record_failure(self, operation_name: str, error: str):
        """Record failed execution"""
        if self.config.enable_circuit_breaker:
            circuit_breaker = self.get_circuit_breaker(operation_name)
            circuit_breaker.record_failure()
        
        self.metrics.increment(
            "operation.failure",
            {"operation": operation_name, "error": error}
        )
    
    def _check_health(self, component: str) -> bool:
        """Check health status of a component"""
        if component not in self.health_statuses:
            return True  # Assume healthy if no status
        
        status = self.health_statuses[component]
        
        # Check if health check is stale
        if time.time() - status.last_check > self.config.health_check_interval * 2:
            logger.warning("Health check for %s is stale", component)
            return False
        
        return status.is_healthy
    
    async def _check_auto_restart(self, operation_name: str):
        """Check if auto-restart should be triggered"""
        if not self.config.enable_auto_restart:
            return
        
        # Increment restart count
        self.restart_counts[operation_name] = (
            self.restart_counts.get(operation_name, 0) + 1
        )
        
        # Check cooldown
        last_restart = self.last_restart_time.get(operation_name, 0)
        if time.time() - last_restart < self.config.restart_cooldown:
            return
        
        # Check threshold
        if self.restart_counts[operation_name] >= self.config.restart_threshold:
            logger.warning(
                "Auto-restart triggered for %s (count: %d)",
                operation_name, self.restart_counts[operation_name]
            )
            
            # Trigger restart handlers
            self._trigger_degradation_handlers(
                DegradationLevel.CRITICAL,
                operation_name
            )
            
            # Update restart tracking
            self.last_restart_time[operation_name] = time.time()
            self.restart_counts[operation_name] = 0
            
            self.metrics.increment(
                "auto_restart.triggered",
                {"operation": operation_name}
            )
    
    async def _run_health_checks(self):
        """Background task for periodic health checks"""
        while self._running:
            try:
                for component in list(self.health_statuses.keys()):
                    # In a real implementation, this would call actual health check endpoints
                    # For now, we just update the timestamp
                    self.health_statuses[component].last_check = time.time()
                
                await asyncio.sleep(self.config.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in health check task: %s", e)
                await asyncio.sleep(5)  # Backoff on error
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current resilience metrics"""
        return {
            "circuit_breakers": {
                name: {
                    "state": cb.state.value,
                    "failure_count": cb.failure_count
                }
                for name, cb in self.circuit_breakers.items()
            },
            "health_statuses": {
                name: {
                    "is_healthy": status.is_healthy,
                    "consecutive_failures": status.consecutive_failures,
                    "degradation_level": status.degradation_level.value
                }
                for name, status in self.health_statuses.items()
            },
            "restart_counts": dict(self.restart_counts),
            "metrics": {
                "counters": dict(self.metrics.counters),
                "histograms": {
                    k: {
                        "count": len(v),
                        "avg": sum(v) / len(v) if v else 0,
                        "min": min(v) if v else 0,
                        "max": max(v) if v else 0
                    }
                    for k, v in self.metrics.histograms.items()
                }
            }
        }


# Decorator for easy resilience wrapping
def with_resilience(
    operation_name: Optional[str] = None,
    fallback_func: Optional[Callable] = None,
    health_check_component: Optional[str] = None,
    config: Optional[FallbackConfig] = None
):
    """
    Decorator to wrap a function with resilience patterns.
    
    Usage:
        @with_resilience(operation_name="api_call", fallback_func=default_response)
        async def call_external_api():
            ...
    """
    def decorator(func):
        nonlocal operation_name
        if operation_name is None:
            operation_name = func.__name__
        
        manager = FallbackManager(config)
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await manager.execute_with_resilience(
                operation_name=operation_name,
                func=func,
                fallback_func=fallback_func,
                health_check_component=health_check_component,
                *args,
                **kwargs
            )
        
        # Attach manager for access
        wrapper.resilience_manager = manager
        return wrapper
    
    return decorator


# Global instance for easy access
_global_fallback_manager: Optional[FallbackManager] = None


def get_fallback_manager(config: Optional[FallbackConfig] = None) -> FallbackManager:
    """Get or create the global fallback manager instance"""
    global _global_fallback_manager
    if _global_fallback_manager is None:
        _global_fallback_manager = FallbackManager(config)
    return _global_fallback_manager


async def execute_with_resilience(
    operation_name: str,
    func: Callable,
    *args,
    fallback_func: Optional[Callable] = None,
    health_check_component: Optional[str] = None,
    config: Optional[FallbackConfig] = None,
    **kwargs
) -> Any:
    """
    Convenience function to execute with resilience using global manager.
    
    This is the main entry point for integrating resilience patterns
    into existing SOVEREIGN agent code.
    """
    manager = get_fallback_manager(config)
    return await manager.execute_with_resilience(
        operation_name=operation_name,
        func=func,
        fallback_func=fallback_func,
        health_check_component=health_check_component,
        *args,
        **kwargs
    )


# Integration helpers for existing codebase
def integrate_with_executor():
    """Integrate fallback manager with the distributed executor"""
    try:
        from core.distributed.executor import DistributedExecutor
        
        original_execute = DistributedExecutor.execute
        
        async def resilient_execute(self, task, *args, **kwargs):
            operation_name = f"executor.{task.__name__}"
            return await execute_with_resilience(
                operation_name=operation_name,
                func=original_execute,
                fallback_func=lambda *a, **kw: None,
                *args,
                **kwargs
            )
        
        DistributedExecutor.execute = resilient_execute
        logger.info("Integrated fallback manager with DistributedExecutor")
    except ImportError:
        logger.warning("Could not integrate with DistributedExecutor")


def integrate_with_state_manager():
    """Integrate fallback manager with the state manager"""
    try:
        from core.distributed.state_manager import StateManager
        
        original_get_state = StateManager.get_state
        original_set_state = StateManager.set_state
        
        async def resilient_get_state(self, key, default=None):
            return await execute_with_resilience(
                operation_name="state_manager.get_state",
                func=original_get_state,
                fallback_func=lambda *a, **kw: default,
                health_check_component="state_manager",
                self=self, key=key, default=default
            )
        
        async def resilient_set_state(self, key, value):
            return await execute_with_resilience(
                operation_name="state_manager.set_state",
                func=original_set_state,
                fallback_func=lambda *a, **kw: None,
                health_check_component="state_manager",
                self=self, key=key, value=value
            )
        
        StateManager.get_state = resilient_get_state
        StateManager.set_state = resilient_set_state
        logger.info("Integrated fallback manager with StateManager")
    except ImportError:
        logger.warning("Could not integrate with StateManager")


# Auto-initialize on import
def initialize_resilience():
    """Initialize resilience patterns for the SOVEREIGN system"""
    manager = get_fallback_manager()
    asyncio.create_task(manager.start())
    
    # Integrate with existing modules
    integrate_with_executor()
    integrate_with_state_manager()
    
    logger.info("Resilience framework initialized")


# Example degradation handlers
async def handle_partial_degradation(component: str, level: DegradationLevel):
    """Handle partial degradation by reducing functionality"""
    logger.warning("Partial degradation for %s", component)
    # Implement partial degradation logic here


async def handle_minimal_degradation(component: str, level: DegradationLevel):
    """Handle minimal degradation by switching to basic mode"""
    logger.error("Minimal degradation for %s", component)
    # Implement minimal degradation logic here


async def handle_critical_degradation(component: str, level: DegradationLevel):
    """Handle critical degradation by entering emergency mode"""
    logger.critical("Critical degradation for %s", component)
    # Implement critical degradation logic here
    # This might include notifying operators, switching to backup systems, etc.


# Register default degradation handlers
def register_default_handlers():
    """Register default degradation handlers"""
    manager = get_fallback_manager()
    manager.register_degradation_handler(DegradationLevel.PARTIAL, handle_partial_degradation)
    manager.register_degradation_handler(DegradationLevel.MINIMAL, handle_minimal_degradation)
    manager.register_degradation_handler(DegradationLevel.CRITICAL, handle_critical_degradation)


# Initialize on module import
try:
    register_default_handlers()
except Exception as e:
    logger.warning("Could not register default handlers: %s", e)