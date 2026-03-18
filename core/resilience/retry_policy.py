"""core/resilience/retry_policy.py - Adaptive Agent Resilience Framework

Implements configurable resilience patterns with exponential backoff, jitter,
graceful degradation, health checks, and automatic agent restart capabilities.
Integrates with existing circuit breaker, monitoring, and distributed systems.
"""

import asyncio
import functools
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union
from datetime import datetime, timedelta

from core.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerState
from monitoring.tracing import get_tracer
from monitoring.metrics_collector import MetricsCollector
from monitoring.cost_tracker import track_cost

logger = logging.getLogger(__name__)
T = TypeVar('T')


class RetryStrategy(Enum):
    """Retry strategy enumeration."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    FIXED_DELAY = "fixed_delay"
    FIBONACCI = "fibonacci"
    LINEAR_BACKOFF = "linear_backoff"


class DegradationLevel(Enum):
    """Service degradation levels."""
    FULL_SERVICE = 0
    REDUCED_FUNCTIONALITY = 1
    ESSENTIAL_ONLY = 2
    READ_ONLY = 3
    OFFLINE = 4


@dataclass
class RetryConfig:
    """Configuration for retry policies."""
    max_retries: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    jitter: bool = True
    jitter_factor: float = 0.1
    retry_on_exceptions: List[Type[Exception]] = field(default_factory=lambda: [Exception])
    stop_on_exceptions: List[Type[Exception]] = field(default_factory=list)
    timeout: Optional[float] = None
    exponential_base: float = 2.0
    fibonacci_max: int = 10


@dataclass
class HealthCheckConfig:
    """Configuration for health checks."""
    check_interval: float = 30.0  # seconds
    failure_threshold: int = 3
    success_threshold: int = 2
    timeout: float = 5.0
    enabled: bool = True


@dataclass
class DegradationConfig:
    """Configuration for graceful degradation."""
    enabled: bool = True
    levels: Dict[DegradationLevel, Dict[str, Any]] = field(default_factory=dict)
    fallback_functions: Dict[str, Callable] = field(default_factory=dict)
    auto_degrade_on_failures: int = 5
    auto_recover_after: float = 300.0  # seconds


class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted."""
    def __init__(self, message: str, last_exception: Exception, attempts: int):
        super().__init__(message)
        self.last_exception = last_exception
        self.attempts = attempts


class HealthCheckFailedError(Exception):
    """Raised when health check fails beyond threshold."""
    pass


class AdaptiveRetryPolicy:
    """Adaptive retry policy with exponential backoff and jitter.
    
    Features:
    - Multiple retry strategies
    - Jitter to prevent thundering herd
    - Integration with circuit breakers
    - Adaptive delays based on error types
    - Cost-aware retries for LLM operations
    """
    
    def __init__(
        self,
        config: Optional[RetryConfig] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        self.config = config or RetryConfig()
        self.circuit_breaker = circuit_breaker
        self.metrics = metrics_collector
        self.tracer = get_tracer()
        self._fibonacci_cache = {}
        
    def calculate_delay(self, attempt: int, error: Optional[Exception] = None) -> float:
        """Calculate delay based on strategy and attempt number."""
        if self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        elif self.config.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.config.base_delay
        elif self.config.strategy == RetryStrategy.FIBONACCI:
            delay = self.config.base_delay * self._fibonacci(attempt)
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.base_delay * (attempt + 1)
        else:
            delay = self.config.base_delay
        
        # Apply maximum delay cap
        delay = min(delay, self.config.max_delay)
        
        # Apply jitter to prevent synchronized retries
        if self.config.jitter:
            jitter_range = delay * self.config.jitter_factor
            delay += random.uniform(-jitter_range, jitter_range)
            delay = max(0.1, delay)  # Ensure positive delay
        
        # Adaptive delay based on error type
        if error:
            delay = self._adapt_delay_for_error(delay, error)
        
        return delay
    
    def _fibonacci(self, n: int) -> int:
        """Calculate fibonacci number with caching."""
        if n in self._fibonacci_cache:
            return self._fibonacci_cache[n]
        
        if n <= 1:
            return n
        
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        
        self._fibonacci_cache[n] = b
        return b
    
    def _adapt_delay_for_error(self, base_delay: float, error: Exception) -> float:
        """Adapt delay based on error type and context."""
        error_type = type(error).__name__
        
        # Longer delays for rate limiting errors
        if "rate" in error_type.lower() or "limit" in error_type.lower():
            return base_delay * 2
        
        # Longer delays for timeout errors
        if "timeout" in error_type.lower():
            return base_delay * 1.5
        
        # Shorter delays for transient network errors
        if "network" in error_type.lower() or "connection" in error_type.lower():
            return base_delay * 0.8
        
        return base_delay
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if operation should be retried."""
        if attempt >= self.config.max_retries:
            return False
        
        # Check stop conditions
        for stop_exception in self.config.stop_on_exceptions:
            if isinstance(exception, stop_exception):
                return False
        
        # Check retry conditions
        for retry_exception in self.config.retry_on_exceptions:
            if isinstance(exception, retry_exception):
                return True
        
        return False
    
    async def execute_async(
        self,
        func: Callable[..., T],
        *args,
        operation_name: str = "unknown",
        **kwargs
    ) -> T:
        """Execute async function with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                # Check circuit breaker if available
                if self.circuit_breaker and not self.circuit_breaker.allow_request():
                    raise CircuitBreakerState(
                        f"Circuit breaker is {self.circuit_breaker.state.value}"
                    )
                
                # Execute with timeout if configured
                if self.config.timeout:
                    result = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=self.config.timeout
                    )
                else:
                    result = await func(*args, **kwargs)
                
                # Record success metrics
                if self.metrics:
                    self.metrics.record_success(
                        operation=operation_name,
                        attempt=attempt + 1
                    )
                
                if self.circuit_breaker:
                    self.circuit_breaker.record_success()
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Record failure metrics
                if self.metrics:
                    self.metrics.record_failure(
                        operation=operation_name,
                        attempt=attempt + 1,
                        error_type=type(e).__name__
                    )
                
                if self.circuit_breaker:
                    self.circuit_breaker.record_failure()
                
                # Check if we should retry
                if not self.should_retry(e, attempt):
                    break
                
                # Calculate and wait for delay
                delay = self.calculate_delay(attempt, e)
                logger.warning(
                    f"Attempt {attempt + 1}/{self.config.max_retries + 1} failed "
                    f"for {operation_name}. Retrying in {delay:.2f}s. Error: {e}"
                )
                
                await asyncio.sleep(delay)
        
        # All retries exhausted
        raise RetryExhaustedError(
            f"All {self.config.max_retries + 1} attempts failed for {operation_name}",
            last_exception,
            self.config.max_retries + 1
        )
    
    def execute_sync(
        self,
        func: Callable[..., T],
        *args,
        operation_name: str = "unknown",
        **kwargs
    ) -> T:
        """Execute sync function with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                # Check circuit breaker if available
                if self.circuit_breaker and not self.circuit_breaker.allow_request():
                    raise CircuitBreakerState(
                        f"Circuit breaker is {self.circuit_breaker.state.value}"
                    )
                
                # Execute with timeout if configured
                if self.config.timeout:
                    # For sync functions, we'd need a different timeout mechanism
                    result = func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Record success metrics
                if self.metrics:
                    self.metrics.record_success(
                        operation=operation_name,
                        attempt=attempt + 1
                    )
                
                if self.circuit_breaker:
                    self.circuit_breaker.record_success()
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Record failure metrics
                if self.metrics:
                    self.metrics.record_failure(
                        operation=operation_name,
                        attempt=attempt + 1,
                        error_type=type(e).__name__
                    )
                
                if self.circuit_breaker:
                    self.circuit_breaker.record_failure()
                
                # Check if we should retry
                if not self.should_retry(e, attempt):
                    break
                
                # Calculate and wait for delay
                delay = self.calculate_delay(attempt, e)
                logger.warning(
                    f"Attempt {attempt + 1}/{self.config.max_retries + 1} failed "
                    f"for {operation_name}. Retrying in {delay:.2f}s. Error: {e}"
                )
                
                time.sleep(delay)
        
        # All retries exhausted
        raise RetryExhaustedError(
            f"All {self.config.max_retries + 1} attempts failed for {operation_name}",
            last_exception,
            self.config.max_retries + 1
        )


class HealthMonitor:
    """Monitors health of nexus and services with automatic restart capabilities."""
    
    def __init__(
        self,
        agent_id: str,
        health_check_func: Callable[[], bool],
        restart_func: Callable[[], None],
        config: Optional[HealthCheckConfig] = None,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        self.agent_id = agent_id
        self.health_check_func = health_check_func
        self.restart_func = restart_func
        self.config = config or HealthCheckConfig()
        self.metrics = metrics_collector
        self.tracer = get_tracer()
        
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._is_healthy = True
        self._last_check_time = None
        self._restart_count = 0
        self._max_restarts = 5
        self._restart_window = 300  # 5 minutes
        self._restart_timestamps = []
        
        self._monitoring_task = None
        self._shutdown_event = asyncio.Event()
    
    async def start_monitoring(self):
        """Start continuous health monitoring."""
        if not self.config.enabled:
            return
        
        self._monitoring_task = asyncio.create_task(self._monitor_loop())
        logger.info(f"Started health monitoring for agent {self.agent_id}")
    
    async def stop_monitoring(self):
        """Stop health monitoring."""
        self._shutdown_event.set()
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info(f"Stopped health monitoring for agent {self.agent_id}")
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.config.check_interval)
                await self._perform_health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitor loop: {e}")
    
    async def _perform_health_check(self):
        """Perform a single health check."""
        self._last_check_time = datetime.now()
        
        try:
            # Execute health check with timeout
            is_healthy = await asyncio.wait_for(
                asyncio.to_thread(self.health_check_func),
                timeout=self.config.timeout
            )
            
            if is_healthy:
                await self._handle_healthy()
            else:
                await self._handle_unhealthy("Health check returned False")
                
        except asyncio.TimeoutError:
            await self._handle_unhealthy("Health check timed out")
        except Exception as e:
            await self._handle_unhealthy(f"Health check exception: {e}")
    
    async def _handle_healthy(self):
        """Handle successful health check."""
        self._consecutive_failures = 0
        self._consecutive_successes += 1
        
        if (not self._is_healthy and 
            self._consecutive_successes >= self.config.success_threshold):
            self._is_healthy = True
            logger.info(f"Agent {self.agent_id} recovered and is now healthy")
            
            if self.metrics:
                self.metrics.record_recovery(self.agent_id)
    
    async def _handle_unhealthy(self, reason: str):
        """Handle failed health check."""
        self._consecutive_successes = 0
        self._consecutive_failures += 1
        
        logger.warning(
            f"Agent {self.agent_id} health check failed ({self._consecutive_failures}/"
            f"{self.config.failure_threshold}): {reason}"
        )
        
        if self._consecutive_failures >= self.config.failure_threshold:
            self._is_healthy = False
            
            if self.metrics:
                self.metrics.record_health_failure(self.agent_id, reason)
            
            # Check if we should restart
            if self._should_restart():
                await self._restart_agent()
            else:
                logger.error(
                    f"Agent {self.agent_id} is unhealthy but restart limit reached"
                )
    
    def _should_restart(self) -> bool:
        """Determine if agent should be restarted."""
        now = time.time()
        
        # Clean old restart timestamps
        self._restart_timestamps = [
            ts for ts in self._restart_timestamps 
            if now - ts < self._restart_window
        ]
        
        # Check restart limits
        if len(self._restart_timestamps) >= self._max_restarts:
            return False
        
        return True
    
    async def _restart_agent(self):
        """Restart the agent."""
        logger.warning(f"Restarting agent {self.agent_id}")
        
        try:
            await asyncio.to_thread(self.restart_func)
            self._restart_timestamps.append(time.time())
            self._restart_count += 1
            self._consecutive_failures = 0
            
            logger.info(
                f"Agent {self.agent_id} restarted successfully "
                f"(restart #{self._restart_count})"
            )
            
            if self.metrics:
                self.metrics.record_restart(self.agent_id)
                
        except Exception as e:
            logger.error(f"Failed to restart agent {self.agent_id}: {e}")
            if self.metrics:
                self.metrics.record_restart_failure(self.agent_id, str(e))
    
    @property
    def is_healthy(self) -> bool:
        """Check if agent is currently healthy."""
        return self._is_healthy
    
    def get_status(self) -> Dict[str, Any]:
        """Get current health status."""
        return {
            "agent_id": self.agent_id,
            "is_healthy": self._is_healthy,
            "consecutive_failures": self._consecutive_failures,
            "consecutive_successes": self._consecutive_successes,
            "last_check_time": self._last_check_time.isoformat() if self._last_check_time else None,
            "restart_count": self._restart_count,
            "restarts_in_window": len(self._restart_timestamps)
        }


class GracefulDegradationManager:
    """Manages graceful degradation of services based on failure patterns."""
    
    def __init__(
        self,
        service_name: str,
        config: Optional[DegradationConfig] = None,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        self.service_name = service_name
        self.config = config or DegradationConfig()
        self.metrics = metrics_collector
        
        self._current_level = DegradationLevel.FULL_SERVICE
        self._failure_counts: Dict[str, int] = {}
        self._last_degradation_time: Optional[datetime] = None
        self._degradation_lock = asyncio.Lock()
    
    async def record_failure(self, operation: str, error: Exception):
        """Record a failure for degradation decisions."""
        if not self.config.enabled:
            return
        
        async with self._degradation_lock:
            self._failure_counts[operation] = self._failure_counts.get(operation, 0) + 1
            
            total_failures = sum(self._failure_counts.values())
            
            if (total_failures >= self.config.auto_degrade_on_failures and
                self._current_level == DegradationLevel.FULL_SERVICE):
                await self._degrade_service()
    
    async def record_success(self, operation: str):
        """Record a success for recovery decisions."""
        if not self.config.enabled:
            return
        
        async with self._degradation_lock:
            if operation in self._failure_counts:
                self._failure_counts[operation] = max(
                    0, self._failure_counts[operation] - 1
                )
            
            # Check if we should recover
            if (self._current_level != DegradationLevel.FULL_SERVICE and
                self._should_recover()):
                await self._recover_service()
    
    async def _degrade_service(self):
        """Degrade service to next level."""
        next_level_value = min(
            self._current_level.value + 1,
            DegradationLevel.OFFLINE.value
        )
        next_level = DegradationLevel(next_level_value)
        
        if next_level != self._current_level:
            logger.warning(
                f"Degrading {self.service_name} from {self._current_level.name} "
                f"to {next_level.name}"
            )
            
            self._current_level = next_level
            self._last_degradation_time = datetime.now()
            self._failure_counts.clear()
            
            if self.metrics:
                self.metrics.record_degradation(
                    self.service_name,
                    self._current_level.name
                )
    
    async def _recover_service(self):
        """Recover service to previous level."""
        prev_level_value = max(
            self._current_level.value - 1,
            DegradationLevel.FULL_SERVICE.value
        )
        prev_level = DegradationLevel(prev_level_value)
        
        if prev_level != self._current_level:
            logger.info(
                f"Recovering {self.service_name} from {self._current_level.name} "
                f"to {prev_level.name}"
            )
            
            self._current_level = prev_level
            self._failure_counts.clear()
            
            if self.metrics:
                self.metrics.record_recovery(
                    self.service_name,
                    self._current_level.name
                )
    
    def _should_recover(self) -> bool:
        """Determine if service should recover."""
        if not self._last_degradation_time:
            return False
        
        time_since_degradation = (
            datetime.now() - self._last_degradation_time
        ).total_seconds()
        
        return time_since_degradation >= self.config.auto_recover_after
    
    def get_fallback(self, operation: str) -> Optional[Callable]:
        """Get fallback function for operation at current degradation level."""
        if self._current_level == DegradationLevel.FULL_SERVICE:
            return None
        
        level_config = self.config.levels.get(self._current_level, {})
        
        # Check if operation is disabled at this level
        if operation in level_config.get("disabled_operations", []):
            return lambda: None
        
        # Return specific fallback if available
        fallback_key = f"{operation}_{self._current_level.name}"
        if fallback_key in self.config.fallback_functions:
            return self.config.fallback_functions[fallback_key]
        
        # Return default fallback for level
        default_fallback = level_config.get("default_fallback")
        if default_fallback and default_fallback in self.config.fallback_functions:
            return self.config.fallback_functions[default_fallback]
        
        return None
    
    def get_current_level(self) -> DegradationLevel:
        """Get current degradation level."""
        return self._current_level
    
    def is_operation_allowed(self, operation: str) -> bool:
        """Check if operation is allowed at current degradation level."""
        if self._current_level == DegradationLevel.FULL_SERVICE:
            return True
        
        level_config = self.config.levels.get(self._current_level, {})
        disabled_ops = level_config.get("disabled_operations", [])
        
        return operation not in disabled_ops


class ResilientAgent:
    """Wrapper that adds resilience patterns to any agent or service."""
    
    def __init__(
        self,
        agent_id: str,
        agent_func: Callable[..., Any],
        retry_policy: Optional[AdaptiveRetryPolicy] = None,
        health_monitor: Optional[HealthMonitor] = None,
        degradation_manager: Optional[GracefulDegradationManager] = None,
        circuit_breaker: Optional[CircuitBreaker] = None
    ):
        self.agent_id = agent_id
        self.agent_func = agent_func
        self.retry_policy = retry_policy or AdaptiveRetryPolicy()
        self.health_monitor = health_monitor
        self.degradation_manager = degradation_manager
        self.circuit_breaker = circuit_breaker
        
        self._tracer = get_tracer()
        self._metrics = MetricsCollector.get_instance()
    
    async def execute(self, *args, operation_name: str = None, **kwargs) -> Any:
        """Execute agent function with full resilience patterns."""
        op_name = operation_name or f"{self.agent_id}_operation"
        
        # Check if operation is allowed at current degradation level
        if (self.degradation_manager and 
            not self.degradation_manager.is_operation_allowed(op_name)):
            
            fallback = self.degradation_manager.get_fallback(op_name)
            if fallback:
                logger.info(f"Using fallback for {op_name} due to degradation")
                return await asyncio.to_thread(fallback, *args, **kwargs)
            else:
                raise RuntimeError(
                    f"Operation {op_name} not available at current degradation level"
                )
        
        # Execute with retry policy
        try:
            result = await self.retry_policy.execute_async(
                self.agent_func,
                *args,
                operation_name=op_name,
                **kwargs
            )
            
            # Record success for degradation manager
            if self.degradation_manager:
                await self.degradation_manager.record_success(op_name)
            
            return result
            
        except Exception as e:
            # Record failure for degradation manager
            if self.degradation_manager:
                await self.degradation_manager.record_failure(op_name, e)
            
            # Try fallback if available
            if self.degradation_manager:
                fallback = self.degradation_manager.get_fallback(op_name)
                if fallback:
                    logger.warning(
                        f"Primary execution failed, using fallback for {op_name}: {e}"
                    )
                    try:
                        return await asyncio.to_thread(fallback, *args, **kwargs)
                    except Exception as fallback_error:
                        logger.error(
                            f"Fallback also failed for {op_name}: {fallback_error}"
                        )
                        raise fallback_error
            
            raise
    
    async def start(self):
        """Start all resilience components."""
        if self.health_monitor:
            await self.health_monitor.start_monitoring()
        
        logger.info(f"Resilient agent {self.agent_id} started")
    
    async def stop(self):
        """Stop all resilience components."""
        if self.health_monitor:
            await self.health_monitor.stop_monitoring()
        
        logger.info(f"Resilient agent {self.agent_id} stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all resilience components."""
        status = {
            "agent_id": self.agent_id,
            "circuit_breaker": None,
            "health_monitor": None,
            "degradation_level": None
        }
        
        if self.circuit_breaker:
            status["circuit_breaker"] = {
                "state": self.circuit_breaker.state.value,
                "failure_count": self.circuit_breaker.failure_count,
                "success_count": self.circuit_breaker.success_count
            }
        
        if self.health_monitor:
            status["health_monitor"] = self.health_monitor.get_status()
        
        if self.degradation_manager:
            status["degradation_level"] = self.degradation_manager.get_current_level().name
        
        return status


# Decorator for easy application of retry logic
def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
    jitter: bool = True
):
    """Decorator to add retry logic to any function."""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            config = RetryConfig(
                max_retries=max_retries,
                base_delay=base_delay,
                strategy=strategy,
                jitter=jitter
            )
            policy = AdaptiveRetryPolicy(config)
            return await policy.execute_async(func, *args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            config = RetryConfig(
                max_retries=max_retries,
                base_delay=base_delay,
                strategy=strategy,
                jitter=jitter
            )
            policy = AdaptiveRetryPolicy(config)
            return policy.execute_sync(func, *args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Integration with cost tracking for LLM operations
@track_cost
def resilient_llm_call(
    prompt: str,
    model: str = "gpt-4",
    max_retries: int = 3,
    **kwargs
) -> str:
    """Example of resilient LLM call with cost tracking."""
    # This would integrate with actual LLM client
    # For now, it's a placeholder showing the pattern
    import openai
    
    config = RetryConfig(
        max_retries=max_retries,
        retry_on_exceptions=[
            openai.error.RateLimitError,
            openai.error.APIError,
            openai.error.Timeout
        ],
        stop_on_exceptions=[
            openai.error.AuthenticationError,
            openai.error.InvalidRequestError
        ]
    )
    
    policy = AdaptiveRetryPolicy(config)
    
    def call_llm():
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content
    
    return policy.execute_sync(call_llm, operation_name="llm_completion")


# Factory function for creating resilient nexus
def create_resilient_agent(
    agent_id: str,
    agent_func: Callable,
    retry_config: Optional[RetryConfig] = None,
    health_check_func: Optional[Callable] = None,
    restart_func: Optional[Callable] = None,
    degradation_config: Optional[DegradationConfig] = None
) -> ResilientAgent:
    """Factory function to create a fully resilient agent."""
    
    # Create retry policy
    retry_policy = AdaptiveRetryPolicy(
        config=retry_config,
        metrics_collector=MetricsCollector.get_instance()
    )
    
    # Create circuit breaker
    circuit_breaker = CircuitBreaker(
        name=f"{agent_id}_circuit_breaker",
        metrics_collector=MetricsCollector.get_instance()
    )
    
    # Create health monitor if functions provided
    health_monitor = None
    if health_check_func and restart_func:
        health_monitor = HealthMonitor(
            agent_id=agent_id,
            health_check_func=health_check_func,
            restart_func=restart_func,
            metrics_collector=MetricsCollector.get_instance()
        )
    
    # Create degradation manager
    degradation_manager = None
    if degradation_config:
        degradation_manager = GracefulDegradationManager(
            service_name=agent_id,
            config=degradation_config,
            metrics_collector=MetricsCollector.get_instance()
        )
    
    # Create and return resilient agent
    return ResilientAgent(
        agent_id=agent_id,
        agent_func=agent_func,
        retry_policy=retry_policy,
        health_monitor=health_monitor,
        degradation_manager=degradation_manager,
        circuit_breaker=circuit_breaker
    )


# Example usage and integration points
if __name__ == "__main__":
    # Example 1: Simple retry decorator
    @with_retry(max_retries=3, base_delay=0.5)
    def flaky_operation():
        import random
        if random.random() < 0.7:
            raise ConnectionError("Network issue")
        return "Success"
    
    # Example 2: Full resilient agent
    async def example_agent_operation(data: str) -> str:
        """Example operation that might fail."""
        # Simulate processing
        await asyncio.sleep(0.1)
        if "fail" in data:
            raise ValueError("Invalid data")
        return f"Processed: {data}"
    
    async def health_check() -> bool:
        """Example health check."""
        # Check if agent can process basic requests
        try:
            result = await example_agent_operation("test")
            return "Processed" in result
        except:
            return False
    
    def restart_agent():
        """Example restart function."""
        print("Restarting agent...")
        # Actual restart logic would go here
    
    # Create degradation config
    degradation_config = DegradationConfig(
        levels={
            DegradationLevel.REDUCED_FUNCTIONALITY: {
                "disabled_operations": ["expensive_operation"],
                "default_fallback": "basic_fallback"
            },
            DegradationLevel.ESSENTIAL_ONLY: {
                "disabled_operations": ["expensive_operation", "secondary_operation"],
                "default_fallback": "minimal_fallback"
            }
        },
        fallback_functions={
            "basic_fallback": lambda x: f"Fallback: {x}",
            "minimal_fallback": lambda x: "Service temporarily unavailable"
        }
    )
    
    # Create resilient agent
    agent = create_resilient_agent(
        agent_id="example_agent",
        agent_func=example_agent_operation,
        retry_config=RetryConfig(max_retries=2),
        health_check_func=health_check,
        restart_func=restart_agent,
        degradation_config=degradation_config
    )
    
    # Example async usage
    async def main():
        await agent.start()
        try:
            result = await agent.execute("test data", operation_name="process_data")
            print(f"Result: {result}")
        finally:
            await agent.stop()
    
    # Run example
    # asyncio.run(main())