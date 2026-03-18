# nexus/resilience/fallback_manager.py
"""
Circuit Breaker & Resilience Layer for Browser Automation
Provides comprehensive error recovery with circuit breakers, automatic retry,
session persistence, and graceful degradation for browser instances.
"""

import asyncio
import json
import logging
import time
import traceback
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, List, Callable, Awaitable, TypeVar, Generic
from datetime import datetime, timedelta
import pickle
import aiofiles
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Circuit tripped, failing fast
    HALF_OPEN = "half_open"  # Testing if service recovered


class OperationType(Enum):
    """Types of browser operations for configurable policies"""
    CDP_COMMAND = "cdp_command"
    NAVIGATION = "navigation"
    ELEMENT_INTERACTION = "element_interaction"
    EVALUATION = "evaluation"
    SCREENSHOT = "screenshot"
    SESSION_MANAGEMENT = "session_management"


@dataclass
class ResiliencePolicy:
    """Configurable resilience policy for different operation types"""
    max_retries: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 30.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0  # seconds
    circuit_breaker_reset_timeout: float = 30.0  # seconds
    health_check_interval: float = 30.0  # seconds
    timeout: float = 30.0  # operation timeout in seconds


@dataclass
class OperationMetrics:
    """Metrics for tracking operation health"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    consecutive_failures: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    average_response_time: float = 0.0
    circuit_trips: int = 0


@dataclass
class SessionState:
    """Serializable browser session state for recovery"""
    session_id: str
    url: Optional[str] = None
    cookies: List[Dict[str, Any]] = field(default_factory=list)
    local_storage: Dict[str, str] = field(default_factory=dict)
    session_storage: Dict[str, str] = field(default_factory=dict)
    viewport: Optional[Dict[str, int]] = None
    user_agent: Optional[str] = None
    last_updated: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionState':
        """Create from dictionary"""
        return cls(**data)


class CircuitBreaker:
    """Circuit breaker implementation for preventing cascading failures"""
    
    def __init__(self, 
                 operation_type: OperationType,
                 policy: ResiliencePolicy,
                 name: str = "default"):
        self.operation_type = operation_type
        self.policy = policy
        self.name = name
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.last_success_time: Optional[float] = None
        self.metrics = OperationMetrics()
        self._lock = asyncio.Lock()
        
    async def execute(self, 
                     operation: Callable[[], Awaitable[T]],
                     operation_name: str = "unknown") -> T:
        """Execute operation with circuit breaker protection"""
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    logger.info(f"Circuit {self.name} transitioning to HALF_OPEN")
                else:
                    self.metrics.circuit_trips += 1
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker {self.name} is OPEN. "
                        f"Last failure: {self.last_failure_time}"
                    )
        
        start_time = time.time()
        self.metrics.total_calls += 1
        
        try:
            result = await asyncio.wait_for(
                operation(),
                timeout=self.policy.timeout
            )
            
            # Success handling
            async with self._lock:
                self.metrics.successful_calls += 1
                self.metrics.consecutive_failures = 0
                self.last_success_time = time.time()
                
                if self.state == CircuitState.HALF_OPEN:
                    self.state = CircuitState.CLOSED
                    logger.info(f"Circuit {self.name} reset to CLOSED")
                
                # Update average response time
                response_time = time.time() - start_time
                total = self.metrics.successful_calls
                self.metrics.average_response_time = (
                    (self.metrics.average_response_time * (total - 1) + response_time) / total
                )
            
            return result
            
        except asyncio.TimeoutError:
            await self._handle_failure(f"Operation {operation_name} timed out")
            raise
        except Exception as e:
            await self._handle_failure(f"Operation {operation_name} failed: {str(e)}")
            raise
    
    async def _handle_failure(self, error_msg: str):
        """Handle operation failure and update circuit state"""
        async with self._lock:
            self.failure_count += 1
            self.metrics.failed_calls += 1
            self.metrics.consecutive_failures += 1
            self.last_failure_time = time.time()
            self.metrics.last_failure_time = self.last_failure_time
            
            logger.warning(f"Circuit {self.name} failure #{self.failure_count}: {error_msg}")
            
            if (self.state == CircuitState.CLOSED and 
                self.failure_count >= self.policy.circuit_breaker_threshold):
                self.state = CircuitState.OPEN
                logger.error(f"Circuit {self.name} tripped to OPEN state")
                
            elif self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                logger.error(f"Circuit {self.name} returned to OPEN state after test failure")
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        return (time.time() - self.last_failure_time) >= self.policy.circuit_breaker_reset_timeout
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "metrics": asdict(self.metrics)
        }


class RetryHandler:
    """Handles retry logic with exponential backoff and jitter"""
    
    def __init__(self, policy: ResiliencePolicy):
        self.policy = policy
    
    async def execute_with_retry(self,
                               operation: Callable[[], Awaitable[T]],
                               operation_name: str = "unknown",
                               on_retry: Optional[Callable[[int, Exception], Awaitable[None]]] = None) -> T:
        """Execute operation with retry logic"""
        last_exception = None
        
        for attempt in range(self.policy.max_retries + 1):
            try:
                return await operation()
            except Exception as e:
                last_exception = e
                
                if attempt == self.policy.max_retries:
                    logger.error(f"Operation {operation_name} failed after {attempt + 1} attempts")
                    raise
                
                # Calculate delay with exponential backoff
                delay = min(
                    self.policy.base_delay * (self.policy.exponential_base ** attempt),
                    self.policy.max_delay
                )
                
                # Add jitter if enabled
                if self.policy.jitter:
                    import random
                    delay = delay * (0.5 + random.random())
                
                logger.warning(
                    f"Operation {operation_name} failed (attempt {attempt + 1}/{self.policy.max_retries + 1}), "
                    f"retrying in {delay:.2f}s: {str(e)}"
                )
                
                if on_retry:
                    await on_retry(attempt, e)
                
                await asyncio.sleep(delay)
        
        raise last_exception


class SessionPersistence:
    """Handles session state serialization and recovery"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path.home() / ".nexus" / "sessions"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
    
    async def save_session(self, session_state: SessionState) -> bool:
        """Save session state to disk"""
        try:
            async with self._lock:
                file_path = self.storage_path / f"{session_state.session_id}.json"
                async with aiofiles.open(file_path, 'w') as f:
                    await f.write(json.dumps(session_state.to_dict(), indent=2))
                logger.debug(f"Session {session_state.session_id} saved to {file_path}")
                return True
        except Exception as e:
            logger.error(f"Failed to save session {session_state.session_id}: {e}")
            return False
    
    async def load_session(self, session_id: str) -> Optional[SessionState]:
        """Load session state from disk"""
        try:
            async with self._lock:
                file_path = self.storage_path / f"{session_id}.json"
                if not file_path.exists():
                    return None
                
                async with aiofiles.open(file_path, 'r') as f:
                    data = json.loads(await f.read())
                    session_state = SessionState.from_dict(data)
                    logger.debug(f"Session {session_id} loaded from {file_path}")
                    return session_state
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None
    
    async def list_sessions(self) -> List[str]:
        """List all saved session IDs"""
        try:
            async with self._lock:
                sessions = []
                for file_path in self.storage_path.glob("*.json"):
                    sessions.append(file_path.stem)
                return sessions
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a saved session"""
        try:
            async with self._lock:
                file_path = self.storage_path / f"{session_id}.json"
                if file_path.exists():
                    file_path.unlink()
                    logger.debug(f"Session {session_id} deleted")
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False


class HealthChecker:
    """Monitors browser health and handles restarts"""
    
    def __init__(self, 
                 check_interval: float = 30.0,
                 max_restart_attempts: int = 3):
        self.check_interval = check_interval
        self.max_restart_attempts = max_restart_attempts
        self.restart_attempts = 0
        self.last_health_check: Optional[float] = None
        self.is_healthy = True
        self._check_task: Optional[asyncio.Task] = None
        self._restart_callback: Optional[Callable[[], Awaitable[bool]]] = None
    
    def set_restart_callback(self, callback: Callable[[], Awaitable[bool]]):
        """Set callback function for restarting browser"""
        self._restart_callback = callback
    
    async def start_monitoring(self):
        """Start health check monitoring"""
        if self._check_task is None or self._check_task.done():
            self._check_task = asyncio.create_task(self._monitor_loop())
            logger.info("Health monitoring started")
    
    async def stop_monitoring(self):
        """Stop health check monitoring"""
        if self._check_task and not self._check_task.done():
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
            logger.info("Health monitoring stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while True:
            await asyncio.sleep(self.check_interval)
            await self.perform_health_check()
    
    async def perform_health_check(self) -> bool:
        """Perform a health check"""
        try:
            # This would be implemented to actually check browser health
            # For now, we'll assume it's healthy if we can call this method
            self.is_healthy = True
            self.last_health_check = time.time()
            self.restart_attempts = 0
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self.is_healthy = False
            return False
    
    async def attempt_restart(self) -> bool:
        """Attempt to restart the browser"""
        if self.restart_attempts >= self.max_restart_attempts:
            logger.error(f"Max restart attempts ({self.max_restart_attempts}) reached")
            return False
        
        self.restart_attempts += 1
        logger.warning(f"Attempting browser restart (attempt {self.restart_attempts})")
        
        if self._restart_callback:
            try:
                success = await self._restart_callback()
                if success:
                    logger.info("Browser restart successful")
                    self.is_healthy = True
                    self.restart_attempts = 0
                    return True
                else:
                    logger.error("Browser restart failed")
                    return False
            except Exception as e:
                logger.error(f"Browser restart callback failed: {e}")
                return False
        else:
            logger.warning("No restart callback configured")
            return False


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""
    pass


class FallbackManager:
    """
    Main resilience manager that coordinates circuit breakers, retries,
    session persistence, and health checks for browser operations.
    """
    
    def __init__(self, 
                 default_policy: Optional[ResiliencePolicy] = None,
                 session_storage_path: Optional[Path] = None):
        self.default_policy = default_policy or ResiliencePolicy()
        self.policies: Dict[OperationType, ResiliencePolicy] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_handlers: Dict[OperationType, RetryHandler] = {}
        self.session_persistence = SessionPersistence(session_storage_path)
        self.health_checker = HealthChecker()
        self._operation_handlers: Dict[str, Callable] = {}
        
        # Initialize default policies for each operation type
        for op_type in OperationType:
            self.policies[op_type] = self.default_policy
            self.retry_handlers[op_type] = RetryHandler(self.policies[op_type])
    
    def set_policy(self, operation_type: OperationType, policy: ResiliencePolicy):
        """Set resilience policy for a specific operation type"""
        self.policies[operation_type] = policy
        self.retry_handlers[operation_type] = RetryHandler(policy)
        logger.info(f"Updated policy for {operation_type.value}")
    
    def get_circuit_breaker(self, name: str, operation_type: OperationType) -> CircuitBreaker:
        """Get or create a circuit breaker for the given name and operation type"""
        key = f"{name}_{operation_type.value}"
        if key not in self.circuit_breakers:
            policy = self.policies.get(operation_type, self.default_policy)
            self.circuit_breakers[key] = CircuitBreaker(operation_type, policy, name)
        return self.circuit_breakers[key]
    
    async def execute_with_resilience(self,
                                    operation: Callable[[], Awaitable[T]],
                                    operation_name: str,
                                    operation_type: OperationType,
                                    session_id: Optional[str] = None,
                                    save_session_on_success: bool = False,
                                    on_retry: Optional[Callable[[int, Exception], Awaitable[None]]] = None) -> T:
        """
        Execute an operation with full resilience protection.
        
        Args:
            operation: Async function to execute
            operation_name: Name for logging/metrics
            operation_type: Type of operation for policy selection
            session_id: Optional session ID for persistence
            save_session_on_success: Whether to save session after successful operation
            on_retry: Optional callback for retry events
            
        Returns:
            Result of the operation
            
        Raises:
            CircuitBreakerOpenError: If circuit breaker is open
            Exception: Last exception if all retries fail
        """
        circuit_breaker = self.get_circuit_breaker(operation_name, operation_type)
        retry_handler = self.retry_handlers.get(operation_type, self.retry_handlers[OperationType.CDP_COMMAND])
        
        async def protected_operation():
            return await circuit_breaker.execute(operation, operation_name)
        
        try:
            result = await retry_handler.execute_with_retry(
                protected_operation,
                operation_name,
                on_retry
            )
            
            # Save session on success if requested
            if save_session_on_success and session_id:
                # Note: Actual session state capture would need browser context
                # This is a placeholder for the integration point
                logger.debug(f"Session save requested for {session_id} after {operation_name}")
            
            return result
            
        except CircuitBreakerOpenError:
            # Attempt graceful degradation
            logger.warning(f"Circuit open for {operation_name}, attempting graceful degradation")
            return await self._graceful_degradation(operation_name, operation_type)
        
        except Exception as e:
            # Log the failure and potentially trigger health check
            logger.error(f"Operation {operation_name} failed after all retries: {e}")
            await self.health_checker.perform_health_check()
            raise
    
    async def _graceful_degradation(self, 
                                  operation_name: str,
                                  operation_type: OperationType) -> Any:
        """
        Handle graceful degradation when circuit is open.
        Returns a default value or raises a specific exception.
        """
        # Define degradation strategies per operation type
        degradation_strategies = {
            OperationType.SCREENSHOT: lambda: None,  # Return None for screenshots
            OperationType.EVALUATION: lambda: {},    # Return empty dict for evaluations
            OperationType.ELEMENT_INTERACTION: lambda: False,  # Return False for interactions
        }
        
        strategy = degradation_strategies.get(operation_type)
        if strategy:
            logger.warning(f"Degrading {operation_name} with default value")
            return strategy()
        
        # Default: raise an exception
        raise CircuitBreakerOpenError(
            f"Operation {operation_name} unavailable due to circuit breaker. "
            f"Service is degraded."
        )
    
    async def save_session_state(self, session_id: str, state_data: Dict[str, Any]) -> bool:
        """Save session state with metadata"""
        session_state = SessionState(
            session_id=session_id,
            metadata=state_data,
            last_updated=time.time()
        )
        return await self.session_persistence.save_session(session_state)
    
    async def load_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session state metadata"""
        session_state = await self.session_persistence.load_session(session_id)
        if session_state:
            return session_state.metadata
        return None
    
    async def recover_session(self, session_id: str) -> bool:
        """
        Attempt to recover a session.
        This would be integrated with actual browser recovery logic.
        """
        logger.info(f"Attempting to recover session {session_id}")
        session_state = await self.session_persistence.load_session(session_id)
        
        if not session_state:
            logger.warning(f"No saved state found for session {session_id}")
            return False
        
        # Check if session state is too old (e.g., > 24 hours)
        if time.time() - session_state.last_updated > 86400:
            logger.warning(f"Session {session_id} state is too old, discarding")
            await self.session_persistence.delete_session(session_id)
            return False
        
        # Actual recovery would happen here
        # This is a placeholder for integration with browser control
        logger.info(f"Session {session_id} recovery initiated")
        return True
    
    def get_all_circuit_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers"""
        return {name: cb.get_status() for name, cb in self.circuit_breakers.items()}
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        summary = {
            "total_circuits": len(self.circuit_breakers),
            "open_circuits": 0,
            "half_open_circuits": 0,
            "closed_circuits": 0,
            "operations_by_type": {}
        }
        
        for cb in self.circuit_breakers.values():
            if cb.state == CircuitState.OPEN:
                summary["open_circuits"] += 1
            elif cb.state == CircuitState.HALF_OPEN:
                summary["half_open_circuits"] += 1
            else:
                summary["closed_circuits"] += 1
            
            op_type = cb.operation_type.value
            if op_type not in summary["operations_by_type"]:
                summary["operations_by_type"][op_type] = {
                    "total_calls": 0,
                    "successful_calls": 0,
                    "failed_calls": 0
                }
            
            summary["operations_by_type"][op_type]["total_calls"] += cb.metrics.total_calls
            summary["operations_by_type"][op_type]["successful_calls"] += cb.metrics.successful_calls
            summary["operations_by_type"][op_type]["failed_calls"] += cb.metrics.failed_calls
        
        return summary
    
    async def cleanup_old_sessions(self, max_age_hours: float = 24.0):
        """Clean up sessions older than specified age"""
        sessions = await self.session_persistence.list_sessions()
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for session_id in sessions:
            session_state = await self.session_persistence.load_session(session_id)
            if session_state and (current_time - session_state.last_updated) > max_age_seconds:
                await self.session_persistence.delete_session(session_id)
                logger.info(f"Cleaned up old session: {session_id}")


# Decorator for easy integration with existing functions
def with_resilience(operation_type: OperationType,
                   operation_name: Optional[str] = None,
                   session_id: Optional[str] = None,
                   fallback_manager: Optional[FallbackManager] = None):
    """
    Decorator to add resilience to async functions.
    
    Usage:
        @with_resilience(OperationType.NAVIGATION)
        async def navigate_to_page(url: str):
            # navigation logic
    """
    def decorator(func: Callable[..., Awaitable[T]]):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            manager = fallback_manager or get_global_fallback_manager()
            name = operation_name or func.__name__
            
            async def operation():
                return await func(*args, **kwargs)
            
            return await manager.execute_with_resilience(
                operation=operation,
                operation_name=name,
                operation_type=operation_type,
                session_id=session_id
            )
        return wrapper
    return decorator


# Global instance for easy access
_global_fallback_manager: Optional[FallbackManager] = None


def get_global_fallback_manager() -> FallbackManager:
    """Get or create global fallback manager instance"""
    global _global_fallback_manager
    if _global_fallback_manager is None:
        _global_fallback_manager = FallbackManager()
    return _global_fallback_manager


def set_global_fallback_manager(manager: FallbackManager):
    """Set global fallback manager instance"""
    global _global_fallback_manager
    _global_fallback_manager = manager


# Integration helpers for existing modules
class BrowserResilienceMixin:
    """
    Mixin class to add resilience to browser-related classes.
    Can be mixed into existing actor or agent classes.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fallback_manager = get_global_fallback_manager()
        self._current_session_id: Optional[str] = None
    
    def set_session_id(self, session_id: str):
        """Set current session ID for persistence"""
        self._current_session_id = session_id
    
    async def execute_browser_operation(self,
                                      operation: Callable[[], Awaitable[T]],
                                      operation_name: str,
                                      operation_type: OperationType,
                                      **kwargs) -> T:
        """Execute a browser operation with resilience"""
        return await self.fallback_manager.execute_with_resilience(
            operation=operation,
            operation_name=operation_name,
            operation_type=operation_type,
            session_id=self._current_session_id,
            **kwargs
        )
    
    async def save_current_session(self, additional_data: Optional[Dict[str, Any]] = None) -> bool:
        """Save current session state"""
        if not self._current_session_id:
            return False
        
        data = additional_data or {}
        data['saved_at'] = datetime.now().isoformat()
        
        return await self.fallback_manager.save_session_state(
            self._current_session_id,
            data
        )


# Example configuration for different environments
def create_production_policy() -> ResiliencePolicy:
    """Create production-ready resilience policy"""
    return ResiliencePolicy(
        max_retries=5,
        base_delay=1.0,
        max_delay=60.0,
        exponential_base=2.0,
        jitter=True,
        circuit_breaker_threshold=10,
        circuit_breaker_timeout=120.0,
        circuit_breaker_reset_timeout=60.0,
        health_check_interval=60.0,
        timeout=45.0
    )


def create_development_policy() -> ResiliencePolicy:
    """Create development-friendly resilience policy"""
    return ResiliencePolicy(
        max_retries=2,
        base_delay=0.5,
        max_delay=10.0,
        exponential_base=1.5,
        jitter=False,
        circuit_breaker_threshold=3,
        circuit_breaker_timeout=30.0,
        circuit_breaker_reset_timeout=15.0,
        health_check_interval=30.0,
        timeout=30.0
    )


# Initialize with sensible defaults
def initialize_resilience_layer(environment: str = "production"):
    """Initialize the resilience layer with environment-specific settings"""
    if environment == "production":
        policy = create_production_policy()
    else:
        policy = create_development_policy()
    
    manager = FallbackManager(default_policy=policy)
    set_global_fallback_manager(manager)
    
    logger.info(f"Resilience layer initialized for {environment} environment")
    return manager