"""Resilience layer for browser automation with circuit breakers and session recovery."""

import asyncio
import json
import logging
import pickle
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union
import aiofiles
import aiofiles.os

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Circuit is open, calls fail fast
    HALF_OPEN = "half_open"  # Testing if service is back


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5  # Failures before opening circuit
    reset_timeout: float = 30.0  # Seconds before trying half-open
    half_open_max_calls: int = 3  # Max calls in half-open state
    excluded_exceptions: Set[Type[Exception]] = field(default_factory=set)
    success_threshold: int = 2  # Successes needed to close circuit from half-open


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    initial_delay: float = 1.0  # seconds
    max_delay: float = 30.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: Set[Type[Exception]] = field(default_factory=lambda: {
        ConnectionError, TimeoutError, OSError
    })


@dataclass
class SessionState:
    """Serializable session state for recovery."""
    session_id: str
    cookies: List[Dict[str, Any]] = field(default_factory=list)
    local_storage: Dict[str, str] = field(default_factory=dict)
    session_storage: Dict[str, str] = field(default_factory=dict)
    current_url: Optional[str] = None
    viewport_size: Optional[Dict[str, int]] = None
    user_agent: Optional[str] = None
    last_activity: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "cookies": self.cookies,
            "local_storage": self.local_storage,
            "session_storage": self.session_storage,
            "current_url": self.current_url,
            "viewport_size": self.viewport_size,
            "user_agent": self.user_agent,
            "last_activity": self.last_activity,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        """Create from dictionary."""
        return cls(
            session_id=data["session_id"],
            cookies=data.get("cookies", []),
            local_storage=data.get("local_storage", {}),
            session_storage=data.get("session_storage", {}),
            current_url=data.get("current_url"),
            viewport_size=data.get("viewport_size"),
            user_agent=data.get("user_agent"),
            last_activity=data.get("last_activity", time.time()),
            metadata=data.get("metadata", {}),
        )


class CircuitBreaker:
    """Circuit breaker pattern implementation for CDP connections."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0
        self._lock = asyncio.Lock()
        
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                    logger.info(f"Circuit {self.name} transitioning to HALF_OPEN")
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit {self.name} is OPEN. "
                        f"Will retry in {self._time_until_reset():.1f}s"
                    )
            
            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls >= self.config.half_open_max_calls:
                    raise CircuitBreakerOpenError(
                        f"Circuit {self.name} is HALF_OPEN with max calls reached"
                    )
                self.half_open_calls += 1
        
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure(e)
            raise
    
    async def _on_success(self) -> None:
        """Handle successful call."""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self._reset()
                    logger.info(f"Circuit {self.name} closed after successful recovery")
            else:
                self.failure_count = 0
    
    async def _on_failure(self, exception: Exception) -> None:
        """Handle failed call."""
        if any(isinstance(exception, exc_type) for exc_type in self.config.excluded_exceptions):
            return
            
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                self._trip()
                logger.warning(f"Circuit {self.name} tripped during HALF_OPEN state")
            elif self.failure_count >= self.config.failure_threshold:
                self._trip()
                logger.warning(f"Circuit {self.name} tripped after {self.failure_count} failures")
    
    def _trip(self) -> None:
        """Trip the circuit breaker (open it)."""
        self.state = CircuitState.OPEN
        self.success_count = 0
    
    def _reset(self) -> None:
        """Reset the circuit breaker to closed state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
        self.last_failure_time = None
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.config.reset_timeout
    
    def _time_until_reset(self) -> float:
        """Calculate time until circuit reset attempt."""
        if self.last_failure_time is None:
            return 0.0
        elapsed = time.time() - self.last_failure_time
        return max(0.0, self.config.reset_timeout - elapsed)
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "half_open_calls": self.half_open_calls,
        }


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class SessionRecoveryManager:
    """Manages session persistence and recovery."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("./browser_sessions")
        self.sessions: Dict[str, SessionState] = {}
        self._lock = asyncio.Lock()
        
    async def initialize(self) -> None:
        """Initialize storage directory."""
        await aiofiles.os.makedirs(self.storage_path, exist_ok=True)
        await self._load_existing_sessions()
    
    async def _load_existing_sessions(self) -> None:
        """Load existing session files from storage."""
        try:
            files = await aiofiles.os.listdir(self.storage_path)
            for file in files:
                if file.endswith(".session"):
                    session_id = file[:-8]  # Remove .session extension
                    try:
                        await self.restore_session(session_id)
                    except Exception as e:
                        logger.warning(f"Failed to load session {session_id}: {e}")
        except Exception as e:
            logger.error(f"Failed to load sessions from {self.storage_path}: {e}")
    
    async def save_session(self, session_state: SessionState) -> None:
        """Save session state to disk."""
        async with self._lock:
            self.sessions[session_state.session_id] = session_state
            session_file = self.storage_path / f"{session_state.session_id}.session"
            
            try:
                async with aiofiles.open(session_file, 'wb') as f:
                    await f.write(pickle.dumps(session_state.to_dict()))
                logger.debug(f"Saved session {session_state.session_id}")
            except Exception as e:
                logger.error(f"Failed to save session {session_state.session_id}: {e}")
                raise
    
    async def restore_session(self, session_id: str) -> Optional[SessionState]:
        """Restore session state from disk."""
        session_file = self.storage_path / f"{session_id}.session"
        
        if not await aiofiles.os.path.exists(session_file):
            return None
        
        try:
            async with aiofiles.open(session_file, 'rb') as f:
                data = pickle.loads(await f.read())
                session_state = SessionState.from_dict(data)
                
                async with self._lock:
                    self.sessions[session_id] = session_state
                
                logger.info(f"Restored session {session_id}")
                return session_state
        except Exception as e:
            logger.error(f"Failed to restore session {session_id}: {e}")
            return None
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a saved session."""
        async with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
            
            session_file = self.storage_path / f"{session_id}.session"
            try:
                if await aiofiles.os.path.exists(session_file):
                    await aiofiles.os.remove(session_file)
                    logger.debug(f"Deleted session {session_id}")
                    return True
            except Exception as e:
                logger.error(f"Failed to delete session {session_id}: {e}")
        
        return False
    
    async def list_sessions(self) -> List[str]:
        """List all saved session IDs."""
        async with self._lock:
            return list(self.sessions.keys())
    
    async def cleanup_old_sessions(self, max_age_hours: float = 24) -> int:
        """Clean up sessions older than specified age."""
        cutoff_time = time.time() - (max_age_hours * 3600)
        deleted_count = 0
        
        async with self._lock:
            sessions_to_delete = []
            for session_id, state in self.sessions.items():
                if state.last_activity < cutoff_time:
                    sessions_to_delete.append(session_id)
            
            for session_id in sessions_to_delete:
                if await self.delete_session(session_id):
                    deleted_count += 1
        
        logger.info(f"Cleaned up {deleted_count} old sessions")
        return deleted_count


class ResilienceLayer:
    """Comprehensive resilience layer for browser automation."""
    
    def __init__(
        self,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        retry_config: Optional[RetryConfig] = None,
        session_storage_path: Optional[Path] = None,
    ):
        self.circuit_breaker_config = circuit_breaker_config or CircuitBreakerConfig()
        self.retry_config = retry_config or RetryConfig()
        self.session_manager = SessionRecoveryManager(session_storage_path)
        
        # Circuit breakers for different operation types
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Health monitoring
        self._health_check_interval = 30.0  # seconds
        self._last_health_check = 0.0
        self._browser_healthy = True
        
        # Operation-specific configurations
        self.operation_configs: Dict[str, Dict[str, Any]] = {
            "navigation": {"timeout": 30.0, "critical": True},
            "element_interaction": {"timeout": 10.0, "critical": False},
            "screenshot": {"timeout": 5.0, "critical": False},
            "evaluate_script": {"timeout": 10.0, "critical": False},
            "session_management": {"timeout": 15.0, "critical": True},
        }
    
    async def initialize(self) -> None:
        """Initialize the resilience layer."""
        await self.session_manager.initialize()
        logger.info("Resilience layer initialized")
    
    def get_circuit_breaker(self, operation_type: str) -> CircuitBreaker:
        """Get or create circuit breaker for operation type."""
        if operation_type not in self.circuit_breakers:
            self.circuit_breakers[operation_type] = CircuitBreaker(
                name=f"cb_{operation_type}",
                config=self.circuit_breaker_config
            )
        return self.circuit_breakers[operation_type]
    
    async def execute_with_resilience(
        self,
        operation_type: str,
        operation: Callable,
        *args,
        session_id: Optional[str] = None,
        save_session: bool = False,
        **kwargs
    ) -> Any:
        """Execute operation with full resilience protection."""
        circuit_breaker = self.get_circuit_breaker(operation_type)
        config = self.operation_configs.get(operation_type, {})
        
        # Check if browser is healthy
        if not self._browser_healthy and config.get("critical", False):
            raise BrowserUnhealthyError("Browser is not healthy for critical operations")
        
        # Execute with retry and circuit breaker
        return await self._execute_with_retry(
            circuit_breaker.execute,
            operation,
            *args,
            session_id=session_id,
            save_session=save_session,
            **kwargs
        )
    
    async def _execute_with_retry(
        self,
        protected_executor: Callable,
        operation: Callable,
        *args,
        session_id: Optional[str] = None,
        save_session: bool = False,
        **kwargs
    ) -> Any:
        """Execute with exponential backoff retry."""
        last_exception = None
        delay = self.retry_config.initial_delay
        
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                result = await protected_executor(operation, *args, **kwargs)
                
                # Save session state if requested
                if save_session and session_id:
                    await self._update_session_state(session_id, result)
                
                return result
                
            except CircuitBreakerOpenError as e:
                last_exception = e
                logger.warning(f"Circuit breaker open on attempt {attempt + 1}: {e}")
                break  # Don't retry if circuit is open
                
            except Exception as e:
                last_exception = e
                
                # Check if exception is retryable
                if not any(isinstance(e, exc_type) for exc_type in self.retry_config.retryable_exceptions):
                    logger.error(f"Non-retryable exception: {e}")
                    break
                
                if attempt == self.retry_config.max_retries:
                    logger.error(f"Max retries ({self.retry_config.max_retries}) exceeded: {e}")
                    break
                
                # Calculate delay with exponential backoff and jitter
                actual_delay = min(delay, self.retry_config.max_delay)
                if self.retry_config.jitter:
                    actual_delay *= (0.5 + asyncio.get_event_loop().time() % 1)
                
                logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {actual_delay:.2f}s..."
                )
                
                await asyncio.sleep(actual_delay)
                delay *= self.retry_config.exponential_base
        
        # All retries exhausted or non-retryable exception
        raise last_exception or RuntimeError("Operation failed after retries")
    
    async def _update_session_state(self, session_id: str, operation_result: Any) -> None:
        """Update session state based on operation result."""
        try:
            # Extract session state from operation result if available
            session_state = None
            
            if isinstance(operation_result, dict):
                # Assume result contains session state
                if "cookies" in operation_result or "local_storage" in operation_result:
                    session_state = SessionState(
                        session_id=session_id,
                        cookies=operation_result.get("cookies", []),
                        local_storage=operation_result.get("local_storage", {}),
                        session_storage=operation_result.get("session_storage", {}),
                        current_url=operation_result.get("current_url"),
                        viewport_size=operation_result.get("viewport_size"),
                        user_agent=operation_result.get("user_agent"),
                        last_activity=time.time(),
                    )
            
            elif hasattr(operation_result, "session_state"):
                session_state = operation_result.session_state
            
            if session_state:
                await self.session_manager.save_session(session_state)
                
        except Exception as e:
            logger.warning(f"Failed to update session state: {e}")
    
    async def recover_session(self, session_id: str) -> Optional[SessionState]:
        """Recover a saved session."""
        return await self.session_manager.restore_session(session_id)
    
    async def check_health(self, health_check_func: Callable) -> bool:
        """Check browser health and update status."""
        try:
            current_time = time.time()
            if current_time - self._last_health_check < self._health_check_interval:
                return self._browser_healthy
            
            self._last_health_check = current_time
            is_healthy = await health_check_func()
            self._browser_healthy = is_healthy
            
            if not is_healthy:
                logger.warning("Browser health check failed")
            
            return is_healthy
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            self._browser_healthy = False
            return False
    
    async def restart_browser(self, restart_func: Callable, *args, **kwargs) -> bool:
        """Attempt to restart the browser process."""
        try:
            logger.info("Attempting browser restart...")
            success = await restart_func(*args, **kwargs)
            
            if success:
                self._browser_healthy = True
                # Reset all circuit breakers
                for cb in self.circuit_breakers.values():
                    cb._reset()
                logger.info("Browser restarted successfully")
            else:
                logger.error("Browser restart failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Browser restart error: {e}")
            self._browser_healthy = False
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get resilience layer metrics."""
        return {
            "browser_healthy": self._browser_healthy,
            "last_health_check": self._last_health_check,
            "circuit_breakers": {
                name: cb.get_state() for name, cb in self.circuit_breakers.items()
            },
            "active_sessions": len(self.session_manager.sessions),
        }
    
    async def graceful_degradation(
        self,
        operation_type: str,
        fallback_operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute with graceful degradation when primary operation fails."""
        try:
            return await self.execute_with_resilience(
                operation_type,
                fallback_operation,
                *args,
                **kwargs
            )
        except Exception as e:
            logger.warning(f"Graceful degradation failed: {e}")
            # Return minimal safe result
            return {"status": "degraded", "error": str(e)}


class BrowserUnhealthyError(Exception):
    """Raised when browser is not healthy for critical operations."""
    pass


# Integration with existing nexus modules
class ResilientPageWrapper:
    """Wrapper for page operations with resilience."""
    
    def __init__(self, page, resilience_layer: ResilienceLayer):
        self.page = page
        self.resilience = resilience_layer
        self.session_id = getattr(page, 'session_id', None) or str(id(page))
    
    async def goto(self, url: str, **kwargs) -> Any:
        """Navigate to URL with resilience."""
        return await self.resilience.execute_with_resilience(
            "navigation",
            self.page.goto,
            url,
            session_id=self.session_id,
            save_session=True,
            **kwargs
        )
    
    async def click(self, selector: str, **kwargs) -> Any:
        """Click element with resilience."""
        return await self.resilience.execute_with_resilience(
            "element_interaction",
            self.page.click,
            selector,
            session_id=self.session_id,
            **kwargs
        )
    
    async def evaluate(self, script: str, **kwargs) -> Any:
        """Evaluate script with resilience."""
        return await self.resilience.execute_with_resilience(
            "evaluate_script",
            self.page.evaluate,
            script,
            session_id=self.session_id,
            **kwargs
        )
    
    async def screenshot(self, **kwargs) -> Any:
        """Take screenshot with resilience."""
        return await self.resilience.execute_with_resilience(
            "screenshot",
            self.page.screenshot,
            session_id=self.session_id,
            **kwargs
        )
    
    async def get_cookies(self) -> List[Dict[str, Any]]:
        """Get cookies with session recovery."""
        try:
            return await self.resilience.execute_with_resilience(
                "session_management",
                self.page.get_cookies,
                session_id=self.session_id
            )
        except Exception as e:
            logger.warning(f"Failed to get cookies: {e}")
            # Try to recover from session
            session = await self.resilience.recover_session(self.session_id)
            return session.cookies if session else []
    
    async def set_cookies(self, cookies: List[Dict[str, Any]]) -> None:
        """Set cookies with session persistence."""
        await self.resilience.execute_with_resilience(
            "session_management",
            self.page.set_cookies,
            cookies,
            session_id=self.session_id,
            save_session=True
        )


# Factory function for easy integration
def create_resilient_browser(
    browser_factory: Callable,
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
    retry_config: Optional[RetryConfig] = None,
    session_storage_path: Optional[Path] = None,
) -> tuple:
    """Create a resilient browser instance with session recovery."""
    resilience_layer = ResilienceLayer(
        circuit_breaker_config=circuit_breaker_config,
        retry_config=retry_config,
        session_storage_path=session_storage_path,
    )
    
    async def create_browser():
        await resilience_layer.initialize()
        browser = await browser_factory()
        return browser, resilience_layer
    
    return create_browser()


# Configuration presets for different environments
PRODUCTION_CONFIG = {
    "circuit_breaker": CircuitBreakerConfig(
        failure_threshold=10,
        reset_timeout=60.0,
        half_open_max_calls=5,
    ),
    "retry": RetryConfig(
        max_retries=5,
        initial_delay=2.0,
        max_delay=60.0,
        exponential_base=2.0,
        jitter=True,
    ),
}

DEVELOPMENT_CONFIG = {
    "circuit_breaker": CircuitBreakerConfig(
        failure_threshold=3,
        reset_timeout=10.0,
        half_open_max_calls=2,
    ),
    "retry": RetryConfig(
        max_retries=2,
        initial_delay=0.5,
        max_delay=5.0,
        exponential_base=1.5,
        jitter=False,
    ),
}