"""
Resilience layer for nexus with circuit breakers, retries, and session persistence.
"""
import asyncio
import json
import logging
import time
import traceback
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar, Union
from functools import wraps
import pickle
import aiofiles
from pathlib import Path

from nexus.actor.browser import Browser
from nexus.agent.service import AgentService
from nexus.exceptions import BrowserCrashError, CDPConnectionError

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3
    excluded_exceptions: Set[type] = field(default_factory=lambda: {
        KeyboardInterrupt,
        SystemExit,
    })


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True


@dataclass
class ResiliencePolicy:
    """Comprehensive resilience policy for different operation types."""
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    timeout: float = 30.0
    session_persist: bool = True
    fallback_enabled: bool = True


@dataclass
class SessionState:
    """Serializable session state for recovery."""
    session_id: str
    url: str
    cookies: List[Dict] = field(default_factory=list)
    local_storage: Dict[str, str] = field(default_factory=dict)
    session_storage: Dict[str, str] = field(default_factory=dict)
    viewport_size: Dict[str, int] = field(default_factory=lambda: {"width": 1280, "height": 720})
    user_agent: str = ""
    last_updated: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SessionState':
        """Create from dictionary."""
        return cls(**data)


class CircuitBreaker:
    """Circuit breaker implementation for CDP connections."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0
        self._lock = asyncio.Lock()
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.acquire()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if exc_type is None:
            await self.record_success()
        else:
            await self.record_failure(exc_type, exc_val)
        
    async def acquire(self) -> None:
        """Check if call is allowed by circuit breaker."""
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                    logger.info(f"Circuit {self.name} transitioning to HALF_OPEN")
                else:
                    raise CDPConnectionError(
                        f"Circuit {self.name} is OPEN. "
                        f"Retry after {self._time_until_reset():.1f}s"
                    )
            
            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls >= self.config.half_open_max_calls:
                    raise CDPConnectionError(
                        f"Circuit {self.name} HALF_OPEN call limit reached"
                    )
                self.half_open_calls += 1
    
    async def record_success(self) -> None:
        """Record successful call."""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                logger.info(f"Circuit {self.name} recovered, transitioning to CLOSED")
            self.failure_count = 0
            self.last_failure_time = None
    
    async def record_failure(self, exc_type: type, exc_val: Exception) -> None:
        """Record failed call."""
        if exc_type in self.config.excluded_exceptions:
            return
            
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit {self.name} failed in HALF_OPEN, transitioning to OPEN")
            elif self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(
                    f"Circuit {self.name} reached failure threshold ({self.config.failure_threshold}), "
                    f"transitioning to OPEN"
                )
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return (time.time() - self.last_failure_time) >= self.config.recovery_timeout
    
    def _time_until_reset(self) -> float:
        """Calculate time until circuit can attempt reset."""
        if self.last_failure_time is None:
            return 0.0
        elapsed = time.time() - self.last_failure_time
        return max(0.0, self.config.recovery_timeout - elapsed)
    
    @property
    def metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "half_open_calls": self.half_open_calls,
        }


class RetryHandler:
    """Handles automatic retries with exponential backoff."""
    
    def __init__(self, config: RetryConfig):
        self.config = config
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt with exponential backoff."""
        delay = min(
            self.config.base_delay * (self.config.exponential_base ** attempt),
            self.config.max_delay
        )
        
        if self.config.jitter:
            import random
            delay = delay * (0.5 + random.random())
        
        return delay
    
    async def execute_with_retry(
        self,
        func: Callable[..., Any],
        *args,
        circuit_breaker: Optional[CircuitBreaker] = None,
        **kwargs
    ) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                if circuit_breaker:
                    async with circuit_breaker:
                        return await func(*args, **kwargs)
                else:
                    return await func(*args, **kwargs)
                    
            except Exception as e:
                last_exception = e
                
                # Don't retry on excluded exceptions
                if circuit_breaker and type(e) in circuit_breaker.config.excluded_exceptions:
                    raise
                
                # Don't retry on last attempt
                if attempt == self.config.max_retries:
                    break
                
                # Calculate and apply delay
                delay = self.calculate_delay(attempt)
                logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                await asyncio.sleep(delay)
        
        raise last_exception


class SessionPersistence:
    """Handles session state persistence and recovery."""
    
    def __init__(self, storage_path: Union[str, Path] = ".browser_sessions"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
    
    async def save_session(self, session_state: SessionState) -> None:
        """Save session state to disk."""
        try:
            file_path = self.storage_path / f"{session_state.session_id}.json"
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(json.dumps(session_state.to_dict(), indent=2))
            logger.debug(f"Saved session {session_state.session_id}")
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
    
    async def load_session(self, session_id: str) -> Optional[SessionState]:
        """Load session state from disk."""
        try:
            file_path = self.storage_path / f"{session_id}.json"
            if not file_path.exists():
                return None
                
            async with aiofiles.open(file_path, 'r') as f:
                data = json.loads(await f.read())
            
            session_state = SessionState.from_dict(data)
            logger.debug(f"Loaded session {session_id}")
            return session_state
        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            return None
    
    async def list_sessions(self) -> List[str]:
        """List all saved session IDs."""
        try:
            return [
                f.stem for f in self.storage_path.glob("*.json")
                if f.is_file()
            ]
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete saved session."""
        try:
            file_path = self.storage_path / f"{session_id}.json"
            if file_path.exists():
                file_path.unlink()
                logger.debug(f"Deleted session {session_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete session: {e}")
            return False


class ResilienceLayer:
    """Main resilience layer that coordinates all resilience mechanisms."""
    
    def __init__(
        self,
        default_policy: Optional[ResiliencePolicy] = None,
        session_storage_path: Union[str, Path] = ".browser_sessions"
    ):
        self.default_policy = default_policy or ResiliencePolicy()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_handlers: Dict[str, RetryHandler] = {}
        self.session_persistence = SessionPersistence(session_storage_path)
        self._browser_health_check_task: Optional[asyncio.Task] = None
        self._active_sessions: Dict[str, SessionState] = {}
        
        # Default policies for different operation types
        self.policies: Dict[str, ResiliencePolicy] = {
            "cdp_connection": ResiliencePolicy(
                circuit_breaker=CircuitBreakerConfig(
                    failure_threshold=3,
                    recovery_timeout=10.0
                ),
                retry=RetryConfig(max_retries=2),
                timeout=15.0
            ),
            "page_navigation": ResiliencePolicy(
                circuit_breaker=CircuitBreakerConfig(
                    failure_threshold=5,
                    recovery_timeout=30.0
                ),
                retry=RetryConfig(max_retries=3, base_delay=2.0),
                timeout=60.0
            ),
            "element_interaction": ResiliencePolicy(
                circuit_breaker=CircuitBreakerConfig(
                    failure_threshold=10,
                    recovery_timeout=5.0
                ),
                retry=RetryConfig(max_retries=2, base_delay=0.5),
                timeout=10.0
            ),
            "script_execution": ResiliencePolicy(
                circuit_breaker=CircuitBreakerConfig(
                    failure_threshold=8,
                    recovery_timeout=15.0
                ),
                retry=RetryConfig(max_retries=2),
                timeout=30.0
            ),
        }
    
    def get_policy(self, operation_type: str) -> ResiliencePolicy:
        """Get resilience policy for operation type."""
        return self.policies.get(operation_type, self.default_policy)
    
    def get_circuit_breaker(self, operation_type: str) -> CircuitBreaker:
        """Get or create circuit breaker for operation type."""
        if operation_type not in self.circuit_breakers:
            policy = self.get_policy(operation_type)
            self.circuit_breakers[operation_type] = CircuitBreaker(
                name=f"cb_{operation_type}",
                config=policy.circuit_breaker
            )
        return self.circuit_breakers[operation_type]
    
    def get_retry_handler(self, operation_type: str) -> RetryHandler:
        """Get or create retry handler for operation type."""
        if operation_type not in self.retry_handlers:
            policy = self.get_policy(operation_type)
            self.retry_handlers[operation_type] = RetryHandler(policy.retry)
        return self.retry_handlers[operation_type]
    
    def resilient(
        self,
        operation_type: str = "default",
        session_aware: bool = False
    ) -> Callable:
        """Decorator for adding resilience to async functions."""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                policy = self.get_policy(operation_type)
                circuit_breaker = self.get_circuit_breaker(operation_type)
                retry_handler = self.get_retry_handler(operation_type)
                
                # Apply timeout
                try:
                    async with asyncio.timeout(policy.timeout):
                        return await retry_handler.execute_with_retry(
                            func, *args,
                            circuit_breaker=circuit_breaker,
                            **kwargs
                        )
                except asyncio.TimeoutError:
                    logger.error(f"Operation {operation_type} timed out after {policy.timeout}s")
                    raise
                except Exception as e:
                    logger.error(f"Operation {operation_type} failed: {e}")
                    raise
            
            return wrapper
        return decorator
    
    async def with_session_recovery(
        self,
        browser: Browser,
        session_id: str,
        operation: Callable[..., T],
        *args,
        **kwargs
    ) -> T:
        """Execute operation with session recovery on failure."""
        policy = self.get_policy("cdp_connection")
        
        # Try to load existing session
        session_state = await self.session_persistence.load_session(session_id)
        if session_state:
            logger.info(f"Recovering session {session_id} from {session_state.url}")
            try:
                await self._restore_session_state(browser, session_state)
            except Exception as e:
                logger.warning(f"Failed to restore session state: {e}")
        
        # Execute operation with resilience
        try:
            result = await self.resilient("cdp_connection")(operation)(*args, **kwargs)
            
            # Save successful session state
            if policy.session_persist:
                current_state = await self._capture_session_state(browser, session_id)
                await self.session_persistence.save_session(current_state)
                self._active_sessions[session_id] = current_state
            
            return result
            
        except Exception as e:
            # Attempt graceful degradation
            if policy.fallback_enabled:
                logger.warning(f"Primary operation failed, attempting fallback: {e}")
                try:
                    return await self._graceful_degradation(browser, operation, *args, **kwargs)
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
            
            raise
    
    async def _capture_session_state(self, browser: Browser, session_id: str) -> SessionState:
        """Capture current browser session state."""
        try:
            # Get current URL
            url = await browser.evaluate("window.location.href")
            
            # Get cookies
            cookies = await browser.send("Network.getCookies")
            
            # Get local storage
            local_storage = await browser.evaluate("""
                () => {
                    const items = {};
                    for (let i = 0; i < localStorage.length; i++) {
                        const key = localStorage.key(i);
                        items[key] = localStorage.getItem(key);
                    }
                    return items;
                }
            """)
            
            # Get session storage
            session_storage = await browser.evaluate("""
                () => {
                    const items = {};
                    for (let i = 0; i < sessionStorage.length; i++) {
                        const key = sessionStorage.key(i);
                        items[key] = sessionStorage.getItem(key);
                    }
                    return items;
                }
            """)
            
            # Get viewport size
            viewport = await browser.evaluate("""
                () => ({
                    width: window.innerWidth,
                    height: window.innerHeight
                })
            """)
            
            return SessionState(
                session_id=session_id,
                url=url,
                cookies=cookies,
                local_storage=local_storage,
                session_storage=session_storage,
                viewport_size=viewport,
                user_agent=await browser.evaluate("navigator.userAgent")
            )
            
        except Exception as e:
            logger.error(f"Failed to capture session state: {e}")
            return SessionState(session_id=session_id, url="about:blank")
    
    async def _restore_session_state(self, browser: Browser, session_state: SessionState) -> None:
        """Restore browser session from saved state."""
        try:
            # Navigate to URL
            if session_state.url and session_state.url != "about:blank":
                await browser.goto(session_state.url)
            
            # Restore cookies
            if session_state.cookies:
                await browser.send("Network.setCookies", {
                    "cookies": session_state.cookies
                })
            
            # Restore local storage
            if session_state.local_storage:
                await browser.evaluate("""
                    (storage) => {
                        for (const [key, value] of Object.entries(storage)) {
                            localStorage.setItem(key, value);
                        }
                    }
                """, session_state.local_storage)
            
            # Restore session storage
            if session_state.session_storage:
                await browser.evaluate("""
                    (storage) => {
                        for (const [key, value] of Object.entries(storage)) {
                            sessionStorage.setItem(key, value);
                        }
                    }
                """, session_state.session_storage)
            
            # Restore viewport
            if session_state.viewport_size:
                await browser.set_viewport_size(
                    session_state.viewport_size["width"],
                    session_state.viewport_size["height"]
                )
            
            logger.info(f"Successfully restored session {session_state.session_id}")
            
        except Exception as e:
            logger.error(f"Failed to restore session state: {e}")
            raise
    
    async def _graceful_degradation(
        self,
        browser: Browser,
        operation: Callable[..., T],
        *args,
        **kwargs
    ) -> T:
        """Implement graceful degradation strategies."""
        # Strategy 1: Try with fresh browser context
        try:
            logger.info("Attempting graceful degradation with fresh context")
            await browser.new_page()
            return await operation(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Fresh context failed: {e}")
        
        # Strategy 2: Try with simplified operation
        try:
            logger.info("Attempting simplified operation")
            # Create a simplified version of the operation
            if hasattr(operation, '__name__'):
                simplified_name = f"simplified_{operation.__name__}"
                if hasattr(self, simplified_name):
                    simplified_op = getattr(self, simplified_name)
                    return await simplified_op(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Simplified operation failed: {e}")
        
        # Strategy 3: Return fallback response
        logger.error("All degradation strategies failed")
        raise BrowserCrashError("Browser operation failed and all recovery strategies exhausted")
    
    async def start_health_monitoring(self, browser: Browser, interval: float = 30.0) -> None:
        """Start periodic health checks for browser."""
        async def health_check_loop():
            while True:
                try:
                    await self._perform_health_check(browser)
                except Exception as e:
                    logger.error(f"Health check failed: {e}")
                    await self._handle_unhealthy_browser(browser)
                
                await asyncio.sleep(interval)
        
        self._browser_health_check_task = asyncio.create_task(health_check_loop())
        logger.info(f"Started browser health monitoring every {interval}s")
    
    async def stop_health_monitoring(self) -> None:
        """Stop health monitoring."""
        if self._browser_health_check_task:
            self._browser_health_check_task.cancel()
            try:
                await self._browser_health_check_task
            except asyncio.CancelledError:
                pass
            self._browser_health_check_task = None
    
    async def _perform_health_check(self, browser: Browser) -> bool:
        """Perform health check on browser."""
        try:
            # Simple health check: evaluate JavaScript
            result = await browser.evaluate("() => true")
            if not result:
                raise Exception("Health check JavaScript returned false")
            
            # Check if we can navigate
            current_url = await browser.evaluate("window.location.href")
            if not current_url:
                raise Exception("Cannot retrieve current URL")
            
            return True
            
        except Exception as e:
            logger.warning(f"Browser health check failed: {e}")
            return False
    
    async def _handle_unhealthy_browser(self, browser: Browser) -> None:
        """Handle unhealthy browser by attempting restart."""
        logger.warning("Browser unhealthy, attempting recovery")
        
        try:
            # Try to close and reopen browser
            await browser.close()
            await asyncio.sleep(2)
            await browser.start()
            
            # Restore active sessions
            for session_id, session_state in self._active_sessions.items():
                try:
                    await self._restore_session_state(browser, session_state)
                except Exception as e:
                    logger.error(f"Failed to restore session {session_id}: {e}")
            
            logger.info("Browser recovery successful")
            
        except Exception as e:
            logger.error(f"Browser recovery failed: {e}")
            raise BrowserCrashError("Browser crashed and could not be recovered")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive resilience metrics."""
        return {
            "circuit_breakers": {
                name: cb.metrics
                for name, cb in self.circuit_breakers.items()
            },
            "active_sessions": list(self._active_sessions.keys()),
            "policy_count": len(self.policies),
        }


# Global resilience layer instance
_resilience_layer: Optional[ResilienceLayer] = None


def get_resilience_layer() -> ResilienceLayer:
    """Get or create global resilience layer instance."""
    global _resilience_layer
    if _resilience_layer is None:
        _resilience_layer = ResilienceLayer()
    return _resilience_layer


def resilient(
    operation_type: str = "default",
    session_aware: bool = False
) -> Callable:
    """Convenience decorator for adding resilience."""
    return get_resilience_layer().resilient(operation_type, session_aware)


async def with_circuit_breaker(
    operation_type: str,
    func: Callable[..., T],
    *args,
    **kwargs
) -> T:
    """Execute function with circuit breaker protection."""
    layer = get_resilience_layer()
    circuit_breaker = layer.get_circuit_breaker(operation_type)
    retry_handler = layer.get_retry_handler(operation_type)
    
    return await retry_handler.execute_with_retry(
        func, *args,
        circuit_breaker=circuit_breaker,
        **kwargs
    )


# Integration helpers for existing modules
class ResilientBrowser:
    """Wrapper for Browser with built-in resilience."""
    
    def __init__(self, browser: Browser, resilience_layer: Optional[ResilienceLayer] = None):
        self.browser = browser
        self.resilience = resilience_layer or get_resilience_layer()
    
    async def goto(self, url: str, **kwargs) -> None:
        """Navigate with resilience."""
        @self.resilience.resilient("page_navigation")
        async def _goto():
            return await self.browser.goto(url, **kwargs)
        
        return await _goto()
    
    async def send(self, method: str, params: Optional[Dict] = None) -> Any:
        """Send CDP command with resilience."""
        @self.resilience.resilient("cdp_connection")
        async def _send():
            return await self.browser.send(method, params)
        
        return await _send()
    
    async def evaluate(self, expression: str, *args) -> Any:
        """Evaluate JavaScript with resilience."""
        @self.resilience.resilient("script_execution")
        async def _evaluate():
            return await self.browser.evaluate(expression, *args)
        
        return await _evaluate()


class ResilientAgentService:
    """Wrapper for AgentService with built-in resilience."""
    
    def __init__(self, agent_service: AgentService, resilience_layer: Optional[ResilienceLayer] = None):
        self.agent_service = agent_service
        self.resilience = resilience_layer or get_resilience_layer()
    
    async def run(self, *args, **kwargs) -> Any:
        """Run agent with resilience."""
        @self.resilience.resilient("agent_execution")
        async def _run():
            return await self.agent_service.run(*args, **kwargs)
        
        return await _run()


# Export public API
__all__ = [
    'CircuitBreaker',
    'CircuitBreakerConfig',
    'CircuitState',
    'RetryConfig',
    'RetryHandler',
    'ResiliencePolicy',
    'ResilienceLayer',
    'SessionPersistence',
    'SessionState',
    'ResilientBrowser',
    'ResilientAgentService',
    'get_resilience_layer',
    'resilient',
    'with_circuit_breaker',
]