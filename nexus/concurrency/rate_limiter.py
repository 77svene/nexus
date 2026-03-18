"""
Adaptive Concurrency Engine for nexus.

Implements intelligent parallelization that dynamically adjusts concurrency based on system resources
and target website rate limits. Uses token-bucket rate limiting per domain with automatic backoff
when detecting rate limit headers or CAPTCHAs.
"""

import asyncio
import time
import logging
import psutil
from typing import Dict, Optional, Tuple, Any, Callable, Awaitable
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import re

logger = logging.getLogger(__name__)


class RateLimitStrategy(Enum):
    """Strategy for handling rate limits."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    TOKEN_BUCKET = "token_bucket"


@dataclass
class TokenBucket:
    """Token bucket implementation for rate limiting."""
    capacity: float
    refill_rate: float  # tokens per second
    tokens: float = 0.0
    last_refill: float = field(default_factory=time.monotonic)
    
    def __post_init__(self):
        self.tokens = self.capacity
    
    def consume(self, tokens: float = 1.0) -> Tuple[bool, float]:
        """Try to consume tokens. Returns (success, wait_time)."""
        self._refill()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True, 0.0
        
        # Calculate wait time for next token
        needed = tokens - self.tokens
        wait_time = needed / self.refill_rate
        return False, wait_time
    
    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now


@dataclass
class DomainState:
    """State tracking for a specific domain."""
    bucket: TokenBucket
    consecutive_errors: int = 0
    last_error_time: float = 0.0
    backoff_until: float = 0.0
    response_times: list = field(default_factory=list)
    rate_limit_headers: Dict[str, str] = field(default_factory=dict)
    
    def update_backoff(self, error_type: str):
        """Update backoff based on error type."""
        self.consecutive_errors += 1
        self.last_error_time = time.monotonic()
        
        if error_type == "rate_limit":
            # Exponential backoff with jitter
            base_delay = min(2 ** self.consecutive_errors, 300)  # Max 5 minutes
            jitter = base_delay * 0.1 * (2 * asyncio.get_event_loop().time() % 1 - 0.5)
            self.backoff_until = time.monotonic() + base_delay + jitter
        elif error_type == "captcha":
            # Longer backoff for CAPTCHAs
            self.backoff_until = time.monotonic() + 60 * (self.consecutive_errors + 1)
        elif error_type == "server_error":
            # Linear backoff for server errors
            self.backoff_until = time.monotonic() + 10 * self.consecutive_errors
    
    def reset_backoff(self):
        """Reset backoff after successful request."""
        self.consecutive_errors = 0
        self.backoff_until = 0.0


class AdaptiveConcurrencyEngine:
    """
    Adaptive concurrency engine that dynamically adjusts parallelization based on:
    1. System resources (CPU, memory)
    2. Target website rate limits
    3. Response patterns and error rates
    """
    
    def __init__(
        self,
        max_concurrent_requests: int = 10,
        min_concurrent_requests: int = 1,
        initial_rate_limit: float = 10.0,  # requests per second per domain
        token_bucket_capacity: float = 20.0,
        cpu_threshold: float = 0.8,  # 80% CPU usage threshold
        memory_threshold: float = 0.8,  # 80% memory usage threshold
        monitoring_interval: float = 5.0,  # seconds
        strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET,
    ):
        self.max_concurrent_requests = max_concurrent_requests
        self.min_concurrent_requests = min_concurrent_requests
        self.current_concurrency = max_concurrent_requests
        self.strategy = strategy
        
        # Rate limiting per domain
        self.domain_states: Dict[str, DomainState] = defaultdict(
            lambda: DomainState(
                bucket=TokenBucket(
                    capacity=token_bucket_capacity,
                    refill_rate=initial_rate_limit
                )
            )
        )
        
        # Resource monitoring
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.monitoring_interval = monitoring_interval
        
        # Concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)
        self._active_requests = 0
        self._request_queue: asyncio.Queue = asyncio.Queue()
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "rate_limited_requests": 0,
            "avg_response_time": 0.0,
            "current_concurrency": self.current_concurrency,
        }
    
    async def start(self):
        """Start the concurrency engine and monitoring."""
        if self._running:
            return
        
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitor_resources())
        logger.info(f"Adaptive concurrency engine started with {self.current_concurrency} max concurrent requests")
    
    async def stop(self):
        """Stop the concurrency engine."""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Adaptive concurrency engine stopped")
    
    async def execute_request(
        self,
        domain: str,
        request_func: Callable[..., Awaitable[Any]],
        *args,
        **kwargs
    ) -> Any:
        """
        Execute a request with adaptive concurrency and rate limiting.
        
        Args:
            domain: Target domain for rate limiting
            request_func: Async function to execute
            *args, **kwargs: Arguments to pass to request_func
        
        Returns:
            Result from request_func
        
        Raises:
            RateLimitError: When rate limited
            ConcurrencyError: When concurrency limit exceeded
        """
        self.stats["total_requests"] += 1
        
        # Check domain backoff
        domain_state = self.domain_states[domain]
        if time.monotonic() < domain_state.backoff_until:
            wait_time = domain_state.backoff_until - time.monotonic()
            logger.warning(f"Domain {domain} in backoff, waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
        
        # Wait for rate limit token
        wait_time = await self._wait_for_rate_limit(domain)
        if wait_time > 0:
            logger.debug(f"Rate limit wait for {domain}: {wait_time:.2f}s")
            await asyncio.sleep(wait_time)
        
        # Acquire concurrency semaphore
        async with self._semaphore:
            self._active_requests += 1
            start_time = time.monotonic()
            
            try:
                # Execute the request
                result = await request_func(*args, **kwargs)
                
                # Update statistics
                response_time = time.monotonic() - start_time
                domain_state.response_times.append(response_time)
                if len(domain_state.response_times) > 100:
                    domain_state.response_times.pop(0)
                
                domain_state.reset_backoff()
                self.stats["successful_requests"] += 1
                
                # Check for rate limit headers in response
                if hasattr(result, 'headers'):
                    self._update_rate_limits_from_headers(domain, result.headers)
                
                return result
                
            except Exception as e:
                self.stats["failed_requests"] += 1
                error_type = self._classify_error(e, domain)
                domain_state.update_backoff(error_type)
                
                if error_type == "rate_limit":
                    self.stats["rate_limited_requests"] += 1
                    # Reduce concurrency when hitting rate limits
                    self._adjust_concurrency_down()
                
                raise
            finally:
                self._active_requests -= 1
                # Update average response time
                if domain_state.response_times:
                    self.stats["avg_response_time"] = sum(domain_state.response_times) / len(domain_state.response_times)
    
    async def _wait_for_rate_limit(self, domain: str) -> float:
        """Wait for rate limit token. Returns wait time if any."""
        domain_state = self.domain_states[domain]
        
        # Check if we're in backoff
        if time.monotonic() < domain_state.backoff_until:
            return domain_state.backoff_until - time.monotonic()
        
        # Try to consume token
        success, wait_time = domain_state.bucket.consume()
        if success:
            return 0.0
        
        # Wait for token to become available
        return wait_time
    
    def _update_rate_limits_from_headers(self, domain: str, headers: Dict[str, str]):
        """Update rate limits based on response headers."""
        domain_state = domain_states[domain]
        domain_state.rate_limit_headers = headers
        
        # Common rate limit headers
        rate_limit_remaining = headers.get('X-RateLimit-Remaining')
        rate_limit_reset = headers.get('X-RateLimit-Reset')
        retry_after = headers.get('Retry-After')
        
        if rate_limit_remaining is not None:
            try:
                remaining = int(rate_limit_remaining)
                if remaining == 0:
                    # We've hit the rate limit
                    domain_state.update_backoff("rate_limit")
                    self._adjust_concurrency_down()
            except ValueError:
                pass
        
        if retry_after is not None:
            try:
                retry_seconds = int(retry_after)
                domain_state.backoff_until = time.monotonic() + retry_seconds
            except ValueError:
                pass
    
    def _classify_error(self, error: Exception, domain: str) -> str:
        """Classify error type for backoff strategy."""
        error_str = str(error).lower()
        
        # Check for rate limit indicators
        if any(indicator in error_str for indicator in ['rate limit', 'too many requests', '429']):
            return "rate_limit"
        
        # Check for CAPTCHA indicators
        if any(indicator in error_str for indicator in ['captcha', 'bot detection', 'suspicious activity']):
            return "captcha"
        
        # Check for server errors
        if any(indicator in error_str for indicator in ['500', '502', '503', '504', 'server error']):
            return "server_error"
        
        return "unknown"
    
    def _adjust_concurrency_down(self):
        """Reduce concurrency when hitting rate limits."""
        new_concurrency = max(
            self.min_concurrent_requests,
            int(self.current_concurrency * 0.7)  # Reduce by 30%
        )
        
        if new_concurrency < self.current_concurrency:
            logger.info(f"Reducing concurrency from {self.current_concurrency} to {new_concurrency}")
            self.current_concurrency = new_concurrency
            self._update_semaphore()
            self.stats["current_concurrency"] = self.current_concurrency
    
    def _adjust_concurrency_up(self):
        """Increase concurrency when resources are available."""
        new_concurrency = min(
            self.max_concurrent_requests,
            int(self.current_concurrency * 1.2)  # Increase by 20%
        )
        
        if new_concurrency > self.current_concurrency:
            logger.info(f"Increasing concurrency from {self.current_concurrency} to {new_concurrency}")
            self.current_concurrency = new_concurrency
            self._update_semaphore()
            self.stats["current_concurrency"] = self.current_concurrency
    
    def _update_semaphore(self):
        """Update the semaphore to match current concurrency."""
        # Create new semaphore with updated value
        old_semaphore = self._semaphore
        self._semaphore = asyncio.Semaphore(self.current_concurrency)
        
        # Transfer any waiting tasks
        # Note: This is a simplified approach. In production, you'd want to
        # properly handle tasks waiting on the old semaphore.
    
    async def _monitor_resources(self):
        """Monitor system resources and adjust concurrency accordingly."""
        while self._running:
            try:
                # Get system resource usage
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent / 100.0
                
                # Adjust concurrency based on resource usage
                if cpu_percent > self.cpu_threshold * 100 or memory_percent > self.memory_threshold:
                    # Resources are constrained, reduce concurrency
                    self._adjust_concurrency_down()
                elif cpu_percent < (self.cpu_threshold * 100 * 0.5) and memory_percent < (self.memory_threshold * 0.5):
                    # Resources are available, increase concurrency
                    self._adjust_concurrency_up()
                
                # Log current state
                logger.debug(
                    f"Resource monitoring: CPU={cpu_percent:.1f}%, "
                    f"Memory={memory_percent:.1f}%, "
                    f"Concurrency={self.current_concurrency}"
                )
                
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        return {
            **self.stats,
            "active_requests": self._active_requests,
            "domain_count": len(self.domain_states),
        }
    
    def get_domain_stats(self, domain: str) -> Dict[str, Any]:
        """Get statistics for a specific domain."""
        if domain not in self.domain_states:
            return {}
        
        state = self.domain_states[domain]
        avg_response_time = sum(state.response_times) / len(state.response_times) if state.response_times else 0
        
        return {
            "consecutive_errors": state.consecutive_errors,
            "backoff_until": state.backoff_until,
            "avg_response_time": avg_response_time,
            "rate_limit_headers": state.rate_limit_headers,
        }


class RateLimitError(Exception):
    """Exception raised when rate limited."""
    pass


class ConcurrencyError(Exception):
    """Exception raised when concurrency limit is exceeded."""
    pass


# Global instance for easy access
_global_engine: Optional[AdaptiveConcurrencyEngine] = None


def get_global_engine() -> AdaptiveConcurrencyEngine:
    """Get or create the global concurrency engine instance."""
    global _global_engine
    if _global_engine is None:
        _global_engine = AdaptiveConcurrencyEngine()
    return _global_engine


async def execute_with_adaptive_concurrency(
    domain: str,
    request_func: Callable[..., Awaitable[Any]],
    *args,
    **kwargs
) -> Any:
    """
    Convenience function to execute a request with adaptive concurrency.
    
    This uses the global engine instance for simplicity.
    """
    engine = get_global_engine()
    if not engine._running:
        await engine.start()
    
    return await engine.execute_request(domain, request_func, *args, **kwargs)


# Example usage with existing nexus modules
async def example_integration():
    """
    Example of how to integrate with existing nexus modules.
    This shows how the rate limiter can be used with the actor/page module.
    """
    from nexus.actor.page import PageActor
    
    engine = AdaptiveConcurrencyEngine(
        max_concurrent_requests=5,
        initial_rate_limit=2.0,  # 2 requests per second per domain
    )
    
    await engine.start()
    
    # Example: Navigate to multiple URLs with rate limiting
    urls = [
        "https://example.com/page1",
        "https://example.com/page2",
        "https://example.com/page3",
    ]
    
    async def navigate_to_url(url: str):
        # This would be integrated with PageActor
        # For now, simulate a request
        await asyncio.sleep(0.5)  # Simulate network delay
        return f"Navigated to {url}"
    
    tasks = []
    for url in urls:
        domain = url.split("//")[1].split("/")[0]
        task = engine.execute_request(domain, navigate_to_url, url)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Check results
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Failed to navigate to {urls[i]}: {result}")
        else:
            logger.info(f"Success: {result}")
    
    # Print statistics
    stats = engine.get_stats()
    logger.info(f"Final stats: {stats}")
    
    await engine.stop()


if __name__ == "__main__":
    # Run example if executed directly
    asyncio.run(example_integration())