import asyncio
import time
import threading
import psutil
import logging
from typing import Dict, Optional, List, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import re
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class RateLimitState(Enum):
    """State of rate limiting for a domain."""
    NORMAL = "normal"
    BACKING_OFF = "backing_off"
    BLOCKED = "blocked"


@dataclass
class TokenBucket:
    """Token bucket implementation for rate limiting."""
    capacity: float
    tokens: float
    refill_rate: float  # tokens per second
    last_refill: float = field(default_factory=time.time)
    
    def consume(self, tokens: float = 1.0) -> bool:
        """Try to consume tokens from the bucket."""
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
    
    def time_until_tokens(self, tokens: float = 1.0) -> float:
        """Calculate time until enough tokens are available."""
        self._refill()
        if self.tokens >= tokens:
            return 0.0
        deficit = tokens - self.tokens
        return deficit / self.refill_rate


@dataclass
class DomainRateLimit:
    """Rate limit configuration and state for a domain."""
    bucket: TokenBucket
    state: RateLimitState = RateLimitState.NORMAL
    consecutive_failures: int = 0
    last_failure_time: float = 0.0
    backoff_until: float = 0.0
    min_delay: float = 0.1  # Minimum delay between requests
    current_delay: float = 0.1
    detected_patterns: Set[str] = field(default_factory=set)


class ResourceMonitor:
    """
    Adaptive Concurrency Engine that dynamically adjusts concurrency based on:
    1. System resources (CPU, memory, network)
    2. Website rate limits and response patterns
    3. Automatic backoff when detecting rate limits or CAPTCHAs
    """
    
    def __init__(self, 
                 max_concurrency: int = 10,
                 min_concurrency: int = 1,
                 target_cpu_percent: float = 70.0,
                 target_memory_percent: float = 80.0,
                 check_interval: float = 1.0):
        """
        Initialize the resource monitor.
        
        Args:
            max_concurrency: Maximum number of concurrent tasks
            min_concurrency: Minimum number of concurrent tasks
            target_cpu_percent: Target CPU usage percentage
            target_memory_percent: Target memory usage percentage
            check_interval: How often to check system resources (seconds)
        """
        self.max_concurrency = max_concurrency
        self.min_concurrency = min_concurrency
        self.current_concurrency = min_concurrency
        self.target_cpu_percent = target_cpu_percent
        self.target_memory_percent = target_memory_percent
        self.check_interval = check_interval
        
        # Domain-specific rate limiting
        self.domain_limits: Dict[str, DomainRateLimit] = {}
        self.domain_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        
        # System monitoring
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False
        self._system_stats = {
            'cpu_percent': 0.0,
            'memory_percent': 0.0,
            'network_bytes_sent': 0,
            'network_bytes_recv': 0,
        }
        
        # Adaptive parameters
        self._concurrency_adjustment_step = 1
        self._backoff_multiplier = 2.0
        self._max_backoff = 60.0  # Maximum backoff in seconds
        
        # Pattern detection for rate limits
        self.rate_limit_patterns = [
            r'rate.?limit',
            r'too.?many.?requests',
            r'please.?wait',
            r'slow.?down',
            r'429',
            r'503',
            r'captcha',
            r'robot',
            r'blocked',
            r'temporarily.?unavailable',
        ]
        
        # Headers that indicate rate limiting
        self.rate_limit_headers = {
            'x-rate-limit-limit',
            'x-rate-limit-remaining',
            'x-rate-limit-reset',
            'retry-after',
            'x-ratelimit-reset',
        }
        
        logger.info(f"ResourceMonitor initialized with max_concurrency={max_concurrency}")
    
    async def start(self):
        """Start the resource monitoring task."""
        if self._running:
            return
        
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_resources())
        logger.info("ResourceMonitor started")
    
    async def stop(self):
        """Stop the resource monitoring task."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("ResourceMonitor stopped")
    
    async def _monitor_resources(self):
        """Background task to monitor system resources and adjust concurrency."""
        prev_net_io = psutil.net_io_counters()
        
        while self._running:
            try:
                # Get current system stats
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                net_io = psutil.net_io_counters()
                
                # Calculate network throughput
                net_sent = net_io.bytes_sent - prev_net_io.bytes_sent
                net_recv = net_io.bytes_recv - prev_net_io.bytes_recv
                prev_net_io = net_io
                
                # Update system stats
                self._system_stats.update({
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'network_bytes_sent': net_sent,
                    'network_bytes_recv': net_recv,
                })
                
                # Adjust concurrency based on system resources
                await self._adjust_concurrency(cpu_percent, memory_percent)
                
                # Log stats periodically
                if int(time.time()) % 30 == 0:  # Log every 30 seconds
                    logger.debug(
                        f"System stats: CPU={cpu_percent:.1f}%, "
                        f"Memory={memory_percent:.1f}%, "
                        f"Concurrency={self.current_concurrency}"
                    )
                
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in resource monitor: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _adjust_concurrency(self, cpu_percent: float, memory_percent: float):
        """Adjust concurrency based on system resource usage."""
        old_concurrency = self.current_concurrency
        
        # Check if we need to reduce concurrency
        if cpu_percent > self.target_cpu_percent or memory_percent > self.target_memory_percent:
            # System is under pressure, reduce concurrency
            self.current_concurrency = max(
                self.min_concurrency,
                self.current_concurrency - self._concurrency_adjustment_step
            )
            if self.current_concurrency != old_concurrency:
                logger.info(
                    f"Reducing concurrency from {old_concurrency} to {self.current_concurrency} "
                    f"(CPU={cpu_percent:.1f}%, Memory={memory_percent:.1f}%)"
                )
        
        # Check if we can increase concurrency
        elif cpu_percent < self.target_cpu_percent * 0.7 and memory_percent < self.target_memory_percent * 0.7:
            # System has capacity, increase concurrency
            self.current_concurrency = min(
                self.max_concurrency,
                self.current_concurrency + self._concurrency_adjustment_step
            )
            if self.current_concurrency != old_concurrency:
                logger.info(
                    f"Increasing concurrency from {old_concurrency} to {self.current_concurrency} "
                    f"(CPU={cpu_percent:.1f}%, Memory={memory_percent:.1f}%)"
                )
    
    def get_domain_from_url(self, url: str) -> str:
        """Extract domain from URL."""
        parsed = urlparse(url)
        return parsed.netloc or parsed.path.split('/')[0]
    
    async def acquire_request_slot(self, url: str) -> Tuple[bool, float]:
        """
        Acquire a slot for making a request to the given URL.
        
        Returns:
            Tuple of (success, wait_time)
            - success: True if slot acquired, False if should skip
            - wait_time: Time to wait before making request (0 if immediate)
        """
        domain = self.get_domain_from_url(url)
        
        async with self.domain_locks[domain]:
            # Initialize domain limit if not exists
            if domain not in self.domain_limits:
                self.domain_limits[domain] = DomainRateLimit(
                    bucket=TokenBucket(
                        capacity=10.0,  # Default: 10 requests burst
                        tokens=10.0,
                        refill_rate=1.0  # Default: 1 request per second
                    )
                )
            
            domain_limit = self.domain_limits[domain]
            
            # Check if domain is blocked
            if domain_limit.state == RateLimitState.BLOCKED:
                logger.warning(f"Domain {domain} is blocked, skipping request")
                return False, 0.0
            
            # Check if in backoff period
            if domain_limit.state == RateLimitState.BACKING_OFF:
                now = time.time()
                if now < domain_limit.backoff_until:
                    wait_time = domain_limit.backoff_until - now
                    logger.debug(f"Domain {domain} in backoff, waiting {wait_time:.1f}s")
                    return False, wait_time
                else:
                    # Backoff period ended, reset to normal
                    domain_limit.state = RateLimitState.NORMAL
                    domain_limit.current_delay = domain_limit.min_delay
            
            # Try to consume token from bucket
            if domain_limit.bucket.consume():
                # Apply minimum delay between requests
                wait_time = max(0.0, domain_limit.current_delay - 
                              (time.time() - domain_limit.bucket.last_refill))
                return True, wait_time
            else:
                # Calculate wait time for token refill
                wait_time = domain_limit.bucket.time_until_tokens()
                logger.debug(f"Rate limited for {domain}, waiting {wait_time:.1f}s")
                return False, wait_time
    
    async def report_request_result(self, 
                                   url: str, 
                                   success: bool, 
                                   status_code: Optional[int] = None,
                                   headers: Optional[Dict[str, str]] = None,
                                   response_text: Optional[str] = None):
        """
        Report the result of a request for adaptive rate limiting.
        
        Args:
            url: The URL that was requested
            success: Whether the request was successful
            status_code: HTTP status code
            headers: Response headers
            response_text: Response body text (for pattern detection)
        """
        domain = self.get_domain_from_url(url)
        
        async with self.domain_locks[domain]:
            if domain not in self.domain_limits:
                return
            
            domain_limit = self.domain_limits[domain]
            now = time.time()
            
            if success:
                # Successful request
                domain_limit.consecutive_failures = 0
                
                # Check for rate limit headers in successful response
                if headers:
                    self._check_rate_limit_headers(domain_limit, headers)
                
                # Gradually reduce delay after successful requests
                if domain_limit.current_delay > domain_limit.min_delay:
                    domain_limit.current_delay = max(
                        domain_limit.min_delay,
                        domain_limit.current_delay * 0.9  # Reduce by 10%
                    )
                
                # Gradually increase token bucket refill rate after success
                if domain_limit.bucket.refill_rate < 10.0:  # Max 10 req/sec
                    domain_limit.bucket.refill_rate = min(
                        10.0,
                        domain_limit.bucket.refill_rate * 1.1  # Increase by 10%
                    )
                    
            else:
                # Failed request
                domain_limit.consecutive_failures += 1
                domain_limit.last_failure_time = now
                
                # Check for rate limit patterns in response
                is_rate_limited = False
                
                # Check status code
                if status_code == 429 or status_code == 503:
                    is_rate_limited = True
                    logger.warning(f"Rate limit detected for {domain}: HTTP {status_code}")
                
                # Check headers
                if headers:
                    if self._check_rate_limit_headers(domain_limit, headers):
                        is_rate_limited = True
                
                # Check response text for patterns
                if response_text:
                    if self._check_rate_limit_patterns(response_text):
                        is_rate_limited = True
                        logger.warning(f"Rate limit pattern detected in response from {domain}")
                
                if is_rate_limited:
                    # Apply exponential backoff
                    backoff_time = min(
                        self._max_backoff,
                        domain_limit.current_delay * (self._backoff_multiplier ** domain_limit.consecutive_failures)
                    )
                    
                    domain_limit.state = RateLimitState.BACKING_OFF
                    domain_limit.backoff_until = now + backoff_time
                    domain_limit.current_delay = backoff_time
                    
                    # Reduce token bucket rate
                    domain_limit.bucket.refill_rate = max(
                        0.1,  # Minimum 0.1 req/sec
                        domain_limit.bucket.refill_rate * 0.5  # Reduce by 50%
                    )
                    
                    logger.warning(
                        f"Rate limiting {domain}: backing off for {backoff_time:.1f}s, "
                        f"reduced rate to {domain_limit.bucket.refill_rate:.2f} req/sec"
                    )
                    
                    # Block domain if too many consecutive failures
                    if domain_limit.consecutive_failures >= 10:
                        domain_limit.state = RateLimitState.BLOCKED
                        logger.error(f"Domain {domain} blocked after {domain_limit.consecutive_failures} failures")
    
    def _check_rate_limit_headers(self, domain_limit: DomainRateLimit, headers: Dict[str, str]) -> bool:
        """Check response headers for rate limit information."""
        headers_lower = {k.lower(): v for k, v in headers.items()}
        
        # Check for rate limit headers
        for header in self.rate_limit_headers:
            if header in headers_lower:
                domain_limit.detected_patterns.add(f"header:{header}")
                
                # Parse Retry-After header
                if header == 'retry-after':
                    try:
                        retry_after = float(headers_lower[header])
                        domain_limit.backoff_until = time.time() + retry_after
                        domain_limit.state = RateLimitState.BACKING_OFF
                        logger.info(f"Retry-After header: {retry_after}s for domain")
                        return True
                    except ValueError:
                        pass
        
        return False
    
    def _check_rate_limit_patterns(self, text: str) -> bool:
        """Check response text for rate limit patterns."""
        text_lower = text.lower()
        
        for pattern in self.rate_limit_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system statistics."""
        return {
            **self._system_stats,
            'current_concurrency': self.current_concurrency,
            'max_concurrency': self.max_concurrency,
            'min_concurrency': self.min_concurrency,
            'monitored_domains': len(self.domain_limits),
        }
    
    def get_domain_stats(self, domain: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific domain."""
        if domain not in self.domain_limits:
            return None
        
        domain_limit = self.domain_limits[domain]
        return {
            'state': domain_limit.state.value,
            'consecutive_failures': domain_limit.consecutive_failures,
            'current_delay': domain_limit.current_delay,
            'token_bucket': {
                'tokens': domain_limit.bucket.tokens,
                'capacity': domain_limit.bucket.capacity,
                'refill_rate': domain_limit.bucket.refill_rate,
            },
            'detected_patterns': list(domain_limit.detected_patterns),
        }
    
    def reset_domain(self, domain: str):
        """Reset rate limiting for a domain."""
        if domain in self.domain_limits:
            self.domain_limits[domain] = DomainRateLimit(
                bucket=TokenBucket(
                    capacity=10.0,
                    tokens=10.0,
                    refill_rate=1.0
                )
            )
            logger.info(f"Reset rate limiting for domain: {domain}")
    
    def set_domain_rate_limit(self, domain: str, requests_per_second: float, burst_capacity: float = 10.0):
        """Manually set rate limit for a domain."""
        if domain not in self.domain_limits:
            self.domain_limits[domain] = DomainRateLimit(
                bucket=TokenBucket(
                    capacity=burst_capacity,
                    tokens=burst_capacity,
                    refill_rate=requests_per_second
                )
            )
        else:
            self.domain_limits[domain].bucket.refill_rate = requests_per_second
            self.domain_limits[domain].bucket.capacity = burst_capacity
        
        logger.info(f"Set rate limit for {domain}: {requests_per_second} req/sec, burst={burst_capacity}")
    
    def get_optimal_concurrency(self) -> int:
        """Get the current optimal concurrency level."""
        return self.current_concurrency
    
    def set_concurrency_limits(self, min_concurrency: int, max_concurrency: int):
        """Set minimum and maximum concurrency limits."""
        self.min_concurrency = min_concurrency
        self.max_concurrency = max_concurrency
        self.current_concurrency = max(min_concurrency, min(max_concurrency, self.current_concurrency))
        logger.info(f"Concurrency limits set: min={min_concurrency}, max={max_concurrency}")


# Global instance for easy access
_global_monitor: Optional[ResourceMonitor] = None


def get_resource_monitor() -> ResourceMonitor:
    """Get the global resource monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = ResourceMonitor()
    return _global_monitor


def set_resource_monitor(monitor: ResourceMonitor):
    """Set the global resource monitor instance."""
    global _global_monitor
    _global_monitor = monitor


# Decorator for rate-limited functions
def rate_limited(url_extractor=None):
    """
    Decorator to apply rate limiting to a function.
    
    Args:
        url_extractor: Function to extract URL from function arguments
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            monitor = get_resource_monitor()
            
            # Extract URL from arguments
            if url_extractor:
                url = url_extractor(*args, **kwargs)
            else:
                # Try to find URL in arguments
                url = None
                for arg in args:
                    if isinstance(arg, str) and ('http://' in arg or 'https://' in arg):
                        url = arg
                        break
                
                if not url:
                    for key, value in kwargs.items():
                        if isinstance(value, str) and ('http://' in value or 'https://' in value):
                            url = value
                            break
            
            if not url:
                # No URL found, execute without rate limiting
                return await func(*args, **kwargs)
            
            # Acquire request slot
            success, wait_time = await monitor.acquire_request_slot(url)
            
            if not success:
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    # Try again after waiting
                    success, wait_time = await monitor.acquire_request_slot(url)
                    if not success:
                        raise Exception(f"Could not acquire request slot for {url}")
                else:
                    raise Exception(f"Request blocked for {url}")
            
            # Wait if needed
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            
            # Execute the function
            try:
                result = await func(*args, **kwargs)
                await monitor.report_request_result(url, success=True)
                return result
            except Exception as e:
                await monitor.report_request_result(url, success=False)
                raise
        
        return wrapper
    return decorator


# Context manager for batch processing with adaptive concurrency
class AdaptiveBatchProcessor:
    """Context manager for processing batches with adaptive concurrency."""
    
    def __init__(self, monitor: Optional[ResourceMonitor] = None):
        self.monitor = monitor or get_resource_monitor()
        self._semaphore: Optional[asyncio.Semaphore] = None
    
    async def __aenter__(self):
        await self.monitor.start()
        self._semaphore = asyncio.Semaphore(self.monitor.get_optimal_concurrency())
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.monitor.stop()
    
    async def process(self, coro):
        """Process a coroutine with adaptive concurrency control."""
        async with self._semaphore:
            # Update semaphore capacity if concurrency changed
            current_limit = self.monitor.get_optimal_concurrency()
            if self._semaphore._value != current_limit:
                # Adjust semaphore (this is a simplified approach)
                self._semaphore = asyncio.Semaphore(current_limit)
            
            return await coro
    
    def adjust_concurrency(self, new_limit: int):
        """Manually adjust concurrency limit."""
        self.monitor.current_concurrency = new_limit
        self._semaphore = asyncio.Semaphore(new_limit)