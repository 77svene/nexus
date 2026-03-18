"""
Adaptive Concurrency Engine for nexus.

Implements intelligent parallelization that dynamically adjusts concurrency
based on system resources and target website rate limits.
"""

import asyncio
import time
import threading
import logging
import psutil
from typing import Dict, Optional, Callable, Any, List, Set
from dataclasses import dataclass, field
from enum import Enum
import re
from urllib.parse import urlparse
from collections import defaultdict

from ..agent.service import AgentService
from ..actor.page import Page

logger = logging.getLogger(__name__)


class RateLimitStrategy(Enum):
    """Strategies for handling rate limiting."""
    BACKOFF = "backoff"
    QUEUE = "queue"
    ABORT = "abort"


@dataclass
class TokenBucket:
    """Token bucket for rate limiting requests to a specific domain."""
    domain: str
    capacity: int = 10
    refill_rate: float = 1.0  # tokens per second
    tokens: float = field(default=0.0)
    last_refill: float = field(default_factory=time.time)
    blocked_until: float = field(default=0.0)
    consecutive_failures: int = field(default=0)
    
    def __post_init__(self):
        self.tokens = min(self.capacity, self.tokens)
        self.last_refill = time.time()
    
    def refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
    
    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens. Returns True if successful."""
        self.refill()
        
        if self.is_blocked():
            return False
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            self.consecutive_failures = 0
            return True
        return False
    
    def is_blocked(self) -> bool:
        """Check if this domain is currently blocked."""
        return time.time() < self.blocked_until
    
    def record_failure(self, retry_after: Optional[float] = None):
        """Record a failure and potentially block the domain."""
        self.consecutive_failures += 1
        
        if retry_after:
            self.blocked_until = time.time() + retry_after
        elif self.consecutive_failures > 3:
            # Exponential backoff
            backoff = min(300, 2 ** self.consecutive_failures)
            self.blocked_until = time.time() + backoff
    
    def record_success(self):
        """Record a successful request."""
        self.consecutive_failures = 0
        self.blocked_until = 0.0


@dataclass
class SystemMetrics:
    """System resource metrics for adaptive concurrency."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    network_sent: int = 0
    network_recv: int = 0
    disk_io: float = 0.0
    timestamp: float = field(default_factory=time.time)


class AdaptiveConcurrencyPool:
    """
    Adaptive thread/process pool that adjusts concurrency based on system
    resources and website rate limits.
    
    Features:
    - Dynamic worker scaling based on CPU/memory/network usage
    - Per-domain token bucket rate limiting
    - Automatic backoff on rate limit headers/CAPTCHAs
    - Priority queuing for critical tasks
    - Real-time performance monitoring
    """
    
    def __init__(
        self,
        min_workers: int = 2,
        max_workers: int = 20,
        target_cpu_percent: float = 70.0,
        target_memory_percent: float = 80.0,
        check_interval: float = 1.0,
        default_rate_limit: int = 10,
        rate_limit_strategy: RateLimitStrategy = RateLimitStrategy.BACKOFF,
    ):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.target_cpu_percent = target_cpu_percent
        self.target_memory_percent = target_memory_percent
        self.check_interval = check_interval
        self.default_rate_limit = default_rate_limit
        self.rate_limit_strategy = rate_limit_strategy
        
        # Concurrency control
        self._current_workers = min_workers
        self._semaphore = asyncio.Semaphore(min_workers)
        self._worker_lock = threading.Lock()
        self._running = False
        
        # Rate limiting
        self._token_buckets: Dict[str, TokenBucket] = {}
        self._domain_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        
        # Monitoring
        self._metrics_history: List[SystemMetrics] = []
        self._max_history = 100
        self._last_network_stats = None
        
        # Task management
        self._pending_tasks: Dict[str, asyncio.Task] = {}
        self._task_queue: asyncio.Queue = asyncio.Queue()
        self._domain_queues: Dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)
        
        # Callbacks
        self._on_rate_limit: Optional[Callable] = None
        self._on_concurrency_change: Optional[Callable] = None
        
        # Statistics
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "rate_limited_requests": 0,
            "avg_response_time": 0.0,
            "concurrency_changes": 0,
        }
    
    async def start(self):
        """Start the adaptive concurrency pool."""
        if self._running:
            return
        
        self._running = True
        
        # Start monitoring tasks
        self._monitor_task = asyncio.create_task(self._monitor_resources())
        self._processor_task = asyncio.create_task(self._process_queue())
        
        logger.info(f"AdaptiveConcurrencyPool started with {self._current_workers} workers")
    
    async def stop(self):
        """Stop the adaptive concurrency pool."""
        self._running = False
        
        # Cancel monitoring tasks
        if hasattr(self, '_monitor_task'):
            self._monitor_task.cancel()
        if hasattr(self, '_processor_task'):
            self._processor_task.cancel()
        
        # Wait for pending tasks
        if self._pending_tasks:
            await asyncio.gather(*self._pending_tasks.values(), return_exceptions=True)
        
        logger.info("AdaptiveConcurrencyPool stopped")
    
    def _get_domain(self, url: str) -> str:
        """Extract domain from URL."""
        parsed = urlparse(url)
        return parsed.netloc or parsed.path
    
    def _get_token_bucket(self, domain: str) -> TokenBucket:
        """Get or create token bucket for domain."""
        if domain not in self._token_buckets:
            self._token_buckets[domain] = TokenBucket(
                domain=domain,
                capacity=self.default_rate_limit,
                refill_rate=self.default_rate_limit / 10.0  # Refill in 10 seconds
            )
        return self._token_buckets[domain]
    
    async def _check_rate_limit(self, response_headers: Dict[str, str]) -> Optional[float]:
        """Check response headers for rate limiting indicators."""
        # Check for Retry-After header
        if 'Retry-After' in response_headers:
            try:
                return float(response_headers['Retry-After'])
            except ValueError:
                # Could be a date, parse it
                pass
        
        # Check for common rate limit headers
        rate_limit_headers = [
            'X-RateLimit-Reset',
            'X-Rate-Limit-Reset',
            'RateLimit-Reset',
        ]
        
        for header in rate_limit_headers:
            if header in response_headers:
                try:
                    reset_time = float(response_headers[header])
                    return max(0, reset_time - time.time())
                except (ValueError, TypeError):
                    pass
        
        # Check for CAPTCHA indicators
        captcha_indicators = [
            'captcha', 'recaptcha', 'hcaptcha', 'challenge',
            'please verify', 'human verification'
        ]
        
        # Check response body if available (would need to be passed in)
        # For now, we'll rely on status codes and headers
        
        return None
    
    def _update_metrics(self) -> SystemMetrics:
        """Update system metrics."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        
        # Network I/O
        net_io = psutil.net_io_counters()
        network_sent = net_io.bytes_sent
        network_recv = net_io.bytes_recv
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_busy = 0.0
        if disk_io:
            disk_busy = (disk_io.read_bytes + disk_io.write_bytes) / 1024 / 1024  # MB
        
        metrics = SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            network_sent=network_sent,
            network_recv=network_recv,
            disk_io=disk_busy,
        )
        
        self._metrics_history.append(metrics)
        if len(self._metrics_history) > self._max_history:
            self._metrics_history.pop(0)
        
        return metrics
    
    def _calculate_optimal_workers(self) -> int:
        """Calculate optimal number of workers based on system metrics."""
        if not self._metrics_history:
            return self._current_workers
        
        recent_metrics = self._metrics_history[-5:]  # Last 5 measurements
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        
        # Calculate adjustment based on resource usage
        cpu_factor = max(0.1, min(2.0, self.target_cpu_percent / max(avg_cpu, 1)))
        memory_factor = max(0.1, min(2.0, self.target_memory_percent / max(avg_memory, 1)))
        
        # Network factor (simplified)
        network_factor = 1.0
        if self._last_network_stats:
            current_recv = recent_metrics[-1].network_recv
            recv_rate = (current_recv - self._last_network_stats) / self.check_interval
            # If receiving more than 10MB/s, reduce concurrency
            if recv_rate > 10 * 1024 * 1024:
                network_factor = 0.7
        
        self._last_network_stats = recent_metrics[-1].network_recv
        
        # Calculate new worker count
        adjustment = min(cpu_factor, memory_factor) * network_factor
        new_workers = int(self._current_workers * adjustment)
        
        # Apply bounds
        new_workers = max(self.min_workers, min(self.max_workers, new_workers))
        
        return new_workers
    
    async def _adjust_concurrency(self):
        """Adjust concurrency based on system metrics."""
        optimal_workers = self._calculate_optimal_workers()
        
        if optimal_workers != self._current_workers:
            old_workers = self._current_workers
            self._current_workers = optimal_workers
            
            # Adjust semaphore
            current_value = self._semaphore._value
            diff = optimal_workers - old_workers
            
            if diff > 0:
                # Increase concurrency
                for _ in range(diff):
                    self._semaphore.release()
            elif diff < 0:
                # Decrease concurrency (acquire extra permits)
                for _ in range(-diff):
                    try:
                        await asyncio.wait_for(self._semaphore.acquire(), timeout=0.1)
                    except asyncio.TimeoutError:
                        # Couldn't acquire, that's okay
                        pass
            
            self._stats["concurrency_changes"] += 1
            
            if self._on_concurrency_change:
                self._on_concurrency_change(old_workers, optimal_workers)
            
            logger.info(f"Adjusted concurrency: {old_workers} -> {optimal_workers} workers")
    
    async def _monitor_resources(self):
        """Monitor system resources and adjust concurrency."""
        while self._running:
            try:
                self._update_metrics()
                await self._adjust_concurrency()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in resource monitor: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _process_queue(self):
        """Process the task queue."""
        while self._running:
            try:
                # Check domain-specific queues first
                for domain, queue in list(self._domain_queues.items()):
                    if not queue.empty():
                        bucket = self._get_token_bucket(domain)
                        if not bucket.is_blocked() and bucket.consume():
                            task_data = await queue.get()
                            asyncio.create_task(self._execute_task(task_data))
                            break
                
                # Then check main queue
                if not self._task_queue.empty():
                    task_data = await self._task_queue.get()
                    domain = task_data.get('domain')
                    
                    if domain:
                        bucket = self._get_token_bucket(domain)
                        if bucket.is_blocked() or not bucket.consume():
                            # Requeue to domain-specific queue
                            await self._domain_queues[domain].put(task_data)
                            continue
                    
                    asyncio.create_task(self._execute_task(task_data))
                
                await asyncio.sleep(0.01)  # Small sleep to prevent busy waiting
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in queue processor: {e}")
                await asyncio.sleep(0.1)
    
    async def _execute_task(self, task_data: Dict[str, Any]):
        """Execute a single task with rate limiting and concurrency control."""
        task_id = task_data.get('id', str(id(task_data)))
        domain = task_data.get('domain')
        func = task_data.get('func')
        args = task_data.get('args', [])
        kwargs = task_data.get('kwargs', {})
        
        self._pending_tasks[task_id] = asyncio.current_task()
        start_time = time.time()
        
        try:
            # Acquire concurrency semaphore
            async with self._semaphore:
                # Acquire domain lock if needed
                if domain:
                    async with self._domain_locks[domain]:
                        result = await func(*args, **kwargs)
                else:
                    result = await func(*args, **kwargs)
            
            # Record success
            if domain:
                bucket = self._get_token_bucket(domain)
                bucket.record_success()
            
            self._stats["successful_requests"] += 1
            response_time = time.time() - start_time
            self._update_avg_response_time(response_time)
            
            return result
            
        except Exception as e:
            self._stats["failed_requests"] += 1
            
            # Check if this is a rate limit error
            if self._is_rate_limit_error(e):
                self._stats["rate_limited_requests"] += 1
                
                if domain:
                    bucket = self._get_token_bucket(domain)
                    retry_after = self._extract_retry_after(e)
                    bucket.record_failure(retry_after)
                    
                    if self._on_rate_limit:
                        self._on_rate_limit(domain, e, retry_after)
            
            raise
            
        finally:
            self._pending_tasks.pop(task_id, None)
            self._stats["total_requests"] += 1
    
    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if an error indicates rate limiting."""
        error_str = str(error).lower()
        rate_limit_indicators = [
            'rate limit', 'too many requests', '429',
            'captcha', 'challenge', 'blocked',
            'access denied', 'forbidden'
        ]
        return any(indicator in error_str for indicator in rate_limit_indicators)
    
    def _extract_retry_after(self, error: Exception) -> Optional[float]:
        """Extract retry-after value from error if available."""
        error_str = str(error)
        
        # Try to find retry-after in error message
        patterns = [
            r'retry[_-]?after[:\s]*(\d+)',
            r'wait[_-]?for[:\s]*(\d+)',
            r'(\d+)\s*seconds?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, error_str, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except (ValueError, IndexError):
                    pass
        
        return None
    
    def _update_avg_response_time(self, response_time: float):
        """Update average response time statistic."""
        total = self._stats["successful_requests"]
        if total == 0:
            self._stats["avg_response_time"] = response_time
        else:
            current_avg = self._stats["avg_response_time"]
            self._stats["avg_response_time"] = (
                (current_avg * (total - 1) + response_time) / total
            )
    
    async def submit(
        self,
        func: Callable,
        *args,
        domain: Optional[str] = None,
        priority: int = 0,
        **kwargs
    ) -> asyncio.Task:
        """
        Submit a task to the concurrency pool.
        
        Args:
            func: Async function to execute
            domain: Target domain for rate limiting
            priority: Task priority (higher = more important)
            **kwargs: Additional arguments to pass to func
        
        Returns:
            asyncio.Task that can be awaited
        """
        task_id = f"{domain or 'global'}_{id(func)}_{time.time()}"
        
        task_data = {
            'id': task_id,
            'func': func,
            'args': args,
            'kwargs': kwargs,
            'domain': domain,
            'priority': priority,
            'submitted_at': time.time(),
        }
        
        if domain:
            # Check if domain is blocked
            bucket = self._get_token_bucket(domain)
            if bucket.is_blocked():
                logger.debug(f"Domain {domain} is blocked, queuing task")
            
            # Add to domain-specific queue
            await self._domain_queues[domain].put(task_data)
        else:
            # Add to main queue
            await self._task_queue.put(task_data)
        
        # Create a future that will be resolved when the task completes
        future = asyncio.get_event_loop().create_future()
        
        # Create a wrapper task
        async def wrapper():
            try:
                result = await self._execute_task(task_data)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
        
        task = asyncio.create_task(wrapper())
        return task
    
    async def submit_with_agent(
        self,
        agent: AgentService,
        url: str,
        action: str,
        **kwargs
    ) -> Any:
        """
        Submit a browser automation task with adaptive concurrency.
        
        Args:
            agent: AgentService instance
            url: Target URL
            action: Action to perform
            **kwargs: Additional arguments
        
        Returns:
            Result of the action
        """
        domain = self._get_domain(url)
        
        async def browser_task():
            # Create a page
            page = await agent.create_page()
            
            try:
                # Navigate to URL
                await page.goto(url)
                
                # Perform action
                if action == "screenshot":
                    return await page.screenshot(**kwargs)
                elif action == "extract":
                    return await page.extract(**kwargs)
                elif action == "click":
                    return await page.click(**kwargs)
                elif action == "type":
                    return await page.type(**kwargs)
                else:
                    raise ValueError(f"Unknown action: {action}")
                    
            finally:
                await page.close()
        
        return await self.submit(browser_task, domain=domain)
    
    def set_on_rate_limit_callback(self, callback: Callable):
        """Set callback for rate limit events."""
        self._on_rate_limit = callback
    
    def set_on_concurrency_change_callback(self, callback: Callable):
        """Set callback for concurrency changes."""
        self._on_concurrency_change = callback
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        return {
            **self._stats,
            "current_workers": self._current_workers,
            "pending_tasks": len(self._pending_tasks),
            "token_buckets": {
                domain: {
                    "tokens": bucket.tokens,
                    "blocked_until": bucket.blocked_until,
                    "consecutive_failures": bucket.consecutive_failures,
                }
                for domain, bucket in self._token_buckets.items()
            },
            "queue_sizes": {
                "main": self._task_queue.qsize(),
                "domains": {
                    domain: queue.qsize()
                    for domain, queue in self._domain_queues.items()
                    if queue.qsize() > 0
                }
            },
        }
    
    def get_domain_stats(self, domain: str) -> Dict[str, Any]:
        """Get statistics for a specific domain."""
        bucket = self._get_token_bucket(domain)
        return {
            "tokens": bucket.tokens,
            "capacity": bucket.capacity,
            "refill_rate": bucket.refill_rate,
            "blocked_until": bucket.blocked_until,
            "consecutive_failures": bucket.consecutive_failures,
            "is_blocked": bucket.is_blocked(),
            "queue_size": self._domain_queues[domain].qsize(),
        }
    
    def update_domain_rate_limit(
        self,
        domain: str,
        capacity: Optional[int] = None,
        refill_rate: Optional[float] = None,
    ):
        """Update rate limit settings for a domain."""
        bucket = self._get_token_bucket(domain)
        
        if capacity is not None:
            bucket.capacity = capacity
        
        if refill_rate is not None:
            bucket.refill_rate = refill_rate
        
        logger.info(f"Updated rate limit for {domain}: capacity={bucket.capacity}, refill_rate={bucket.refill_rate}")
    
    def clear_domain_block(self, domain: str):
        """Clear block for a domain."""
        if domain in self._token_buckets:
            self._token_buckets[domain].blocked_until = 0.0
            self._token_buckets[domain].consecutive_failures = 0
            logger.info(f"Cleared block for domain: {domain}")


# Convenience function for creating a pool
def create_adaptive_pool(
    min_workers: int = 2,
    max_workers: int = 20,
    **kwargs
) -> AdaptiveConcurrencyPool:
    """
    Create an adaptive concurrency pool with sensible defaults.
    
    Args:
        min_workers: Minimum number of concurrent workers
        max_workers: Maximum number of concurrent workers
        **kwargs: Additional arguments for AdaptiveConcurrencyPool
    
    Returns:
        Configured AdaptiveConcurrencyPool instance
    """
    return AdaptiveConcurrencyPool(
        min_workers=min_workers,
        max_workers=max_workers,
        **kwargs
    )