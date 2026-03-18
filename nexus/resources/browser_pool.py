"""
Cost-Optimized Resource Management for Browser Automation
Dynamically optimizes browser instances, memory usage, and network bandwidth
"""

import asyncio
import psutil
import time
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class TaskComplexity(Enum):
    """Task complexity levels for resource estimation"""
    LIGHT = "light"      # Simple navigation, minimal JavaScript
    MEDIUM = "medium"    # Standard web interactions, moderate JS
    HEAVY = "heavy"      # Complex SPAs, heavy JavaScript execution
    EXTREME = "extreme"  # Media-heavy sites, WebGL, video processing

@dataclass
class ResourceEstimate:
    """Estimated resource requirements for a task"""
    memory_mb: float
    cpu_percent: float
    network_bandwidth_mbps: float
    estimated_duration_seconds: float
    complexity: TaskComplexity
    confidence: float = 0.8  # Confidence in the estimate

@dataclass
class BrowserInstance:
    """Represents a managed browser instance with resource tracking"""
    id: str
    browser: Any  # Playwright browser instance
    context: Any  # Browser context
    pages: List[Any] = field(default_factory=list)
    memory_usage_mb: float = 0.0
    cpu_percent: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_received: int = 0
    last_activity: float = field(default_factory=time.time)
    task_count: int = 0
    is_healthy: bool = True
    created_at: float = field(default_factory=time.time)

class ResourceMonitor:
    """Monitors system and browser resource usage"""
    
    def __init__(self, check_interval: float = 5.0):
        self.check_interval = check_interval
        self.system_memory_percent = 0.0
        self.system_cpu_percent = 0.0
        self.browser_processes: Dict[int, psutil.Process] = {}
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        
    async def start_monitoring(self):
        """Start background monitoring"""
        if self._monitoring:
            return
            
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        
    async def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
                
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self._monitoring:
            try:
                await self._update_system_metrics()
                await self._update_browser_metrics()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in resource monitor: {e}")
                await asyncio.sleep(self.check_interval)
                
    async def _update_system_metrics(self):
        """Update system-wide resource metrics"""
        self.system_memory_percent = psutil.virtual_memory().percent
        self.system_cpu_percent = psutil.cpu_percent(interval=0.1)
        
    async def _update_browser_metrics(self):
        """Update metrics for tracked browser processes"""
        for pid, process in list(self.browser_processes.items()):
            try:
                if process.is_running():
                    with process.oneshot():
                        memory_info = process.memory_info()
                        self.browser_processes[pid] = {
                            'memory_mb': memory_info.rss / 1024 / 1024,
                            'cpu_percent': process.cpu_percent(),
                            'threads': process.num_threads()
                        }
                else:
                    del self.browser_processes[pid]
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                del self.browser_processes[pid]
                
    def register_browser_process(self, pid: int):
        """Register a browser process for monitoring"""
        try:
            self.browser_processes[pid] = psutil.Process(pid)
        except psutil.NoSuchProcess:
            logger.warning(f"Browser process {pid} not found")
            
    def get_browser_metrics(self, pid: int) -> Dict[str, Any]:
        """Get metrics for a specific browser process"""
        return self.browser_processes.get(pid, {})
        
    def should_throttle(self) -> bool:
        """Determine if we should throttle based on system load"""
        return self.system_memory_percent > 85 or self.system_cpu_percent > 80

class CostEstimator:
    """Estimates resource costs for different task types"""
    
    # Resource profiles for different task types
    TASK_PROFILES = {
        TaskComplexity.LIGHT: ResourceEstimate(
            memory_mb=50,
            cpu_percent=5,
            network_bandwidth_mbps=1,
            estimated_duration_seconds=10,
            complexity=TaskComplexity.LIGHT
        ),
        TaskComplexity.MEDIUM: ResourceEstimate(
            memory_mb=150,
            cpu_percent=15,
            network_bandwidth_mbps=5,
            estimated_duration_seconds=30,
            complexity=TaskComplexity.MEDIUM
        ),
        TaskComplexity.HEAVY: ResourceEstimate(
            memory_mb=300,
            cpu_percent=30,
            network_bandwidth_mbps=10,
            estimated_duration_seconds=60,
            complexity=TaskComplexity.HEAVY
        ),
        TaskComplexity.EXTREME: ResourceEstimate(
            memory_mb=500,
            cpu_percent=50,
            network_bandwidth_mbps=20,
            estimated_duration_seconds=120,
            complexity=TaskComplexity.EXTREME
        )
    }
    
    @classmethod
    def estimate_task(cls, task_type: str, url: Optional[str] = None) -> ResourceEstimate:
        """Estimate resource needs for a task"""
        complexity = cls._classify_task(task_type, url)
        base_estimate = cls.TASK_PROFILES[complexity]
        
        # Adjust based on URL characteristics if available
        if url:
            base_estimate = cls._adjust_for_url(base_estimate, url)
            
        return base_estimate
        
    @classmethod
    def _classify_task(cls, task_type: str, url: Optional[str] = None) -> TaskComplexity:
        """Classify task complexity based on type and URL"""
        task_type_lower = task_type.lower()
        
        # Simple classification based on task type keywords
        if any(kw in task_type_lower for kw in ['simple', 'navigate', 'basic', 'login']):
            return TaskComplexity.LIGHT
        elif any(kw in task_type_lower for kw in ['form', 'search', 'click', 'input']):
            return TaskComplexity.MEDIUM
        elif any(kw in task_type_lower for kw in ['scrape', 'extract', 'parse', 'analyze']):
            return TaskComplexity.HEAVY
        elif any(kw in task_type_lower for kw in ['video', 'stream', 'game', '3d', 'webgl']):
            return TaskComplexity.EXTREME
        else:
            return TaskComplexity.MEDIUM  # Default
            
    @classmethod
    def _adjust_for_url(cls, estimate: ResourceEstimate, url: str) -> ResourceEstimate:
        """Adjust estimate based on URL characteristics"""
        url_lower = url.lower()
        
        # Adjust for known heavy sites
        heavy_domains = ['youtube.com', 'facebook.com', 'twitter.com', 'netflix.com']
        if any(domain in url_lower for domain in heavy_domains):
            estimate.memory_mb *= 1.5
            estimate.cpu_percent *= 1.3
            estimate.network_bandwidth_mbps *= 2.0
            
        # Adjust for video content
        if any(ext in url_lower for ext in ['.mp4', '.webm', '.m3u8', 'video']):
            estimate.memory_mb *= 2.0
            estimate.network_bandwidth_mbps *= 3.0
            
        return estimate
        
    @classmethod
    def calculate_cost(cls, estimate: ResourceEstimate, duration_hours: float = 1.0) -> Dict[str, float]:
        """Calculate estimated cost based on resource usage"""
        # Simple cost model (in abstract units)
        memory_cost = estimate.memory_mb * 0.001  # Cost per MB-hour
        cpu_cost = estimate.cpu_percent * 0.01    # Cost per CPU%-hour
        network_cost = estimate.network_bandwidth_mbps * 0.05  # Cost per Mbps-hour
        
        total_cost = (memory_cost + cpu_cost + network_cost) * duration_hours
        
        return {
            'memory_cost': memory_cost * duration_hours,
            'cpu_cost': cpu_cost * duration_hours,
            'network_cost': network_cost * duration_hours,
            'total_cost': total_cost,
            'cost_per_hour': total_cost / duration_hours
        }

class BrowserPool:
    """Manages a pool of browser instances with auto-scaling"""
    
    def __init__(
        self,
        min_instances: int = 1,
        max_instances: int = 10,
        max_pages_per_instance: int = 5,
        idle_timeout: float = 300.0,  # 5 minutes
        memory_threshold_mb: float = 1024.0,  # 1GB
        scale_up_threshold: float = 0.8,  # Scale up when 80% utilized
        scale_down_threshold: float = 0.3   # Scale down when 30% utilized
    ):
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.max_pages_per_instance = max_pages_per_instance
        self.idle_timeout = idle_timeout
        self.memory_threshold_mb = memory_threshold_mb
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        
        self.instances: Dict[str, BrowserInstance] = {}
        self.available_instances: deque = deque()
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.resource_monitor = ResourceMonitor()
        self.cost_estimator = CostEstimator()
        
        self._scaling_lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._scaling_task: Optional[asyncio.Task] = None
        self._playwright = None
        
    async def initialize(self):
        """Initialize the browser pool"""
        await self.resource_monitor.start_monitoring()
        
        # Start background tasks
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._scaling_task = asyncio.create_task(self._scaling_loop())
        
        # Create minimum instances
        for _ in range(self.min_instances):
            await self._create_instance()
            
        logger.info(f"Browser pool initialized with {self.min_instances} instances")
        
    async def shutdown(self):
        """Shutdown the browser pool"""
        # Cancel background tasks
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._scaling_task:
            self._scaling_task.cancel()
            
        # Stop monitoring
        await self.resource_monitor.stop_monitoring()
        
        # Close all instances
        for instance_id in list(self.instances.keys()):
            await self._close_instance(instance_id)
            
        logger.info("Browser pool shut down")
        
    @asynccontextmanager
    async def get_browser_page(self, task_estimate: Optional[ResourceEstimate] = None):
        """Get a browser page from the pool"""
        instance = await self._acquire_instance(task_estimate)
        page = None
        
        try:
            # Create a new page in the instance
            page = await instance.context.new_page()
            instance.pages.append(page)
            instance.task_count += 1
            instance.last_activity = time.time()
            
            yield page
            
        finally:
            if page:
                await self._release_page(instance, page)
                
    async def _acquire_instance(self, task_estimate: Optional[ResourceEstimate] = None) -> BrowserInstance:
        """Acquire a browser instance for use"""
        # Try to get an available instance
        if self.available_instances:
            instance_id = self.available_instances.popleft()
            instance = self.instances[instance_id]
            
            # Check if instance can handle more pages
            if len(instance.pages) < self.max_pages_per_instance:
                return instance
                
        # No suitable instance available, create or wait
        async with self._scaling_lock:
            # Try to create a new instance if under limit
            if len(self.instances) < self.max_instances:
                return await self._create_instance()
                
        # Wait for an instance to become available
        while True:
            if self.available_instances:
                instance_id = self.available_instances.popleft()
                instance = self.instances[instance_id]
                if len(instance.pages) < self.max_pages_per_instance:
                    return instance
                    
            await asyncio.sleep(0.1)  # Small delay before retry
            
    async def _release_page(self, instance: BrowserInstance, page: Any):
        """Release a page back to the pool"""
        try:
            # Close the page to free resources
            await page.close()
            if page in instance.pages:
                instance.pages.remove(page)
                
            # Return instance to available pool if healthy
            if instance.is_healthy and len(instance.pages) < self.max_pages_per_instance:
                self.available_instances.append(instance.id)
                
        except Exception as e:
            logger.error(f"Error releasing page: {e}")
            instance.is_healthy = False
            
    async def _create_instance(self) -> BrowserInstance:
        """Create a new browser instance"""
        try:
            # Import playwright here to avoid circular imports
            from playwright.async_api import async_playwright
            
            if not self._playwright:
                self._playwright = await async_playwright().start()
                
            browser = await self._playwright.chromium.launch(
                headless=True,
                args=['--disable-dev-shm-usage', '--no-sandbox']
            )
            
            context = await browser.new_context(
                viewport={'width': 1280, 'height': 720},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            
            instance_id = f"browser_{int(time.time() * 1000)}_{len(self.instances)}"
            
            instance = BrowserInstance(
                id=instance_id,
                browser=browser,
                context=context
            )
            
            self.instances[instance_id] = instance
            self.available_instances.append(instance_id)
            
            # Register browser process for monitoring
            if hasattr(browser, 'pid'):
                self.resource_monitor.register_browser_process(browser.pid)
                
            logger.info(f"Created new browser instance: {instance_id}")
            return instance
            
        except Exception as e:
            logger.error(f"Failed to create browser instance: {e}")
            raise
            
    async def _close_instance(self, instance_id: str):
        """Close and remove a browser instance"""
        if instance_id not in self.instances:
            return
            
        instance = self.instances[instance_id]
        
        try:
            # Close all pages
            for page in instance.pages[:]:  # Copy list to avoid modification during iteration
                try:
                    await page.close()
                except:
                    pass
                    
            # Close context and browser
            await instance.context.close()
            await instance.browser.close()
            
        except Exception as e:
            logger.error(f"Error closing browser instance {instance_id}: {e}")
            
        finally:
            # Remove from pools
            if instance_id in self.available_instances:
                self.available_instances.remove(instance_id)
            del self.instances[instance_id]
            
            logger.info(f"Closed browser instance: {instance_id}")
            
    async def _cleanup_loop(self):
        """Background task to clean up idle instances and tabs"""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                await self._cleanup_idle_resources()
                await self._cleanup_excess_memory()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                
    async def _cleanup_idle_resources(self):
        """Clean up idle instances and pages"""
        current_time = time.time()
        
        for instance_id, instance in list(self.instances.items()):
            # Check for idle pages
            idle_pages = []
            for page in instance.pages[:]:
                try:
                    # Simple heuristic: if page hasn't been used in a while
                    # In production, you'd track actual page activity
                    if current_time - instance.last_activity > self.idle_timeout:
                        idle_pages.append(page)
                except:
                    pass
                    
            # Close idle pages
            for page in idle_pages:
                try:
                    await page.close()
                    instance.pages.remove(page)
                except:
                    pass
                    
            # Check if instance is idle and we have more than minimum
            if (not instance.pages and 
                current_time - instance.last_activity > self.idle_timeout and
                len(self.instances) > self.min_instances):
                
                await self._close_instance(instance_id)
                
    async def _cleanup_excess_memory(self):
        """Clean up resources if memory usage is high"""
        if self.resource_monitor.system_memory_percent > 85:
            logger.warning(f"High memory usage: {self.resource_monitor.system_memory_percent}%")
            
            # Close oldest instances first
            instances_by_age = sorted(
                self.instances.values(),
                key=lambda x: x.created_at
            )
            
            for instance in instances_by_age:
                if len(self.instances) <= self.min_instances:
                    break
                    
                if not instance.pages:  # Only close idle instances
                    await self._close_instance(instance.id)
                    
    async def _scaling_loop(self):
        """Background task to scale instances based on demand"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                await self._auto_scale()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}")
                
    async def _auto_scale(self):
        """Automatically scale instances based on load"""
        total_instances = len(self.instances)
        total_pages = sum(len(inst.pages) for inst in self.instances.values())
        avg_utilization = total_pages / (total_instances * self.max_pages_per_instance) if total_instances > 0 else 0
        
        # Scale up if high utilization
        if (avg_utilization > self.scale_up_threshold and 
            total_instances < self.max_instances and
            not self.resource_monitor.should_throttle()):
            
            async with self._scaling_lock:
                if len(self.instances) < self.max_instances:
                    await self._create_instance()
                    logger.info(f"Scaled up to {len(self.instances)} instances")
                    
        # Scale down if low utilization
        elif (avg_utilization < self.scale_down_threshold and 
              total_instances > self.min_instances):
              
            async with self._scaling_lock:
                # Find an idle instance to remove
                for instance_id, instance in list(self.instances.items()):
                    if not instance.pages and len(self.instances) > self.min_instances:
                        await self._close_instance(instance_id)
                        logger.info(f"Scaled down to {len(self.instances)} instances")
                        break
                        
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get current pool statistics"""
        total_instances = len(self.instances)
        total_pages = sum(len(inst.pages) for inst in self.instances.values())
        available_instances = len(self.available_instances)
        
        return {
            'total_instances': total_instances,
            'total_pages': total_pages,
            'available_instances': available_instances,
            'avg_pages_per_instance': total_pages / total_instances if total_instances > 0 else 0,
            'system_memory_percent': self.resource_monitor.system_memory_percent,
            'system_cpu_percent': self.resource_monitor.system_cpu_percent,
            'estimated_cost_per_hour': self._estimate_current_cost()
        }
        
    def _estimate_current_cost(self) -> float:
        """Estimate current hourly cost of the pool"""
        total_cost = 0.0
        
        for instance in self.instances.values():
            # Estimate based on instance activity
            # In production, you'd use actual resource measurements
            estimate = ResourceEstimate(
                memory_mb=100,  # Base memory per instance
                cpu_percent=10,  # Base CPU per instance
                network_bandwidth_mbps=2,
                estimated_duration_seconds=3600,
                complexity=TaskComplexity.MEDIUM
            )
            
            cost = self.cost_estimator.calculate_cost(estimate, 1.0)
            total_cost += cost['total_cost']
            
        return total_cost
        
    async def estimate_task_cost(self, task_type: str, url: Optional[str] = None) -> Dict[str, Any]:
        """Estimate cost for a specific task"""
        estimate = self.cost_estimator.estimate_task(task_type, url)
        cost = self.cost_estimator.calculate_cost(estimate, estimate.estimated_duration_seconds / 3600)
        
        return {
            'resource_estimate': {
                'memory_mb': estimate.memory_mb,
                'cpu_percent': estimate.cpu_percent,
                'network_bandwidth_mbps': estimate.network_bandwidth_mbps,
                'estimated_duration_seconds': estimate.estimated_duration_seconds,
                'complexity': estimate.complexity.value,
                'confidence': estimate.confidence
            },
            'cost_estimate': cost,
            'recommendations': self._get_recommendations(estimate)
        }
        
    def _get_recommendations(self, estimate: ResourceEstimate) -> List[str]:
        """Get optimization recommendations based on estimate"""
        recommendations = []
        
        if estimate.complexity == TaskComplexity.EXTREME:
            recommendations.append("Consider running during off-peak hours to reduce costs")
            recommendations.append("Use headless mode to reduce memory usage")
            
        if estimate.memory_mb > 300:
            recommendations.append("Enable tab cleanup to manage memory")
            
        if estimate.network_bandwidth_mbps > 10:
            recommendations.append("Consider caching static resources")
            
        if self.resource_monitor.should_throttle():
            recommendations.append("System load is high, consider delaying non-critical tasks")
            
        return recommendations

# Global browser pool instance
_browser_pool: Optional[BrowserPool] = None

async def get_browser_pool() -> BrowserPool:
    """Get or create the global browser pool instance"""
    global _browser_pool
    
    if _browser_pool is None:
        _browser_pool = BrowserPool()
        await _browser_pool.initialize()
        
    return _browser_pool

async def shutdown_browser_pool():
    """Shutdown the global browser pool"""
    global _browser_pool
    
    if _browser_pool:
        await _browser_pool.shutdown()
        _browser_pool = None