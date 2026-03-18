"""
nexus/resources/cost_optimizer.py

Cost-Optimized Resource Management System
Dynamically optimizes browser instances, memory usage, and network bandwidth
based on task requirements with predictive scaling and automatic cleanup.
"""

import asyncio
import psutil
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import json
from datetime import datetime, timedelta

from playwright.async_api import Browser, BrowserContext, Page, async_playwright
from ..actor.page import Page as ActorPage
from ..agent.service import AgentService

logger = logging.getLogger(__name__)


class TaskComplexity(Enum):
    """Task complexity levels for resource estimation"""
    LIGHT = "light"      # Simple page loads, minimal JS
    MEDIUM = "medium"    # Standard web interactions
    HEAVY = "heavy"      # Complex SPAs, heavy JS, multiple tabs
    EXTREME = "extreme"  # Resource-intensive automation


class ResourceType(Enum):
    """Types of resources managed"""
    BROWSER_INSTANCE = "browser_instance"
    MEMORY = "memory"
    NETWORK = "network"
    CPU = "cpu"


@dataclass
class ResourceEstimate:
    """Estimated resource requirements for a task"""
    task_type: str
    complexity: TaskComplexity
    estimated_memory_mb: float
    estimated_duration_seconds: float
    estimated_network_mb: float
    browser_instances_needed: int
    max_concurrent_tabs: int
    priority: int = 1  # 1-10, higher = more important


@dataclass
class BrowserInstance:
    """Managed browser instance with usage tracking"""
    id: str
    browser: Browser
    context: Optional[BrowserContext] = None
    active_pages: List[Page] = field(default_factory=list)
    memory_usage_mb: float = 0.0
    last_used: datetime = field(default_factory=datetime.now)
    total_requests: int = 0
    is_healthy: bool = True
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ResourceUsage:
    """Current resource usage snapshot"""
    timestamp: datetime
    total_memory_mb: float
    browser_memory_mb: float
    cpu_percent: float
    active_browsers: int
    active_pages: int
    network_sent_mb: float
    network_recv_mb: float
    queue_depth: int


class CostOptimizer:
    """
    Main cost optimization engine that manages browser instances,
    memory usage, and network bandwidth dynamically.
    """
    
    def __init__(self, 
                 min_browsers: int = 2,
                 max_browsers: int = 20,
                 memory_threshold_mb: float = 4096,
                 cleanup_interval_seconds: int = 300,
                 scale_check_interval: int = 30):
        
        self.min_browsers = min_browsers
        self.max_browsers = max_browsers
        self.memory_threshold_mb = memory_threshold_mb
        self.cleanup_interval = cleanup_interval_seconds
        self.scale_check_interval = scale_check_interval
        
        # Resource pools
        self.browser_pool: Dict[str, BrowserInstance] = {}
        self.available_browsers: deque = deque()
        self.busy_browsers: Dict[str, BrowserInstance] = {}
        
        # Task queues
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.priority_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        
        # Monitoring
        self.resource_history: deque = deque(maxlen=1000)
        self.task_history: deque = deque(maxlen=5000)
        self.cost_metrics: Dict[str, Any] = {
            "total_browsers_created": 0,
            "total_browsers_destroyed": 0,
            "total_pages_created": 0,
            "total_pages_closed": 0,
            "total_memory_saved_mb": 0.0,
            "total_estimated_cost_usd": 0.0
        }
        
        # Predictive models (simplified for MVP)
        self.task_patterns: Dict[str, ResourceEstimate] = {}
        self._initialize_task_patterns()
        
        # Control flags
        self._running = False
        self._playwright = None
        self._monitor_task = None
        self._cleanup_task = None
        self._scaling_task = None
        
        # Network monitoring
        self._last_net_io = psutil.net_io_counters()
        self._last_net_check = time.time()
    
    def _initialize_task_patterns(self):
        """Initialize default task patterns for common operations"""
        self.task_patterns = {
            "page_load": ResourceEstimate(
                task_type="page_load",
                complexity=TaskComplexity.LIGHT,
                estimated_memory_mb=100,
                estimated_duration_seconds=5,
                estimated_network_mb=2,
                browser_instances_needed=1,
                max_concurrent_tabs=3
            ),
            "form_fill": ResourceEstimate(
                task_type="form_fill",
                complexity=TaskComplexity.MEDIUM,
                estimated_memory_mb=200,
                estimated_duration_seconds=15,
                estimated_network_mb=5,
                browser_instances_needed=1,
                max_concurrent_tabs=5
            ),
            "data_scrape": ResourceEstimate(
                task_type="data_scrape",
                complexity=TaskComplexity.HEAVY,
                estimated_memory_mb=500,
                estimated_duration_seconds=60,
                estimated_network_mb=50,
                browser_instances_needed=2,
                max_concurrent_tabs=10
            ),
            "automation_workflow": ResourceEstimate(
                task_type="automation_workflow",
                complexity=TaskComplexity.EXTREME,
                estimated_memory_mb=1024,
                estimated_duration_seconds=300,
                estimated_network_mb=200,
                browser_instances_needed=4,
                max_concurrent_tabs=20
            )
        }
    
    async def start(self):
        """Start the cost optimizer and all background tasks"""
        if self._running:
            return
        
        self._running = True
        self._playwright = await async_playwright().start()
        
        # Initialize minimum browser pool
        await self._initialize_browser_pool()
        
        # Start background tasks
        self._monitor_task = asyncio.create_task(self._monitor_resources())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._scaling_task = asyncio.create_task(self._scaling_loop())
        
        logger.info(f"CostOptimizer started with {len(self.browser_pool)} browsers")
    
    async def stop(self):
        """Stop the cost optimizer and cleanup resources"""
        self._running = False
        
        # Cancel background tasks
        for task in [self._monitor_task, self._cleanup_task, self._scaling_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Close all browsers
        for instance in list(self.browser_pool.values()):
            await self._destroy_browser_instance(instance.id)
        
        if self._playwright:
            await self._playwright.stop()
        
        logger.info("CostOptimizer stopped")
    
    async def _initialize_browser_pool(self):
        """Initialize the minimum number of browser instances"""
        for i in range(self.min_browsers):
            await self._create_browser_instance()
    
    async def _create_browser_instance(self) -> str:
        """Create a new browser instance and add to pool"""
        try:
            browser = await self._playwright.chromium.launch(
                headless=True,
                args=[
                    '--disable-dev-shm-usage',
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-accelerated-2d-canvas',
                    '--disable-gpu',
                    '--window-size=1920,1080'
                ]
            )
            
            instance_id = f"browser_{int(time.time())}_{len(self.browser_pool)}"
            instance = BrowserInstance(
                id=instance_id,
                browser=browser
            )
            
            self.browser_pool[instance_id] = instance
            self.available_browsers.append(instance_id)
            self.cost_metrics["total_browsers_created"] += 1
            
            logger.debug(f"Created browser instance {instance_id}")
            return instance_id
            
        except Exception as e:
            logger.error(f"Failed to create browser instance: {e}")
            raise
    
    async def _destroy_browser_instance(self, instance_id: str):
        """Destroy a browser instance and remove from pool"""
        if instance_id not in self.browser_pool:
            return
        
        instance = self.browser_pool[instance_id]
        
        try:
            # Close all pages in the instance
            for page in instance.active_pages[:]:
                try:
                    await page.close()
                except:
                    pass
            
            # Close the browser
            await instance.browser.close()
            
            # Remove from tracking
            if instance_id in self.available_browsers:
                self.available_browsers.remove(instance_id)
            if instance_id in self.busy_browsers:
                del self.busy_browsers[instance_id]
            
            del self.browser_pool[instance_id]
            self.cost_metrics["total_browsers_destroyed"] += 1
            
            logger.debug(f"Destroyed browser instance {instance_id}")
            
        except Exception as e:
            logger.error(f"Error destroying browser instance {instance_id}: {e}")
    
    async def get_browser_for_task(self, 
                                  task_type: str, 
                                  priority: int = 1) -> Tuple[str, BrowserContext]:
        """
        Get a browser instance optimized for the given task type.
        Returns (instance_id, context)
        """
        # Estimate resources needed
        estimate = self.estimate_resources(task_type)
        
        # Check if we need to scale up
        await self._check_scaling_needs(estimate)
        
        # Wait for available browser
        instance_id = await self._acquire_browser(priority)
        instance = self.browser_pool[instance_id]
        
        # Create optimized context
        context = await self._create_optimized_context(instance, estimate)
        instance.context = context
        
        # Move to busy pool
        if instance_id in self.available_browsers:
            self.available_browsers.remove(instance_id)
        self.busy_browsers[instance_id] = instance
        
        # Track usage
        instance.last_used = datetime.now()
        instance.total_requests += 1
        
        return instance_id, context
    
    async def release_browser(self, instance_id: str, force: bool = False):
        """Release a browser instance back to the pool"""
        if instance_id not in self.busy_browsers:
            return
        
        instance = self.busy_browsers[instance_id]
        
        # Check if instance is still healthy
        if not force and not await self._check_browser_health(instance):
            await self._destroy_browser_instance(instance_id)
            await self._create_browser_instance()  # Replace with new instance
            return
        
        # Close context but keep browser
        if instance.context:
            try:
                await instance.context.close()
            except:
                pass
            instance.context = None
        
        # Clear pages
        instance.active_pages.clear()
        
        # Move back to available pool
        del self.busy_browsers[instance_id]
        self.available_browsers.append(instance_id)
        
        logger.debug(f"Released browser instance {instance_id}")
    
    async def _acquire_browser(self, priority: int) -> str:
        """Acquire an available browser instance, waiting if necessary"""
        # Try to get from available pool
        if self.available_browsers:
            return self.available_browsers.popleft()
        
        # If none available, wait or create new one
        if len(self.browser_pool) < self.max_browsers:
            return await self._create_browser_instance()
        
        # Wait for one to become available
        logger.warning(f"No browsers available, waiting... (queue depth: {self.task_queue.qsize()})")
        while not self.available_browsers:
            await asyncio.sleep(0.1)
        
        return self.available_browsers.popleft()
    
    async def _create_optimized_context(self, 
                                       instance: BrowserInstance, 
                                       estimate: ResourceEstimate) -> BrowserContext:
        """Create an optimized browser context based on task requirements"""
        # Configure context based on task complexity
        context_options = {
            "viewport": {"width": 1920, "height": 1080},
            "ignore_https_errors": True,
            "java_script_enabled": True,
        }
        
        # Adjust based on complexity
        if estimate.complexity in [TaskComplexity.HEAVY, TaskComplexity.EXTREME]:
            context_options["bypass_csp"] = True
            context_options["user_agent"] = (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        
        # Limit resources for lighter tasks
        if estimate.complexity == TaskComplexity.LIGHT:
            context_options["java_script_enabled"] = False
        
        context = await instance.browser.new_context(**context_options)
        
        # Set resource limits
        page = await context.new_page()
        await page.route("**/*.{png,jpg,jpeg,gif,svg,ico}", lambda route: route.abort())
        
        instance.active_pages.append(page)
        self.cost_metrics["total_pages_created"] += 1
        
        return context
    
    async def _check_scaling_needs(self, estimate: ResourceEstimate):
        """Check if we need to scale up based on current load"""
        queue_depth = self.task_queue.qsize()
        available_count = len(self.available_browsers)
        
        # Scale up if queue is building and we have capacity
        if queue_depth > available_count * 2 and len(self.browser_pool) < self.max_browsers:
            scale_count = min(
                estimate.browser_instances_needed,
                self.max_browsers - len(self.browser_pool)
            )
            
            for _ in range(scale_count):
                await self._create_browser_instance()
            
            logger.info(f"Scaled up by {scale_count} browsers (total: {len(self.browser_pool)})")
    
    async def _monitor_resources(self):
        """Background task to monitor resource usage"""
        while self._running:
            try:
                usage = await self._collect_resource_usage()
                self.resource_history.append(usage)
                
                # Update cost metrics
                self._update_cost_metrics(usage)
                
                # Check for memory pressure
                if usage.browser_memory_mb > self.memory_threshold_mb * 0.8:
                    await self._emergency_cleanup()
                
                await asyncio.sleep(5)  # Collect every 5 seconds
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _collect_resource_usage(self) -> ResourceUsage:
        """Collect current resource usage metrics"""
        # System metrics
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Network metrics
        current_time = time.time()
        current_net = psutil.net_io_counters()
        time_diff = current_time - self._last_net_check
        
        if time_diff > 0:
            sent_mb = (current_net.bytes_sent - self._last_net_io.bytes_sent) / (1024 * 1024)
            recv_mb = (current_net.bytes_recv - self._last_net_io.bytes_recv) / (1024 * 1024)
        else:
            sent_mb = recv_mb = 0
        
        self._last_net_io = current_net
        self._last_net_check = current_time
        
        # Browser-specific metrics
        browser_memory = 0
        total_pages = 0
        
        for instance in self.browser_pool.values():
            # Estimate memory per browser (simplified)
            instance.memory_usage_mb = len(instance.active_pages) * 50  # ~50MB per page
            browser_memory += instance.memory_usage_mb
            total_pages += len(instance.active_pages)
        
        return ResourceUsage(
            timestamp=datetime.now(),
            total_memory_mb=memory.used / (1024 * 1024),
            browser_memory_mb=browser_memory,
            cpu_percent=cpu_percent,
            active_browsers=len(self.busy_browsers),
            active_pages=total_pages,
            network_sent_mb=sent_mb,
            network_recv_mb=recv_mb,
            queue_depth=self.task_queue.qsize()
        )
    
    async def _cleanup_loop(self):
        """Background task for periodic cleanup"""
        while self._running:
            try:
                await self._perform_cleanup()
                await asyncio.sleep(self.cleanup_interval)
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(60)
    
    async def _perform_cleanup(self):
        """Perform cleanup of idle resources"""
        now = datetime.now()
        cleanup_threshold = timedelta(minutes=10)
        
        # Clean up idle browsers
        for instance_id in list(self.available_browsers):
            instance = self.browser_pool.get(instance_id)
            if not instance:
                continue
            
            idle_time = now - instance.last_used
            if idle_time > cleanup_threshold and len(self.available_browsers) > self.min_browsers:
                await self._destroy_browser_instance(instance_id)
                logger.info(f"Cleaned up idle browser {instance_id} (idle: {idle_time})")
        
        # Clean up old tabs in busy browsers
        for instance in self.busy_browsers.values():
            if len(instance.active_pages) > 10:  # Arbitrary limit
                # Close oldest pages
                pages_to_close = instance.active_pages[:-5]  # Keep last 5
                for page in pages_to_close:
                    try:
                        await page.close()
                        instance.active_pages.remove(page)
                        self.cost_metrics["total_pages_closed"] += 1
                    except:
                        pass
    
    async def _emergency_cleanup(self):
        """Emergency cleanup when memory pressure is high"""
        logger.warning("Performing emergency cleanup due to high memory usage")
        
        # Close all pages in available browsers
        for instance_id in list(self.available_browsers):
            instance = self.browser_pool.get(instance_id)
            if instance:
                for page in instance.active_pages[:]:
                    try:
                        await page.close()
                        instance.active_pages.remove(page)
                        self.cost_metrics["total_pages_closed"] += 1
                    except:
                        pass
        
        # Destroy excess browsers
        while len(self.available_browsers) > self.min_browsers:
            instance_id = self.available_browsers.pop()
            await self._destroy_browser_instance(instance_id)
    
    async def _scaling_loop(self):
        """Background task for dynamic scaling"""
        while self._running:
            try:
                await self._adjust_pool_size()
                await asyncio.sleep(self.scale_check_interval)
            except Exception as e:
                logger.error(f"Scaling error: {e}")
                await asyncio.sleep(60)
    
    async def _adjust_pool_size(self):
        """Dynamically adjust browser pool size based on load"""
        queue_depth = self.task_queue.qsize()
        available = len(self.available_browsers)
        busy = len(self.busy_browsers)
        total = len(self.browser_pool)
        
        # Scale up conditions
        if queue_depth > available * 3 and total < self.max_browsers:
            scale_up = min(3, self.max_browsers - total)
            for _ in range(scale_up):
                await self._create_browser_instance()
            logger.info(f"Auto-scaled up by {scale_up} browsers")
        
        # Scale down conditions
        elif available > self.min_browsers * 2 and queue_depth == 0:
            # Remove excess idle browsers
            excess = available - self.min_browsers
            for _ in range(min(excess, 2)):  # Remove up to 2 at a time
                if self.available_browsers:
                    instance_id = self.available_browsers.pop()
                    await self._destroy_browser_instance(instance_id)
            logger.info(f"Auto-scaled down, removed {min(excess, 2)} browsers")
    
    async def _check_browser_health(self, instance: BrowserInstance) -> bool:
        """Check if a browser instance is healthy"""
        try:
            # Try to create a new page as health check
            page = await instance.context.new_page()
            await page.close()
            return True
        except:
            instance.is_healthy = False
            return False
    
    def estimate_resources(self, task_type: str) -> ResourceEstimate:
        """Estimate resource requirements for a task type"""
        if task_type in self.task_patterns:
            return self.task_patterns[task_type]
        
        # Default estimate for unknown task types
        return ResourceEstimate(
            task_type=task_type,
            complexity=TaskComplexity.MEDIUM,
            estimated_memory_mb=300,
            estimated_duration_seconds=30,
            estimated_network_mb=10,
            browser_instances_needed=1,
            max_concurrent_tabs=5
        )
    
    def update_task_pattern(self, 
                           task_type: str, 
                           actual_memory_mb: float,
                           actual_duration_seconds: float,
                           actual_network_mb: float):
        """Update task patterns based on actual usage (for learning)"""
        if task_type not in self.task_patterns:
            return
        
        pattern = self.task_patterns[task_type]
        
        # Simple exponential moving average update
        alpha = 0.3  # Learning rate
        pattern.estimated_memory_mb = (
            alpha * actual_memory_mb + 
            (1 - alpha) * pattern.estimated_memory_mb
        )
        pattern.estimated_duration_seconds = (
            alpha * actual_duration_seconds + 
            (1 - alpha) * pattern.estimated_duration_seconds
        )
        pattern.estimated_network_mb = (
            alpha * actual_network_mb + 
            (1 - alpha) * pattern.estimated_network_mb
        )
        
        logger.debug(f"Updated pattern for {task_type}: memory={pattern.estimated_memory_mb:.1f}MB")
    
    def _update_cost_metrics(self, usage: ResourceUsage):
        """Update cost metrics based on current usage"""
        # Simplified cost calculation (in USD)
        # In production, this would use actual cloud provider pricing
        
        # Browser instance cost: $0.05 per hour per instance
        browser_cost = len(self.browser_pool) * 0.05 * (5 / 3600)  # 5 second interval
        
        # Memory cost: $0.01 per GB-hour
        memory_gb = usage.browser_memory_mb / 1024
        memory_cost = memory_gb * 0.01 * (5 / 3600)
        
        # Network cost: $0.09 per GB
        network_cost = (usage.network_sent_mb + usage.network_recv_mb) / 1024 * 0.09
        
        total_cost = browser_cost + memory_cost + network_cost
        self.cost_metrics["total_estimated_cost_usd"] += total_cost
    
    def get_cost_report(self) -> Dict[str, Any]:
        """Generate a cost optimization report"""
        current_usage = self.resource_history[-1] if self.resource_history else None
        
        return {
            "timestamp": datetime.now().isoformat(),
            "pool_size": len(self.browser_pool),
            "available_browsers": len(self.available_browsers),
            "busy_browsers": len(self.busy_browsers),
            "queue_depth": self.task_queue.qsize(),
            "current_usage": {
                "memory_mb": current_usage.browser_memory_mb if current_usage else 0,
                "cpu_percent": current_usage.cpu_percent if current_usage else 0,
                "active_pages": current_usage.active_pages if current_usage else 0
            },
            "cost_metrics": self.cost_metrics,
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on usage patterns"""
        recommendations = []
        
        if not self.resource_history:
            return recommendations
        
        recent_usage = list(self.resource_history)[-10:]  # Last 10 samples
        
        avg_memory = sum(u.browser_memory_mb for u in recent_usage) / len(recent_usage)
        avg_cpu = sum(u.cpu_percent for u in recent_usage) / len(recent_usage)
        
        if avg_memory > self.memory_threshold_mb * 0.7:
            recommendations.append(
                f"High memory usage ({avg_memory:.0f}MB). "
                f"Consider reducing max_browsers or increasing cleanup frequency."
            )
        
        if avg_cpu > 80:
            recommendations.append(
                f"High CPU usage ({avg_cpu:.1f}%). "
                f"Consider reducing concurrent tasks or optimizing scripts."
            )
        
        if len(self.available_browsers) > self.min_browsers * 2:
            recommendations.append(
                "Many idle browsers. Consider reducing min_browsers to save costs."
            )
        
        return recommendations


# Singleton instance for global access
_cost_optimizer: Optional[CostOptimizer] = None


async def get_cost_optimizer() -> CostOptimizer:
    """Get or create the global cost optimizer instance"""
    global _cost_optimizer
    if _cost_optimizer is None:
        _cost_optimizer = CostOptimizer()
        await _cost_optimizer.start()
    return _cost_optimizer


async def shutdown_cost_optimizer():
    """Shutdown the global cost optimizer"""
    global _cost_optimizer
    if _cost_optimizer:
        await _cost_optimizer.stop()
        _cost_optimizer = None


# Integration with existing ActorPage
class OptimizedActorPage(ActorPage):
    """Extended ActorPage with cost optimization"""
    
    def __init__(self, page: Page, cost_optimizer: CostOptimizer, instance_id: str):
        super().__init__(page)
        self.cost_optimizer = cost_optimizer
        self.instance_id = instance_id
        self.task_type = "unknown"
        self.start_time = time.time()
    
    async def close(self):
        """Override close to track resource usage"""
        duration = time.time() - self.start_time
        
        # Update cost optimizer with actual usage
        self.cost_optimizer.update_task_pattern(
            task_type=self.task_type,
            actual_memory_mb=50,  # Estimated
            actual_duration_seconds=duration,
            actual_network_mb=10  # Estimated
        )
        
        # Release browser back to pool
        await self.cost_optimizer.release_browser(self.instance_id)
        
        await super().close()


# Integration with AgentService
class CostAwareAgentService(AgentService):
    """AgentService with cost optimization"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cost_optimizer = None
    
    async def initialize(self):
        """Initialize with cost optimizer"""
        await super().initialize()
        self.cost_optimizer = await get_cost_optimizer()
    
    async def execute_task(self, task_type: str, *args, **kwargs):
        """Execute task with optimized resources"""
        # Get optimized browser
        instance_id, context = await self.cost_optimizer.get_browser_for_task(task_type)
        
        try:
            # Create optimized page
            page = await context.new_page()
            actor_page = OptimizedActorPage(page, self.cost_optimizer, instance_id)
            actor_page.task_type = task_type
            
            # Execute the task
            result = await super().execute_task(task_type, actor_page, *args, **kwargs)
            
            return result
            
        finally:
            # Ensure browser is released
            await self.cost_optimizer.release_browser(instance_id)