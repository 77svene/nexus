"""
nexus/resources/memory_manager.py

Cost-Optimized Resource Management for Browser Automation
Dynamically optimizes browser instances, memory usage, and network bandwidth based on task requirements.
"""

import asyncio
import time
import psutil
import gc
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class TaskComplexity(Enum):
    """Task complexity levels for resource estimation"""
    SIMPLE = "simple"  # Basic navigation, form filling
    MODERATE = "moderate"  # Multi-page flows, moderate JS
    COMPLEX = "complex"  # Heavy JS, multiple tabs, data extraction
    EXTREME = "extreme"  # Long-running, memory-intensive tasks


@dataclass
class ResourceEstimate:
    """Predicted resource requirements for a task"""
    memory_mb: float
    cpu_percent: float
    network_bandwidth_kbps: float
    estimated_duration_seconds: float
    max_concurrent_tabs: int
    priority: int = 1  # 1-10 scale


@dataclass
class BrowserInstance:
    """Represents a managed browser instance with resource tracking"""
    id: str
    browser: Any  # Playwright browser instance
    context: Any  # Playwright browser context
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    network_usage_kbps: float = 0.0
    active_tabs: int = 0
    total_tabs_created: int = 0
    last_activity: float = field(default_factory=time.time)
    task_type: Optional[str] = None
    is_busy: bool = False
    created_at: float = field(default_factory=time.time)
    
    def update_metrics(self):
        """Update resource usage metrics for this instance"""
        try:
            # Get memory usage of browser process
            if hasattr(self.browser, 'process') and self.browser.process:
                process = psutil.Process(self.browser.process.pid)
                self.memory_usage_mb = process.memory_info().rss / 1024 / 1024
                self.cpu_usage_percent = process.cpu_percent()
        except (psutil.NoSuchProcess, AttributeError):
            # Process might have been terminated
            self.memory_usage_mb = 0.0
            self.cpu_usage_percent = 0.0


@dataclass
class ResourceThresholds:
    """Configurable thresholds for resource management"""
    max_memory_mb_per_instance: float = 2048.0  # 2GB per browser instance
    max_total_memory_mb: float = 8192.0  # 8GB total system memory limit
    max_cpu_percent_per_instance: float = 50.0
    max_total_cpu_percent: float = 80.0
    max_concurrent_instances: int = 10
    min_instances: int = 1
    max_tabs_per_instance: int = 20
    memory_cleanup_threshold_mb: float = 1500.0  # Trigger cleanup above this
    idle_timeout_seconds: float = 300.0  # 5 minutes
    scale_up_queue_depth: int = 5  # Scale up when queue depth exceeds this
    scale_down_idle_instances: int = 2  # Keep at least this many idle instances


class MemoryManager:
    """
    Cost-Optimized Resource Manager for browser automation.
    
    Features:
    1. Browser instance pooling with automatic scaling
    2. Memory usage monitoring with automatic tab cleanup
    3. Cost estimation API for different task types
    4. Resource-aware task scheduling
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern for global resource management"""
        if cls._instance is None:
            cls._instance = super(MemoryManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the memory manager with default configuration"""
        if self._initialized:
            return
            
        self._initialized = True
        self.thresholds = ResourceThresholds()
        
        # Instance pools
        self.available_instances: asyncio.Queue = asyncio.Queue()
        self.busy_instances: Dict[str, BrowserInstance] = {}
        self.all_instances: Dict[str, BrowserInstance] = {}
        
        # Task queues by priority
        self.task_queues: Dict[int, asyncio.Queue] = defaultdict(asyncio.Queue)
        self.pending_tasks: Dict[str, Dict] = {}
        
        # Resource tracking
        self.total_memory_usage_mb: float = 0.0
        self.total_cpu_usage_percent: float = 0.0
        self.total_network_usage_kbps: float = 0.0
        
        # Cost estimation models (simplified - could be ML-based)
        self.task_profiles: Dict[str, ResourceEstimate] = {
            "navigation": ResourceEstimate(
                memory_mb=100, cpu_percent=5, network_bandwidth_kbps=50,
                estimated_duration_seconds=2, max_concurrent_tabs=1
            ),
            "form_filling": ResourceEstimate(
                memory_mb=150, cpu_percent=10, network_bandwidth_kbps=30,
                estimated_duration_seconds=5, max_concurrent_tabs=1
            ),
            "data_extraction": ResourceEstimate(
                memory_mb=300, cpu_percent=25, network_bandwidth_kbps=100,
                estimated_duration_seconds=10, max_concurrent_tabs=3
            ),
            "multi_tab_automation": ResourceEstimate(
                memory_mb=500, cpu_percent=40, network_bandwidth_kbps=200,
                estimated_duration_seconds=30, max_concurrent_tabs=10
            ),
            "heavy_js_execution": ResourceEstimate(
                memory_mb=800, cpu_percent=60, network_bandwidth_kbps=150,
                estimated_duration_seconds=20, max_concurrent_tabs=5
            ),
        }
        
        # Monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._scaling_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Statistics
        self.stats = {
            "total_instances_created": 0,
            "total_instances_destroyed": 0,
            "total_tasks_completed": 0,
            "total_memory_saved_mb": 0.0,
            "total_cost_savings_percent": 0.0,
            "average_utilization_percent": 0.0,
        }
        
        logger.info("MemoryManager initialized with singleton pattern")
    
    async def start(self):
        """Start background monitoring and scaling tasks"""
        if self._running:
            return
            
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitor_resources())
        self._scaling_task = asyncio.create_task(self._auto_scale_instances())
        self._cleanup_task = asyncio.create_task(self._cleanup_idle_resources())
        logger.info("MemoryManager background tasks started")
    
    async def stop(self):
        """Stop all background tasks and cleanup"""
        self._running = False
        
        # Cancel background tasks
        for task in [self._monitoring_task, self._scaling_task, self._cleanup_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Cleanup all instances
        await self._cleanup_all_instances()
        logger.info("MemoryManager stopped")
    
    @asynccontextmanager
    async def acquire_browser(self, task_type: str = "navigation", priority: int = 1):
        """
        Context manager to acquire a browser instance with automatic resource management.
        
        Args:
            task_type: Type of task (navigation, form_filling, etc.)
            priority: Task priority (1-10, higher is more important)
            
        Yields:
            BrowserInstance: Managed browser instance
        """
        instance = await self._get_or_create_instance(task_type, priority)
        instance.is_busy = True
        instance.last_activity = time.time()
        
        try:
            yield instance
        finally:
            await self._release_instance(instance)
    
    async def _get_or_create_instance(self, task_type: str, priority: int) -> BrowserInstance:
        """
        Get an available instance or create a new one if needed.
        Implements intelligent pooling and scaling.
        """
        # Check if we can reuse an available instance
        try:
            # Try to get from available pool with timeout
            instance = await asyncio.wait_for(
                self.available_instances.get(), 
                timeout=0.1
            )
            
            # Check if instance is still valid and has capacity
            if (instance.id in self.all_instances and 
                instance.active_tabs < self.thresholds.max_tabs_per_instance and
                not instance.is_busy):
                
                instance.task_type = task_type
                instance.is_busy = True
                self.busy_instances[instance.id] = instance
                logger.debug(f"Reusing instance {instance.id} for {task_type}")
                return instance
            else:
                # Instance invalid or at capacity, create new one
                await self._destroy_instance(instance)
                
        except (asyncio.TimeoutError, asyncio.QueueEmpty):
            pass
        
        # Check resource limits before creating new instance
        if len(self.all_instances) >= self.thresholds.max_concurrent_instances:
            # Wait for an available instance
            logger.warning("Max instances reached, waiting for available instance")
            instance = await self.available_instances.get()
            instance.task_type = task_type
            instance.is_busy = True
            self.busy_instances[instance.id] = instance
            return instance
        
        # Create new instance
        instance = await self._create_instance(task_type)
        self.busy_instances[instance.id] = instance
        return instance
    
    async def _create_instance(self, task_type: str) -> BrowserInstance:
        """Create a new browser instance with optimized settings"""
        from playwright.async_api import async_playwright
        
        # Import here to avoid circular imports
        playwright = await async_playwright().start()
        
        # Launch browser with memory-optimized settings
        browser = await playwright.chromium.launch(
            headless=True,
            args=[
                '--disable-dev-shm-usage',
                '--disable-gpu',
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-accelerated-2d-canvas',
                '--disable-accelerated-jpeg-decoding',
                '--disable-accelerated-mjpeg-decode',
                '--disable-accelerated-video-decode',
                '--disable-gpu-compositing',
                '--disable-gpu-rasterization',
                '--disable-gpu-sandbox',
                '--js-flags=--max-old-space-size=512',  # Limit JS heap
            ]
        )
        
        # Create context with optimized settings
        context = await browser.new_context(
            viewport={'width': 1280, 'height': 720},
            bypass_csp=True,
            ignore_https_errors=True,
            java_script_enabled=True,
        )
        
        # Block unnecessary resources to save bandwidth
        await context.route("**/*.{png,jpg,jpeg,gif,svg,ico,woff,woff2,ttf,eot}", 
                          lambda route: route.abort())
        
        instance_id = f"browser_{int(time.time() * 1000)}_{len(self.all_instances)}"
        instance = BrowserInstance(
            id=instance_id,
            browser=browser,
            context=context,
            task_type=task_type
        )
        
        self.all_instances[instance_id] = instance
        self.stats["total_instances_created"] += 1
        
        logger.info(f"Created new browser instance {instance_id} for {task_type}")
        return instance
    
    async def _release_instance(self, instance: BrowserInstance):
        """Release an instance back to the pool"""
        instance.is_busy = False
        instance.last_activity = time.time()
        
        # Remove from busy instances
        if instance.id in self.busy_instances:
            del self.busy_instances[instance.id]
        
        # Check if instance should be kept or destroyed
        if (len(self.all_instances) > self.thresholds.min_instances and
            instance.memory_usage_mb > self.thresholds.memory_cleanup_threshold_mb):
            # Instance using too much memory, destroy it
            await self._destroy_instance(instance)
        else:
            # Return to available pool
            await self.available_instances.put(instance)
            logger.debug(f"Released instance {instance.id} back to pool")
    
    async def _destroy_instance(self, instance: BrowserInstance):
        """Safely destroy a browser instance and free resources"""
        try:
            if instance.context:
                await instance.context.close()
            if instance.browser:
                await instance.browser.close()
        except Exception as e:
            logger.error(f"Error closing browser instance {instance.id}: {e}")
        
        # Remove from tracking
        if instance.id in self.all_instances:
            del self.all_instances[instance.id]
        if instance.id in self.busy_instances:
            del self.busy_instances[instance.id]
        
        self.stats["total_instances_destroyed"] += 1
        logger.info(f"Destroyed browser instance {instance.id}")
    
    async def _monitor_resources(self):
        """Background task to monitor resource usage"""
        while self._running:
            try:
                total_memory = 0.0
                total_cpu = 0.0
                
                for instance in list(self.all_instances.values()):
                    instance.update_metrics()
                    total_memory += instance.memory_usage_mb
                    total_cpu += instance.cpu_usage_percent
                    
                    # Auto-cleanup tabs if memory usage is high
                    if instance.memory_usage_mb > self.thresholds.memory_cleanup_threshold_mb:
                        await self._cleanup_tabs(instance)
                
                self.total_memory_usage_mb = total_memory
                self.total_cpu_usage_percent = total_cpu
                
                # Update statistics
                total_instances = len(self.all_instances)
                if total_instances > 0:
                    self.stats["average_utilization_percent"] = (
                        len(self.busy_instances) / total_instances * 100
                    )
                
                # Log warnings if thresholds exceeded
                if total_memory > self.thresholds.max_total_memory_mb:
                    logger.warning(f"Total memory usage {total_memory:.1f}MB exceeds limit")
                
                await asyncio.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                await asyncio.sleep(10)
    
    async def _auto_scale_instances(self):
        """Automatically scale browser instances based on queue depth"""
        while self._running:
            try:
                # Calculate total pending tasks
                total_pending = sum(q.qsize() for q in self.task_queues.values())
                available_count = self.available_instances.qsize()
                
                # Scale up if queue depth is high
                if (total_pending > self.thresholds.scale_up_queue_depth and
                    len(self.all_instances) < self.thresholds.max_concurrent_instances):
                    
                    instances_to_create = min(
                        total_pending - self.thresholds.scale_up_queue_depth,
                        self.thresholds.max_concurrent_instances - len(self.all_instances)
                    )
                    
                    for _ in range(instances_to_create):
                        # Create with default task type
                        instance = await self._create_instance("navigation")
                        await self.available_instances.put(instance)
                    
                    logger.info(f"Scaled up: created {instances_to_create} instances")
                
                # Scale down if too many idle instances
                elif (available_count > self.thresholds.scale_down_idle_instances and
                      len(self.all_instances) > self.thresholds.min_instances):
                    
                    instances_to_remove = min(
                        available_count - self.thresholds.scale_down_idle_instances,
                        len(self.all_instances) - self.thresholds.min_instances
                    )
                    
                    for _ in range(instances_to_remove):
                        try:
                            instance = await asyncio.wait_for(
                                self.available_instances.get(), timeout=0.1
                            )
                            await self._destroy_instance(instance)
                        except asyncio.TimeoutError:
                            break
                    
                    if instances_to_remove > 0:
                        logger.info(f"Scaled down: removed {instances_to_remove} instances")
                
                await asyncio.sleep(10)  # Check scaling every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in auto-scaling: {e}")
                await asyncio.sleep(15)
    
    async def _cleanup_idle_resources(self):
        """Clean up idle browser instances and tabs"""
        while self._running:
            try:
                current_time = time.time()
                instances_to_destroy = []
                
                for instance in list(self.all_instances.values()):
                    # Destroy idle instances that have timed out
                    idle_time = current_time - instance.last_activity
                    if (not instance.is_busy and 
                        idle_time > self.thresholds.idle_timeout_seconds and
                        len(self.all_instances) > self.thresholds.min_instances):
                        
                        instances_to_destroy.append(instance)
                
                # Destroy idle instances
                for instance in instances_to_destroy:
                    if instance.id in self.all_instances:
                        await self._destroy_instance(instance)
                
                # Cleanup tabs in busy instances with high memory
                for instance in list(self.busy_instances.values()):
                    if instance.memory_usage_mb > self.thresholds.memory_cleanup_threshold_mb:
                        await self._cleanup_tabs(instance)
                
                # Force garbage collection
                gc.collect()
                
                await asyncio.sleep(30)  # Cleanup every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in cleanup: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_tabs(self, instance: BrowserInstance):
        """Clean up excess tabs in a browser instance to free memory"""
        try:
            if not instance.context:
                return
            
            pages = instance.context.pages
            if len(pages) > 2:  # Keep at least 2 tabs (main + one spare)
                # Close oldest tabs first
                pages_to_close = pages[:-2]  # Keep last 2 tabs
                for page in pages_to_close:
                    try:
                        await page.close()
                        instance.active_tabs = max(0, instance.active_tabs - 1)
                    except Exception:
                        pass
                
                logger.debug(f"Cleaned up {len(pages_to_close)} tabs in instance {instance.id}")
                
                # Update memory estimate
                memory_saved = len(pages_to_close) * 50  # Estimate 50MB per tab
                self.stats["total_memory_saved_mb"] += memory_saved
                
        except Exception as e:
            logger.error(f"Error cleaning up tabs for instance {instance.id}: {e}")
    
    async def _cleanup_all_instances(self):
        """Clean up all browser instances"""
        instances_to_destroy = list(self.all_instances.values())
        for instance in instances_to_destroy:
            await self._destroy_instance(instance)
        
        # Clear queues
        while not self.available_instances.empty():
            try:
                self.available_instances.get_nowait()
            except asyncio.QueueEmpty:
                break
    
    def estimate_resources(self, task_type: str, 
                          task_complexity: TaskComplexity = TaskComplexity.MODERATE,
                          estimated_pages: int = 1) -> ResourceEstimate:
        """
        Estimate resource requirements for a given task.
        
        Args:
            task_type: Type of task (navigation, form_filling, etc.)
            task_complexity: Complexity level of the task
            estimated_pages: Estimated number of pages to process
            
        Returns:
            ResourceEstimate with predicted resource needs
        """
        base_estimate = self.task_profiles.get(
            task_type, 
            self.task_profiles["navigation"]
        )
        
        # Adjust based on complexity
        complexity_multipliers = {
            TaskComplexity.SIMPLE: 0.7,
            TaskComplexity.MODERATE: 1.0,
            TaskComplexity.COMPLEX: 1.5,
            TaskComplexity.EXTREME: 2.5,
        }
        
        multiplier = complexity_multipliers.get(task_complexity, 1.0)
        
        # Adjust for multiple pages
        page_multiplier = max(1.0, estimated_pages * 0.8)
        
        return ResourceEstimate(
            memory_mb=base_estimate.memory_mb * multiplier * page_multiplier,
            cpu_percent=base_estimate.cpu_percent * multiplier,
            network_bandwidth_kbps=base_estimate.network_bandwidth_kbps * page_multiplier,
            estimated_duration_seconds=base_estimate.estimated_duration_seconds * page_multiplier,
            max_concurrent_tabs=min(
                base_estimate.max_concurrent_tabs * estimated_pages,
                self.thresholds.max_tabs_per_instance
            ),
            priority=base_estimate.priority
        )
    
    def get_cost_savings(self) -> Dict[str, Any]:
        """
        Calculate cost savings from optimized resource management.
        
        Returns:
            Dictionary with cost savings metrics
        """
        # Calculate theoretical costs without optimization
        theoretical_memory = len(self.all_instances) * self.thresholds.max_memory_mb_per_instance
        theoretical_cpu = len(self.all_instances) * 100  # Assume 100% CPU without optimization
        
        # Calculate actual costs
        actual_memory = self.total_memory_usage_mb
        actual_cpu = self.total_cpu_usage_percent
        
        # Calculate savings
        memory_savings_percent = (
            (theoretical_memory - actual_memory) / theoretical_memory * 100
            if theoretical_memory > 0 else 0
        )
        
        cpu_savings_percent = (
            (theoretical_cpu - actual_cpu) / theoretical_cpu * 100
            if theoretical_cpu > 0 else 0
        )
        
        self.stats["total_cost_savings_percent"] = (memory_savings_percent + cpu_savings_percent) / 2
        
        return {
            "memory_savings_percent": memory_savings_percent,
            "cpu_savings_percent": cpu_savings_percent,
            "total_cost_savings_percent": self.stats["total_cost_savings_percent"],
            "memory_saved_mb": theoretical_memory - actual_memory,
            "instances_optimized": len(self.all_instances),
            "total_tasks_completed": self.stats["total_tasks_completed"],
        }
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status and statistics"""
        return {
            "total_instances": len(self.all_instances),
            "busy_instances": len(self.busy_instances),
            "available_instances": self.available_instances.qsize(),
            "total_memory_usage_mb": self.total_memory_usage_mb,
            "total_cpu_usage_percent": self.total_cpu_usage_percent,
            "memory_utilization_percent": (
                self.total_memory_usage_mb / self.thresholds.max_total_memory_mb * 100
                if self.thresholds.max_total_memory_mb > 0 else 0
            ),
            "stats": self.stats.copy(),
            "thresholds": {
                "max_memory_mb_per_instance": self.thresholds.max_memory_mb_per_instance,
                "max_total_memory_mb": self.thresholds.max_total_memory_mb,
                "max_concurrent_instances": self.thresholds.max_concurrent_instances,
                "max_tabs_per_instance": self.thresholds.max_tabs_per_instance,
            }
        }
    
    async def optimize_for_task(self, task_type: str, 
                               estimated_duration: float = None) -> Dict[str, Any]:
        """
        Optimize resource allocation for a specific task type.
        
        Args:
            task_type: Type of task to optimize for
            estimated_duration: Estimated duration in seconds
            
        Returns:
            Optimization recommendations
        """
        estimate = self.estimate_resources(task_type)
        
        # Check current resource availability
        available_memory = self.thresholds.max_total_memory_mb - self.total_memory_usage_mb
        available_cpu = 100 - self.total_cpu_usage_percent
        
        recommendations = {
            "task_type": task_type,
            "estimated_resources": estimate,
            "current_availability": {
                "memory_mb": available_memory,
                "cpu_percent": available_cpu,
                "instances": self.available_instances.qsize(),
            },
            "recommendations": [],
            "warnings": [],
        }
        
        # Generate recommendations
        if estimate.memory_mb > available_memory:
            recommendations["warnings"].append(
                f"Insufficient memory: need {estimate.memory_mb:.1f}MB, "
                f"available {available_memory:.1f}MB"
            )
            recommendations["recommendations"].append(
                "Consider scaling down other tasks or increasing memory limits"
            )
        
        if estimate.cpu_percent > available_cpu:
            recommendations["warnings"].append(
                f"High CPU demand: need {estimate.cpu_percent:.1f}%, "
                f"available {available_cpu:.1f}%"
            )
        
        if estimate.max_concurrent_tabs > self.thresholds.max_tabs_per_instance:
            recommendations["recommendations"].append(
                f"Task requires {estimate.max_concurrent_tabs} tabs, "
                f"but max per instance is {self.thresholds.max_tabs_per_instance}. "
                f"Consider using multiple instances."
            )
        
        return recommendations


# Global instance for easy import
memory_manager = MemoryManager()


# Integration functions for existing codebase
async def get_managed_browser(task_type: str = "navigation", priority: int = 1):
    """
    Integration function for existing codebase.
    Returns a managed browser instance with automatic resource optimization.
    """
    return memory_manager.acquire_browser(task_type, priority)


def estimate_task_resources(task_type: str, complexity: str = "moderate", 
                          pages: int = 1) -> Dict[str, Any]:
    """
    Integration function for cost estimation.
    
    Args:
        task_type: Type of task
        complexity: simple, moderate, complex, or extreme
        pages: Estimated number of pages
        
    Returns:
        Resource estimate dictionary
    """
    complexity_map = {
        "simple": TaskComplexity.SIMPLE,
        "moderate": TaskComplexity.MODERATE,
        "complex": TaskComplexity.COMPLEX,
        "extreme": TaskComplexity.EXTREME,
    }
    
    estimate = memory_manager.estimate_resources(
        task_type,
        complexity_map.get(complexity, TaskComplexity.MODERATE),
        pages
    )
    
    return {
        "memory_mb": estimate.memory_mb,
        "cpu_percent": estimate.cpu_percent,
        "network_kbps": estimate.network_bandwidth_kbps,
        "duration_seconds": estimate.estimated_duration_seconds,
        "max_tabs": estimate.max_concurrent_tabs,
        "priority": estimate.priority,
    }


# Cleanup on module exit
import atexit

def _cleanup():
    """Cleanup function for module exit"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(memory_manager.stop())
        else:
            asyncio.run(memory_manager.stop())
    except Exception:
        pass

atexit.register(_cleanup)