"""
Parallel Execution Engine — Enable concurrent browser automation with intelligent task parallelization, shared browser contexts, and resource-aware scheduling. Dramatically improves throughput for batch operations.
"""

import asyncio
import logging
import time
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict
import psutil
import uuid

from playwright.async_api import BrowserContext, Page

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels for scheduling."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class TaskState(Enum):
    """Task execution states."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    WAITING_DEPENDENCY = "waiting_dependency"


@dataclass
class ResourceLimits:
    """Resource constraints for execution."""
    max_concurrent_pages: int = 10
    max_concurrent_contexts: int = 5
    max_memory_mb: int = 2048
    max_cpu_percent: float = 80.0
    max_tasks_per_context: int = 20
    context_timeout_seconds: int = 300
    page_timeout_seconds: int = 60


@dataclass
class TaskDependency:
    """Task dependency definition."""
    task_id: str
    required_state: TaskState = TaskState.COMPLETED
    timeout_seconds: int = 300


@dataclass
class BrowserTask:
    """Executable browser automation task."""
    task_id: str
    func: Callable
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    dependencies: List[TaskDependency] = field(default_factory=list)
    context_id: Optional[str] = None
    state: TaskState = TaskState.PENDING
    result: Any = None
    error: Optional[Exception] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 2
    resource_requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BrowserContextPool:
    """Pool of shared browser contexts."""
    contexts: Dict[str, BrowserContext] = field(default_factory=dict)
    context_usage: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    context_locks: Dict[str, asyncio.Lock] = field(default_factory=dict)
    available_contexts: asyncio.Queue = field(default_factory=asyncio.Queue)
    
    async def acquire_context(self, task: BrowserTask) -> Tuple[str, BrowserContext]:
        """Acquire a browser context for task execution."""
        if task.context_id and task.context_id in self.contexts:
            # Use specified context
            context_id = task.context_id
        else:
            # Get least used available context
            context_id = await self.available_contexts.get()
        
        async with self.context_locks[context_id]:
            self.context_usage[context_id] += 1
        
        return context_id, self.contexts[context_id]
    
    async def release_context(self, context_id: str):
        """Release a browser context back to pool."""
        async with self.context_locks[context_id]:
            self.context_usage[context_id] -= 1
            if self.context_usage[context_id] <= 0:
                self.context_usage[context_id] = 0
                await self.available_contexts.put(context_id)


class DependencyGraph:
    """Analyzes and manages task dependencies for parallelization."""
    
    def __init__(self):
        self.graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_graph: Dict[str, Set[str]] = defaultdict(set)
        self.task_map: Dict[str, BrowserTask] = {}
    
    def add_task(self, task: BrowserTask):
        """Add task to dependency graph."""
        self.task_map[task.task_id] = task
        for dep in task.dependencies:
            self.graph[task.task_id].add(dep.task_id)
            self.reverse_graph[dep.task_id].add(task.task_id)
    
    def get_ready_tasks(self, completed_task_ids: Set[str]) -> List[BrowserTask]:
        """Get tasks whose dependencies are satisfied."""
        ready_tasks = []
        for task_id, task in self.task_map.items():
            if task.state != TaskState.PENDING:
                continue
            
            dependencies_met = True
            for dep in task.dependencies:
                if dep.task_id not in completed_task_ids:
                    dependencies_met = False
                    break
            
            if dependencies_met:
                ready_tasks.append(task)
        
        # Sort by priority
        ready_tasks.sort(key=lambda t: t.priority.value)
        return ready_tasks
    
    def get_dependent_tasks(self, task_id: str) -> List[BrowserTask]:
        """Get tasks that depend on the given task."""
        dependent_ids = self.reverse_graph.get(task_id, set())
        return [self.task_map[tid] for tid in dependent_ids if tid in self.task_map]


class ResourceMonitor:
    """Monitors system resources to prevent overload."""
    
    def __init__(self, limits: ResourceLimits):
        self.limits = limits
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._resource_warnings: List[str] = []
    
    async def start_monitoring(self):
        """Start resource monitoring."""
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
    
    async def stop_monitoring(self):
        """Stop resource monitoring."""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
    
    async def _monitor_loop(self):
        """Resource monitoring loop."""
        while self._monitoring:
            try:
                # Check memory usage
                memory = psutil.virtual_memory()
                memory_mb = memory.used / 1024 / 1024
                if memory_mb > self.limits.max_memory_mb:
                    warning = f"Memory usage {memory_mb:.0f}MB exceeds limit {self.limits.max_memory_mb}MB"
                    if warning not in self._resource_warnings:
                        self._resource_warnings.append(warning)
                        logger.warning(warning)
                
                # Check CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                if cpu_percent > self.limits.max_cpu_percent:
                    warning = f"CPU usage {cpu_percent:.1f}% exceeds limit {self.limits.max_cpu_percent}%"
                    if warning not in self._resource_warnings:
                        self._resource_warnings.append(warning)
                        logger.warning(warning)
                
                await asyncio.sleep(5)  # Check every 5 seconds
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(10)
    
    def can_accept_task(self, current_pages: int, current_contexts: int) -> bool:
        """Check if system can accept another task."""
        # Check page limits
        if current_pages >= self.limits.max_concurrent_pages:
            return False
        
        # Check context limits
        if current_contexts >= self.limits.max_concurrent_contexts:
            return False
        
        # Check system resources
        memory = psutil.virtual_memory()
        memory_mb = memory.used / 1024 / 1024
        if memory_mb > self.limits.max_memory_mb * 0.9:  # 90% threshold
            return False
        
        cpu_percent = psutil.cpu_percent(interval=0.1)
        if cpu_percent > self.limits.max_cpu_percent * 0.9:  # 90% threshold
            return False
        
        return True


class WorkerPool:
    """Manages worker processes/threads for task execution."""
    
    def __init__(self, 
                 max_workers: int = 4,
                 use_processes: bool = False,
                 resource_limits: ResourceLimits = None):
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.resource_limits = resource_limits or ResourceLimits()
        self._executor: Optional[Union[ProcessPoolExecutor, ThreadPoolExecutor]] = None
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._task_queue: asyncio.PriorityQueue = None
        self._shutdown = False
    
    async def initialize(self):
        """Initialize worker pool."""
        if self.use_processes:
            self._executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Priority queue: (priority, timestamp, task_id)
        self._task_queue = asyncio.PriorityQueue()
    
    async def submit_task(self, task: BrowserTask, context: BrowserContext) -> str:
        """Submit a task for execution."""
        if self._shutdown:
            raise RuntimeError("Worker pool is shutting down")
        
        # Create async task
        task_id = task.task_id
        async_task = asyncio.create_task(
            self._execute_task(task, context)
        )
        self._active_tasks[task_id] = async_task
        
        # Add to priority queue
        await self._task_queue.put((
            task.priority.value,
            task.created_at,
            task_id
        ))
        
        return task_id
    
    async def _execute_task(self, task: BrowserTask, context: BrowserContext):
        """Execute a single task."""
        try:
            task.state = TaskState.RUNNING
            task.started_at = time.time()
            
            # Create a new page in the context
            page = await context.new_page()
            
            try:
                # Execute task function
                if asyncio.iscoroutinefunction(task.func):
                    result = await task.func(page, *task.args, **task.kwargs)
                else:
                    # Run sync function in thread pool
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        self._executor,
                        lambda: task.func(page, *task.args, **task.kwargs)
                    )
                
                task.result = result
                task.state = TaskState.COMPLETED
                task.completed_at = time.time()
                
            finally:
                # Always close the page
                await page.close()
                
        except Exception as e:
            task.error = e
            task.state = TaskState.FAILED
            task.completed_at = time.time()
            logger.error(f"Task {task.task_id} failed: {e}")
    
    async def cancel_task(self, task_id: str):
        """Cancel a running task."""
        if task_id in self._active_tasks:
            self._active_tasks[task_id].cancel()
            try:
                await self._active_tasks[task_id]
            except asyncio.CancelledError:
                pass
            del self._active_tasks[task_id]
    
    async def wait_for_task(self, task_id: str, timeout: float = None) -> BrowserTask:
        """Wait for a task to complete."""
        if task_id in self._active_tasks:
            try:
                await asyncio.wait_for(self._active_tasks[task_id], timeout)
            except asyncio.TimeoutError:
                raise TimeoutError(f"Task {task_id} timed out after {timeout} seconds")
            except asyncio.CancelledError:
                pass
    
    async def shutdown(self):
        """Shutdown worker pool."""
        self._shutdown = True
        
        # Cancel all active tasks
        for task_id in list(self._active_tasks.keys()):
            await self.cancel_task(task_id)
        
        # Shutdown executor
        if self._executor:
            self._executor.shutdown(wait=True)


class ResourceManager:
    """
    Main resource manager for parallel browser automation.
    
    Features:
    - Intelligent task parallelization with dependency analysis
    - Shared browser contexts for efficient resource usage
    - Resource-aware scheduling to prevent overload
    - Support for both asyncio and process/thread-based concurrency
    - Automatic retry and error recovery
    """
    
    def __init__(self,
                 max_workers: int = 4,
                 use_processes: bool = False,
                 resource_limits: ResourceLimits = None,
                 context_factory: Callable = None):
        """
        Initialize resource manager.
        
        Args:
            max_workers: Maximum number of concurrent workers
            use_processes: Use process pool instead of thread pool
            resource_limits: Resource constraints
            context_factory: Async function to create browser contexts
        """
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.resource_limits = resource_limits or ResourceLimits()
        self.context_factory = context_factory
        
        # Core components
        self.worker_pool = WorkerPool(max_workers, use_processes, self.resource_limits)
        self.context_pool = BrowserContextPool()
        self.dependency_graph = DependencyGraph()
        self.resource_monitor = ResourceMonitor(self.resource_limits)
        
        # Task management
        self.tasks: Dict[str, BrowserTask] = {}
        self.completed_task_ids: Set[str] = set()
        self._task_counter = 0
        self._lock = asyncio.Lock()
        
        # Execution state
        self._running = False
        self._execution_complete = asyncio.Event()
        self._scheduler_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0,
            "context_reuses": 0
        }
    
    async def initialize(self, num_contexts: int = None):
        """
        Initialize resource manager and create browser contexts.
        
        Args:
            num_contexts: Number of browser contexts to create (default: max_concurrent_contexts)
        """
        if self._running:
            raise RuntimeError("ResourceManager is already running")
        
        num_contexts = num_contexts or self.resource_limits.max_concurrent_contexts
        
        # Initialize components
        await self.worker_pool.initialize()
        await self.resource_monitor.start_monitoring()
        
        # Create browser contexts
        if self.context_factory:
            for i in range(num_contexts):
                context_id = f"context_{i}"
                context = await self.context_factory()
                self.context_pool.contexts[context_id] = context
                self.context_pool.context_locks[context_id] = asyncio.Lock()
                await self.context_pool.available_contexts.put(context_id)
        
        self._running = True
        logger.info(f"ResourceManager initialized with {num_contexts} contexts")
    
    def create_task(self,
                    func: Callable,
                    args: Tuple = None,
                    kwargs: Dict = None,
                    priority: TaskPriority = TaskPriority.NORMAL,
                    dependencies: List[TaskDependency] = None,
                    context_id: str = None,
                    max_retries: int = 2,
                    resource_requirements: Dict = None) -> str:
        """
        Create a new browser automation task.
        
        Args:
            func: Async or sync function to execute
            args: Function arguments
            kwargs: Function keyword arguments
            priority: Task priority
            dependencies: Task dependencies
            context_id: Specific context to use (optional)
            max_retries: Maximum retry attempts
            resource_requirements: Resource requirements for scheduling
            
        Returns:
            Task ID
        """
        with self._lock:
            self._task_counter += 1
            task_id = f"task_{self._task_counter}_{uuid.uuid4().hex[:8]}"
        
        task = BrowserTask(
            task_id=task_id,
            func=func,
            args=args or (),
            kwargs=kwargs or {},
            priority=priority,
            dependencies=dependencies or [],
            context_id=context_id,
            max_retries=max_retries,
            resource_requirements=resource_requirements or {}
        )
        
        self.tasks[task_id] = task
        self.dependency_graph.add_task(task)
        self.stats["tasks_submitted"] += 1
        
        return task_id
    
    async def execute_all(self, timeout: float = None) -> Dict[str, BrowserTask]:
        """
        Execute all pending tasks with dependency-aware parallelization.
        
        Args:
            timeout: Maximum execution time in seconds
            
        Returns:
            Dictionary of task results
        """
        if not self._running:
            raise RuntimeError("ResourceManager not initialized")
        
        # Start scheduler
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        
        try:
            # Wait for completion or timeout
            if timeout:
                await asyncio.wait_for(self._execution_complete.wait(), timeout)
            else:
                await self._execution_complete.wait()
        except asyncio.TimeoutError:
            logger.warning(f"Execution timed out after {timeout} seconds")
            await self.cancel_all()
        finally:
            # Stop scheduler
            if self._scheduler_task:
                self._scheduler_task.cancel()
                try:
                    await self._scheduler_task
                except asyncio.CancelledError:
                    pass
        
        return self.tasks
    
    async def _scheduler_loop(self):
        """Main scheduler loop for task execution."""
        while self._running:
            try:
                # Check if all tasks are done
                all_done = all(
                    task.state in (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED)
                    for task in self.tasks.values()
                )
                
                if all_done and not any(t.state == TaskState.RUNNING for t in self.tasks.values()):
                    self._execution_complete.set()
                    break
                
                # Get ready tasks based on dependencies
                ready_tasks = self.dependency_graph.get_ready_tasks(self.completed_task_ids)
                
                # Filter tasks that can be executed based on resources
                executable_tasks = []
                current_pages = len(self.worker_pool._active_tasks)
                current_contexts = len(self.context_pool.contexts) - self.context_pool.available_contexts.qsize()
                
                for task in ready_tasks:
                    if self.resource_monitor.can_accept_task(current_pages, current_contexts):
                        executable_tasks.append(task)
                        current_pages += 1
                    else:
                        break  # Stop if resources are exhausted
                
                # Execute tasks
                for task in executable_tasks:
                    await self._execute_task(task)
                
                # Wait a bit before next scheduling cycle
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(1)
    
    async def _execute_task(self, task: BrowserTask):
        """Execute a single task."""
        try:
            # Acquire context
            context_id, context = await self.context_pool.acquire_context(task)
            
            # Submit to worker pool
            await self.worker_pool.submit_task(task, context)
            
            # Monitor task completion
            asyncio.create_task(self._monitor_task(task, context_id))
            
            # Update stats
            self.stats["context_reuses"] += 1
            
        except Exception as e:
            task.state = TaskState.FAILED
            task.error = e
            logger.error(f"Failed to execute task {task.task_id}: {e}")
    
    async def _monitor_task(self, task: BrowserTask, context_id: str):
        """Monitor task execution and handle completion."""
        try:
            await self.worker_pool.wait_for_task(task.task_id)
            
            # Handle retry on failure
            if task.state == TaskState.FAILED and task.retry_count < task.max_retries:
                task.retry_count += 1
                task.state = TaskState.PENDING
                logger.info(f"Retrying task {task.task_id} (attempt {task.retry_count})")
            else:
                # Mark as completed
                if task.state == TaskState.COMPLETED:
                    self.completed_task_ids.add(task.task_id)
                    self.stats["tasks_completed"] += 1
                    if task.started_at and task.completed_at:
                        self.stats["total_execution_time"] += task.completed_at - task.started_at
                elif task.state == TaskState.FAILED:
                    self.stats["tasks_failed"] += 1
                
                # Notify dependent tasks
                dependent_tasks = self.dependency_graph.get_dependent_tasks(task.task_id)
                for dep_task in dependent_tasks:
                    # Check if all dependencies are met
                    deps_met = all(
                        dep.task_id in self.completed_task_ids
                        for dep in dep_task.dependencies
                    )
                    if deps_met and dep_task.state == TaskState.WAITING_DEPENDENCY:
                        dep_task.state = TaskState.PENDING
        
        finally:
            # Release context
            await self.context_pool.release_context(context_id)
    
    async def cancel_task(self, task_id: str):
        """Cancel a specific task."""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            if task.state == TaskState.RUNNING:
                await self.worker_pool.cancel_task(task_id)
            task.state = TaskState.CANCELLED
    
    async def cancel_all(self):
        """Cancel all pending and running tasks."""
        for task_id, task in self.tasks.items():
            if task.state in (TaskState.PENDING, TaskState.RUNNING, TaskState.QUEUED):
                await self.cancel_task(task_id)
    
    async def get_task_result(self, task_id: str, timeout: float = None) -> Any:
        """
        Get result of a specific task.
        
        Args:
            task_id: Task identifier
            timeout: Wait timeout in seconds
            
        Returns:
            Task result
            
        Raises:
            TimeoutError: If task doesn't complete within timeout
            Exception: If task failed
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task = self.tasks[task_id]
        
        # Wait for task completion
        start_time = time.time()
        while task.state not in (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED):
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Task {task_id} timed out")
            await asyncio.sleep(0.1)
        
        if task.state == TaskState.FAILED:
            raise task.error
        elif task.state == TaskState.CANCELLED:
            raise asyncio.CancelledError(f"Task {task_id} was cancelled")
        
        return task.result
    
    async def get_all_results(self, timeout: float = None) -> Dict[str, Any]:
        """
        Get results of all tasks.
        
        Args:
            timeout: Wait timeout in seconds
            
        Returns:
            Dictionary mapping task IDs to results
        """
        results = {}
        for task_id in self.tasks:
            try:
                results[task_id] = await self.get_task_result(task_id, timeout)
            except Exception as e:
                results[task_id] = e
        return results
    
    async def shutdown(self):
        """Shutdown resource manager and cleanup resources."""
        self._running = False
        
        # Cancel all tasks
        await self.cancel_all()
        
        # Shutdown components
        await self.worker_pool.shutdown()
        await self.resource_monitor.stop_monitoring()
        
        # Close all contexts
        for context_id, context in self.context_pool.contexts.items():
            try:
                await context.close()
            except Exception as e:
                logger.error(f"Error closing context {context_id}: {e}")
        
        logger.info("ResourceManager shutdown complete")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        stats = self.stats.copy()
        stats.update({
            "active_tasks": len(self.worker_pool._active_tasks),
            "pending_tasks": sum(1 for t in self.tasks.values() if t.state == TaskState.PENDING),
            "completed_tasks": len(self.completed_task_ids),
            "total_tasks": len(self.tasks),
            "context_pool_size": len(self.context_pool.contexts),
            "available_contexts": self.context_pool.available_contexts.qsize()
        })
        return stats


# Convenience functions for common patterns
async def run_parallel_tasks(tasks: List[Dict[str, Any]],
                            max_workers: int = 4,
                            resource_limits: ResourceLimits = None,
                            context_factory: Callable = None) -> Dict[str, Any]:
    """
    Convenience function to run multiple tasks in parallel.
    
    Args:
        tasks: List of task definitions with 'func', 'args', 'kwargs', etc.
        max_workers: Maximum concurrent workers
        resource_limits: Resource constraints
        context_factory: Function to create browser contexts
        
    Returns:
        Dictionary of task results
    """
    manager = ResourceManager(
        max_workers=max_workers,
        resource_limits=resource_limits,
        context_factory=context_factory
    )
    
    try:
        await manager.initialize()
        
        # Create tasks
        task_ids = []
        for task_def in tasks:
            task_id = manager.create_task(
                func=task_def['func'],
                args=task_def.get('args', ()),
                kwargs=task_def.get('kwargs', {}),
                priority=task_def.get('priority', TaskPriority.NORMAL),
                dependencies=task_def.get('dependencies', []),
                context_id=task_def.get('context_id'),
                max_retries=task_def.get('max_retries', 2)
            )
            task_ids.append(task_id)
        
        # Execute all tasks
        results = await manager.execute_all()
        
        # Return results
        return {task_id: results[task_id].result for task_id in task_ids}
    
    finally:
        await manager.shutdown()


def create_task_batch(func: Callable,
                     args_list: List[Tuple],
                     kwargs_list: List[Dict] = None,
                     priority: TaskPriority = TaskPriority.NORMAL,
                     max_retries: int = 2) -> List[Dict[str, Any]]:
    """
    Create a batch of tasks from a single function with different arguments.
    
    Args:
        func: Function to execute
        args_list: List of argument tuples
        kwargs_list: List of keyword argument dicts
        priority: Task priority
        max_retries: Maximum retry attempts
        
    Returns:
        List of task definitions
    """
    kwargs_list = kwargs_list or [{}] * len(args_list)
    
    tasks = []
    for args, kwargs in zip(args_list, kwargs_list):
        tasks.append({
            'func': func,
            'args': args,
            'kwargs': kwargs,
            'priority': priority,
            'max_retries': max_retries
        })
    
    return tasks