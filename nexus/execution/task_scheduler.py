"""
nexus/execution/task_scheduler.py

Parallel Execution Engine — Enable concurrent browser automation with intelligent task parallelization, shared browser contexts, and resource-aware scheduling. Dramatically improves throughput for batch operations.
"""

import asyncio
import multiprocessing
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from ..actor.page import Page
from ..agent.service import AgentService
from ..config import settings


class TaskPriority(Enum):
    """Task priority levels for scheduling."""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    BACKGROUND = 4


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class ConcurrencyMode(Enum):
    """Concurrency execution modes."""
    ASYNCIO = "asyncio"
    THREAD = "thread"
    PROCESS = "process"


@dataclass
class ResourceLimits:
    """Resource limits for browser execution."""
    max_concurrent_pages: int = 10
    max_concurrent_contexts: int = 5
    max_memory_mb: int = 4096
    max_cpu_percent: int = 80
    max_network_requests: int = 100
    timeout_seconds: int = 300


@dataclass
class TaskDependency:
    """Task dependency definition."""
    task_id: str
    required_status: TaskStatus = TaskStatus.COMPLETED
    timeout_seconds: int = 60


@dataclass
class BrowserContext:
    """Shared browser context for task execution."""
    context_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pages: List[Page] = field(default_factory=list)
    agent_service: Optional[AgentService] = None
    cookies: Dict[str, Any] = field(default_factory=dict)
    storage_state: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    active_tasks: int = 0
    max_pages: int = 5


@dataclass
class TaskDefinition:
    """Task definition for parallel execution."""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    priority: TaskPriority = TaskPriority.MEDIUM
    function: Optional[Callable] = None
    args: Tuple = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[TaskDependency] = field(default_factory=list)
    context_id: Optional[str] = None
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    timeout_seconds: int = 300
    max_retries: int = 3
    retry_delay: float = 1.0
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Task execution result."""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    execution_time: Optional[float] = None
    context_id: Optional[str] = None
    worker_id: Optional[str] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResourceMonitor:
    """Monitors system resources for scheduling decisions."""
    
    def __init__(self, limits: ResourceLimits):
        self.limits = limits
        self.current_usage = {
            "pages": 0,
            "contexts": 0,
            "memory_mb": 0,
            "cpu_percent": 0,
            "network_requests": 0,
        }
        self._lock = asyncio.Lock()
    
    async def can_allocate(self, requirements: Dict[str, float]) -> bool:
        """Check if resources can be allocated."""
        async with self._lock:
            for resource, required in requirements.items():
                if resource in self.current_usage:
                    limit = getattr(self.limits, f"max_{resource}", float('inf'))
                    if self.current_usage[resource] + required > limit:
                        return False
            return True
    
    async def allocate(self, requirements: Dict[str, float]) -> bool:
        """Allocate resources if available."""
        async with self._lock:
            if not await self.can_allocate(requirements):
                return False
            
            for resource, required in requirements.items():
                if resource in self.current_usage:
                    self.current_usage[resource] += required
            return True
    
    async def release(self, requirements: Dict[str, float]):
        """Release allocated resources."""
        async with self._lock:
            for resource, required in requirements.items():
                if resource in self.current_usage:
                    self.current_usage[resource] = max(0, self.current_usage[resource] - required)
    
    def get_utilization(self) -> Dict[str, float]:
        """Get current resource utilization percentages."""
        utilization = {}
        for resource, current in self.current_usage.items():
            limit = getattr(self.limits, f"max_{resource}", 1)
            utilization[resource] = (current / limit) * 100 if limit > 0 else 0
        return utilization


class DependencyAnalyzer:
    """Analyzes task dependencies for safe parallelization."""
    
    def __init__(self):
        self.task_graph: Dict[str, Set[str]] = {}
        self.reverse_graph: Dict[str, Set[str]] = {}
    
    def add_task(self, task: TaskDefinition):
        """Add task to dependency graph."""
        if task.task_id not in self.task_graph:
            self.task_graph[task.task_id] = set()
            self.reverse_graph[task.task_id] = set()
        
        for dep in task.dependencies:
            self.task_graph[task.task_id].add(dep.task_id)
            if dep.task_id not in self.reverse_graph:
                self.reverse_graph[dep.task_id] = set()
            self.reverse_graph[dep.task_id].add(task.task_id)
    
    def get_ready_tasks(self, completed_tasks: Set[str], 
                       running_tasks: Set[str]) -> List[str]:
        """Get tasks that are ready to run (dependencies satisfied)."""
        ready = []
        for task_id, deps in self.task_graph.items():
            if task_id in completed_tasks or task_id in running_tasks:
                continue
            
            # Check if all dependencies are satisfied
            all_deps_satisfied = True
            for dep_id in deps:
                if dep_id not in completed_tasks:
                    all_deps_satisfied = False
                    break
            
            if all_deps_satisfied:
                ready.append(task_id)
        
        return ready
    
    def detect_cycles(self) -> bool:
        """Detect cycles in dependency graph."""
        visited = set()
        rec_stack = set()
        
        def visit(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self.task_graph.get(node, set()):
                if neighbor not in visited:
                    if visit(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in self.task_graph:
            if node not in visited:
                if visit(node):
                    return True
        return False
    
    def get_critical_path(self) -> List[str]:
        """Get the critical path of tasks (longest dependency chain)."""
        # Simplified implementation - in production would use proper DAG analysis
        visited = set()
        path = []
        
        def dfs(node, current_path):
            visited.add(node)
            current_path.append(node)
            
            max_path = current_path.copy()
            for neighbor in self.reverse_graph.get(node, set()):
                if neighbor not in visited:
                    new_path = dfs(neighbor, current_path.copy())
                    if len(new_path) > len(max_path):
                        max_path = new_path
            
            return max_path
        
        # Find nodes with no incoming edges (start nodes)
        start_nodes = set(self.task_graph.keys()) - set().union(*self.reverse_graph.values())
        for start in start_nodes:
            path = dfs(start, [])
            if len(path) > len(path):
                path = path
        
        return path


class ContextManager:
    """Manages shared browser contexts for task execution."""
    
    def __init__(self, max_contexts: int = 5):
        self.max_contexts = max_contexts
        self.contexts: Dict[str, BrowserContext] = {}
        self.available_contexts: asyncio.Queue[str] = asyncio.Queue()
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def initialize(self):
        """Initialize context manager."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def shutdown(self):
        """Shutdown context manager."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close all contexts
        for context_id in list(self.contexts.keys()):
            await self.release_context(context_id)
    
    async def get_context(self, task: TaskDefinition) -> BrowserContext:
        """Get or create a browser context for task execution."""
        async with self._lock:
            # Try to reuse existing context if specified
            if task.context_id and task.context_id in self.contexts:
                context = self.contexts[task.context_id]
                context.active_tasks += 1
                context.last_used = time.time()
                return context
            
            # Find available context with capacity
            for context_id, context in self.contexts.items():
                if (context.active_tasks < context.max_pages and 
                    len(context.pages) < context.max_pages):
                    context.active_tasks += 1
                    context.last_used = time.time()
                    return context
            
            # Create new context if under limit
            if len(self.contexts) < self.max_contexts:
                context = BrowserContext()
                self.contexts[context.context_id] = context
                context.active_tasks += 1
                return context
            
            # Wait for available context
            while True:
                try:
                    context_id = await asyncio.wait_for(
                        self.available_contexts.get(), 
                        timeout=1.0
                    )
                    if context_id in self.contexts:
                        context = self.contexts[context_id]
                        if context.active_tasks < context.max_pages:
                            context.active_tasks += 1
                            context.last_used = time.time()
                            return context
                except asyncio.TimeoutError:
                    continue
    
    async def release_context(self, context_id: str):
        """Release a browser context."""
        async with self._lock:
            if context_id in self.contexts:
                context = self.contexts[context_id]
                context.active_tasks = max(0, context.active_tasks - 1)
                
                # Put back in queue if has capacity
                if context.active_tasks < context.max_pages:
                    await self.available_contexts.put(context_id)
    
    async def _cleanup_loop(self):
        """Cleanup idle contexts periodically."""
        while True:
            await asyncio.sleep(60)  # Check every minute
            async with self._lock:
                current_time = time.time()
                to_remove = []
                
                for context_id, context in self.contexts.items():
                    # Remove contexts idle for more than 5 minutes
                    if (context.active_tasks == 0 and 
                        current_time - context.last_used > 300):
                        to_remove.append(context_id)
                
                for context_id in to_remove:
                    context = self.contexts.pop(context_id)
                    # Close pages
                    for page in context.pages:
                        try:
                            await page.close()
                        except:
                            pass


class WorkerPool:
    """Worker pool for task execution with different concurrency modes."""
    
    def __init__(self, 
                 max_workers: int = 10,
                 concurrency_mode: ConcurrencyMode = ConcurrencyMode.ASYNCIO):
        self.max_workers = max_workers
        self.concurrency_mode = concurrency_mode
        self.workers: Dict[str, asyncio.Task] = {}
        self.worker_semaphore = asyncio.Semaphore(max_workers)
        self._executor = None
        self._worker_count = 0
        
        if concurrency_mode == ConcurrencyMode.THREAD:
            self._executor = ThreadPoolExecutor(max_workers=max_workers)
        elif concurrency_mode == ConcurrencyMode.PROCESS:
            self._executor = ProcessPoolExecutor(max_workers=max_workers)
    
    async def execute_task(self, 
                          task_func: Callable,
                          task_args: Tuple,
                          task_kwargs: Dict,
                          timeout: int = 300) -> Any:
        """Execute a task using the configured concurrency mode."""
        async with self.worker_semaphore:
            worker_id = f"worker_{self._worker_count}"
            self._worker_count += 1
            
            try:
                if self.concurrency_mode == ConcurrencyMode.ASYNCIO:
                    return await asyncio.wait_for(
                        task_func(*task_args, **task_kwargs),
                        timeout=timeout
                    )
                
                elif self.concurrency_mode == ConcurrencyMode.THREAD:
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        self._executor,
                        lambda: asyncio.run(task_func(*task_args, **task_kwargs))
                    )
                
                elif self.concurrency_mode == ConcurrencyMode.PROCESS:
                    # Process-based execution requires picklable functions
                    # This is a simplified implementation
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        self._executor,
                        task_func,
                        *task_args
                    )
            
            finally:
                self._worker_count -= 1
    
    async def shutdown(self):
        """Shutdown worker pool."""
        if self._executor:
            self._executor.shutdown(wait=True)


class TaskScheduler:
    """
    Parallel Execution Engine for browser automation.
    
    Features:
    - Intelligent task parallelization with dependency analysis
    - Shared browser contexts for efficient resource usage
    - Resource-aware scheduling to prevent overload
    - Support for multiple concurrency modes
    - Priority-based task scheduling
    - Automatic retries and error handling
    """
    
    def __init__(self,
                 max_concurrent_tasks: int = 20,
                 max_contexts: int = 5,
                 concurrency_mode: ConcurrencyMode = ConcurrencyMode.ASYNCIO,
                 resource_limits: Optional[ResourceLimits] = None):
        
        self.max_concurrent_tasks = max_concurrent_tasks
        self.concurrency_mode = concurrency_mode
        
        # Core components
        self.resource_monitor = ResourceMonitor(resource_limits or ResourceLimits())
        self.dependency_analyzer = DependencyAnalyzer()
        self.context_manager = ContextManager(max_contexts=max_contexts)
        self.worker_pool = WorkerPool(
            max_workers=max_concurrent_tasks,
            concurrency_mode=concurrency_mode
        )
        
        # Task management
        self.tasks: Dict[str, TaskDefinition] = {}
        self.task_results: Dict[str, TaskResult] = {}
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.running_tasks: Set[str] = set()
        self.completed_tasks: Set[str] = set()
        self.failed_tasks: Set[str] = set()
        
        # Scheduling state
        self._scheduler_task: Optional[asyncio.Task] = None
        self._running = False
        self._lock = asyncio.Lock()
        self._task_complete_events: Dict[str, asyncio.Event] = {}
        
        # Statistics
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0,
            "average_execution_time": 0,
            "context_reuses": 0,
            "resource_warnings": 0,
        }
    
    async def start(self):
        """Start the task scheduler."""
        if self._running:
            return
        
        self._running = True
        await self.context_manager.initialize()
        self._scheduler_task = asyncio.create_task(self._scheduling_loop())
        
        print(f"Task scheduler started with {self.max_concurrent_tasks} workers "
              f"using {self.concurrency_mode.value} concurrency")
    
    async def stop(self):
        """Stop the task scheduler."""
        self._running = False
        
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all running tasks
        for task_id in list(self.running_tasks):
            await self.cancel_task(task_id)
        
        await self.context_manager.shutdown()
        await self.worker_pool.shutdown()
        
        print("Task scheduler stopped")
    
    async def submit_task(self, task: TaskDefinition) -> str:
        """Submit a task for execution."""
        async with self._lock:
            # Validate task
            if not task.function:
                raise ValueError("Task must have a callable function")
            
            # Store task
            self.tasks[task.task_id] = task
            self.task_results[task.task_id] = TaskResult(
                task_id=task.task_id,
                status=TaskStatus.PENDING
            )
            self._task_complete_events[task.task_id] = asyncio.Event()
            
            # Add to dependency graph
            self.dependency_analyzer.add_task(task)
            
            # Check for dependency cycles
            if self.dependency_analyzer.detect_cycles():
                raise ValueError("Task dependency cycle detected")
            
            # Queue task with priority
            priority_value = task.priority.value
            await self.task_queue.put((priority_value, time.time(), task.task_id))
            
            self.stats["tasks_submitted"] += 1
            
            return task.task_id
    
    async def submit_batch(self, tasks: List[TaskDefinition]) -> List[str]:
        """Submit multiple tasks for execution."""
        task_ids = []
        for task in tasks:
            task_id = await self.submit_task(task)
            task_ids.append(task_id)
        return task_ids
    
    async def cancel_task(self, task_id: str):
        """Cancel a pending or running task."""
        async with self._lock:
            if task_id in self.running_tasks:
                # Task is running - mark for cancellation
                result = self.task_results[task_id]
                result.status = TaskStatus.CANCELLED
                result.end_time = time.time()
                self.running_tasks.remove(task_id)
                self.failed_tasks.add(task_id)
                
                # Trigger completion event
                if task_id in self._task_complete_events:
                    self._task_complete_events[task_id].set()
            
            elif task_id in self.tasks:
                # Task is pending - mark as cancelled
                result = self.task_results[task_id]
                result.status = TaskStatus.CANCELLED
    
    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> TaskResult:
        """Wait for a task to complete."""
        if task_id not in self._task_complete_events:
            raise ValueError(f"Task {task_id} not found")
        
        try:
            await asyncio.wait_for(
                self._task_complete_events[task_id].wait(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            pass
        
        return self.task_results[task_id]
    
    async def wait_for_batch(self, task_ids: List[str], 
                           timeout: Optional[float] = None) -> Dict[str, TaskResult]:
        """Wait for multiple tasks to complete."""
        tasks = []
        for task_id in task_ids:
            if task_id in self._task_complete_events:
                tasks.append(self._task_complete_events[task_id].wait())
        
        if tasks:
            try:
                await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout)
            except asyncio.TimeoutError:
                pass
        
        return {task_id: self.task_results[task_id] 
                for task_id in task_ids if task_id in self.task_results}
    
    async def _scheduling_loop(self):
        """Main scheduling loop."""
        while self._running:
            try:
                # Check for ready tasks based on dependencies
                ready_task_ids = self.dependency_analyzer.get_ready_tasks(
                    completed_tasks=self.completed_tasks,
                    running_tasks=self.running_tasks
                )
                
                # Schedule ready tasks
                for task_id in ready_task_ids:
                    if (task_id not in self.running_tasks and 
                        task_id not in self.completed_tasks and
                        task_id not in self.failed_tasks):
                        
                        task = self.tasks[task_id]
                        
                        # Check resource availability
                        if await self.resource_monitor.can_allocate(task.resource_requirements):
                            await self._schedule_task(task_id)
                
                # Process queued tasks
                await self._process_queue()
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.01)
                
            except Exception as e:
                print(f"Error in scheduling loop: {e}")
                await asyncio.sleep(1)
    
    async def _process_queue(self):
        """Process tasks from the priority queue."""
        try:
            # Non-blocking get from queue
            priority, timestamp, task_id = await asyncio.wait_for(
                self.task_queue.get(), 
                timeout=0.001
            )
            
            if (task_id not in self.running_tasks and 
                task_id not in self.completed_tasks and
                task_id not in self.failed_tasks):
                
                task = self.tasks[task_id]
                
                # Check dependencies and resources
                deps_satisfied = all(
                    dep.task_id in self.completed_tasks 
                    for dep in task.dependencies
                )
                
                if deps_satisfied and await self.resource_monitor.can_allocate(task.resource_requirements):
                    await self._schedule_task(task_id)
                else:
                    # Re-queue if not ready
                    await self.task_queue.put((priority, timestamp, task_id))
        
        except asyncio.TimeoutError:
            pass
    
    async def _schedule_task(self, task_id: str):
        """Schedule a task for execution."""
        task = self.tasks[task_id]
        
        # Allocate resources
        await self.resource_monitor.allocate(task.resource_requirements)
        
        # Get browser context
        context = await self.context_manager.get_context(task)
        
        # Update task status
        result = self.task_results[task_id]
        result.status = TaskStatus.RUNNING
        result.start_time = time.time()
        result.context_id = context.context_id
        
        self.running_tasks.add(task_id)
        
        # Execute task in worker pool
        asyncio.create_task(self._execute_task(task_id, task, context))
    
    async def _execute_task(self, task_id: str, task: TaskDefinition, context: BrowserContext):
        """Execute a task with retries and error handling."""
        retry_count = 0
        last_error = None
        
        while retry_count <= task.max_retries:
            try:
                # Execute task
                task_result = await self.worker_pool.execute_task(
                    task.function,
                    (context,) + task.args,
                    task_kwargs=task.kwargs,
                    timeout=task.timeout_seconds
                )
                
                # Task succeeded
                async with self._lock:
                    result = self.task_results[task_id]
                    result.status = TaskStatus.COMPLETED
                    result.result = task_result
                    result.end_time = time.time()
                    result.execution_time = result.end_time - result.start_time
                    result.retry_count = retry_count
                    
                    self.running_tasks.remove(task_id)
                    self.completed_tasks.add(task_id)
                    
                    # Update statistics
                    self.stats["tasks_completed"] += 1
                    self.stats["total_execution_time"] += result.execution_time
                    self.stats["average_execution_time"] = (
                        self.stats["total_execution_time"] / self.stats["tasks_completed"]
                    )
                    
                    # Release resources
                    await self.resource_monitor.release(task.resource_requirements)
                    await self.context_manager.release_context(context.context_id)
                    
                    # Trigger completion event
                    if task_id in self._task_complete_events:
                        self._task_complete_events[task_id].set()
                
                return
                
            except Exception as e:
                last_error = str(e)
                retry_count += 1
                
                if retry_count <= task.max_retries:
                    # Wait before retry
                    await asyncio.sleep(task.retry_delay * retry_count)
                else:
                    # Max retries exceeded
                    async with self._lock:
                        result = self.task_results[task_id]
                        result.status = TaskStatus.FAILED
                        result.error = last_error
                        result.end_time = time.time()
                        result.execution_time = result.end_time - result.start_time
                        result.retry_count = retry_count
                        
                        self.running_tasks.remove(task_id)
                        self.failed_tasks.add(task_id)
                        
                        # Update statistics
                        self.stats["tasks_failed"] += 1
                        
                        # Release resources
                        await self.resource_monitor.release(task.resource_requirements)
                        await self.context_manager.release_context(context.context_id)
                        
                        # Trigger completion event
                        if task_id in self._task_complete_events:
                            self._task_complete_events[task_id].set()
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status and statistics."""
        resource_util = self.resource_monitor.get_utilization()
        
        return {
            "running": self._running,
            "total_tasks": len(self.tasks),
            "pending_tasks": len(self.tasks) - len(self.running_tasks) - len(self.completed_tasks) - len(self.failed_tasks),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "resource_utilization": resource_util,
            "statistics": self.stats,
            "contexts": len(self.context_manager.contexts),
            "concurrency_mode": self.concurrency_mode.value,
        }
    
    def get_task_status(self, task_id: str) -> Optional[TaskResult]:
        """Get status of a specific task."""
        return self.task_results.get(task_id)
    
    def get_failed_tasks(self) -> List[TaskResult]:
        """Get all failed tasks."""
        return [
            self.task_results[task_id] 
            for task_id in self.failed_tasks 
            if task_id in self.task_results
        ]
    
    async def retry_failed_tasks(self) -> List[str]:
        """Retry all failed tasks."""
        retried_tasks = []
        
        for task_id in list(self.failed_tasks):
            task = self.tasks[task_id]
            
            # Reset task state
            async with self._lock:
                self.failed_tasks.remove(task_id)
                result = self.task_results[task_id]
                result.status = TaskStatus.PENDING
                result.error = None
                result.retry_count = 0
            
            # Re-submit task
            new_task_id = await self.submit_task(task)
            retried_tasks.append(new_task_id)
        
        return retried_tasks


# Convenience functions for common use cases
async def execute_parallel_tasks(
    tasks: List[Callable],
    max_concurrent: int = 10,
    timeout: int = 300,
    **shared_kwargs
) -> List[Any]:
    """
    Execute multiple tasks in parallel with automatic resource management.
    
    Args:
        tasks: List of async callables to execute
        max_concurrent: Maximum concurrent executions
        timeout: Timeout per task in seconds
        **shared_kwargs: Shared keyword arguments for all tasks
    
    Returns:
        List of results in order of task submission
    """
    scheduler = TaskScheduler(max_concurrent_tasks=max_concurrent)
    await scheduler.start()
    
    try:
        # Submit all tasks
        task_definitions = []
        for i, task_func in enumerate(tasks):
            task_def = TaskDefinition(
                name=f"task_{i}",
                function=task_func,
                kwargs=shared_kwargs,
                timeout_seconds=timeout,
                resource_requirements={"pages": 1}
            )
            task_definitions.append(task_def)
        
        task_ids = await scheduler.submit_batch(task_definitions)
        
        # Wait for all tasks to complete
        results = await scheduler.wait_for_batch(task_ids, timeout=timeout * 2)
        
        # Extract results in order
        ordered_results = []
        for task_id in task_ids:
            result = results.get(task_id)
            if result and result.status == TaskStatus.COMPLETED:
                ordered_results.append(result.result)
            else:
                ordered_results.append(None)
        
        return ordered_results
        
    finally:
        await scheduler.stop()


async def execute_with_dependencies(
    task_graph: Dict[str, Dict[str, Any]],
    max_concurrent: int = 10
) -> Dict[str, Any]:
    """
    Execute tasks with complex dependency relationships.
    
    Args:
        task_graph: Dictionary mapping task_id to task definition with dependencies
        max_concurrent: Maximum concurrent executions
    
    Returns:
        Dictionary mapping task_id to result
    """
    scheduler = TaskScheduler(max_concurrent_tasks=max_concurrent)
    await scheduler.start()
    
    try:
        # Submit all tasks
        task_definitions = []
        for task_id, task_spec in task_graph.items():
            task_def = TaskDefinition(
                task_id=task_id,
                name=task_spec.get("name", task_id),
                function=task_spec["function"],
                args=task_spec.get("args", ()),
                kwargs=task_spec.get("kwargs", {}),
                dependencies=[
                    TaskDependency(dep_id) 
                    for dep_id in task_spec.get("dependencies", [])
                ],
                resource_requirements=task_spec.get("resources", {"pages": 1})
            )
            task_definitions.append(task_def)
        
        await scheduler.submit_batch(task_definitions)
        
        # Wait for all tasks
        all_task_ids = list(task_graph.keys())
        results = await scheduler.wait_for_batch(all_task_ids, timeout=600)
        
        # Convert to simple result dictionary
        return {
            task_id: result.result if result else None
            for task_id, result in results.items()
        }
        
    finally:
        await scheduler.stop()


# Integration with existing nexus modules
class BrowserTaskScheduler(TaskScheduler):
    """
    Specialized task scheduler for browser automation tasks.
    Integrates with nexus actor and agent modules.
    """
    
    async def execute_page_tasks(self, 
                                page_tasks: List[Tuple[Page, Callable, Tuple, Dict]],
                                max_concurrent: int = 5) -> List[Any]:
        """
        Execute tasks that operate on browser pages.
        
        Args:
            page_tasks: List of (page, function, args, kwargs) tuples
            max_concurrent: Maximum concurrent page operations
        
        Returns:
            List of results
        """
        task_definitions = []
        
        for i, (page, func, args, kwargs) in enumerate(page_tasks):
            # Wrap function to include page as first argument
            async def task_wrapper(context, page=page, func=func, args=args, kwargs=kwargs):
                return await func(page, *args, **kwargs)
            
            task_def = TaskDefinition(
                name=f"page_task_{i}",
                function=task_wrapper,
                resource_requirements={"pages": 1},
                tags={"page_operation"}
            )
            task_definitions.append(task_def)
        
        task_ids = await self.submit_batch(task_definitions)
        results = await self.wait_for_batch(task_ids)
        
        return [
            results[task_id].result 
            for task_id in task_ids 
            if results.get(task_id)
        ]
    
    async def execute_agent_tasks(self,
                                 agent_tasks: List[Tuple[AgentService, str, Dict]],
                                 max_concurrent: int = 3) -> List[Any]:
        """
        Execute agent automation tasks.
        
        Args:
            agent_tasks: List of (agent, instruction, context) tuples
            max_concurrent: Maximum concurrent agent operations
        
        Returns:
            List of agent results
        """
        task_definitions = []
        
        for i, (agent, instruction, context) in enumerate(agent_tasks):
            async def agent_task(context, agent=agent, instruction=instruction, ctx=context):
                return await agent.execute(instruction, context=ctx)
            
            task_def = TaskDefinition(
                name=f"agent_task_{i}",
                function=agent_task,
                resource_requirements={"pages": 2, "memory_mb": 500},
                tags={"agent_operation"},
                timeout_seconds=600
            )
            task_definitions.append(task_def)
        
        task_ids = await self.submit_batch(task_definitions)
        results = await self.wait_for_batch(task_ids)
        
        return [
            results[task_id].result 
            for task_id in task_ids 
            if results.get(task_id)
        ]