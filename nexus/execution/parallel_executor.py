"""
Parallel Execution Engine for Browser-Use
Enables concurrent browser automation with intelligent task parallelization,
shared browser contexts, and resource-aware scheduling.
"""

import asyncio
import logging
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import psutil
from playwright.async_api import BrowserContext, Page

from nexus.actor.page import ActorPage
from nexus.agent.service import AgentService

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels for scheduling."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BrowserTask:
    """Represents a single browser automation task."""
    task_id: str
    func: Callable
    args: Tuple = ()
    kwargs: Dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    dependencies: Set[str] = field(default_factory=set)
    timeout: Optional[float] = 30.0
    retry_count: int = 0
    max_retries: int = 3
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[Exception] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    context_id: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.task_id:
            self.task_id = str(uuid.uuid4())


@dataclass
class ExecutionContext:
    """Shared browser context for parallel execution."""
    context_id: str
    browser_context: BrowserContext
    active_pages: int = 0
    max_pages: int = 10
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    is_shared: bool = True


class ResourceMonitor:
    """Monitors system resources to prevent browser overload."""
    
    def __init__(self, max_cpu_percent: float = 80.0, max_memory_percent: float = 75.0):
        self.max_cpu_percent = max_cpu_percent
        self.max_memory_percent = max_memory_percent
        self._last_check = 0
        self._check_interval = 1.0
        self._cached_status = True
    
    def check_resources(self) -> bool:
        """Check if system resources are available for new tasks."""
        current_time = time.time()
        
        # Use cached status if checked recently
        if current_time - self._last_check < self._check_interval:
            return self._cached_status
        
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            
            self._cached_status = (
                cpu_percent < self.max_cpu_percent and
                memory_percent < self.max_memory_percent
            )
            self._last_check = current_time
            
            if not self._cached_status:
                logger.warning(
                    f"Resource limits exceeded: CPU {cpu_percent}% (max {self.max_cpu_percent}%), "
                    f"Memory {memory_percent}% (max {self.max_memory_percent}%)"
                )
            
            return self._cached_status
            
        except Exception as e:
            logger.error(f"Resource monitoring failed: {e}")
            return True  # Fail open
    
    def get_resource_stats(self) -> Dict[str, float]:
        """Get current resource statistics."""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": psutil.virtual_memory().percent,
                "available_memory_gb": psutil.virtual_memory().available / (1024**3),
            }
        except Exception:
            return {}


class TaskDependencyAnalyzer:
    """Analyzes task dependencies for safe parallelization."""
    
    @staticmethod
    def analyze_dependencies(tasks: List[BrowserTask]) -> Dict[str, Set[str]]:
        """Build dependency graph and detect cycles."""
        graph = {task.task_id: set(task.dependencies) for task in tasks}
        
        # Check for cycles using DFS
        visited = set()
        temp_visited = set()
        
        def visit(node: str, path: List[str]):
            if node in temp_visited:
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                raise ValueError(f"Dependency cycle detected: {' -> '.join(cycle)}")
            
            if node not in visited:
                temp_visited.add(node)
                path.append(node)
                
                for dependency in graph.get(node, []):
                    if dependency in graph:
                        visit(dependency, path.copy())
                
                temp_visited.remove(node)
                visited.add(node)
        
        for task_id in graph:
            if task_id not in visited:
                visit(task_id, [])
        
        return graph
    
    @staticmethod
    def get_ready_tasks(
        tasks: Dict[str, BrowserTask],
        completed_tasks: Set[str]
    ) -> List[BrowserTask]:
        """Get tasks whose dependencies are all satisfied."""
        ready_tasks = []
        
        for task in tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            
            if task.dependencies.issubset(completed_tasks):
                ready_tasks.append(task)
        
        # Sort by priority (higher priority first)
        ready_tasks.sort(key=lambda t: t.priority.value, reverse=True)
        return ready_tasks


class ContextPool:
    """Pool of shared browser contexts for efficient resource usage."""
    
    def __init__(self, max_contexts: int = 5, max_pages_per_context: int = 10):
        self.max_contexts = max_contexts
        self.max_pages_per_context = max_pages_per_context
        self.contexts: Dict[str, ExecutionContext] = {}
        self._lock = asyncio.Lock()
    
    async def acquire_context(self) -> Tuple[str, BrowserContext]:
        """Acquire a browser context from the pool or create a new one."""
        async with self._lock:
            # Try to find an available context
            for context_id, context in self.contexts.items():
                if context.active_pages < context.max_pages:
                    context.active_pages += 1
                    context.last_used = time.time()
                    return context_id, context.browser_context
            
            # Create new context if pool not full
            if len(self.contexts) < self.max_contexts:
                return await self._create_context()
            
            # Find least recently used context
            lru_context_id = min(
                self.contexts.keys(),
                key=lambda cid: self.contexts[cid].last_used
            )
            context = self.contexts[lru_context_id]
            context.active_pages += 1
            context.last_used = time.time()
            return lru_context_id, context.browser_context
    
    async def _create_context(self) -> Tuple[str, BrowserContext]:
        """Create a new browser context."""
        # This is a placeholder - actual implementation would use your browser factory
        from playwright.async_api import async_playwright
        
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch()
        browser_context = await browser.new_context()
        
        context_id = str(uuid.uuid4())
        self.contexts[context_id] = ExecutionContext(
            context_id=context_id,
            browser_context=browser_context,
            active_pages=1,
            max_pages=self.max_pages_per_context,
        )
        
        logger.info(f"Created new browser context: {context_id}")
        return context_id, browser_context
    
    async def release_context(self, context_id: str):
        """Release a context back to the pool."""
        async with self._lock:
            if context_id in self.contexts:
                self.contexts[context_id].active_pages = max(
                    0, self.contexts[context_id].active_pages - 1
                )
    
    async def cleanup(self):
        """Close all browser contexts."""
        async with self._lock:
            for context in self.contexts.values():
                try:
                    await context.browser_context.close()
                except Exception as e:
                    logger.error(f"Error closing context {context.context_id}: {e}")
            self.contexts.clear()


class ParallelExecutor:
    """
    Parallel execution engine for browser automation tasks.
    Supports both asyncio and process-based concurrency.
    """
    
    def __init__(
        self,
        max_concurrent_tasks: int = 5,
        concurrency_type: str = "asyncio",  # "asyncio", "thread", "process"
        max_contexts: int = 3,
        max_pages_per_context: int = 8,
        enable_resource_monitoring: bool = True,
        task_timeout: float = 60.0,
    ):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.concurrency_type = concurrency_type
        self.task_timeout = task_timeout
        
        # Task management
        self.tasks: Dict[str, BrowserTask] = {}
        self.completed_tasks: Set[str] = set()
        self.failed_tasks: Set[str] = set()
        
        # Resource management
        self.resource_monitor = ResourceMonitor() if enable_resource_monitoring else None
        self.context_pool = ContextPool(max_contexts, max_pages_per_context)
        
        # Concurrency primitives
        self._semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self._task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._active_tasks: Set[asyncio.Task] = set()
        self._running = False
        self._executor = None
        
        # Statistics
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0,
            "avg_execution_time": 0.0,
        }
        
        # Initialize executor based on concurrency type
        if concurrency_type == "thread":
            self._executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)
        elif concurrency_type == "process":
            self._executor = ProcessPoolExecutor(max_workers=max_concurrent_tasks)
    
    async def submit_task(
        self,
        func: Callable,
        *args,
        task_id: Optional[str] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        dependencies: Optional[Set[str]] = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> str:
        """Submit a task for parallel execution."""
        task = BrowserTask(
            task_id=task_id or str(uuid.uuid4()),
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            dependencies=dependencies or set(),
            timeout=timeout or self.task_timeout,
        )
        
        self.tasks[task.task_id] = task
        self.stats["tasks_submitted"] += 1
        
        # Check if task is immediately ready
        if task.dependencies.issubset(self.completed_tasks):
            task.status = TaskStatus.READY
            await self._task_queue.put((-priority.value, time.time(), task.task_id))
        
        logger.debug(f"Task submitted: {task.task_id} with priority {priority.name}")
        return task.task_id
    
    async def submit_batch(
        self,
        tasks: List[Tuple[Callable, Tuple, Dict]],
        priorities: Optional[List[TaskPriority]] = None,
        dependencies_map: Optional[Dict[int, Set[int]]] = None,
    ) -> List[str]:
        """Submit multiple tasks as a batch with dependency mapping."""
        if priorities is None:
            priorities = [TaskPriority.NORMAL] * len(tasks)
        
        if dependencies_map is None:
            dependencies_map = {}
        
        task_ids = []
        id_mapping = {}
        
        # First pass: create all tasks
        for idx, (func, args, kwargs) in enumerate(tasks):
            task_id = str(uuid.uuid4())
            id_mapping[idx] = task_id
            
            task = BrowserTask(
                task_id=task_id,
                func=func,
                args=args,
                kwargs=kwargs,
                priority=priorities[idx],
            )
            self.tasks[task_id] = task
            task_ids.append(task_id)
            self.stats["tasks_submitted"] += 1
        
        # Second pass: set up dependencies
        for idx, dependencies in dependencies_map.items():
            if idx in id_mapping:
                task_id = id_mapping[idx]
                dependency_ids = {id_mapping[dep_idx] for dep_idx in dependencies if dep_idx in id_mapping}
                self.tasks[task_id].dependencies = dependency_ids
        
        # Queue ready tasks
        for task_id in task_ids:
            task = self.tasks[task_id]
            if task.dependencies.issubset(self.completed_tasks):
                task.status = TaskStatus.READY
                await self._task_queue.put((-task.priority.value, time.time(), task_id))
        
        return task_ids
    
    async def start(self):
        """Start the parallel executor."""
        if self._running:
            return
        
        self._running = True
        logger.info(f"Starting parallel executor with {self.concurrency_type} concurrency")
        
        # Start worker tasks
        workers = []
        for _ in range(self.max_concurrent_tasks):
            worker = asyncio.create_task(self._worker())
            workers.append(worker)
        
        # Wait for all tasks to complete
        try:
            await self._task_queue.join()
        finally:
            # Cancel workers
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            self._running = False
    
    async def _worker(self):
        """Worker coroutine that processes tasks from the queue."""
        while self._running:
            try:
                # Get next task from queue
                _, _, task_id = await asyncio.wait_for(
                    self._task_queue.get(), timeout=1.0
                )
                
                task = self.tasks.get(task_id)
                if not task or task.status != TaskStatus.READY:
                    self._task_queue.task_done()
                    continue
                
                # Check resources if monitoring is enabled
                if self.resource_monitor and not self.resource_monitor.check_resources():
                    logger.warning(f"Resource limits exceeded, requeueing task {task_id}")
                    task.status = TaskStatus.PENDING
                    await asyncio.sleep(1.0)  # Back off
                    await self._task_queue.put((-task.priority.value, time.time(), task_id))
                    self._task_queue.task_done()
                    continue
                
                # Execute task
                async with self._semaphore:
                    await self._execute_task(task)
                
                self._task_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}")
    
    async def _execute_task(self, task: BrowserTask):
        """Execute a single task with timeout and error handling."""
        task.status = TaskStatus.RUNNING
        task.start_time = time.time()
        
        context_id = None
        try:
            # Acquire browser context
            context_id, browser_context = await self.context_pool.acquire_context()
            task.context_id = context_id
            
            # Create page in context
            page = await browser_context.new_page()
            
            # Wrap the function to work with our page
            async def execute_with_page():
                if asyncio.iscoroutinefunction(task.func):
                    # If it's an async function, pass the page directly
                    return await task.func(page, *task.args, **task.kwargs)
                else:
                    # If it's a sync function, run in thread pool
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        self._executor,
                        lambda: task.func(page, *task.args, **task.kwargs)
                    )
            
            # Execute with timeout
            task.result = await asyncio.wait_for(
                execute_with_page(),
                timeout=task.timeout
            )
            
            task.status = TaskStatus.COMPLETED
            task.end_time = time.time()
            
            # Update statistics
            execution_time = task.end_time - task.start_time
            self.stats["tasks_completed"] += 1
            self.stats["total_execution_time"] += execution_time
            self.stats["avg_execution_time"] = (
                self.stats["total_execution_time"] / self.stats["tasks_completed"]
            )
            
            # Mark as completed and check dependent tasks
            self.completed_tasks.add(task.task_id)
            await self._check_dependent_tasks(task.task_id)
            
            logger.debug(
                f"Task completed: {task.task_id} in {execution_time:.2f}s"
            )
            
        except asyncio.TimeoutError:
            task.status = TaskStatus.FAILED
            task.error = TimeoutError(f"Task timed out after {task.timeout} seconds")
            task.end_time = time.time()
            self.failed_tasks.add(task.task_id)
            self.stats["tasks_failed"] += 1
            logger.warning(f"Task timed out: {task.task_id}")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = e
            task.end_time = time.time()
            self.failed_tasks.add(task.task_id)
            self.stats["tasks_failed"] += 1
            logger.error(f"Task failed: {task.task_id} - {e}")
            
            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                logger.info(f"Retrying task {task.task_id} (attempt {task.retry_count})")
                await self._task_queue.put((-task.priority.value, time.time(), task.task_id))
            
        finally:
            # Clean up page
            try:
                if 'page' in locals():
                    await page.close()
            except Exception:
                pass
            
            # Release context
            if context_id:
                await self.context_pool.release_context(context_id)
    
    async def _check_dependent_tasks(self, completed_task_id: str):
        """Check if any pending tasks can now be executed."""
        for task in self.tasks.values():
            if (
                task.status == TaskStatus.PENDING and
                completed_task_id in task.dependencies and
                task.dependencies.issubset(self.completed_tasks)
            ):
                task.status = TaskStatus.READY
                await self._task_queue.put((-task.priority.value, time.time(), task.task_id))
    
    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Wait for a specific task to complete and return its result."""
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        start_time = time.time()
        while True:
            if task.status == TaskStatus.COMPLETED:
                return task.result
            elif task.status == TaskStatus.FAILED:
                raise task.error or Exception(f"Task {task_id} failed")
            elif task.status == TaskStatus.CANCELLED:
                raise asyncio.CancelledError(f"Task {task_id} was cancelled")
            
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Timeout waiting for task {task_id}")
            
            await asyncio.sleep(0.1)
    
    async def wait_all(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Wait for all tasks to complete and return results."""
        start_time = time.time()
        
        while True:
            all_done = all(
                task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
                for task in self.tasks.values()
            )
            
            if all_done:
                break
            
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError("Timeout waiting for all tasks")
            
            await asyncio.sleep(0.1)
        
        return {
            task_id: task.result
            for task_id, task in self.tasks.items()
            if task.status == TaskStatus.COMPLETED
        }
    
    async def cancel_task(self, task_id: str):
        """Cancel a pending or ready task."""
        task = self.tasks.get(task_id)
        if task and task.status in (TaskStatus.PENDING, TaskStatus.READY):
            task.status = TaskStatus.CANCELLED
            logger.info(f"Task cancelled: {task_id}")
    
    async def cancel_all(self):
        """Cancel all pending tasks."""
        for task in self.tasks.values():
            if task.status in (TaskStatus.PENDING, TaskStatus.READY):
                task.status = TaskStatus.CANCELLED
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            **self.stats,
            "pending_tasks": sum(
                1 for t in self.tasks.values() if t.status == TaskStatus.PENDING
            ),
            "running_tasks": sum(
                1 for t in self.tasks.values() if t.status == TaskStatus.RUNNING
            ),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "resource_stats": self.resource_monitor.get_resource_stats() if self.resource_monitor else {},
        }
    
    async def cleanup(self):
        """Clean up resources."""
        await self.cancel_all()
        await self.context_pool.cleanup()
        
        if self._executor:
            self._executor.shutdown(wait=True)
        
        logger.info("Parallel executor cleaned up")


# Integration with existing AgentService
class ParallelAgentService:
    """
    Wrapper for AgentService that enables parallel execution of agent tasks.
    """
    
    def __init__(
        self,
        agent_service: AgentService,
        executor: Optional[ParallelExecutor] = None,
        **executor_kwargs
    ):
        self.agent_service = agent_service
        self.executor = executor or ParallelExecutor(**executor_kwargs)
    
    async def run_parallel(
        self,
        tasks: List[Dict[str, Any]],
        max_concurrent: Optional[int] = None,
    ) -> List[Any]:
        """
        Run multiple agent tasks in parallel.
        
        Args:
            tasks: List of task dictionaries with 'prompt' and optional 'context'
            max_concurrent: Override max concurrent tasks for this run
        
        Returns:
            List of results in the same order as input tasks
        """
        if max_concurrent:
            self.executor.max_concurrent_tasks = max_concurrent
        
        # Submit all tasks
        task_ids = []
        for task_config in tasks:
            task_id = await self.executor.submit_task(
                self._run_agent_task,
                task_config,
                priority=TaskPriority.NORMAL,
            )
            task_ids.append(task_id)
        
        # Wait for all tasks
        results = await self.executor.wait_all()
        
        # Return results in original order
        return [results.get(task_id) for task_id in task_ids]
    
    async def _run_agent_task(self, page: Page, task_config: Dict[str, Any]) -> Any:
        """Run a single agent task with the given page."""
        # Create a temporary agent service instance for this task
        from nexus.agent.service import AgentService
        
        agent = AgentService(
            page=page,
            **{k: v for k, v in task_config.items() if k != 'prompt'}
        )
        
        return await agent.run(task_config['prompt'])
    
    async def cleanup(self):
        """Clean up the executor."""
        await self.executor.cleanup()


# Utility functions for common parallel patterns
async def parallel_map(
    func: Callable,
    items: List[Any],
    max_concurrent: int = 5,
    **executor_kwargs
) -> List[Any]:
    """
    Apply a function to each item in parallel.
    
    Args:
        func: Async function to apply (receives page as first argument)
        items: List of items to process
        max_concurrent: Maximum concurrent executions
    
    Returns:
        List of results
    """
    executor = ParallelExecutor(
        max_concurrent_tasks=max_concurrent,
        **executor_kwargs
    )
    
    try:
        # Submit all tasks
        task_ids = []
        for item in items:
            task_id = await executor.submit_task(func, item)
            task_ids.append(task_id)
        
        # Wait for completion
        results = await executor.wait_all()
        
        # Return in order
        return [results.get(task_id) for task_id in task_ids]
    
    finally:
        await executor.cleanup()


async def parallel_starmap(
    func: Callable,
    items: List[Tuple],
    max_concurrent: int = 5,
    **executor_kwargs
) -> List[Any]:
    """
    Apply a function to each tuple of arguments in parallel.
    
    Args:
        func: Async function to apply (receives page as first argument)
        items: List of argument tuples
        max_concurrent: Maximum concurrent executions
    
    Returns:
        List of results
    """
    executor = ParallelExecutor(
        max_concurrent_tasks=max_concurrent,
        **executor_kwargs
    )
    
    try:
        # Submit all tasks
        task_ids = []
        for args in items:
            task_id = await executor.submit_task(func, *args)
            task_ids.append(task_id)
        
        # Wait for completion
        results = await executor.wait_all()
        
        # Return in order
        return [results.get(task_id) for task_id in task_ids]
    
    finally:
        await executor.cleanup()