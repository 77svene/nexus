import asyncio
import uuid
import time
import logging
from typing import Dict, List, Optional, Any, Callable, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime, timedelta

from nexus.agent.service import Agent
from nexus.actor.page import Page
from nexus.agent.views import AgentConfig
from nexus.actor.utils import BrowserContext

logger = logging.getLogger(__name__)

class WorkerStatus(Enum):
    IDLE = "idle"
    BUSY = "busy"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

@dataclass
class WorkerNode:
    worker_id: str
    status: WorkerStatus = WorkerStatus.IDLE
    last_heartbeat: float = field(default_factory=time.time)
    current_task_id: Optional[str] = None
    browser_context: Optional[BrowserContext] = None
    agent: Optional[Agent] = None
    capabilities: Dict[str, Any] = field(default_factory=dict)
    load_score: float = 0.0
    region: str = "local"
    cloud_provider: Optional[str] = None
    instance_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "status": self.status.value,
            "last_heartbeat": self.last_heartbeat,
            "current_task_id": self.current_task_id,
            "capabilities": self.capabilities,
            "load_score": self.load_score,
            "region": self.region,
            "cloud_provider": self.cloud_provider,
            "instance_id": self.instance_id
        }

@dataclass
class BrowserTask:
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: TaskStatus = TaskStatus.PENDING
    worker_id: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    timeout: float = 300.0
    retry_count: int = 0
    max_retries: int = 3
    session_id: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "priority": self.priority,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "status": self.status.value,
            "worker_id": self.worker_id,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "session_id": self.session_id
        }

class LoadBalancer:
    def __init__(self, strategy: str = "least_loaded"):
        self.strategy = strategy
        self.worker_scores: Dict[str, float] = {}
        
    def select_worker(self, workers: Dict[str, WorkerNode], task: BrowserTask) -> Optional[str]:
        available_workers = [
            w for w in workers.values() 
            if w.status in [WorkerStatus.IDLE, WorkerStatus.BUSY] and self._meets_requirements(w, task)
        ]
        
        if not available_workers:
            return None
            
        if self.strategy == "round_robin":
            return self._round_robin(available_workers)
        elif self.strategy == "least_loaded":
            return self._least_loaded(available_workers)
        elif self.strategy == "random":
            import random
            return random.choice(available_workers).worker_id
        else:
            return self._least_loaded(available_workers)
    
    def _meets_requirements(self, worker: WorkerNode, task: BrowserTask) -> bool:
        # Check if worker meets task requirements
        if task.payload.get("requires_gpu") and not worker.capabilities.get("gpu"):
            return False
        if task.payload.get("region") and task.payload["region"] != worker.region:
            return False
        return True
    
    def _round_robin(self, workers: List[WorkerNode]) -> str:
        # Simple round-robin implementation
        if not hasattr(self, '_last_index'):
            self._last_index = 0
        
        self._last_index = (self._last_index + 1) % len(workers)
        return workers[self._last_index].worker_id
    
    def _least_loaded(self, workers: List[WorkerNode]) -> str:
        # Select worker with lowest load score
        return min(workers, key=lambda w: w.load_score).worker_id
    
    def update_load_score(self, worker_id: str, score: float):
        self.worker_scores[worker_id] = score

class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_workers: Dict[str, str] = {}  # session_id -> worker_id
        
    def create_session(self, session_id: str, worker_id: str, metadata: Dict[str, Any] = None):
        self.sessions[session_id] = {
            "created_at": time.time(),
            "last_accessed": time.time(),
            "metadata": metadata or {},
            "state": {}
        }
        self.session_workers[session_id] = worker_id
        
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        if session_id in self.sessions:
            self.sessions[session_id]["last_accessed"] = time.time()
            return self.sessions[session_id]
        return None
    
    def get_session_worker(self, session_id: str) -> Optional[str]:
        return self.session_workers.get(session_id)
    
    def update_session_state(self, session_id: str, state: Dict[str, Any]):
        if session_id in self.sessions:
            self.sessions[session_id]["state"].update(state)
            self.sessions[session_id]["last_accessed"] = time.time()
    
    def cleanup_expired_sessions(self, max_age: float = 3600):
        current_time = time.time()
        expired = [
            sid for sid, session in self.sessions.items()
            if current_time - session["last_accessed"] > max_age
        ]
        for sid in expired:
            self.delete_session(sid)
    
    def delete_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]
        if session_id in self.session_workers:
            del self.session_workers[session_id]

class HealthChecker:
    def __init__(self, check_interval: float = 30.0, timeout: float = 10.0):
        self.check_interval = check_interval
        self.timeout = timeout
        self.unhealthy_threshold = 3
        
    async def check_worker_health(self, worker: WorkerNode) -> bool:
        """Check if worker is healthy"""
        try:
            # Check if worker has sent heartbeat recently
            if time.time() - worker.last_heartbeat > self.timeout * 2:
                return False
            
            # If worker has a browser context, check if it's responsive
            if worker.browser_context:
                # Simple health check - try to get page title
                try:
                    page = await worker.browser_context.new_page()
                    await page.goto("about:blank", timeout=self.timeout)
                    title = await page.title()
                    await page.close()
                    return True
                except Exception:
                    return False
            return True
        except Exception:
            return False
    
    async def monitor_workers(self, coordinator: 'BrowserFarmCoordinator'):
        """Continuously monitor worker health"""
        while True:
            await asyncio.sleep(self.check_interval)
            
            for worker_id, worker in list(coordinator.workers.items()):
                is_healthy = await self.check_worker_health(worker)
                
                if not is_healthy:
                    worker.status = WorkerStatus.UNHEALTHY
                    logger.warning(f"Worker {worker_id} is unhealthy")
                    
                    # If worker has a current task, mark it for retry
                    if worker.current_task_id:
                        task = coordinator.tasks.get(worker.current_task_id)
                        if task and task.status == TaskStatus.RUNNING:
                            task.status = TaskStatus.FAILED
                            task.error = "Worker became unhealthy"
                            coordinator.task_queue.put_nowait(task)
                    
                    # Try to recover the worker
                    await coordinator.recover_worker(worker_id)
                else:
                    if worker.status == WorkerStatus.UNHEALTHY:
                        worker.status = WorkerStatus.IDLE
                        logger.info(f"Worker {worker_id} recovered")

class BrowserFarmCoordinator:
    def __init__(
        self,
        max_workers: int = 10,
        task_timeout: float = 300.0,
        load_balancing_strategy: str = "least_loaded",
        enable_cloud_scaling: bool = False,
        cloud_config: Optional[Dict[str, Any]] = None
    ):
        self.workers: Dict[str, WorkerNode] = {}
        self.tasks: Dict[str, BrowserTask] = {}
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.max_workers = max_workers
        self.task_timeout = task_timeout
        self.load_balancer = LoadBalancer(load_balancing_strategy)
        self.session_manager = SessionManager()
        self.health_checker = HealthChecker()
        self.enable_cloud_scaling = enable_cloud_scaling
        self.cloud_config = cloud_config or {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.running = False
        self.task_counter = 0
        self.worker_counter = 0
        self._lock = asyncio.Lock()
        
        # Metrics
        self.metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0,
            "worker_utilization": 0.0
        }
        
        # Cloud provider interfaces
        self.cloud_providers = {}
        if enable_cloud_scaling:
            self._init_cloud_providers()
    
    def _init_cloud_providers(self):
        """Initialize cloud provider interfaces"""
        try:
            if "aws" in self.cloud_config:
                from nexus.farm.cloud.aws import AWSProvider
                self.cloud_providers["aws"] = AWSProvider(self.cloud_config["aws"])
            if "gcp" in self.cloud_config:
                from nexus.farm.cloud.gcp import GCPProvider
                self.cloud_providers["gcp"] = GCPProvider(self.cloud_config["gcp"])
        except ImportError as e:
            logger.warning(f"Cloud provider not available: {e}")
    
    async def start(self):
        """Start the coordinator"""
        self.running = True
        logger.info("Starting Browser Farm Coordinator")
        
        # Start background tasks
        asyncio.create_task(self._process_tasks())
        asyncio.create_task(self.health_checker.monitor_workers(self))
        asyncio.create_task(self._monitor_scaling())
        asyncio.create_task(self._cleanup_sessions())
        
        # Initialize local workers
        await self._initialize_local_workers()
    
    async def stop(self):
        """Stop the coordinator"""
        self.running = False
        logger.info("Stopping Browser Farm Coordinator")
        
        # Stop all workers
        for worker_id in list(self.workers.keys()):
            await self.remove_worker(worker_id)
        
        # Shutdown cloud instances
        if self.enable_cloud_scaling:
            await self._shutdown_cloud_instances()
        
        self.executor.shutdown(wait=True)
    
    async def _initialize_local_workers(self, count: int = 3):
        """Initialize local browser workers"""
        for i in range(min(count, self.max_workers)):
            await self.add_worker(region="local")
    
    async def add_worker(
        self,
        region: str = "local",
        cloud_provider: Optional[str] = None,
        instance_type: Optional[str] = None,
        capabilities: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a new worker to the farm"""
        async with self._lock:
            worker_id = f"worker_{self.worker_counter}_{uuid.uuid4().hex[:8]}"
            self.worker_counter += 1
            
            if cloud_provider and cloud_provider in self.cloud_providers:
                # Create cloud instance
                instance_id = await self.cloud_providers[cloud_provider].create_instance(
                    instance_type=instance_type or "small",
                    region=region
                )
                worker = WorkerNode(
                    worker_id=worker_id,
                    region=region,
                    cloud_provider=cloud_provider,
                    instance_id=instance_id,
                    capabilities=capabilities or {}
                )
            else:
                # Create local worker
                worker = WorkerNode(
                    worker_id=worker_id,
                    region=region,
                    capabilities=capabilities or {}
                )
                # Initialize browser context for local worker
                try:
                    browser_context = await BrowserContext.create()
                    worker.browser_context = browser_context
                    agent = Agent(
                        config=AgentConfig(
                            browser_context=browser_context,
                            headless=True
                        )
                    )
                    worker.agent = agent
                except Exception as e:
                    logger.error(f"Failed to initialize worker {worker_id}: {e}")
                    return None
            
            self.workers[worker_id] = worker
            logger.info(f"Added worker {worker_id} in region {region}")
            
            # Start worker loop
            asyncio.create_task(self._worker_loop(worker_id))
            
            return worker_id
    
    async def remove_worker(self, worker_id: str):
        """Remove a worker from the farm"""
        async with self._lock:
            if worker_id not in self.workers:
                return
            
            worker = self.workers[worker_id]
            
            # If worker has a current task, mark it for retry
            if worker.current_task_id:
                task = self.tasks.get(worker.current_task_id)
                if task and task.status == TaskStatus.RUNNING:
                    task.status = TaskStatus.FAILED
                    task.error = "Worker removed"
                    await self.retry_task(task.task_id)
            
            # Cleanup browser context
            if worker.browser_context:
                try:
                    await worker.browser_context.close()
                except Exception:
                    pass
            
            # Terminate cloud instance
            if worker.cloud_provider and worker.instance_id:
                if worker.cloud_provider in self.cloud_providers:
                    await self.cloud_providers[worker.cloud_provider].terminate_instance(
                        worker.instance_id
                    )
            
            del self.workers[worker_id]
            logger.info(f"Removed worker {worker_id}")
    
    async def submit_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: int = 0,
        timeout: Optional[float] = None,
        session_id: Optional[str] = None,
        dependencies: Optional[List[str]] = None
    ) -> str:
        """Submit a new task to the farm"""
        task_id = f"task_{self.task_counter}_{uuid.uuid4().hex[:8]}"
        self.task_counter += 1
        
        task = BrowserTask(
            task_id=task_id,
            task_type=task_type,
            payload=payload,
            priority=priority,
            timeout=timeout or self.task_timeout,
            session_id=session_id,
            dependencies=dependencies or []
        )
        
        self.tasks[task_id] = task
        await self.task_queue.put((-priority, task_id))  # Higher priority first
        
        logger.info(f"Submitted task {task_id} of type {task_type}")
        return task_id
    
    async def _process_tasks(self):
        """Main task processing loop"""
        while self.running:
            try:
                # Get next task from queue
                priority, task_id = await asyncio.wait_for(
                    self.task_queue.get(),
                    timeout=1.0
                )
                
                task = self.tasks.get(task_id)
                if not task:
                    continue
                
                # Check dependencies
                if not await self._check_dependencies(task):
                    # Requeue task with lower priority
                    await self.task_queue.put((priority - 1, task_id))
                    await asyncio.sleep(1)
                    continue
                
                # Select worker for task
                worker_id = self.load_balancer.select_worker(self.workers, task)
                if not worker_id:
                    # No available workers, requeue
                    await self.task_queue.put((priority, task_id))
                    
                    # Try to scale up if enabled
                    if self.enable_cloud_scaling:
                        await self._scale_up()
                    
                    await asyncio.sleep(1)
                    continue
                
                # Assign task to worker
                await self._assign_task_to_worker(task_id, worker_id)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing tasks: {e}")
                await asyncio.sleep(1)
    
    async def _assign_task_to_worker(self, task_id: str, worker_id: str):
        """Assign a task to a specific worker"""
        task = self.tasks[task_id]
        worker = self.workers[worker_id]
        
        task.status = TaskStatus.RUNNING
        task.worker_id = worker_id
        task.started_at = time.time()
        
        worker.status = WorkerStatus.BUSY
        worker.current_task_id = task_id
        
        # Update load score
        busy_workers = sum(1 for w in self.workers.values() if w.status == WorkerStatus.BUSY)
        load_score = busy_workers / len(self.workers) if self.workers else 0
        self.load_balancer.update_load_score(worker_id, load_score)
        
        logger.info(f"Assigned task {task_id} to worker {worker_id}")
    
    async def _worker_loop(self, worker_id: str):
        """Main loop for a worker"""
        worker = self.workers.get(worker_id)
        if not worker:
            return
        
        while self.running and worker_id in self.workers:
            try:
                if worker.current_task_id:
                    task = self.tasks.get(worker.current_task_id)
                    if task:
                        await self._execute_task(worker, task)
                    worker.current_task_id = None
                    worker.status = WorkerStatus.IDLE
                else:
                    # Send heartbeat
                    worker.last_heartbeat = time.time()
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                worker.status = WorkerStatus.UNHEALTHY
                await asyncio.sleep(5)
    
    async def _execute_task(self, worker: WorkerNode, task: BrowserTask):
        """Execute a task on a worker"""
        try:
            # Handle session persistence
            if task.session_id:
                session = self.session_manager.get_session(task.session_id)
                if session and worker.agent:
                    # Restore session state
                    if "cookies" in session["state"]:
                        await worker.agent.browser_context.add_cookies(
                            session["state"]["cookies"]
                        )
            
            # Execute task based on type
            if task.task_type == "navigate":
                result = await self._execute_navigate_task(worker, task)
            elif task.task_type == "scrape":
                result = await self._execute_scrape_task(worker, task)
            elif task.task_type == "interact":
                result = await self._execute_interact_task(worker, task)
            elif task.task_type == "custom":
                result = await self._execute_custom_task(worker, task)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            
            # Update session state
            if task.session_id and worker.agent:
                cookies = await worker.agent.browser_context.get_cookies()
                self.session_manager.update_session_state(
                    task.session_id,
                    {"cookies": cookies}
                )
            
            # Update metrics
            execution_time = task.completed_at - task.started_at
            self.metrics["tasks_completed"] += 1
            self.metrics["total_execution_time"] += execution_time
            
            logger.info(f"Task {task.task_id} completed in {execution_time:.2f}s")
            
        except asyncio.TimeoutError:
            task.status = TaskStatus.TIMEOUT
            task.error = "Task execution timed out"
            logger.warning(f"Task {task.task_id} timed out")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            self.metrics["tasks_failed"] += 1
            logger.error(f"Task {task.task_id} failed: {e}")
            
            # Retry task if retries remaining
            if task.retry_count < task.max_retries:
                await self.retry_task(task.task_id)
    
    async def _execute_navigate_task(self, worker: WorkerNode, task: BrowserTask) -> Any:
        """Execute navigation task"""
        url = task.payload["url"]
        timeout = task.payload.get("timeout", 30)
        
        page = await worker.browser_context.new_page()
        try:
            await page.goto(url, timeout=timeout)
            title = await page.title()
            content = await page.content()
            
            return {
                "title": title,
                "url": url,
                "content_length": len(content)
            }
        finally:
            await page.close()
    
    async def _execute_scrape_task(self, worker: WorkerNode, task: BrowserTask) -> Any:
        """Execute scraping task"""
        url = task.payload["url"]
        selectors = task.payload.get("selectors", {})
        
        page = await worker.browser_context.new_page()
        try:
            await page.goto(url)
            
            results = {}
            for name, selector in selectors.items():
                elements = await page.query_selector_all(selector)
                results[name] = [await el.text_content() for el in elements]
            
            return results
        finally:
            await page.close()
    
    async def _execute_interact_task(self, worker: WorkerNode, task: BrowserTask) -> Any:
        """Execute interaction task"""
        url = task.payload["url"]
        actions = task.payload["actions"]
        
        if not worker.agent:
            raise ValueError("Worker agent not initialized")
        
        # Use agent to perform actions
        result = await worker.agent.run(
            url=url,
            actions=actions
        )
        
        return result
    
    async def _execute_custom_task(self, worker: WorkerNode, task: BrowserTask) -> Any:
        """Execute custom task using provided function"""
        func_path = task.payload["function"]
        args = task.payload.get("args", [])
        kwargs = task.payload.get("kwargs", {})
        
        # Import and execute custom function
        import importlib
        module_path, func_name = func_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        func = getattr(module, func_name)
        
        # Execute in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            lambda: func(worker, *args, **kwargs)
        )
        
        return result
    
    async def retry_task(self, task_id: str):
        """Retry a failed task"""
        task = self.tasks.get(task_id)
        if not task:
            return
        
        task.retry_count += 1
        task.status = TaskStatus.PENDING
        task.started_at = None
        task.completed_at = None
        task.error = None
        
        # Requeue with lower priority
        await self.task_queue.put((-task.priority + task.retry_count, task_id))
        logger.info(f"Retrying task {task_id} (attempt {task.retry_count})")
    
    async def _check_dependencies(self, task: BrowserTask) -> bool:
        """Check if task dependencies are satisfied"""
        for dep_id in task.dependencies:
            dep_task = self.tasks.get(dep_id)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                return False
        return True
    
    async def _scale_up(self):
        """Scale up by adding more workers"""
        if len(self.workers) >= self.max_workers:
            return
        
        # Check if we should scale based on load
        busy_workers = sum(1 for w in self.workers.values() if w.status == WorkerStatus.BUSY)
        total_workers = len(self.workers)
        
        if total_workers == 0 or busy_workers / total_workers > 0.8:
            # Add cloud worker if enabled
            if self.enable_cloud_scaling and self.cloud_providers:
                provider = list(self.cloud_providers.keys())[0]
                await self.add_worker(
                    cloud_provider=provider,
                    region="us-east-1"  # Default region
                )
            else:
                # Add local worker
                await self.add_worker()
    
    async def _scale_down(self):
        """Scale down by removing idle workers"""
        idle_workers = [
            w for w in self.workers.values()
            if w.status == WorkerStatus.IDLE and w.cloud_provider
        ]
        
        # Keep at least 2 workers
        if len(self.workers) - len(idle_workers) >= 2:
            for worker in idle_workers[:2]:  # Remove up to 2 idle cloud workers
                await self.remove_worker(worker.worker_id)
    
    async def _monitor_scaling(self):
        """Monitor and adjust scaling"""
        while self.running:
            await asyncio.sleep(60)  # Check every minute
            
            try:
                # Update metrics
                busy_workers = sum(1 for w in self.workers.values() if w.status == WorkerStatus.BUSY)
                total_workers = len(self.workers)
                self.metrics["worker_utilization"] = busy_workers / total_workers if total_workers > 0 else 0
                
                # Scale based on utilization
                if self.metrics["worker_utilization"] > 0.8:
                    await self._scale_up()
                elif self.metrics["worker_utilization"] < 0.2:
                    await self._scale_down()
                    
            except Exception as e:
                logger.error(f"Error in scaling monitor: {e}")
    
    async def _cleanup_sessions(self):
        """Clean up expired sessions"""
        while self.running:
            await asyncio.sleep(300)  # Every 5 minutes
            self.session_manager.cleanup_expired_sessions()
    
    async def _shutdown_cloud_instances(self):
        """Shutdown all cloud instances"""
        for worker in list(self.workers.values()):
            if worker.cloud_provider:
                await self.remove_worker(worker.worker_id)
    
    async def recover_worker(self, worker_id: str):
        """Attempt to recover an unhealthy worker"""
        worker = self.workers.get(worker_id)
        if not worker:
            return
        
        logger.info(f"Attempting to recover worker {worker_id}")
        
        # Try to restart browser context
        if worker.browser_context:
            try:
                await worker.browser_context.close()
            except Exception:
                pass
        
        # Create new browser context
        try:
            browser_context = await BrowserContext.create()
            worker.browser_context = browser_context
            
            if worker.agent:
                worker.agent.browser_context = browser_context
            
            worker.status = WorkerStatus.IDLE
            worker.last_heartbeat = time.time()
            logger.info(f"Worker {worker_id} recovered successfully")
            
        except Exception as e:
            logger.error(f"Failed to recover worker {worker_id}: {e}")
            # Remove and replace worker
            await self.remove_worker(worker_id)
            await self.add_worker(
                region=worker.region,
                cloud_provider=worker.cloud_provider,
                capabilities=worker.capabilities
            )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current farm status"""
        return {
            "workers": {
                worker_id: worker.to_dict()
                for worker_id, worker in self.workers.items()
            },
            "tasks": {
                task_id: task.to_dict()
                for task_id, task in self.tasks.items()
            },
            "metrics": self.metrics,
            "queue_size": self.task_queue.qsize(),
            "total_workers": len(self.workers),
            "busy_workers": sum(1 for w in self.workers.values() if w.status == WorkerStatus.BUSY)
        }
    
    async def get_task_result(self, task_id: str, timeout: float = 30.0) -> Optional[Any]:
        """Wait for and return task result"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            task = self.tasks.get(task_id)
            if not task:
                return None
            
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.TIMEOUT]:
                return task.result if task.status == TaskStatus.COMPLETED else None
            
            await asyncio.sleep(0.5)
        
        return None  # Timeout

# Global coordinator instance
_coordinator_instance: Optional[BrowserFarmCoordinator] = None

def get_coordinator(**kwargs) -> BrowserFarmCoordinator:
    """Get or create global coordinator instance"""
    global _coordinator_instance
    if _coordinator_instance is None:
        _coordinator_instance = BrowserFarmCoordinator(**kwargs)
    return _coordinator_instance

async def initialize_farm(**kwargs) -> BrowserFarmCoordinator:
    """Initialize and start the browser farm"""
    coordinator = get_coordinator(**kwargs)
    await coordinator.start()
    return coordinator