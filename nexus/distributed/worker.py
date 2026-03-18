"""
Distributed Task Orchestration for nexus
Enables horizontal scaling across multiple machines with fault tolerance,
task sharding, and result aggregation.
"""
import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Awaitable, Union
from dataclasses import dataclass, asdict, field
import pickle
import hashlib

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not installed. Install with: pip install redis")

try:
    import aio_pika
    RABBITMQ_AVAILABLE = True
except ImportError:
    RABBITMQ_AVAILABLE = False
    logging.warning("RabbitMQ not installed. Install with: pip install aio-pika")

from ..agent.service import Agent
from ..agent.views import AgentResult, AgentState
from ..actor.page import Page


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class WorkerStatus(str, Enum):
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"
    ERROR = "error"


@dataclass
class TaskCheckpoint:
    """Checkpoint data for long-running tasks"""
    task_id: str
    step_index: int
    state: Dict[str, Any]
    timestamp: datetime
    url: Optional[str] = None
    screenshot_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskCheckpoint':
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class DistributedTask:
    """Represents a task in the distributed system"""
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    priority: int = 0
    max_retries: int = 3
    retry_count: int = 0
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    worker_id: Optional[str] = None
    checkpoint: Optional[TaskCheckpoint] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    shard_id: Optional[str] = None
    parent_task_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        if self.started_at:
            data['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        if self.checkpoint:
            data['checkpoint'] = self.checkpoint.to_dict()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DistributedTask':
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('started_at'):
            data['started_at'] = datetime.fromisoformat(data['started_at'])
        if data.get('completed_at'):
            data['completed_at'] = datetime.fromisoformat(data['completed_at'])
        if data.get('checkpoint'):
            data['checkpoint'] = TaskCheckpoint.from_dict(data['checkpoint'])
        return cls(**data)


@dataclass
class WorkerInfo:
    """Information about a worker node"""
    worker_id: str
    status: WorkerStatus
    capabilities: List[str]
    current_task_id: Optional[str] = None
    last_heartbeat: datetime = field(default_factory=datetime.now)
    tasks_completed: int = 0
    tasks_failed: int = 0
    hostname: Optional[str] = None
    ip_address: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['last_heartbeat'] = self.last_heartbeat.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkerInfo':
        data['last_heartbeat'] = datetime.fromisoformat(data['last_heartbeat'])
        return cls(**data)


class TaskQueueBackend:
    """Abstract base class for task queue backends"""
    
    async def connect(self) -> None:
        raise NotImplementedError
    
    async def disconnect(self) -> None:
        raise NotImplementedError
    
    async def push_task(self, task: DistributedTask) -> None:
        raise NotImplementedError
    
    async def pop_task(self, worker_id: str, capabilities: List[str] = None) -> Optional[DistributedTask]:
        raise NotImplementedError
    
    async def complete_task(self, task_id: str, result: Dict[str, Any]) -> None:
        raise NotImplementedError
    
    async def fail_task(self, task_id: str, error: str, retry: bool = True) -> None:
        raise NotImplementedError
    
    async def get_task(self, task_id: str) -> Optional[DistributedTask]:
        raise NotImplementedError
    
    async def update_checkpoint(self, task_id: str, checkpoint: TaskCheckpoint) -> None:
        raise NotImplementedError
    
    async def register_worker(self, worker: WorkerInfo) -> None:
        raise NotImplementedError
    
    async def update_worker_heartbeat(self, worker_id: str) -> None:
        raise NotImplementedError
    
    async def get_worker(self, worker_id: str) -> Optional[WorkerInfo]:
        raise NotImplementedError
    
    async def get_all_workers(self) -> List[WorkerInfo]:
        raise NotImplementedError


class RedisBackend(TaskQueueBackend):
    """Redis-based task queue backend"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", 
                 task_queue_key: str = "nexus:tasks",
                 task_prefix: str = "nexus:task:",
                 worker_prefix: str = "nexus:worker:"):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis is not installed")
        
        self.redis_url = redis_url
        self.task_queue_key = task_queue_key
        self.task_prefix = task_prefix
        self.worker_prefix = worker_prefix
        self.redis: Optional[redis.Redis] = None
    
    async def connect(self) -> None:
        self.redis = redis.from_url(self.redis_url, decode_responses=True)
        await self.redis.ping()
    
    async def disconnect(self) -> None:
        if self.redis:
            await self.redis.close()
    
    async def push_task(self, task: DistributedTask) -> None:
        task_key = f"{self.task_prefix}{task.task_id}"
        task_data = json.dumps(task.to_dict())
        
        # Store task data
        await self.redis.set(task_key, task_data)
        
        # Add to priority queue (sorted set with priority as score)
        await self.redis.zadd(self.task_queue_key, {task.task_id: task.priority})
        
        # Set expiration for completed/failed tasks (7 days)
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            await self.redis.expire(task_key, 7 * 24 * 3600)
    
    async def pop_task(self, worker_id: str, capabilities: List[str] = None) -> Optional[DistributedTask]:
        # Get highest priority pending task
        task_ids = await self.redis.zrevrange(self.task_queue_key, 0, 0)
        
        if not task_ids:
            return None
        
        task_id = task_ids[0]
        task_key = f"{self.task_prefix}{task_id}"
        
        # Get task data
        task_data = await self.redis.get(task_key)
        if not task_data:
            await self.redis.zrem(self.task_queue_key, task_id)
            return None
        
        task = DistributedTask.from_dict(json.loads(task_data))
        
        # Check if task is still pending
        if task.status != TaskStatus.PENDING:
            await self.redis.zrem(self.task_queue_key, task_id)
            return None
        
        # Check capabilities if specified
        if capabilities and task.task_type not in capabilities:
            return None
        
        # Update task status
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        task.worker_id = worker_id
        
        # Save updated task
        await self.redis.set(task_key, json.dumps(task.to_dict()))
        
        # Remove from queue
        await self.redis.zrem(self.task_queue_key, task_id)
        
        return task
    
    async def complete_task(self, task_id: str, result: Dict[str, Any]) -> None:
        task_key = f"{self.task_prefix}{task_id}"
        task_data = await self.redis.get(task_key)
        
        if not task_data:
            return
        
        task = DistributedTask.from_dict(json.loads(task_data))
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now()
        task.result = result
        
        await self.redis.set(task_key, json.dumps(task.to_dict()))
        await self.redis.expire(task_key, 7 * 24 * 3600)  # 7 days
    
    async def fail_task(self, task_id: str, error: str, retry: bool = True) -> None:
        task_key = f"{self.task_prefix}{task_id}"
        task_data = await self.redis.get(task_key)
        
        if not task_data:
            return
        
        task = DistributedTask.from_dict(json.loads(task_data))
        task.error = error
        
        if retry and task.retry_count < task.max_retries:
            task.status = TaskStatus.RETRYING
            task.retry_count += 1
            task.worker_id = None
            
            # Exponential backoff: 2^retry_count seconds
            delay = 2 ** task.retry_count
            task.priority = int(time.time()) + delay  # Re-queue with delayed priority
            
            await self.redis.set(task_key, json.dumps(task.to_dict()))
            await self.redis.zadd(self.task_queue_key, {task.task_id: task.priority})
        else:
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            await self.redis.set(task_key, json.dumps(task.to_dict()))
            await self.redis.expire(task_key, 7 * 24 * 3600)
    
    async def get_task(self, task_id: str) -> Optional[DistributedTask]:
        task_key = f"{self.task_prefix}{task_id}"
        task_data = await self.redis.get(task_key)
        
        if not task_data:
            return None
        
        return DistributedTask.from_dict(json.loads(task_data))
    
    async def update_checkpoint(self, task_id: str, checkpoint: TaskCheckpoint) -> None:
        task_key = f"{self.task_prefix}{task_id}"
        task_data = await self.redis.get(task_key)
        
        if not task_data:
            return
        
        task = DistributedTask.from_dict(json.loads(task_data))
        task.checkpoint = checkpoint
        
        await self.redis.set(task_key, json.dumps(task.to_dict()))
    
    async def register_worker(self, worker: WorkerInfo) -> None:
        worker_key = f"{self.worker_prefix}{worker.worker_id}"
        await self.redis.set(worker_key, json.dumps(worker.to_dict()))
        await self.redis.expire(worker_key, 30)  # 30 second TTL
    
    async def update_worker_heartbeat(self, worker_id: str) -> None:
        worker_key = f"{self.worker_prefix}{worker_id}"
        worker_data = await self.redis.get(worker_key)
        
        if not worker_data:
            return
        
        worker = WorkerInfo.from_dict(json.loads(worker_data))
        worker.last_heartbeat = datetime.now()
        
        await self.redis.set(worker_key, json.dumps(worker.to_dict()))
        await self.redis.expire(worker_key, 30)  # Reset TTL
    
    async def get_worker(self, worker_id: str) -> Optional[WorkerInfo]:
        worker_key = f"{self.worker_prefix}{worker_id}"
        worker_data = await self.redis.get(worker_key)
        
        if not worker_data:
            return None
        
        return WorkerInfo.from_dict(json.loads(worker_data))
    
    async def get_all_workers(self) -> List[WorkerInfo]:
        pattern = f"{self.worker_prefix}*"
        keys = []
        
        async for key in self.redis.scan_iter(match=pattern):
            keys.append(key)
        
        workers = []
        for key in keys:
            worker_data = await self.redis.get(key)
            if worker_data:
                workers.append(WorkerInfo.from_dict(json.loads(worker_data)))
        
        return workers


class RabbitMQBackend(TaskQueueBackend):
    """RabbitMQ-based task queue backend"""
    
    def __init__(self, rabbitmq_url: str = "amqp://guest:guest@localhost:5672/",
                 task_queue_name: str = "nexus_tasks",
                 result_exchange: str = "nexus_results"):
        if not RABBITMQ_AVAILABLE:
            raise ImportError("RabbitMQ is not installed")
        
        self.rabbitmq_url = rabbitmq_url
        self.task_queue_name = task_queue_name
        self.result_exchange = result_exchange
        self.connection = None
        self.channel = None
        self.task_queue = None
        self.result_queue = None
    
    async def connect(self) -> None:
        self.connection = await aio_pika.connect_robust(self.rabbitmq_url)
        self.channel = await self.connection.channel()
        
        # Declare task queue
        self.task_queue = await self.channel.declare_queue(
            self.task_queue_name,
            durable=True,
            arguments={'x-max-priority': 10}  # Support priorities
        )
        
        # Declare result exchange and queue
        await self.channel.declare_exchange(self.result_exchange, aio_pika.ExchangeType.TOPIC)
        self.result_queue = await self.channel.declare_queue(
            f"{self.task_queue_name}_results",
            durable=True
        )
        await self.result_queue.bind(self.result_exchange, routing_key="task.*")
    
    async def disconnect(self) -> None:
        if self.connection:
            await self.connection.close()
    
    async def push_task(self, task: DistributedTask) -> None:
        message = aio_pika.Message(
            body=json.dumps(task.to_dict()).encode(),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            priority=task.priority
        )
        
        await self.channel.default_exchange.publish(
            message,
            routing_key=self.task_queue_name
        )
    
    async def pop_task(self, worker_id: str, capabilities: List[str] = None) -> Optional[DistributedTask]:
        # RabbitMQ uses consumer-based approach, this would be handled differently
        # For now, return None as this is a simplified implementation
        return None
    
    async def complete_task(self, task_id: str, result: Dict[str, Any]) -> None:
        message = aio_pika.Message(
            body=json.dumps({
                'task_id': task_id,
                'status': TaskStatus.COMPLETED,
                'result': result,
                'timestamp': datetime.now().isoformat()
            }).encode(),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT
        )
        
        await self.channel.default_exchange.publish(
            message,
            routing_key=f"task.{task_id}.completed"
        )
    
    async def fail_task(self, task_id: str, error: str, retry: bool = True) -> None:
        message = aio_pika.Message(
            body=json.dumps({
                'task_id': task_id,
                'status': TaskStatus.FAILED,
                'error': error,
                'retry': retry,
                'timestamp': datetime.now().isoformat()
            }).encode(),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT
        )
        
        await self.channel.default_exchange.publish(
            message,
            routing_key=f"task.{task_id}.failed"
        )
    
    async def get_task(self, task_id: str) -> Optional[DistributedTask]:
        # Not directly supported in RabbitMQ, would need a separate store
        return None
    
    async def update_checkpoint(self, task_id: str, checkpoint: TaskCheckpoint) -> None:
        message = aio_pika.Message(
            body=json.dumps({
                'task_id': task_id,
                'checkpoint': checkpoint.to_dict(),
                'timestamp': datetime.now().isoformat()
            }).encode(),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT
        )
        
        await self.channel.default_exchange.publish(
            message,
            routing_key=f"task.{task_id}.checkpoint"
        )
    
    async def register_worker(self, worker: WorkerInfo) -> None:
        # Would need a separate registry service
        pass
    
    async def update_worker_heartbeat(self, worker_id: str) -> None:
        # Would need a separate registry service
        pass
    
    async def get_worker(self, worker_id: str) -> Optional[WorkerInfo]:
        # Would need a separate registry service
        return None
    
    async def get_all_workers(self) -> List[WorkerInfo]:
        # Would need a separate registry service
        return []


class DistributedWorker:
    """
    Distributed worker node for nexus tasks
    Can join/leave the cluster dynamically and handle browser automation tasks
    """
    
    def __init__(
        self,
        worker_id: Optional[str] = None,
        backend: Optional[TaskQueueBackend] = None,
        capabilities: List[str] = None,
        heartbeat_interval: int = 10,
        checkpoint_interval: int = 30,
        max_concurrent_tasks: int = 1,
        browser_pool_size: int = 3
    ):
        self.worker_id = worker_id or f"worker_{uuid.uuid4().hex[:8]}"
        self.backend = backend or RedisBackend()
        self.capabilities = capabilities or ["browser_automation", "scraping", "testing"]
        self.heartbeat_interval = heartbeat_interval
        self.checkpoint_interval = checkpoint_interval
        self.max_concurrent_tasks = max_concurrent_tasks
        self.browser_pool_size = browser_pool_size
        
        self.status = WorkerStatus.IDLE
        self.current_tasks: Dict[str, asyncio.Task] = {}
        self.browser_pool: List[Page] = []
        self.agent_pool: List[Agent] = []
        self.running = False
        self.logger = logging.getLogger(f"nexus.worker.{self.worker_id}")
        
        # Statistics
        self.stats = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_processing_time': 0,
            'start_time': datetime.now()
        }
    
    async def start(self) -> None:
        """Start the worker node"""
        self.logger.info(f"Starting worker {self.worker_id}")
        
        try:
            await self.backend.connect()
        except Exception as e:
            self.logger.error(f"Failed to connect to backend: {e}")
            self.status = WorkerStatus.ERROR
            return
        
        # Register worker
        worker_info = WorkerInfo(
            worker_id=self.worker_id,
            status=self.status,
            capabilities=self.capabilities,
            hostname=self._get_hostname()
        )
        await self.backend.register_worker(worker_info)
        
        # Initialize browser pool
        await self._initialize_browser_pool()
        
        self.running = True
        
        # Start background tasks
        asyncio.create_task(self._heartbeat_loop())
        asyncio.create_task(self._task_polling_loop())
        asyncio.create_task(self._checkpoint_loop())
        
        self.logger.info(f"Worker {self.worker_id} started successfully")
    
    async def stop(self) -> None:
        """Stop the worker node gracefully"""
        self.logger.info(f"Stopping worker {self.worker_id}")
        self.running = False
        self.status = WorkerStatus.OFFLINE
        
        # Cancel all running tasks
        for task_id, task in self.current_tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Close browser pool
        await self._close_browser_pool()
        
        # Update worker status
        worker_info = WorkerInfo(
            worker_id=self.worker_id,
            status=self.status,
            capabilities=self.capabilities,
            hostname=self._get_hostname()
        )
        await self.backend.register_worker(worker_info)
        
        await self.backend.disconnect()
        self.logger.info(f"Worker {self.worker_id} stopped")
    
    async def _initialize_browser_pool(self) -> None:
        """Initialize pool of browser instances"""
        try:
            from ..actor.page import Page
            from ..actor.utils import create_browser_context
            
            for i in range(self.browser_pool_size):
                context = await create_browser_context()
                page = await context.new_page()
                self.browser_pool.append(page)
                
                # Create agent for each page
                agent = Agent(page=page)
                self.agent_pool.append(agent)
            
            self.logger.info(f"Initialized browser pool with {len(self.browser_pool)} instances")
        except Exception as e:
            self.logger.error(f"Failed to initialize browser pool: {e}")
            raise
    
    async def _close_browser_pool(self) -> None:
        """Close all browser instances in the pool"""
        for page in self.browser_pool:
            try:
                await page.close()
            except Exception as e:
                self.logger.warning(f"Error closing browser page: {e}")
        
        self.browser_pool.clear()
        self.agent_pool.clear()
    
    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to the backend"""
        while self.running:
            try:
                await self.backend.update_worker_heartbeat(self.worker_id)
                await asyncio.sleep(self.heartbeat_interval)
            except Exception as e:
                self.logger.error(f"Heartbeat failed: {e}")
                await asyncio.sleep(5)  # Backoff on error
    
    async def _task_polling_loop(self) -> None:
        """Poll for new tasks and execute them"""
        while self.running:
            try:
                # Check if we can accept more tasks
                if len(self.current_tasks) >= self.max_concurrent_tasks:
                    await asyncio.sleep(1)
                    continue
                
                # Update status
                if self.current_tasks:
                    self.status = WorkerStatus.BUSY
                else:
                    self.status = WorkerStatus.IDLE
                
                # Try to get a task
                task = await self.backend.pop_task(self.worker_id, self.capabilities)
                
                if task:
                    self.logger.info(f"Received task {task.task_id} (type: {task.task_type})")
                    
                    # Execute task in background
                    task_coroutine = self._execute_task(task)
                    asyncio_task = asyncio.create_task(task_coroutine)
                    self.current_tasks[task.task_id] = asyncio_task
                    
                    # Clean up completed tasks
                    done_tasks = [t for t in self.current_tasks.values() if t.done()]
                    for done_task in done_tasks:
                        task_id = [k for k, v in self.current_tasks.items() if v == done_task][0]
                        del self.current_tasks[task_id]
                
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                self.logger.error(f"Task polling error: {e}")
                await asyncio.sleep(5)  # Backoff on error
    
    async def _checkpoint_loop(self) -> None:
        """Periodically save checkpoints for long-running tasks"""
        while self.running:
            try:
                await asyncio.sleep(self.checkpoint_interval)
                
                # Save checkpoints for all running tasks
                for task_id in list(self.current_tasks.keys()):
                    task = await self.backend.get_task(task_id)
                    if task and task.status == TaskStatus.RUNNING:
                        # Create checkpoint
                        checkpoint = TaskCheckpoint(
                            task_id=task_id,
                            step_index=task.checkpoint.step_index + 1 if task.checkpoint else 0,
                            state={'progress': 'in_progress'},  # Would be actual state in real implementation
                            timestamp=datetime.now()
                        )
                        
                        await self.backend.update_checkpoint(task_id, checkpoint)
                        self.logger.debug(f"Saved checkpoint for task {task_id}")
                
            except Exception as e:
                self.logger.error(f"Checkpoint error: {e}")
    
    async def _execute_task(self, task: DistributedTask) -> None:
        """Execute a browser automation task"""
        start_time = time.time()
        
        try:
            self.logger.info(f"Executing task {task.task_id}")
            
            # Get an available agent from the pool
            if not self.agent_pool:
                raise RuntimeError("No available browser agents")
            
            agent = self.agent_pool.pop(0)
            
            try:
                # Execute based on task type
                if task.task_type == "browser_automation":
                    result = await self._execute_browser_task(agent, task)
                elif task.task_type == "scraping":
                    result = await self._execute_scraping_task(agent, task)
                elif task.task_type == "testing":
                    result = await self._execute_testing_task(agent, task)
                else:
                    raise ValueError(f"Unknown task type: {task.task_type}")
                
                # Mark task as completed
                await self.backend.complete_task(task.task_id, result)
                
                # Update statistics
                self.stats['tasks_completed'] += 1
                processing_time = time.time() - start_time
                self.stats['total_processing_time'] += processing_time
                
                self.logger.info(f"Task {task.task_id} completed in {processing_time:.2f}s")
                
            finally:
                # Return agent to pool
                self.agent_pool.append(agent)
                
        except asyncio.CancelledError:
            self.logger.info(f"Task {task.task_id} was cancelled")
            raise
            
        except Exception as e:
            self.logger.error(f"Task {task.task_id} failed: {e}")
            
            # Mark task as failed (with retry)
            await self.backend.fail_task(task.task_id, str(e), retry=True)
            
            # Update statistics
            self.stats['tasks_failed'] += 1
    
    async def _execute_browser_task(self, agent: Agent, task: DistributedTask) -> Dict[str, Any]:
        """Execute a general browser automation task"""
        payload = task.payload
        
        # Navigate to URL if specified
        if 'url' in payload:
            await agent.page.goto(payload['url'])
        
        # Execute steps if provided
        results = []
        if 'steps' in payload:
            for step in payload['steps']:
                # Each step could be a function call or action
                if step['type'] == 'click':
                    await agent.page.click(step['selector'])
                elif step['type'] == 'fill':
                    await agent.page.fill(step['selector'], step['value'])
                elif step['type'] == 'screenshot':
                    screenshot = await agent.page.screenshot()
                    results.append({
                        'type': 'screenshot',
                        'data': screenshot,
                        'timestamp': datetime.now().isoformat()
                    })
                elif step['type'] == 'extract':
                    data = await agent.page.evaluate(step['script'])
                    results.append({
                        'type': 'data',
                        'data': data,
                        'timestamp': datetime.now().isoformat()
                    })
        
        # Take final screenshot
        final_screenshot = await agent.page.screenshot()
        
        return {
            'status': 'success',
            'results': results,
            'final_screenshot': final_screenshot,
            'url': agent.page.url,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _execute_scraping_task(self, agent: Agent, task: DistributedTask) -> Dict[str, Any]:
        """Execute a web scraping task"""
        payload = task.payload
        
        # Navigate to URL
        await agent.page.goto(payload['url'])
        
        # Wait for content if specified
        if 'wait_for' in payload:
            await agent.page.wait_for_selector(payload['wait_for'])
        
        # Extract data based on selectors
        extracted_data = {}
        if 'selectors' in payload:
            for field_name, selector in payload['selectors'].items():
                try:
                    elements = await agent.page.query_selector_all(selector)
                    if len(elements) == 1:
                        extracted_data[field_name] = await elements[0].text_content()
                    else:
                        extracted_data[field_name] = [await el.text_content() for el in elements]
                except Exception as e:
                    self.logger.warning(f"Failed to extract {field_name}: {e}")
                    extracted_data[field_name] = None
        
        # Execute custom extraction script if provided
        if 'extraction_script' in payload:
            custom_data = await agent.page.evaluate(payload['extraction_script'])
            extracted_data.update(custom_data)
        
        return {
            'status': 'success',
            'data': extracted_data,
            'url': agent.page.url,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _execute_testing_task(self, agent: Agent, task: DistributedTask) -> Dict[str, Any]:
        """Execute a browser testing task"""
        payload = task.payload
        
        # Navigate to URL
        await agent.page.goto(payload['url'])
        
        test_results = []
        
        # Run test cases
        if 'test_cases' in payload:
            for test_case in payload['test_cases']:
                test_result = {
                    'name': test_case['name'],
                    'passed': False,
                    'error': None,
                    'timestamp': datetime.now().isoformat()
                }
                
                try:
                    # Execute test steps
                    for step in test_case['steps']:
                        if step['action'] == 'assert_visible':
                            element = await agent.page.wait_for_selector(step['selector'], timeout=5000)
                            if not element:
                                raise AssertionError(f"Element {step['selector']} not visible")
                        
                        elif step['action'] == 'assert_text':
                            element = await agent.page.wait_for_selector(step['selector'])
                            text = await element.text_content()
                            if text != step['expected']:
                                raise AssertionError(f"Text mismatch: expected '{step['expected']}', got '{text}'")
                        
                        elif step['action'] == 'click':
                            await agent.page.click(step['selector'])
                        
                        elif step['action'] == 'fill':
                            await agent.page.fill(step['selector'], step['value'])
                    
                    test_result['passed'] = True
                    
                except Exception as e:
                    test_result['error'] = str(e)
                
                test_results.append(test_result)
        
        # Calculate summary
        passed = sum(1 for r in test_results if r['passed'])
        total = len(test_results)
        
        return {
            'status': 'success',
            'test_results': test_results,
            'summary': {
                'passed': passed,
                'total': total,
                'pass_rate': passed / total if total > 0 else 0
            },
            'url': agent.page.url,
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_hostname(self) -> str:
        """Get the hostname of the worker machine"""
        import socket
        try:
            return socket.gethostname()
        except:
            return "unknown"


class TaskSharder:
    """
    Handles sharding of large tasks across multiple workers
    and aggregation of results
    """
    
    def __init__(self, backend: TaskQueueBackend):
        self.backend = backend
    
    async def shard_task(
        self,
        parent_task: DistributedTask,
        shard_count: int,
        shard_strategy: str = "url_list"
    ) -> List[DistributedTask]:
        """
        Shard a large task into smaller subtasks
        
        Args:
            parent_task: The original task to shard
            shard_count: Number of shards to create
            shard_strategy: Strategy for sharding (url_list, pagination, etc.)
        
        Returns:
            List of sharded tasks
        """
        shards = []
        
        if shard_strategy == "url_list" and 'urls' in parent_task.payload:
            urls = parent_task.payload['urls']
            urls_per_shard = len(urls) // shard_count
            remainder = len(urls) % shard_count
            
            start_idx = 0
            for i in range(shard_count):
                # Calculate shard size (distribute remainder)
                shard_size = urls_per_shard + (1 if i < remainder else 0)
                end_idx = start_idx + shard_size
                
                shard_urls = urls[start_idx:end_idx]
                
                # Create shard task
                shard_task = DistributedTask(
                    task_id=f"{parent_task.task_id}_shard_{i}",
                    task_type=parent_task.task_type,
                    payload={
                        **parent_task.payload,
                        'urls': shard_urls,
                        'shard_index': i,
                        'total_shards': shard_count
                    },
                    priority=parent_task.priority,
                    max_retries=parent_task.max_retries,
                    shard_id=f"{parent_task.task_id}_shards",
                    parent_task_id=parent_task.task_id
                )
                
                shards.append(shard_task)
                start_idx = end_idx
        
        elif shard_strategy == "pagination" and 'total_pages' in parent_task.payload:
            total_pages = parent_task.payload['total_pages']
            pages_per_shard = total_pages // shard_count
            remainder = total_pages % shard_count
            
            start_page = 1
            for i in range(shard_count):
                shard_pages = pages_per_shard + (1 if i < remainder else 0)
                end_page = start_page + shard_pages - 1
                
                shard_task = DistributedTask(
                    task_id=f"{parent_task.task_id}_shard_{i}",
                    task_type=parent_task.task_type,
                    payload={
                        **parent_task.payload,
                        'start_page': start_page,
                        'end_page': end_page,
                        'shard_index': i,
                        'total_shards': shard_count
                    },
                    priority=parent_task.priority,
                    max_retries=parent_task.max_retries,
                    shard_id=f"{parent_task.task_id}_shards",
                    parent_task_id=parent_task.task_id
                )
                
                shards.append(shard_task)
                start_page = end_page + 1
        
        # Push all shards to the queue
        for shard in shards:
            await self.backend.push_task(shard)
        
        return shards
    
    async def aggregate_results(self, parent_task_id: str) -> Dict[str, Any]:
        """
        Aggregate results from all shards of a parent task
        
        Args:
            parent_task_id: ID of the parent task
        
        Returns:
            Aggregated results
        """
        # Get parent task
        parent_task = await self.backend.get_task(parent_task_id)
        if not parent_task:
            raise ValueError(f"Parent task {parent_task_id} not found")
        
        # Find all shard tasks
        # In a real implementation, we'd need to query for tasks with parent_task_id
        # For now, we'll assume we can get them by pattern
        all_workers = await self.backend.get_all_workers()
        
        # This is a simplified implementation
        # In production, you'd want to track shard completion in Redis or a database
        
        # For now, return a placeholder
        return {
            'parent_task_id': parent_task_id,
            'status': 'aggregation_pending',
            'message': 'Aggregation logic to be implemented based on backend capabilities'
        }


class DistributedOrchestrator:
    """
    Main orchestrator for distributed nexus tasks
    Manages task distribution, monitoring, and result aggregation
    """
    
    def __init__(self, backend: Optional[TaskQueueBackend] = None):
        self.backend = backend or RedisBackend()
        self.sharder = TaskSharder(self.backend)
        self.workers: Dict[str, DistributedWorker] = {}
        self.logger = logging.getLogger("nexus.orchestrator")
    
    async def start(self) -> None:
        """Start the orchestrator"""
        await self.backend.connect()
        self.logger.info("Distributed orchestrator started")
    
    async def stop(self) -> None:
        """Stop the orchestrator"""
        # Stop all managed workers
        for worker in self.workers.values():
            await worker.stop()
        
        await self.backend.disconnect()
        self.logger.info("Distributed orchestrator stopped")
    
    async def submit_task(self, task: DistributedTask) -> str:
        """Submit a task to the distributed system"""
        await self.backend.push_task(task)
        self.logger.info(f"Task {task.task_id} submitted to queue")
        return task.task_id
    
    async def submit_sharded_task(
        self,
        task: DistributedTask,
        shard_count: int,
        shard_strategy: str = "url_list"
    ) -> str:
        """Submit a task that will be automatically sharded"""
        # First, shard the task
        shards = await self.sharder.shard_task(task, shard_count, shard_strategy)
        
        # Then submit the parent task (for tracking)
        await self.backend.push_task(task)
        
        self.logger.info(f"Task {task.task_id} sharded into {len(shards)} subtasks")
        return task.task_id
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a task"""
        task = await self.backend.get_task(task_id)
        if not task:
            return None
        
        return {
            'task_id': task.task_id,
            'status': task.status,
            'created_at': task.created_at.isoformat(),
            'started_at': task.started_at.isoformat() if task.started_at else None,
            'completed_at': task.completed_at.isoformat() if task.completed_at else None,
            'worker_id': task.worker_id,
            'retry_count': task.retry_count,
            'error': task.error,
            'result': task.result
        }
    
    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get the status of the entire cluster"""
        workers = await self.backend.get_all_workers()
        
        return {
            'total_workers': len(workers),
            'active_workers': len([w for w in workers if w.status != WorkerStatus.OFFLINE]),
            'busy_workers': len([w for w in workers if w.status == WorkerStatus.BUSY]),
            'idle_workers': len([w for w in workers if w.status == WorkerStatus.IDLE]),
            'workers': [w.to_dict() for w in workers],
            'timestamp': datetime.now().isoformat()
        }
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task"""
        task = await self.backend.get_task(task_id)
        if not task:
            return False
        
        if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now()
            
            # Update in backend
            task_key = f"{self.backend.task_prefix}{task_id}"
            if hasattr(self.backend, 'redis'):
                await self.backend.redis.set(task_key, json.dumps(task.to_dict()))
            
            self.logger.info(f"Task {task_id} cancelled")
            return True
        
        return False


# Example usage and factory functions
async def create_redis_worker(
    redis_url: str = "redis://localhost:6379",
    worker_id: Optional[str] = None,
    capabilities: List[str] = None
) -> DistributedWorker:
    """Create a worker with Redis backend"""
    backend = RedisBackend(redis_url=redis_url)
    worker = DistributedWorker(
        worker_id=worker_id,
        backend=backend,
        capabilities=capabilities
    )
    return worker


async def create_rabbitmq_worker(
    rabbitmq_url: str = "amqp://guest:guest@localhost:5672/",
    worker_id: Optional[str] = None,
    capabilities: List[str] = None
) -> DistributedWorker:
    """Create a worker with RabbitMQ backend"""
    backend = RabbitMQBackend(rabbitmq_url=rabbitmq_url)
    worker = DistributedWorker(
        worker_id=worker_id,
        backend=backend,
        capabilities=capabilities
    )
    return worker


def create_browser_automation_task(
    url: str,
    steps: List[Dict[str, Any]],
    priority: int = 0,
    max_retries: int = 3
) -> DistributedTask:
    """Create a browser automation task"""
    return DistributedTask(
        task_id=f"browser_{uuid.uuid4().hex[:8]}",
        task_type="browser_automation",
        payload={
            'url': url,
            'steps': steps
        },
        priority=priority,
        max_retries=max_retries
    )


def create_scraping_task(
    url: str,
    selectors: Dict[str, str],
    extraction_script: Optional[str] = None,
    priority: int = 0,
    max_retries: int = 3
) -> DistributedTask:
    """Create a web scraping task"""
    return DistributedTask(
        task_id=f"scrape_{uuid.uuid4().hex[:8]}",
        task_type="scraping",
        payload={
            'url': url,
            'selectors': selectors,
            'extraction_script': extraction_script
        },
        priority=priority,
        max_retries=max_retries
    )


def create_testing_task(
    url: str,
    test_cases: List[Dict[str, Any]],
    priority: int = 0,
    max_retries: int = 3
) -> DistributedTask:
    """Create a browser testing task"""
    return DistributedTask(
        task_id=f"test_{uuid.uuid4().hex[:8]}",
        task_type="testing",
        payload={
            'url': url,
            'test_cases': test_cases
        },
        priority=priority,
        max_retries=max_retries
    )


# CLI entry point for running a worker
async def run_worker_cli():
    """Command-line interface for running a worker"""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Run a nexus distributed worker")
    parser.add_argument("--worker-id", help="Unique worker ID")
    parser.add_argument("--backend", choices=["redis", "rabbitmq"], default="redis",
                       help="Backend to use (default: redis)")
    parser.add_argument("--redis-url", default="redis://localhost:6379",
                       help="Redis URL (default: redis://localhost:6379)")
    parser.add_argument("--rabbitmq-url", default="amqp://guest:guest@localhost:5672/",
                       help="RabbitMQ URL (default: amqp://guest:guest@localhost:5672/)")
    parser.add_argument("--capabilities", nargs="+", 
                       default=["browser_automation", "scraping", "testing"],
                       help="Task capabilities this worker can handle")
    parser.add_argument("--max-concurrent", type=int, default=1,
                       help="Maximum concurrent tasks (default: 1)")
    parser.add_argument("--browser-pool", type=int, default=3,
                       help="Browser pool size (default: 3)")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create worker
    if args.backend == "redis":
        worker = await create_redis_worker(
            redis_url=args.redis_url,
            worker_id=args.worker_id,
            capabilities=args.capabilities
        )
    else:
        worker = await create_rabbitmq_worker(
            rabbitmq_url=args.rabbitmq_url,
            worker_id=args.worker_id,
            capabilities=args.capabilities
        )
    
    worker.max_concurrent_tasks = args.max_concurrent
    worker.browser_pool_size = args.browser_pool
    
    # Handle shutdown signals
    import signal
    
    def signal_handler():
        print("\nShutting down worker...")
        asyncio.create_task(worker.stop())
    
    if sys.platform != 'win32':
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)
    
    try:
        await worker.start()
        
        # Keep running until stopped
        while worker.running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received")
    finally:
        await worker.stop()


if __name__ == "__main__":
    asyncio.run(run_worker_cli())