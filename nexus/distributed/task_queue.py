"""
nexus/distributed/task_queue.py

Distributed Task Orchestration for nexus.
Enables horizontal scaling across multiple machines with fault tolerance,
task sharding, and result aggregation. Supports Redis/RabbitMQ backends.
"""

import asyncio
import json
import logging
import pickle
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import aio_pika
    RABBITMQ_AVAILABLE = True
except ImportError:
    RABBITMQ_AVAILABLE = False

from nexus.agent.service import Agent
from nexus.agent.views import AgentTask, AgentResult
from nexus.actor.page import PageActor
from nexus.actor.utils import retry_with_exponential_backoff

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Status of a distributed task."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    CHECKPOINTED = "checkpointed"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    AGGREGATING = "aggregating"


class TaskPriority(int, Enum):
    """Task priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class TaskCheckpoint:
    """Checkpoint for long-running tasks."""
    task_id: str
    checkpoint_id: str
    state: Dict[str, Any]
    progress: float  # 0.0 to 1.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DistributedTask:
    """A task to be executed in distributed environment."""
    task_id: str
    name: str
    func_name: str
    args: Tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: Optional[float] = None
    shard_id: Optional[str] = None
    shard_total: int = 1
    parent_task_id: Optional[str] = None
    checkpoint: Optional[TaskCheckpoint] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization."""
        data = asdict(self)
        # Handle non-serializable types
        if self.args:
            data['args'] = pickle.dumps(self.args)
        if self.result is not None:
            try:
                json.dumps(self.result)  # Test if JSON serializable
                data['result'] = self.result
            except (TypeError, ValueError):
                data['result'] = pickle.dumps(self.result)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DistributedTask':
        """Create task from dictionary."""
        if 'args' in data and isinstance(data['args'], bytes):
            data['args'] = pickle.loads(data['args'])
        if 'result' in data and isinstance(data['result'], bytes):
            data['result'] = pickle.loads(data['result'])
        return cls(**data)


@dataclass
class AggregatedResult:
    """Result from aggregated sharded tasks."""
    task_id: str
    results: List[Any]
    errors: List[Optional[str]]
    shard_count: int
    completed_shards: int
    failed_shards: int
    aggregated_result: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskBackend(ABC):
    """Abstract base class for task queue backends."""
    
    @abstractmethod
    async def connect(self) -> None:
        """Connect to the backend."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the backend."""
        pass
    
    @abstractmethod
    async def enqueue_task(self, task: DistributedTask) -> str:
        """Enqueue a task."""
        pass
    
    @abstractmethod
    async def dequeue_task(self, worker_id: str, timeout: float = 30.0) -> Optional[DistributedTask]:
        """Dequeue a task for a worker."""
        pass
    
    @abstractmethod
    async def update_task_status(self, task_id: str, status: TaskStatus, **kwargs) -> None:
        """Update task status."""
        pass
    
    @abstractmethod
    async def get_task(self, task_id: str) -> Optional[DistributedTask]:
        """Get task by ID."""
        pass
    
    @abstractmethod
    async def save_checkpoint(self, checkpoint: TaskCheckpoint) -> None:
        """Save task checkpoint."""
        pass
    
    @abstractmethod
    async def get_checkpoint(self, task_id: str) -> Optional[TaskCheckpoint]:
        """Get latest checkpoint for task."""
        pass
    
    @abstractmethod
    async def register_worker(self, worker_id: str, capabilities: Dict[str, Any]) -> None:
        """Register a worker."""
        pass
    
    @abstractmethod
    async def unregister_worker(self, worker_id: str) -> None:
        """Unregister a worker."""
        pass
    
    @abstractmethod
    async def get_workers(self) -> List[Dict[str, Any]]:
        """Get all registered workers."""
        pass


class RedisBackend(TaskBackend):
    """Redis-based task queue backend."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", 
                 queue_prefix: str = "nexus:tasks"):
        if not REDIS_AVAILABLE:
            raise ImportError("redis package not installed. Install with: pip install redis")
        
        self.redis_url = redis_url
        self.queue_prefix = queue_prefix
        self.redis: Optional[redis.Redis] = None
        self._queues: Dict[TaskPriority, str] = {
            TaskPriority.LOW: f"{queue_prefix}:queue:low",
            TaskPriority.NORMAL: f"{queue_prefix}:queue:normal",
            TaskPriority.HIGH: f"{queue_prefix}:queue:high",
            TaskPriority.CRITICAL: f"{queue_prefix}:queue:critical",
        }
    
    async def connect(self) -> None:
        """Connect to Redis."""
        self.redis = redis.from_url(self.redis_url, decode_responses=False)
        await self.redis.ping()
        logger.info("Connected to Redis backend")
    
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.redis:
            await self.redis.close()
            logger.info("Disconnected from Redis backend")
    
    async def enqueue_task(self, task: DistributedTask) -> str:
        """Enqueue task to Redis."""
        task.status = TaskStatus.QUEUED
        task_key = f"{self.queue_prefix}:task:{task.task_id}"
        
        # Store task data
        await self.redis.hset(task_key, mapping=task.to_dict())
        await self.redis.expire(task_key, 86400)  # 24 hours TTL
        
        # Add to priority queue
        queue_key = self._queues[task.priority]
        await self.redis.lpush(queue_key, task.task_id)
        
        # Update task index
        await self.redis.sadd(f"{self.queue_prefix}:tasks:all", task.task_id)
        
        logger.debug(f"Enqueued task {task.task_id} with priority {task.priority}")
        return task.task_id
    
    async def dequeue_task(self, worker_id: str, timeout: float = 30.0) -> Optional[DistributedTask]:
        """Dequeue task from Redis with priority support."""
        # Try queues in priority order
        for priority in [TaskPriority.CRITICAL, TaskPriority.HIGH, 
                        TaskPriority.NORMAL, TaskPriority.LOW]:
            queue_key = self._queues[priority]
            
            # Use BRPOPLPUSH for reliable queue pattern
            task_id = await self.redis.brpoplpush(
                queue_key, 
                f"{self.queue_prefix}:processing:{worker_id}",
                timeout=timeout
            )
            
            if task_id:
                task_id = task_id.decode() if isinstance(task_id, bytes) else task_id
                task = await self.get_task(task_id)
                
                if task:
                    task.status = TaskStatus.RUNNING
                    task.started_at = time.time()
                    task.metadata['worker_id'] = worker_id
                    await self.update_task_status(task_id, TaskStatus.RUNNING, 
                                                worker_id=worker_id, started_at=task.started_at)
                    
                    # Remove from processing queue after successful dequeue
                    await self.redis.lrem(f"{self.queue_prefix}:processing:{worker_id}", 1, task_id)
                    
                    logger.debug(f"Dequeued task {task_id} for worker {worker_id}")
                    return task
        
        return None
    
    async def update_task_status(self, task_id: str, status: TaskStatus, **kwargs) -> None:
        """Update task status in Redis."""
        task_key = f"{self.queue_prefix}:task:{task_id}"
        
        updates = {'status': status.value, 'updated_at': time.time()}
        updates.update(kwargs)
        
        # Handle special fields
        if 'result' in updates:
            result = updates.pop('result')
            try:
                json.dumps(result)
                updates['result'] = result
            except (TypeError, ValueError):
                updates['result'] = pickle.dumps(result)
        
        await self.redis.hset(task_key, mapping=updates)
        
        # Update status index
        await self.redis.sadd(f"{self.queue_prefix}:tasks:status:{status.value}", task_id)
        
        # Remove from other status sets
        for other_status in TaskStatus:
            if other_status != status:
                await self.redis.srem(f"{self.queue_prefix}:tasks:status:{other_status.value}", task_id)
    
    async def get_task(self, task_id: str) -> Optional[DistributedTask]:
        """Get task from Redis."""
        task_key = f"{self.queue_prefix}:task:{task_id}"
        data = await self.redis.hgetall(task_key)
        
        if not data:
            return None
        
        # Convert bytes to strings/objects
        decoded = {}
        for key, value in data.items():
            key = key.decode() if isinstance(key, bytes) else key
            if isinstance(value, bytes):
                try:
                    decoded[key] = value.decode()
                except UnicodeDecodeError:
                    decoded[key] = value
            else:
                decoded[key] = value
        
        # Handle special fields
        if 'args' in decoded and isinstance(decoded['args'], bytes):
            decoded['args'] = pickle.loads(decoded['args'])
        if 'result' in decoded and isinstance(decoded['result'], bytes):
            decoded['result'] = pickle.loads(decoded['result'])
        
        # Convert enum fields
        if 'status' in decoded:
            decoded['status'] = TaskStatus(decoded['status'])
        if 'priority' in decoded:
            decoded['priority'] = TaskPriority(int(decoded['priority']))
        
        return DistributedTask.from_dict(decoded)
    
    async def save_checkpoint(self, checkpoint: TaskCheckpoint) -> None:
        """Save checkpoint to Redis."""
        checkpoint_key = f"{self.queue_prefix}:checkpoint:{checkpoint.task_id}"
        checkpoint_data = {
            'checkpoint_id': checkpoint.checkpoint_id,
            'state': pickle.dumps(checkpoint.state),
            'progress': checkpoint.progress,
            'timestamp': checkpoint.timestamp,
            'metadata': json.dumps(checkpoint.metadata)
        }
        
        await self.redis.hset(checkpoint_key, mapping=checkpoint_data)
        await self.redis.expire(checkpoint_key, 86400)  # 24 hours TTL
        
        # Update task with checkpoint reference
        await self.update_task_status(checkpoint.task_id, TaskStatus.CHECKPOINTED,
                                    checkpoint_id=checkpoint.checkpoint_id)
        
        logger.debug(f"Saved checkpoint {checkpoint.checkpoint_id} for task {checkpoint.task_id}")
    
    async def get_checkpoint(self, task_id: str) -> Optional[TaskCheckpoint]:
        """Get latest checkpoint for task."""
        checkpoint_key = f"{self.queue_prefix}:checkpoint:{task_id}"
        data = await self.redis.hgetall(checkpoint_key)
        
        if not data:
            return None
        
        return TaskCheckpoint(
            task_id=task_id,
            checkpoint_id=data[b'checkpoint_id'].decode(),
            state=pickle.loads(data[b'state']),
            progress=float(data[b'progress']),
            timestamp=float(data[b'timestamp']),
            metadata=json.loads(data[b'metadata'].decode())
        )
    
    async def register_worker(self, worker_id: str, capabilities: Dict[str, Any]) -> None:
        """Register worker in Redis."""
        worker_key = f"{self.queue_prefix}:worker:{worker_id}"
        worker_data = {
            'worker_id': worker_id,
            'capabilities': json.dumps(capabilities),
            'registered_at': time.time(),
            'last_heartbeat': time.time()
        }
        
        await self.redis.hset(worker_key, mapping=worker_data)
        await self.redis.sadd(f"{self.queue_prefix}:workers", worker_id)
        
        logger.info(f"Registered worker {worker_id}")
    
    async def unregister_worker(self, worker_id: str) -> None:
        """Unregister worker from Redis."""
        worker_key = f"{self.queue_prefix}:worker:{worker_id}"
        await self.redis.delete(worker_key)
        await self.redis.srem(f"{self.queue_prefix}:workers", worker_id)
        
        # Clean up processing queue
        processing_key = f"{self.queue_prefix}:processing:{worker_id}"
        await self.redis.delete(processing_key)
        
        logger.info(f"Unregistered worker {worker_id}")
    
    async def get_workers(self) -> List[Dict[str, Any]]:
        """Get all registered workers."""
        worker_ids = await self.redis.smembers(f"{self.queue_prefix}:workers")
        workers = []
        
        for worker_id in worker_ids:
            worker_id = worker_id.decode() if isinstance(worker_id, bytes) else worker_id
            worker_key = f"{self.queue_prefix}:worker:{worker_id}"
            data = await self.redis.hgetall(worker_key)
            
            if data:
                worker_data = {}
                for key, value in data.items():
                    key = key.decode() if isinstance(key, bytes) else key
                    if isinstance(value, bytes):
                        try:
                            worker_data[key] = value.decode()
                        except UnicodeDecodeError:
                            worker_data[key] = value
                    else:
                        worker_data[key] = value
                
                if 'capabilities' in worker_data:
                    worker_data['capabilities'] = json.loads(worker_data['capabilities'])
                workers.append(worker_data)
        
        return workers


class RabbitMQBackend(TaskBackend):
    """RabbitMQ-based task queue backend."""
    
    def __init__(self, rabbitmq_url: str = "amqp://guest:guest@localhost:5672/",
                 exchange_name: str = "nexus.tasks"):
        if not RABBITMQ_AVAILABLE:
            raise ImportError("aio-pika package not installed. Install with: pip install aio-pika")
        
        self.rabbitmq_url = rabbitmq_url
        self.exchange_name = exchange_name
        self.connection: Optional[aio_pika.Connection] = None
        self.channel: Optional[aio_pika.Channel] = None
        self.exchange: Optional[aio_pika.Exchange] = None
        self.queues: Dict[TaskPriority, aio_pika.Queue] = {}
        self._task_store: Dict[str, DistributedTask] = {}  # In-memory store for simplicity
        self._checkpoints: Dict[str, TaskCheckpoint] = {}
    
    async def connect(self) -> None:
        """Connect to RabbitMQ."""
        self.connection = await aio_pika.connect_robust(self.rabbitmq_url)
        self.channel = await self.connection.channel()
        
        # Declare exchange
        self.exchange = await self.channel.declare_exchange(
            self.exchange_name, 
            aio_pika.ExchangeType.DIRECT,
            durable=True
        )
        
        # Declare priority queues
        for priority in TaskPriority:
            queue_name = f"{self.exchange_name}.{priority.name.lower()}"
            queue = await self.channel.declare_queue(
                queue_name,
                durable=True,
                arguments={'x-max-priority': 10}  # RabbitMQ priority support
            )
            await queue.bind(self.exchange, routing_key=queue_name)
            self.queues[priority] = queue
        
        logger.info("Connected to RabbitMQ backend")
    
    async def disconnect(self) -> None:
        """Disconnect from RabbitMQ."""
        if self.connection:
            await self.connection.close()
            logger.info("Disconnected from RabbitMQ backend")
    
    async def enqueue_task(self, task: DistributedTask) -> str:
        """Enqueue task to RabbitMQ."""
        task.status = TaskStatus.QUEUED
        self._task_store[task.task_id] = task
        
        # Serialize task
        task_data = pickle.dumps(task)
        
        # Publish to appropriate queue
        queue_name = f"{self.exchange_name}.{task.priority.name.lower()}"
        await self.exchange.publish(
            aio_pika.Message(
                body=task_data,
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                priority=task.priority.value
            ),
            routing_key=queue_name
        )
        
        logger.debug(f"Enqueued task {task.task_id} with priority {task.priority}")
        return task.task_id
    
    async def dequeue_task(self, worker_id: str, timeout: float = 30.0) -> Optional[DistributedTask]:
        """Dequeue task from RabbitMQ."""
        # Try queues in priority order
        for priority in [TaskPriority.CRITICAL, TaskPriority.HIGH, 
                        TaskPriority.NORMAL, TaskPriority.LOW]:
            queue = self.queues[priority]
            
            try:
                # Get message with timeout
                message = await queue.get(timeout=timeout, fail=False)
                if message:
                    async with message.process():
                        task = pickle.loads(message.body)
                        task.status = TaskStatus.RUNNING
                        task.started_at = time.time()
                        task.metadata['worker_id'] = worker_id
                        
                        # Update in-memory store
                        self._task_store[task.task_id] = task
                        
                        logger.debug(f"Dequeued task {task.task_id} for worker {worker_id}")
                        return task
            except asyncio.TimeoutError:
                continue
        
        return None
    
    async def update_task_status(self, task_id: str, status: TaskStatus, **kwargs) -> None:
        """Update task status in memory store."""
        if task_id in self._task_store:
            task = self._task_store[task_id]
            task.status = status
            for key, value in kwargs.items():
                setattr(task, key, value)
    
    async def get_task(self, task_id: str) -> Optional[DistributedTask]:
        """Get task from memory store."""
        return self._task_store.get(task_id)
    
    async def save_checkpoint(self, checkpoint: TaskCheckpoint) -> None:
        """Save checkpoint to memory store."""
        self._checkpoints[checkpoint.task_id] = checkpoint
        await self.update_task_status(checkpoint.task_id, TaskStatus.CHECKPOINTED,
                                    checkpoint_id=checkpoint.checkpoint_id)
        logger.debug(f"Saved checkpoint {checkpoint.checkpoint_id} for task {checkpoint.task_id}")
    
    async def get_checkpoint(self, task_id: str) -> Optional[TaskCheckpoint]:
        """Get checkpoint from memory store."""
        return self._checkpoints.get(task_id)
    
    async def register_worker(self, worker_id: str, capabilities: Dict[str, Any]) -> None:
        """Register worker (in-memory for RabbitMQ backend)."""
        # In a production system, you might use a separate Redis instance for worker registry
        logger.info(f"Registered worker {worker_id}")
    
    async def unregister_worker(self, worker_id: str) -> None:
        """Unregister worker."""
        logger.info(f"Unregistered worker {worker_id}")
    
    async def get_workers(self) -> List[Dict[str, Any]]:
        """Get workers (empty for in-memory RabbitMQ backend)."""
        return []


class TaskSharder:
    """Handles task sharding for distributed execution."""
    
    @staticmethod
    def shard_by_url(urls: List[str], shard_count: int) -> List[List[str]]:
        """Shard URLs across workers."""
        return [urls[i::shard_count] for i in range(shard_count)]
    
    @staticmethod
    def shard_by_range(start: int, end: int, shard_count: int) -> List[Tuple[int, int]]:
        """Shard numeric range across workers."""
        step = (end - start) // shard_count
        ranges = []
        for i in range(shard_count):
            shard_start = start + i * step
            shard_end = start + (i + 1) * step if i < shard_count - 1 else end
            ranges.append((shard_start, shard_end))
        return ranges
    
    @staticmethod
    def shard_by_hash(items: List[Any], shard_count: int, 
                     hash_func: Callable[[Any], str] = None) -> List[List[Any]]:
        """Shard items by hash for even distribution."""
        if hash_func is None:
            hash_func = lambda x: hashlib.md5(str(x).encode()).hexdigest()
        
        shards = [[] for _ in range(shard_count)]
        for item in items:
            hash_val = int(hash_func(item), 16)
            shard_idx = hash_val % shard_count
            shards[shard_idx].append(item)
        
        return shards


class ResultAggregator:
    """Aggregates results from sharded tasks."""
    
    @staticmethod
    def concatenate_results(results: List[Any]) -> Any:
        """Concatenate list results."""
        if not results:
            return []
        
        if isinstance(results[0], list):
            return [item for sublist in results for item in sublist]
        return results
    
    @staticmethod
    def merge_dicts(results: List[Dict]) -> Dict:
        """Merge dictionary results."""
        merged = {}
        for result in results:
            if isinstance(result, dict):
                merged.update(result)
        return merged
    
    @staticmethod
    def average_numeric(results: List[Union[int, float]]) -> float:
        """Calculate average of numeric results."""
        numeric_results = [r for r in results if isinstance(r, (int, float))]
        if not numeric_results:
            return 0.0
        return sum(numeric_results) / len(numeric_results)
    
    @staticmethod
    def custom_aggregate(results: List[Any], 
                        aggregate_func: Callable[[List[Any]], Any]) -> Any:
        """Apply custom aggregation function."""
        return aggregate_func(results)


class DistributedTaskOrchestrator:
    """Main orchestrator for distributed task execution."""
    
    def __init__(self, 
                 backend: TaskBackend,
                 worker_id: Optional[str] = None,
                 max_concurrent_tasks: int = 10,
                 checkpoint_interval: float = 30.0):
        self.backend = backend
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.max_concurrent_tasks = max_concurrent_tasks
        self.checkpoint_interval = checkpoint_interval
        
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._task_handlers: Dict[str, Callable] = {}
        self._shutdown_event = asyncio.Event()
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)
        
        # Register built-in handlers
        self._register_builtin_handlers()
    
    def _register_builtin_handlers(self):
        """Register built-in task handlers."""
        # These will be populated from nexus modules
        pass
    
    def register_handler(self, func_name: str, handler: Callable):
        """Register a task handler function."""
        self._task_handlers[func_name] = handler
        logger.debug(f"Registered handler for {func_name}")
    
    async def start(self):
        """Start the orchestrator (worker mode)."""
        await self.backend.connect()
        
        # Register worker
        capabilities = {
            'max_concurrent_tasks': self.max_concurrent_tasks,
            'worker_id': self.worker_id,
            'handlers': list(self._task_handlers.keys())
        }
        await self.backend.register_worker(self.worker_id, capabilities)
        
        logger.info(f"Started distributed task orchestrator as worker {self.worker_id}")
        
        # Start task processing loop
        asyncio.create_task(self._process_tasks())
    
    async def stop(self):
        """Stop the orchestrator."""
        self._shutdown_event.set()
        
        # Cancel running tasks
        for task_id, task in self._running_tasks.items():
            task.cancel()
        
        # Wait for tasks to complete
        if self._running_tasks:
            await asyncio.gather(*self._running_tasks.values(), return_exceptions=True)
        
        # Unregister worker
        await self.backend.unregister_worker(self.worker_id)
        await self.backend.disconnect()
        
        self._executor.shutdown(wait=True)
        logger.info(f"Stopped distributed task orchestrator {self.worker_id}")
    
    async def _process_tasks(self):
        """Main task processing loop."""
        while not self._shutdown_event.is_set():
            try:
                # Check if we can accept more tasks
                if len(self._running_tasks) >= self.max_concurrent_tasks:
                    await asyncio.sleep(1)
                    continue
                
                # Dequeue task
                task = await self.backend.dequeue_task(self.worker_id, timeout=5.0)
                
                if task:
                    # Process task in background
                    task_coro = self._execute_task(task)
                    running_task = asyncio.create_task(task_coro)
                    self._running_tasks[task.task_id] = running_task
                    
                    # Clean up completed tasks
                    done_tasks = [tid for tid, t in self._running_tasks.items() if t.done()]
                    for tid in done_tasks:
                        del self._running_tasks[tid]
                
            except Exception as e:
                logger.error(f"Error in task processing loop: {e}")
                await asyncio.sleep(5)  # Back off on error
    
    async def _execute_task(self, task: DistributedTask):
        """Execute a single task with retry and checkpointing."""
        task_id = task.task_id
        
        try:
            # Check for existing checkpoint
            checkpoint = await self.backend.get_checkpoint(task_id)
            if checkpoint and checkpoint.state:
                logger.info(f"Resuming task {task_id} from checkpoint {checkpoint.checkpoint_id}")
                # Merge checkpoint state into task kwargs
                task.kwargs['checkpoint_state'] = checkpoint.state
            
            # Get handler
            handler = self._task_handlers.get(task.func_name)
            if not handler:
                raise ValueError(f"No handler registered for function: {task.func_name}")
            
            # Execute with retry logic
            result = await retry_with_exponential_backoff(
                lambda: self._run_task_handler(handler, task),
                max_retries=task.max_retries,
                initial_delay=1.0,
                max_delay=60.0
            )
            
            # Update task status
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            task.result = result
            
            await self.backend.update_task_status(
                task_id, 
                TaskStatus.COMPLETED,
                result=result,
                completed_at=task.completed_at
            )
            
            logger.info(f"Completed task {task_id}")
            
            # Handle parent task aggregation if this was a shard
            if task.parent_task_id:
                await self._handle_shard_completion(task)
            
        except asyncio.CancelledError:
            logger.warning(f"Task {task_id} was cancelled")
            raise
        
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            
            # Update task status
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = time.time()
            
            await self.backend.update_task_status(
                task_id,
                TaskStatus.FAILED,
                error=str(e),
                completed_at=task.completed_at
            )
            
            # Re-raise for retry logic
            raise
    
    async def _run_task_handler(self, handler: Callable, task: DistributedTask) -> Any:
        """Run task handler with checkpointing support."""
        # Create checkpoint callback
        checkpoint_callback = lambda state, progress: self._save_checkpoint(
            task.task_id, state, progress
        )
        
        # Add checkpoint callback to kwargs
        task.kwargs['checkpoint_callback'] = checkpoint_callback
        
        # Run handler
        if inspect.iscoroutinefunction(handler):
            return await handler(*task.args, **task.kwargs)
        else:
            # Run sync handler in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._executor,
                lambda: handler(*task.args, **task.kwargs)
            )
    
    async def _save_checkpoint(self, task_id: str, state: Dict[str, Any], progress: float):
        """Save task checkpoint."""
        checkpoint = TaskCheckpoint(
            task_id=task_id,
            checkpoint_id=f"checkpoint-{uuid.uuid4().hex[:8]}",
            state=state,
            progress=progress,
            metadata={'worker_id': self.worker_id}
        )
        
        await self.backend.save_checkpoint(checkpoint)
    
    async def _handle_shard_completion(self, task: DistributedTask):
        """Handle completion of a sharded task."""
        parent_task_id = task.parent_task_id
        
        # Get parent task
        parent_task = await self.backend.get_task(parent_task_id)
        if not parent_task:
            return
        
        # Check if all shards are completed
        # This would require tracking shard completion in a real implementation
        # For now, we'll just update the parent task status
        await self.backend.update_task_status(
            parent_task_id,
            TaskStatus.AGGREGATING,
            metadata={'completed_shard': task.shard_id}
        )
    
    async def submit_task(self, 
                         func_name: str,
                         args: Tuple[Any, ...] = (),
                         kwargs: Dict[str, Any] = None,
                         priority: TaskPriority = TaskPriority.NORMAL,
                         timeout: Optional[float] = None,
                         max_retries: int = 3,
                         task_id: Optional[str] = None,
                         metadata: Dict[str, Any] = None) -> str:
        """Submit a task for distributed execution."""
        if kwargs is None:
            kwargs = {}
        if metadata is None:
            metadata = {}
        
        task = DistributedTask(
            task_id=task_id or f"task-{uuid.uuid4().hex}",
            name=func_name,
            func_name=func_name,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout,
            max_retries=max_retries,
            metadata=metadata
        )
        
        task_id = await self.backend.enqueue_task(task)
        logger.info(f"Submitted task {task_id}: {func_name}")
        return task_id
    
    async def submit_sharded_task(self,
                                 func_name: str,
                                 shards: List[Dict[str, Any]],
                                 aggregation_func: Optional[str] = None,
                                 priority: TaskPriority = TaskPriority.NORMAL,
                                 timeout: Optional[float] = None,
                                 max_retries: int = 3,
                                 metadata: Dict[str, Any] = None) -> str:
        """Submit a sharded task for distributed execution."""
        if metadata is None:
            metadata = {}
        
        # Create parent task
        parent_task_id = f"parent-{uuid.uuid4().hex}"
        parent_task = DistributedTask(
            task_id=parent_task_id,
            name=f"sharded-{func_name}",
            func_name="aggregate_results",  # Special handler for aggregation
            priority=priority,
            timeout=timeout,
            max_retries=max_retries,
            shard_total=len(shards),
            metadata={
                **metadata,
                'aggregation_func': aggregation_func,
                'shard_func_name': func_name
            }
        )
        
        # Enqueue parent task
        await self.backend.enqueue_task(parent_task)
        
        # Create and enqueue shard tasks
        shard_tasks = []
        for i, shard_kwargs in enumerate(shards):
            shard_task = DistributedTask(
                task_id=f"shard-{parent_task_id}-{i}",
                name=f"{func_name}-shard-{i}",
                func_name=func_name,
                kwargs=shard_kwargs,
                priority=priority,
                timeout=timeout,
                max_retries=max_retries,
                shard_id=str(i),
                shard_total=len(shards),
                parent_task_id=parent_task_id,
                metadata={'shard_index': i, **metadata}
            )
            shard_tasks.append(shard_task)
        
        # Enqueue all shard tasks
        for shard_task in shard_tasks:
            await self.backend.enqueue_task(shard_task)
        
        logger.info(f"Submitted sharded task {parent_task_id} with {len(shards)} shards")
        return parent_task_id
    
    async def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get task result, optionally waiting for completion."""
        start_time = time.time()
        
        while True:
            task = await self.backend.get_task(task_id)
            
            if not task:
                raise ValueError(f"Task {task_id} not found")
            
            if task.status == TaskStatus.COMPLETED:
                return task.result
            elif task.status == TaskStatus.FAILED:
                raise RuntimeError(f"Task {task_id} failed: {task.error}")
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")
            
            await asyncio.sleep(1)  # Poll interval
    
    async def get_aggregated_result(self, parent_task_id: str, 
                                  timeout: Optional[float] = None) -> AggregatedResult:
        """Get aggregated result from sharded task."""
        start_time = time.time()
        
        while True:
            parent_task = await self.backend.get_task(parent_task_id)
            
            if not parent_task:
                raise ValueError(f"Parent task {parent_task_id} not found")
            
            if parent_task.status == TaskStatus.COMPLETED:
                # Get all shard results
                # This would require tracking shard tasks in a real implementation
                # For now, return the parent task result
                return AggregatedResult(
                    task_id=parent_task_id,
                    results=[parent_task.result] if parent_task.result else [],
                    errors=[parent_task.error],
                    shard_count=parent_task.shard_total,
                    completed_shards=1 if parent_task.result else 0,
                    failed_shards=1 if parent_task.error else 0,
                    aggregated_result=parent_task.result
                )
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Aggregation for {parent_task_id} did not complete within {timeout} seconds")
            
            await asyncio.sleep(1)  # Poll interval


# Integration with nexus agents
class DistributedAgentOrchestrator:
    """Orchestrator for distributing nexus agent tasks."""
    
    def __init__(self, 
                 orchestrator: DistributedTaskOrchestrator,
                 agent_factory: Callable[..., Agent] = None):
        self.orchestrator = orchestrator
        self.agent_factory = agent_factory or Agent
        
        # Register agent-related handlers
        self._register_agent_handlers()
    
    def _register_agent_handlers(self):
        """Register handlers for agent tasks."""
        self.orchestrator.register_handler("run_agent_task", self._run_agent_task_handler)
        self.orchestrator.register_handler("run_agent_on_url", self._run_agent_on_url_handler)
        self.orchestrator.register_handler("run_agent_batch", self._run_agent_batch_handler)
    
    async def _run_agent_task_handler(self, 
                                     task_config: Dict[str, Any],
                                     checkpoint_callback: Optional[Callable] = None) -> AgentResult:
        """Handler for running an agent task."""
        # Create agent
        agent = self.agent_factory(**task_config.get('agent_config', {}))
        
        # Run task
        task = AgentTask(**task_config['task'])
        result = await agent.run(task, checkpoint_callback=checkpoint_callback)
        
        return result
    
    async def _run_agent_on_url_handler(self,
                                       url: str,
                                       instructions: str,
                                       checkpoint_callback: Optional[Callable] = None) -> AgentResult:
        """Handler for running agent on a single URL."""
        agent = self.agent_factory()
        task = AgentTask(
            url=url,
            instructions=instructions
        )
        return await agent.run(task, checkpoint_callback=checkpoint_callback)
    
    async def _run_agent_batch_handler(self,
                                      urls: List[str],
                                      instructions: str,
                                      checkpoint_callback: Optional[Callable] = None) -> List[AgentResult]:
        """Handler for running agent on multiple URLs."""
        agent = self.agent_factory()
        results = []
        
        for i, url in enumerate(urls):
            # Checkpoint progress
            if checkpoint_callback:
                checkpoint_callback(
                    {'processed_urls': i, 'total_urls': len(urls)},
                    i / len(urls)
                )
            
            task = AgentTask(
                url=url,
                instructions=instructions
            )
            result = await agent.run(task)
            results.append(result)
        
        return results
    
    async def distribute_agent_task(self,
                                   task: AgentTask,
                                   priority: TaskPriority = TaskPriority.NORMAL,
                                   **kwargs) -> str:
        """Distribute an agent task for execution."""
        task_config = {
            'task': asdict(task),
            'agent_config': kwargs.get('agent_config', {})
        }
        
        return await self.orchestrator.submit_task(
            func_name="run_agent_task",
            kwargs={'task_config': task_config},
            priority=priority,
            **{k: v for k, v in kwargs.items() if k != 'agent_config'}
        )
    
    async def distribute_url_batch(self,
                                  urls: List[str],
                                  instructions: str,
                                  shard_count: int = 4,
                                  priority: TaskPriority = TaskPriority.NORMAL,
                                  **kwargs) -> str:
        """Distribute a batch of URLs across multiple workers."""
        # Shard URLs
        url_shards = TaskSharder.shard_by_url(urls, shard_count)
        
        # Create shard configurations
        shards = []
        for shard_urls in url_shards:
            shards.append({
                'urls': shard_urls,
                'instructions': instructions
            })
        
        # Submit sharded task
        return await self.orchestrator.submit_sharded_task(
            func_name="run_agent_batch",
            shards=shards,
            aggregation_func="concatenate_results",
            priority=priority,
            **kwargs
        )


# Factory functions for easy setup
def create_redis_orchestrator(redis_url: str = "redis://localhost:6379",
                            worker_id: Optional[str] = None,
                            **kwargs) -> DistributedTaskOrchestrator:
    """Create a Redis-based distributed task orchestrator."""
    backend = RedisBackend(redis_url)
    return DistributedTaskOrchestrator(backend, worker_id=worker_id, **kwargs)


def create_rabbitmq_orchestrator(rabbitmq_url: str = "amqp://guest:guest@localhost:5672/",
                               worker_id: Optional[str] = None,
                               **kwargs) -> DistributedTaskOrchestrator:
    """Create a RabbitMQ-based distributed task orchestrator."""
    backend = RabbitMQBackend(rabbitmq_url)
    return DistributedTaskOrchestrator(backend, worker_id=worker_id, **kwargs)


# Example usage
async def example_usage():
    """Example of using the distributed task orchestrator."""
    # Create orchestrator
    orchestrator = create_redis_orchestrator()
    
    # Register custom handler
    async def custom_handler(data: Dict[str, Any], checkpoint_callback=None):
        # Simulate work with checkpointing
        for i in range(10):
            await asyncio.sleep(1)
            if checkpoint_callback:
                checkpoint_callback({'progress': i}, i / 10)
        return {'result': 'completed', 'data': data}
    
    orchestrator.register_handler('custom_handler', custom_handler)
    
    # Start orchestrator (as worker)
    await orchestrator.start()
    
    try:
        # Submit a task
        task_id = await orchestrator.submit_task(
            func_name='custom_handler',
            kwargs={'data': 'test'},
            priority=TaskPriority.HIGH
        )
        
        # Wait for result
        result = await orchestrator.get_task_result(task_id, timeout=30)
        print(f"Task result: {result}")
        
        # Submit sharded task
        parent_id = await orchestrator.submit_sharded_task(
            func_name='custom_handler',
            shards=[
                {'data': f'shard-{i}'} for i in range(4)
            ],
            aggregation_func='merge_dicts'
        )
        
        # Get aggregated result
        agg_result = await orchestrator.get_aggregated_result(parent_id, timeout=60)
        print(f"Aggregated result: {agg_result}")
        
    finally:
        await orchestrator.stop()


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())