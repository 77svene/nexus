import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path

import redis.asyncio as redis
from redis.asyncio import Redis
from pydantic import BaseModel, Field
from typing_extensions import Self

# Configure logging for the distributed module
logger = logging.getLogger("nexus.distributed.coordinator")


# ==============================================================================
# CONSTANTS & CONFIGURATION
# ==============================================================================

TASK_QUEUE_KEY = "nexus:tasks"
WORKER_REGISTRY_KEY = "nexus:workers"
WORKER_HEARTBEAT_KEY = "nexus:worker_heartbeat"
TASK_STATE_COMPLETED = "completed"
TASK_STATE_FAILED = "failed"
TASK_STATE_QUEUED = "queued"
TASK_STATE_RETRYING = "retrying"
CHECKPOINT_PREFIX = "nexus:checkpoint"
MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 2.0
WORKER_TIMEOUT_SECONDS = 300


# ==============================================================================
# DATA MODELS
# ==============================================================================

class TaskStatus(str, Enum):
    QUEUED = TASK_STATE_QUEUED
    RUNNING = "running"
    COMPLETED = TASK_STATE_COMPLETED
    FAILED = TASK_STATE_FAILED
    CANCELLED = "cancelled"

@dataclass
class WorkerInfo:
    node_id: str
    ip_address: str
    last_heartbeat: float
    current_task_id: Optional[str] = None
    capacity: int = 1  # Number of concurrent tasks this worker can handle

class TaskPayload(BaseModel):
    task_id: str
    task_type: str  # e.g., "navigate", "scrape", "simulate"
    payload_data: Dict[str, Any]
    priority: int = 5
    created_at: float = field(default_factory=time.time)
    status: TaskStatus = TASK_STATE_QUEUED
    retries_left: int = MAX_RETRIES
    worker_id: Optional[str] = None

class TaskResult(BaseModel):
    task_id: str
    status: TaskStatus
    result_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    checkpoint: Optional[Dict[str, Any]] = None

# ==============================================================================
# CHECKPOINT MANAGER
# ==============================================================================

class CheckpointManager:
    """
    Handles saving and restoring task state for long-running distributed tasks.
    Stores state in Redis Hashes keyed by task_id.
    """
    def __init__(self, redis_client: Redis):
        self.redis = redis_client

    async def save_checkpoint(self, task_id: str, state: Dict[str, Any]) -> None:
        """Saves current state of a task to Redis."""
        key = f"{CHECKPOINT_PREFIX}:{task_id}"
        try:
            await self.redis.hset(key, mapping=state)
            await self.redis.expire(key, 86400 * 7)  # Expire after 7 days
            logger.debug(f"Checkpoint saved for task {task_id}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint for {task_id}: {e}")

    async def load_checkpoint(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Loads state of a task from Redis."""
        key = f"{CHECKPOINT_PREFIX}:{task_id}"
        try:
            data = await self.redis.hgetall(key)
            if data:
                return json.loads(json.dumps(data))
            return None
        except Exception as e:
            logger.error(f"Failed to load checkpoint for {task_id}: {e}")
            return None

    async def delete_checkpoint(self, task_id: str) -> None:
        """Removes checkpoint data after successful completion or failure."""
        key = f"{CHECKPOINT_PREFIX}:{task_id}"
        try:
            await self.redis.delete(key)
        except Exception as e:
            logger.error(f"Failed to delete checkpoint for {task_id}: {e}")

# ==============================================================================
# COORDINATOR CLASS
# ==============================================================================

class DistributedCoordinator:
    """
    Distributed Task Orchestration for nexus.
    
    Features:
    - Horizontal scaling across multiple machines.
    - Redis-based task queue with worker nodes.
    - Dynamic worker join/leave support.
    - Checkpointing for long-running tasks.
    - Automatic retry with exponential backoff.
    - Result aggregation.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """
        Initialize the coordinator.
        
        Args:
            redis_url: Connection string for the Redis broker.
        """
        self.redis_url = redis_url
        self.redis: Optional[Redis] = None
        self.checkpoint_manager: Optional[CheckpointManager] = None
        self.task_queue: Optional[redis.StrictRedis] = None  # Pub/Sub or Stream
        self.worker_registry: Dict[str, WorkerInfo] = {}
        self.task_callbacks: Dict[str, List[Callable]] = {}
        self._running = False
        self._running_tasks: Set[str] = set()

    async def connect(self) -> None:
        """Establish connection to the Redis broker."""
        try:
            self.redis = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False,
                socket_connect_timeout=10,
                socket_timeout=10,
                retry_on_timeout=True
            )
            # Subscribe to task queue
            await self.redis.ping()
            self.task_queue = self.redis
            self.checkpoint_manager = CheckpointManager(self.redis)
            logger.info("Distributed Coordinator connected to Redis")
            self._running = True
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise ConnectionError(f"Cannot connect to Redis: {e}")

    async def disconnect(self) -> None:
        """Close all connections and cleanup."""
        self._running = False
        if self.redis:
            await self.redis.close()
            self.redis = None
            self.checkpoint_manager = None
            self.task_queue = None
            logger.info("Distributed Coordinator disconnected")

    async def register_worker(self, node_id: str, ip_address: str, capacity: int = 1) -> None:
        """
        Register a new worker node to the cluster.
        Used by worker nodes joining the distributed system.
        """
        worker_info = WorkerInfo(
            node_id=node_id,
            ip_address=ip_address,
            last_heartbeat=time.time(),
            capacity=capacity
        )
        # Store worker in Redis Hash
        await self.redis.hset(WORKER_REGISTRY_KEY, member=node_id, mapping={
            "node_id": node_id,
            "ip_address": ip_address,
            "last_heartbeat": time.time(),
            "capacity": capacity,
            "status": "active"
        })
        self.worker_registry[node_id] = worker_info
        logger.info(f"Worker registered: {node_id} on {ip_address}")

    async def heartbeat(self, node_id: str) -> None:
        """
        Update worker heartbeat.
        Called periodically by worker processes.
        """
        if not self.redis:
            return
        worker_data = await self.redis.hget