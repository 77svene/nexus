"""core/distributed/executor.py - Distributed Agent Execution Engine"""

import os
import sys
import time
import json
import asyncio
import logging
import hashlib
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import grpc
import redis
import raft
from raft import RaftNode, State as RaftState
from raft.messages import AppendEntries, RequestVote

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import existing modules
from plugins.backend_development.skills.api_design_principles.assets.rest_api_template import APIResponse
from plugins.llm_application_dev.skills.prompt_engineering_patterns.scripts.optimize_prompt import PromptOptimizer
from tools.yt_design_extractor import DesignExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Node status in the distributed cluster."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    DRAINING = "draining"
    FAILED = "failed"
    RECOVERING = "recovering"


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class NodeInfo:
    """Information about a node in the cluster."""
    node_id: str
    host: str
    port: int
    status: NodeStatus
    capacity: int  # Maximum concurrent tasks
    current_load: int
    last_heartbeat: float
    metadata: Dict[str, Any]


@dataclass
class Task:
    """Represents an executable task."""
    task_id: str
    agent_type: str
    payload: Dict[str, Any]
    priority: int = 0
    timeout: int = 300  # seconds
    retry_count: int = 3
    created_at: float = None
    assigned_to: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


class DistributedStateManager:
    """Redis-backed state management for distributed coordination."""
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.redis_client = None
        self._connect()
        
    def _connect(self):
        """Establish Redis connection with retry logic."""
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                self.redis_client = redis.Redis.from_url(
                    self.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True
                )
                # Test connection
                self.redis_client.ping()
                logger.info("Connected to Redis state manager")
                return
            except redis.ConnectionError as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Redis connection attempt {attempt + 1} failed: {e}")
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    logger.error(f"Failed to connect to Redis after {max_retries} attempts")
                    raise
    
    def store_node(self, node: NodeInfo) -> bool:
        """Store node information in Redis."""
        try:
            key = f"node:{node.node_id"
            data = asdict(node)
            # Convert enum to string
            data['status'] = node.status.value
            self.redis_client.hset(key, mapping=data)
            # Add to node set
            self.redis_client.sadd("nodes", node.node_id)
            return True
        except Exception as e:
            logger.error(f"Failed to store node {node.node_id}: {e}")
            return False
    
    def get_node(self, node_id: str) -> Optional[NodeInfo]:
        """Retrieve node information from Redis."""
        try:
            key = f"node:{node_id}"
            data = self.redis_client.hgetall(key)
            if not data:
                return None
            
            # Convert string values back to appropriate types
            data['port'] = int(data['port'])
            data['capacity'] = int(data['capacity'])
            data['current_load'] = int(data['current_load'])
            data['last_heartbeat'] = float(data['last_heartbeat'])
            data['status'] = NodeStatus(data['status'])
            data['metadata'] = json.loads(data['metadata']) if data.get('metadata') else {}
            
            return NodeInfo(**data)
        except Exception as e:
            logger.error(f"Failed to get node {node_id}: {e}")
            return None
    
    def get_all_nodes(self) -> List[NodeInfo]:
        """Retrieve all nodes from Redis."""
        nodes = []
        try:
            node_ids = self.redis_client.smembers("nodes")
            for node_id in node_ids:
                node = self.get_node(node_id)
                if node:
                    nodes.append(node)
        except Exception as e:
            logger.error(f"Failed to get all nodes: {e}")
        return nodes
    
    def store_task(self, task: Task) -> bool:
        """Store task information in Redis."""
        try:
            key = f"task:{task.task_id}"
            data = asdict(task)
            # Convert enums to strings
            data['status'] = task.status.value
            self.redis_client.hset(key, mapping=data)
            # Add to task set
            self.redis_client.sadd("tasks", task.task_id)
            # Add to priority queue
            self.redis_client.zadd("task_queue", {task.task_id: task.priority})
            return True
        except Exception as e:
            logger.error(f"Failed to store task {task.task_id}: {e}")
            return False
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Retrieve task information from Redis."""
        try:
            key = f"task:{task_id}"
            data = self.redis_client.hgetall(key)
            if not data:
                return None
            
            # Convert string values back to appropriate types
            data['priority'] = int(data['priority'])
            data['timeout'] = int(data['timeout'])
            data['retry_count'] = int(data['retry_count'])
            data['created_at'] = float(data['created_at'])
            data['status'] = TaskStatus(data['status'])
            data['payload'] = json.loads(data['payload']) if data.get('payload') else {}
            data['result'] = json.loads(data['result']) if data.get('result') else None
            
            return Task(**data)
        except Exception as e:
            logger.error(f"Failed to get task {task_id}: {e}")
            return None
    
    def update_task_status(self, task_id: str, status: TaskStatus, 
                          result: Optional[Dict] = None, error: Optional[str] = None) -> bool:
        """Update task status in Redis."""
        try:
            key = f"task:{task_id}"
            updates = {"status": status.value}
            if result is not None:
                updates["result"] = json.dumps(result)
            if error is not None:
                updates["error"] = error
            
            self.redis_client.hset(key, mapping=updates)
            
            # Remove from queue if completed or failed
            if status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                self.redis_client.zrem("task_queue", task_id)
            
            return True
        except Exception as e:
            logger.error(f"Failed to update task {task_id}: {e}")
            return False
    
    def get_pending_tasks(self, limit: int = 10) -> List[Task]:
        """Get pending tasks from priority queue."""
        tasks = []
        try:
            # Get task IDs from priority queue (highest priority first)
            task_ids = self.redis_client.zrevrange("task_queue", 0, limit - 1)
            for task_id in task_ids:
                task = self.get_task(task_id)
                if task and task.status == TaskStatus.PENDING:
                    tasks.append(task)
        except Exception as e:
            logger.error(f"Failed to get pending tasks: {e}")
        return tasks


class RaftConsensusLayer:
    """Raft-based consensus layer for agent coordination."""
    
    def __init__(self, node_id: str, host: str, port: int, peers: List[Tuple[str, str, int]]):
        """
        Initialize Raft consensus layer.
        
        Args:
            node_id: Unique identifier for this node
            host: Host address of this node
            port: Port for Raft communication
            peers: List of (node_id, host, port) for other nodes
        """
        self.node_id = node_id
        self.host = host
        self.port = port
        self.peers = peers
        
        # Raft configuration
        self.raft_node = RaftNode(
            node_id=node_id,
            address=(host, port),
            peers=[(peer_id, (peer_host, peer_port)) 
                   for peer_id, peer_host, peer_port in peers]
        )
        
        # State
        self.is_leader = False
        self.leader_id = None
        self.state_machine = {}
        self._lock = threading.RLock()
        
        # Start Raft node
        self.raft_node.start()
        logger.info(f"Raft node {node_id} started on {host}:{port}")
    
    def propose(self, key: str, value: Any) -> bool:
        """Propose a state change through Raft consensus."""
        if not self.is_leader:
            logger.warning("Cannot propose: not the leader")
            return False
        
        try:
            # Serialize the operation
            operation = json.dumps({"key": key, "value": value})
            
            # Append to Raft log (this would be implemented in the Raft library)
            # For now, simulate consensus
            with self._lock:
                self.state_machine[key] = value
            
            logger.debug(f"Proposed operation: {key} = {value}")
            return True
        except Exception as e:
            logger.error(f"Failed to propose operation: {e}")
            return False
    
    def get(self, key: str) -> Any:
        """Get value from state machine."""
        with self._lock:
            return self.state_machine.get(key)
    
    def is_current_leader(self) -> bool:
        """Check if this node is the current leader."""
        return self.raft_node.state == RaftState.LEADER
    
    def get_leader_id(self) -> Optional[str]:
        """Get the current leader's node ID."""
        return self.raft_node.leader_id
    
    def stop(self):
        """Stop the Raft node."""
        self.raft_node.stop()
        logger.info(f"Raft node {self.node_id} stopped")


# gRPC Service Definition
# Note: In production, this would be defined in a .proto file
# For this implementation, we'll define the service interface

class DistributedExecutorServicer:
    """gRPC service implementation for distributed executor."""
    
    def __init__(self, executor: 'DistributedExecutor'):
        self.executor = executor
    
    async def ExecuteTask(self, request, context):
        """Execute a task on this node."""
        try:
            task_id = request.task_id
            agent_type = request.agent_type
            payload = json.loads(request.payload)
            
            # Execute the task
            result = await self.executor.execute_task_locally(task_id, agent_type, payload)
            
            # Return response
            return {
                "success": True,
                "result": json.dumps(result),
                "error": ""
            }
        except Exception as e:
            logger.error(f"Failed to execute task: {e}")
            return {
                "success": False,
                "result": "",
                "error": str(e)
            }
    
    async def Heartbeat(self, request, context):
        """Handle heartbeat from another node."""
        node_id = request.node_id
        load = request.current_load
        
        # Update node information
        self.executor.update_node_heartbeat(node_id, load)
        
        return {"success": True}
    
    async def GetClusterStatus(self, request, context):
        """Get cluster status information."""
        nodes = self.executor.get_cluster_nodes()
        tasks = self.executor.get_pending_tasks()
        
        return {
            "nodes": [asdict(node) for node in nodes],
            "pending_tasks": len(tasks),
            "leader_id": self.executor.consensus_layer.get_leader_id()
        }


class DistributedExecutor:
    """Distributed Agent Execution Engine."""
    
    def __init__(self, 
                 node_id: str = None,
                 host: str = None,
                 port: int = None,
                 redis_url: str = None,
                 peers: List[Tuple[str, str, int]] = None):
        """
        Initialize the distributed executor.
        
        Args:
            node_id: Unique identifier for this node
            host: Host address for this node
            port: Port for gRPC communication
            redis_url: Redis connection URL
            peers: List of (node_id, host, port) for other nodes in cluster
        """
        # Configuration
        self.node_id = node_id or f"node-{hashlib.md5(os.urandom(16)).hexdigest()[:8]}"
        self.host = host or os.getenv("NODE_HOST", "localhost")
        self.port = port or int(os.getenv("NODE_PORT", "50051"))
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.peers = peers or []
        
        # Initialize components
        self.state_manager = DistributedStateManager(self.redis_url)
        self.consensus_layer = RaftConsensusLayer(
            node_id=self.node_id,
            host=self.host,
            port=self.port + 1000,  # Raft uses different port
            peers=self.peers
        )
        
        # Node information
        self.node_info = NodeInfo(
            node_id=self.node_id,
            host=self.host,
            port=self.port,
            status=NodeStatus.INITIALIZING,
            capacity=int(os.getenv("NODE_CAPACITY", "10")),
            current_load=0,
            last_heartbeat=time.time(),
            metadata={
                "version": "1.0.0",
                "capabilities": ["agent_execution", "state_sync"]
            }
        )
        
        # Task execution
        self.task_executor = ThreadPoolExecutor(
            max_workers=self.node_info.capacity,
            thread_name_prefix=f"task-executor-{self.node_id}"
        )
        self.active_tasks: Dict[str, asyncio.Task] = {}
        
        # Agent registry
        self.agent_registry: Dict[str, Callable] = {}
        self._register_default_nexus()
        
        # gRPC server
        self.grpc_server = None
        self._init_grpc_server()
        
        # Background tasks
        self._background_tasks = []
        self._running = False
        
        logger.info(f"Distributed executor initialized: {self.node_id}")
    
    def _register_default_nexus(self):
        """Register default agent types."""
        # Register existing modules as agent types
        self.register_agent("prompt_optimizer", self._execute_prompt_optimizer)
        self.register_agent("design_extractor", self._execute_design_extractor)
        self.register_agent("api_designer", self._execute_api_designer)
        
        # Register custom agent executor
        self.register_agent("custom", self._execute_custom_agent)
    
    def register_agent(self, agent_type: str, executor: Callable):
        """Register an agent type with its executor function."""
        self.agent_registry[agent_type] = executor
        logger.info(f"Registered agent type: {agent_type}")
    
    def _init_grpc_server(self):
        """Initialize gRPC server."""
        self.grpc_server = grpc.aio.server()
        
        # Add servicer to server
        # Note: In production, this would use generated gRPC code
        # For now, we'll implement the server interface directly
        
        # Bind server to port
        self.grpc_server.add_insecure_port(f"{self.host}:{self.port}")
        logger.info(f"gRPC server initialized on {self.host}:{self.port}")
    
    async def start(self):
        """Start the distributed executor."""
        self._running = True
        
        # Register node in cluster
        await self._register_node()
        
        # Start gRPC server
        await self.grpc_server.start()
        
        # Start background tasks
        self._background_tasks = [
            asyncio.create_task(self._heartbeat_loop()),
            asyncio.create_task(self._task_scheduler_loop()),
            asyncio.create_task(self._health_check_loop()),
            asyncio.create_task(self._leader_election_loop())
        ]
        
        # Update node status
        self.node_info.status = NodeStatus.ACTIVE
        await self._update_node_status()
        
        logger.info(f"Distributed executor started: {self.node_id}")
    
    async def stop(self):
        """Stop the distributed executor."""
        self._running = False
        
        # Update node status to draining
        self.node_info.status = NodeStatus.DRAINING
        await self._update_node_status()
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for active tasks to complete
        if self.active_tasks:
            logger.info(f"Waiting for {len(self.active_tasks)} active tasks to complete...")
            await asyncio.gather(*self.active_tasks.values(), return_exceptions=True)
        
        # Stop gRPC server
        await self.grpc_server.stop(grace=5)
        
        # Stop Raft consensus
        self.consensus_layer.stop()
        
        # Shutdown task executor
        self.task_executor.shutdown(wait=True)
        
        # Remove node from cluster
        await self._unregister_node()
        
        logger.info(f"Distributed executor stopped: {self.node_id}")
    
    async def _register_node(self):
        """Register this node in the cluster."""
        self.node_info.last_heartbeat = time.time()
        success = self.state_manager.store_node(self.node_info)
        if success:
            logger.info(f"Node {self.node_id} registered in cluster")
        else:
            logger.error(f"Failed to register node {self.node_id}")
    
    async def _unregister_node(self):
        """Remove this node from the cluster."""
        try:
            key = f"node:{self.node_id}"
            self.state_manager.redis_client.delete(key)
            self.state_manager.redis_client.srem("nodes", self.node_id)
            logger.info(f"Node {self.node_id} unregistered from cluster")
        except Exception as e:
            logger.error(f"Failed to unregister node {self.node_id}: {e}")
    
    async def _update_node_status(self):
        """Update node status in cluster."""
        self.node_info.last_heartbeat = time.time()
        self.state_manager.store_node(self.node_info)
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats to cluster."""
        while self._running:
            try:
                await self._update_node_status()
                await asyncio.sleep(5)  # Heartbeat every 5 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(5)
    
    async def _task_scheduler_loop(self):
        """Schedule tasks from queue to available nodes."""
        while self._running:
            try:
                # Only leader schedules tasks
                if self.consensus_layer.is_current_leader():
                    await self._schedule_pending_tasks()
                await asyncio.sleep(1)  # Check every second
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Task scheduler error: {e}")
                await asyncio.sleep(1)
    
    async def _health_check_loop(self):
        """Monitor health of cluster nodes."""
        while self._running:
            try:
                await self._check_node_health()
                await asyncio.sleep(10)  # Check every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(10)
    
    async def _leader_election_loop(self):
        """Monitor and handle leader changes."""
        while self._running:
            try:
                current_leader = self.consensus_layer.get_leader_id()
                if current_leader != self.consensus_layer.leader_id:
                    self.consensus_layer.leader_id = current_leader
                    self.consensus_layer.is_leader = (current_leader == self.node_id)
                    
                    if self.consensus_layer.is_leader:
                        logger.info(f"Node {self.node_id} became leader")
                    else:
                        logger.info(f"Leader changed to {current_leader}")
                
                await asyncio.sleep(2)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Leader election error: {e}")
                await asyncio.sleep(2)
    
    async def _schedule_pending_tasks(self):
        """Schedule pending tasks to available nodes."""
        pending_tasks = self.state_manager.get_pending_tasks(limit=5)
        
        for task in pending_tasks:
            # Find best node for task
            best_node = await self._select_node_for_task(task)
            
            if best_node:
                # Assign task to node
                task.assigned_to = best_node.node_id
                task.status = TaskStatus.ASSIGNED
                self.state_manager.store_task(task)
                
                # Send task to node
                await self._send_task_to_node(task, best_node)
                
                logger.info(f"Task {task.task_id} assigned to node {best_node.node_id}")
    
    async def _select_node_for_task(self, task: Task) -> Optional[NodeInfo]:
        """Select the best node for a task based on load balancing."""
        nodes = self.state_manager.get_all_nodes()
        active_nodes = [n for n in nodes if n.status == NodeStatus.ACTIVE]
        
        if not active_nodes:
            return None
        
        # Simple load balancing: select node with lowest load
        active_nodes.sort(key=lambda n: n.current_load / n.capacity)
        selected_node = active_nodes[0]
        
        # Check if node has capacity
        if selected_node.current_load < selected_node.capacity:
            return selected_node
        
        return None
    
    async def _send_task_to_node(self, task: Task, node: NodeInfo):
        """Send task to node for execution."""
        try:
            # Create gRPC channel to node
            channel = grpc.aio.insecure_channel(f"{node.host}:{node.port}")
            
            # Create stub (in production, use generated gRPC stub)
            # For now, we'll simulate the RPC call
            
            # Prepare request
            request = {
                "task_id": task.task_id,
                "agent_type": task.agent_type,
                "payload": json.dumps(task.payload)
            }
            
            # Make RPC call
            # In production: response = await stub.ExecuteTask(request)
            
            # Update node load
            node.current_load += 1
            self.state_manager.store_node(node)
            
            await channel.close()
            
        except Exception as e:
            logger.error(f"Failed to send task to node {node.node_id}: {e}")
            # Mark task as failed
            task.status = TaskStatus.FAILED
            task.error = f"Failed to send to node: {e}"
            self.state_manager.store_task(task)
    
    async def _check_node_health(self):
        """Check health of cluster nodes and handle failures."""
        nodes = self.state_manager.get_all_nodes()
        current_time = time.time()
        
        for node in nodes:
            # Check if node is stale (no heartbeat for 30 seconds)
            if current_time - node.last_heartbeat > 30:
                if node.status != NodeStatus.FAILED:
                    logger.warning(f"Node {node.node_id} appears to be down")
                    node.status = NodeStatus.FAILED
                    self.state_manager.store_node(node)
                    
                    # Reassign tasks from failed node
                    await self._reassign_tasks_from_node(node.node_id)
    
    async def _reassign_tasks_from_node(self, node_id: str):
        """Reassign tasks from a failed node."""
        # Get all tasks assigned to this node
        # In production, we'd query Redis for tasks with assigned_to = node_id
        # For now, we'll implement a simple version
        
        logger.info(f"Reassigning tasks from failed node {node_id}")
        
        # Update node load
        node = self.state_manager.get_node(node_id)
        if node:
            node.current_load = 0
            self.state_manager.store_node(node)
    
    async def execute_task_locally(self, task_id: str, agent_type: str, 
                                  payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task on this node."""
        # Update node load
        self.node_info.current_load += 1
        await self._update_node_status()
        
        try:
            # Get agent executor
            executor = self.agent_registry.get(agent_type)
            if not executor:
                raise ValueError(f"Unknown agent type: {agent_type}")
            
            # Execute agent
            result = await executor(payload)
            
            # Update task status
            self.state_manager.update_task_status(
                task_id, TaskStatus.COMPLETED, result=result
            )
            
            return result
            
        except Exception as e:
            # Update task status with error
            self.state_manager.update_task_status(
                task_id, TaskStatus.FAILED, error=str(e)
            )
            raise
            
        finally:
            # Update node load
            self.node_info.current_load -= 1
            await self._update_node_status()
    
    async def _execute_prompt_optimizer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute prompt optimizer agent."""
        optimizer = PromptOptimizer()
        result = optimizer.optimize(
            prompt=payload.get("prompt", ""),
            context=payload.get("context", ""),
            constraints=payload.get("constraints", [])
        )
        return {"optimized_prompt": result}
    
    async def _execute_design_extractor(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute design extractor agent."""
        extractor = DesignExtractor()
        result = extractor.extract(
            url=payload.get("url", ""),
            options=payload.get("options", {})
        )
        return {"design_data": result}
    
    async def _execute_api_designer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute API designer agent."""
        # Use API design template
        api_response = APIResponse(
            status="success",
            data=payload.get("data", {}),
            message="API design generated"
        )
        return {"api_design": asdict(api_response)}
    
    async def _execute_custom_agent(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute custom agent with dynamic code execution."""
        # This is a simplified version - in production, use sandboxed execution
        code = payload.get("code", "")
        inputs = payload.get("inputs", {})
        
        try:
            # Create a restricted globals dictionary
            restricted_globals = {
                "__builtins__": {
                    "print": print,
                    "len": len,
                    "str": str,
                    "int": int,
                    "float": float,
                    "dict": dict,
                    "list": list,
                    "tuple": tuple,
                    "bool": bool,
                    "True": True,
                    "False": False,
                    "None": None,
                }
            }
            
            # Execute code
            exec(code, restricted_globals)
            
            # Get result from executed code
            if "execute" in restricted_globals:
                result = restricted_globals["execute"](**inputs)
                return {"result": result}
            else:
                return {"result": "Code executed but no execute function found"}
                
        except Exception as e:
            raise RuntimeError(f"Custom agent execution failed: {e}")
    
    def update_node_heartbeat(self, node_id: str, load: int):
        """Update heartbeat from another node."""
        node = self.state_manager.get_node(node_id)
        if node:
            node.last_heartbeat = time.time()
            node.current_load = load
            self.state_manager.store_node(node)
    
    def get_cluster_nodes(self) -> List[NodeInfo]:
        """Get all nodes in the cluster."""
        return self.state_manager.get_all_nodes()
    
    def get_pending_tasks(self) -> List[Task]:
        """Get pending tasks."""
        return self.state_manager.get_pending_tasks()
    
    async def submit_task(self, agent_type: str, payload: Dict[str, Any], 
                         priority: int = 0, timeout: int = 300) -> str:
        """Submit a task for execution."""
        # Generate task ID
        task_id = f"task-{hashlib.md5(json.dumps(payload).encode()).hexdigest()[:12]}"
        
        # Create task
        task = Task(
            task_id=task_id,
            agent_type=agent_type,
            payload=payload,
            priority=priority,
            timeout=timeout
        )
        
        # Store task
        self.state_manager.store_task(task)
        
        logger.info(f"Task submitted: {task_id} (agent: {agent_type})")
        return task_id
    
    async def get_task_result(self, task_id: str, timeout: int = 30) -> Optional[Dict[str, Any]]:
        """Get task result with timeout."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            task = self.state_manager.get_task(task_id)
            
            if not task:
                return None
            
            if task.status == TaskStatus.COMPLETED:
                return task.result
            elif task.status == TaskStatus.FAILED:
                raise RuntimeError(f"Task failed: {task.error}")
            
            await asyncio.sleep(0.5)
        
        raise TimeoutError(f"Task {task_id} timed out after {timeout} seconds")
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get cluster status information."""
        nodes = self.get_cluster_nodes()
        tasks = self.get_pending_tasks()
        
        return {
            "cluster_size": len(nodes),
            "active_nodes": len([n for n in nodes if n.status == NodeStatus.ACTIVE]),
            "total_capacity": sum(n.capacity for n in nodes),
            "current_load": sum(n.current_load for n in nodes),
            "pending_tasks": len(tasks),
            "leader_id": self.consensus_layer.get_leader_id(),
            "nodes": [asdict(node) for node in nodes]
        }


# Factory function for easy initialization
def create_distributed_executor(
    node_id: str = None,
    host: str = None,
    port: int = None,
    redis_url: str = None,
    peers: List[Tuple[str, str, int]] = None
) -> DistributedExecutor:
    """Create and configure a distributed executor instance."""
    return DistributedExecutor(
        node_id=node_id,
        host=host,
        port=port,
        redis_url=redis_url,
        peers=peers
    )


# CLI interface for testing
async def main():
    """Main function for CLI testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Distributed Agent Executor")
    parser.add_argument("--node-id", help="Node ID")
    parser.add_argument("--host", default="localhost", help="Host address")
    parser.add_argument("--port", type=int, default=50051, help="Port number")
    parser.add_argument("--redis-url", help="Redis URL")
    parser.add_argument("--peers", nargs="+", help="Peer nodes (format: node_id:host:port)")
    
    args = parser.parse_args()
    
    # Parse peers
    peers = []
    if args.peers:
        for peer in args.peers:
            parts = peer.split(":")
            if len(parts) == 3:
                peers.append((parts[0], parts[1], int(parts[2])))
    
    # Create executor
    executor = create_distributed_executor(
        node_id=args.node_id,
        host=args.host,
        port=args.port,
        redis_url=args.redis_url,
        peers=peers
    )
    
    try:
        # Start executor
        await executor.start()
        
        # Keep running
        print(f"Distributed executor running on {args.host}:{args.port}")
        print("Press Ctrl+C to stop")
        
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await executor.stop()


if __name__ == "__main__":
    asyncio.run(main())