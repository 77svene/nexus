"""
Distributed Agent Execution Engine with Raft Consensus
SOVEREIGN - Autonomous Open-Source Project Builder
"""

import asyncio
import json
import logging
import pickle
import random
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor
import threading
import hashlib

import grpc
from grpc import aio as aio_grpc
import redis
from redis import Redis
from redis.exceptions import RedisError

# Import existing modules
from core.distributed.executor import AgentExecutor, ExecutionContext, TaskResult
from plugins.llm_application_dev.skills.prompt_engineering_patterns.scripts.optimize_prompt import PromptOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
HEARTBEAT_INTERVAL = 0.5  # seconds
ELECTION_TIMEOUT_MIN = 1.5  # seconds
ELECTION_TIMEOUT_MAX = 3.0  # seconds
MAX_LOG_ENTRIES = 10000
STATE_SYNC_INTERVAL = 2.0  # seconds
NODE_TIMEOUT = 10.0  # seconds

class NodeState(Enum):
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"

@dataclass
class LogEntry:
    """Raft log entry for agent task coordination"""
    term: int
    index: int
    command: str
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    committed: bool = False
    applied: bool = False

@dataclass
class NodeInfo:
    """Information about a cluster node"""
    node_id: str
    address: str
    port: int
    state: NodeState = NodeState.FOLLOWER
    last_heartbeat: float = field(default_factory=time.time)
    current_term: int = 0
    voted_for: Optional[str] = None
    load: float = 0.0
    capabilities: List[str] = field(default_factory=list)
    active_tasks: int = 0

class DistributedConsensus:
    """
    Raft-based consensus layer for distributed agent coordination.
    Manages leader election, log replication, and state synchronization.
    """
    
    def __init__(self, node_id: str, redis_host: str = "localhost", redis_port: int = 6379):
        self.node_id = node_id
        self.state = NodeState.FOLLOWER
        self.current_term = 0
        self.voted_for = None
        self.log: List[LogEntry] = []
        self.commit_index = 0
        self.last_applied = 0
        self.next_index: Dict[str, int] = {}
        self.match_index: Dict[str, int] = {}
        
        # Redis for persistent state and coordination
        self.redis_client = Redis(host=redis_host, port=redis_port, decode_responses=False)
        self._init_redis_state()
        
        # Cluster management
        self.nodes: Dict[str, NodeInfo] = {}
        self.leader_id: Optional[str] = None
        self.election_timeout = random.uniform(ELECTION_TIMEOUT_MIN, ELECTION_TIMEOUT_MAX)
        self.last_heartbeat = time.time()
        
        # Task execution
        self.executor = AgentExecutor()
        self.task_queue = asyncio.Queue()
        self.completed_tasks: Dict[str, TaskResult] = {}
        self.task_assignments: Dict[str, str] = {}  # task_id -> node_id
        
        # State synchronization
        self.state_lock = threading.RLock()
        self.state_version = 0
        
        # Performance metrics
        self.metrics = {
            "tasks_processed": 0,
            "tasks_failed": 0,
            "avg_task_duration": 0.0,
            "cluster_load": 0.0
        }
        
        # gRPC server for inter-node communication
        self.grpc_server = None
        self.grpc_port = None
        
        # Background tasks
        self._running = False
        self._background_tasks = []
        
        logger.info(f"Initialized distributed consensus node: {node_id}")
    
    def _init_redis_state(self):
        """Initialize Redis state structures"""
        try:
            # Cluster state
            self.redis_client.hset("cluster:state", "version", "0")
            self.redis_client.hset("cluster:state", "leader", "")
            
            # Node registry
            self.redis_client.delete("cluster:nodes")
            
            # Task registry
            self.redis_client.delete("cluster:tasks")
            
            # Log storage
            self.redis_client.delete("cluster:log")
            
            logger.info("Redis state initialized")
        except RedisError as e:
            logger.error(f"Failed to initialize Redis: {e}")
            raise
    
    async def start(self, grpc_port: int = 50051):
        """Start the consensus node"""
        self._running = True
        self.grpc_port = grpc_port
        
        # Register node in cluster
        await self._register_node()
        
        # Start gRPC server
        await self._start_grpc_server()
        
        # Start background tasks
        self._background_tasks = [
            asyncio.create_task(self._election_timer()),
            asyncio.create_task(self._heartbeat_sender()),
            asyncio.create_task(self._state_synchronizer()),
            asyncio.create_task(self._task_processor()),
            asyncio.create_task(self._metrics_collector())
        ]
        
        logger.info(f"Consensus node started on port {grpc_port}")
    
    async def stop(self):
        """Stop the consensus node"""
        self._running = False
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Stop gRPC server
        if self.grpc_server:
            await self.grpc_server.stop(grace=5)
        
        # Unregister node
        await self._unregister_node()
        
        logger.info("Consensus node stopped")
    
    async def _register_node(self):
        """Register this node in the cluster"""
        node_info = {
            "node_id": self.node_id,
            "address": f"localhost:{self.grpc_port}",
            "state": self.state.value,
            "capabilities": ["agent_execution", "task_scheduling"],
            "load": 0.0,
            "last_seen": time.time()
        }
        
        try:
            self.redis_client.hset(
                "cluster:nodes",
                self.node_id,
                pickle.dumps(node_info)
            )
            logger.info(f"Node {self.node_id} registered in cluster")
        except RedisError as e:
            logger.error(f"Failed to register node: {e}")
    
    async def _unregister_node(self):
        """Remove this node from the cluster"""
        try:
            self.redis_client.hdel("cluster:nodes", self.node_id)
            logger.info(f"Node {self.node_id} unregistered from cluster")
        except RedisError as e:
            logger.error(f"Failed to unregister node: {e}")
    
    async def _start_grpc_server(self):
        """Start gRPC server for inter-node communication"""
        from concurrent import futures
        
        self.grpc_server = aio_grpc.server(
            futures.ThreadPoolExecutor(max_workers=10)
        )
        
        # Add service to server
        # Note: In production, we would add the actual gRPC service here
        # For now, we'll create a simple service
        
        self.grpc_server.add_insecure_port(f'[::]:{self.grpc_port}')
        await self.grpc_server.start()
        logger.info(f"gRPC server started on port {self.grpc_port}")
    
    async def _election_timer(self):
        """Handle election timeout and start elections if needed"""
        while self._running:
            await asyncio.sleep(0.1)  # Check every 100ms
            
            if self.state == NodeState.LEADER:
                continue
            
            time_since_heartbeat = time.time() - self.last_heartbeat
            
            if time_since_heartbeat > self.election_timeout:
                await self._start_election()
    
    async def _start_election(self):
        """Start a new election"""
        logger.info(f"Node {self.node_id} starting election for term {self.current_term + 1}")
        
        self.state = NodeState.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id
        self.last_heartbeat = time.time()
        
        # Request votes from other nodes
        votes_received = 1  # Vote for self
        votes_needed = (len(self.nodes) // 2) + 1
        
        for node_id, node_info in self.nodes.items():
            if node_id == self.node_id:
                continue
            
            if await self._request_vote(node_id):
                votes_received += 1
            
            if votes_received >= votes_needed:
                await self._become_leader()
                return
        
        # Election failed, return to follower state
        self.state = NodeState.FOLLOWER
        logger.info(f"Election failed for term {self.current_term}")
    
    async def _request_vote(self, node_id: str) -> bool:
        """Request vote from another node"""
        # In production, this would make a gRPC call
        # For now, simulate with Redis
        try:
            vote_key = f"cluster:votes:{self.current_term}"
            vote_data = {
                "candidate_id": self.node_id,
                "term": self.current_term,
                "last_log_index": len(self.log) - 1 if self.log else 0,
                "last_log_term": self.log[-1].term if self.log else 0
            }
            
            self.redis_client.hset(vote_key, node_id, pickle.dumps(vote_data))
            
            # Simulate vote response (in production, this would be async gRPC)
            # For simplicity, assume vote is granted if node is available
            return True
            
        except RedisError as e:
            logger.error(f"Failed to request vote from {node_id}: {e}")
            return False
    
    async def _become_leader(self):
        """Transition to leader state"""
        logger.info(f"Node {self.node_id} becoming leader for term {self.current_term}")
        
        self.state = NodeState.LEADER
        self.leader_id = self.node_id
        
        # Initialize leader state
        for node_id in self.nodes:
            if node_id != self.node_id:
                self.next_index[node_id] = len(self.log)
                self.match_index[node_id] = 0
        
        # Update cluster state
        try:
            self.redis_client.hset("cluster:state", "leader", self.node_id)
            self.redis_client.hset("cluster:state", "term", str(self.current_term))
        except RedisError as e:
            logger.error(f"Failed to update cluster state: {e}")
    
    async def _heartbeat_sender(self):
        """Send heartbeats to all nodes (leader only)"""
        while self._running:
            if self.state == NodeState.LEADER:
                await self._send_heartbeats()
            
            await asyncio.sleep(HEARTBEAT_INTERVAL)
    
    async def _send_heartbeats(self):
        """Send heartbeats to all followers"""
        for node_id, node_info in self.nodes.items():
            if node_id == self.node_id:
                continue
            
            try:
                # In production, this would be a gRPC call
                heartbeat_data = {
                    "term": self.current_term,
                    "leader_id": self.node_id,
                    "prev_log_index": len(self.log) - 1 if self.log else 0,
                    "prev_log_term": self.log[-1].term if self.log else 0,
                    "entries": [],
                    "leader_commit": self.commit_index
                }
                
                self.redis_client.hset(
                    f"cluster:heartbeats:{node_id}",
                    self.node_id,
                    pickle.dumps(heartbeat_data)
                )
                
                node_info.last_heartbeat = time.time()
                
            except RedisError as e:
                logger.error(f"Failed to send heartbeat to {node_id}: {e}")
    
    async def _state_synchronizer(self):
        """Synchronize state across cluster"""
        while self._running:
            await asyncio.sleep(STATE_SYNC_INTERVAL)
            
            try:
                # Get current cluster state
                cluster_state = self.redis_client.hgetall("cluster:state")
                current_version = int(cluster_state.get(b"version", b"0"))
                
                if current_version > self.state_version:
                    # State has changed, update local state
                    await self._sync_cluster_state()
                    self.state_version = current_version
                
                # Update our node info
                await self._update_node_info()
                
            except RedisError as e:
                logger.error(f"State synchronization failed: {e}")
    
    async def _sync_cluster_state(self):
        """Synchronize with cluster state"""
        try:
            # Get all nodes
            nodes_data = self.redis_client.hgetall("cluster:nodes")
            self.nodes.clear()
            
            for node_id, node_data in nodes_data.items():
                node_info = pickle.loads(node_data)
                self.nodes[node_info["node_id"]] = NodeInfo(
                    node_id=node_info["node_id"],
                    address=node_info["address"],
                    port=int(node_info["address"].split(":")[1]),
                    state=NodeState(node_info["state"]),
                    last_heartbeat=node_info["last_heartbeat"],
                    load=node_info["load"],
                    capabilities=node_info["capabilities"],
                    active_tasks=node_info["active_tasks"]
                )
            
            # Get leader
            leader_id = self.redis_client.hget("cluster:state", "leader")
            if leader_id:
                self.leader_id = leader_id.decode()
            
            logger.debug(f"Synced cluster state: {len(self.nodes)} nodes, leader: {self.leader_id}")
            
        except RedisError as e:
            logger.error(f"Failed to sync cluster state: {e}")
    
    async def _update_node_info(self):
        """Update this node's information in Redis"""
        try:
            node_info = {
                "node_id": self.node_id,
                "address": f"localhost:{self.grpc_port}",
                "state": self.state.value,
                "capabilities": ["agent_execution", "task_scheduling"],
                "load": self._calculate_load(),
                "last_seen": time.time(),
                "active_tasks": len(self.task_assignments)
            }
            
            self.redis_client.hset(
                "cluster:nodes",
                self.node_id,
                pickle.dumps(node_info)
            )
            
        except RedisError as e:
            logger.error(f"Failed to update node info: {e}")
    
    def _calculate_load(self) -> float:
        """Calculate current node load"""
        if not self.nodes:
            return 0.0
        
        total_tasks = sum(node.active_tasks for node in self.nodes.values())
        if total_tasks == 0:
            return 0.0
        
        return self.nodes[self.node_id].active_tasks / total_tasks
    
    async def _task_processor(self):
        """Process tasks from the queue"""
        while self._running:
            try:
                # Wait for tasks with timeout
                task_data = await asyncio.wait_for(
                    self.task_queue.get(),
                    timeout=1.0
                )
                
                await self._process_task(task_data)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Task processing error: {e}")
    
    async def _process_task(self, task_data: Dict[str, Any]):
        """Process a single task"""
        task_id = task_data.get("task_id")
        agent_type = task_data.get("agent_type")
        context = task_data.get("context")
        
        logger.info(f"Processing task {task_id} on node {self.node_id}")
        
        try:
            # Create execution context
            execution_context = ExecutionContext(
                task_id=task_id,
                agent_type=agent_type,
                context=context,
                node_id=self.node_id
            )
            
            # Execute the task
            result = await self.executor.execute(execution_context)
            
            # Store result
            self.completed_tasks[task_id] = result
            
            # Update metrics
            self.metrics["tasks_processed"] += 1
            if result.success:
                self.metrics["avg_task_duration"] = (
                    (self.metrics["avg_task_duration"] * (self.metrics["tasks_processed"] - 1) + result.duration)
                    / self.metrics["tasks_processed"]
                )
            else:
                self.metrics["tasks_failed"] += 1
            
            # Replicate result to cluster
            await self._replicate_task_result(task_id, result)
            
            logger.info(f"Task {task_id} completed: {result.success}")
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            self.metrics["tasks_failed"] += 1
    
    async def _replicate_task_result(self, task_id: str, result: TaskResult):
        """Replicate task result to cluster"""
        try:
            # Create log entry for result
            log_entry = LogEntry(
                term=self.current_term,
                index=len(self.log),
                command="task_result",
                data={
                    "task_id": task_id,
                    "result": pickle.dumps(result),
                    "node_id": self.node_id,
                    "timestamp": time.time()
                }
            )
            
            # Append to log
            self.log.append(log_entry)
            
            # Store in Redis
            self.redis_client.hset(
                "cluster:tasks",
                task_id,
                pickle.dumps({
                    "result": result,
                    "node_id": self.node_id,
                    "timestamp": time.time()
                })
            )
            
            # If leader, replicate to followers
            if self.state == NodeState.LEADER:
                await self._replicate_log()
            
        except RedisError as e:
            logger.error(f"Failed to replicate task result: {e}")
    
    async def _replicate_log(self):
        """Replicate log entries to followers"""
        for node_id, node_info in self.nodes.items():
            if node_id == self.node_id:
                continue
            
            try:
                # Get entries to send
                next_idx = self.next_index.get(node_id, 0)
                entries = self.log[next_idx:] if next_idx < len(self.log) else []
                
                if not entries:
                    continue
                
                # Prepare append entries request
                prev_log_index = next_idx - 1
                prev_log_term = self.log[prev_log_index].term if prev_log_index >= 0 else 0
                
                # Send to follower (in production, this would be gRPC)
                append_data = {
                    "term": self.current_term,
                    "leader_id": self.node_id,
                    "prev_log_index": prev_log_index,
                    "prev_log_term": prev_log_term,
                    "entries": entries,
                    "leader_commit": self.commit_index
                }
                
                self.redis_client.hset(
                    f"cluster:replication:{node_id}",
                    self.node_id,
                    pickle.dumps(append_data)
                )
                
                # Update next_index
                self.next_index[node_id] = len(self.log)
                self.match_index[node_id] = len(self.log) - 1
                
            except RedisError as e:
                logger.error(f"Failed to replicate log to {node_id}: {e}")
    
    async def _metrics_collector(self):
        """Collect and update cluster metrics"""
        while self._running:
            await asyncio.sleep(5.0)  # Collect every 5 seconds
            
            try:
                # Calculate cluster load
                total_load = sum(node.load for node in self.nodes.values())
                avg_load = total_load / len(self.nodes) if self.nodes else 0.0
                self.metrics["cluster_load"] = avg_load
                
                # Store metrics in Redis
                self.redis_client.hset(
                    "cluster:metrics",
                    self.node_id,
                    pickle.dumps(self.metrics)
                )
                
                logger.debug(f"Metrics updated: {self.metrics}")
                
            except RedisError as e:
                logger.error(f"Failed to collect metrics: {e}")
    
    async def submit_task(self, task_data: Dict[str, Any]) -> str:
        """Submit a task to the cluster"""
        task_id = str(uuid.uuid4())
        task_data["task_id"] = task_id
        
        # If we're the leader, assign the task
        if self.state == NodeState.LEADER:
            assigned_node = await self._assign_task(task_data)
            task_data["assigned_node"] = assigned_node
        
        # Add to queue
        await self.task_queue.put(task_data)
        
        # Store task metadata
        try:
            self.redis_client.hset(
                "cluster:task_queue",
                task_id,
                pickle.dumps({
                    "task_data": task_data,
                    "submitted_by": self.node_id,
                    "submitted_at": time.time(),
                    "status": "queued"
                })
            )
        except RedisError as e:
            logger.error(f"Failed to store task metadata: {e}")
        
        return task_id
    
    async def _assign_task(self, task_data: Dict[str, Any]) -> str:
        """Assign task to optimal node"""
        # Simple load-based assignment
        available_nodes = [
            node for node in self.nodes.values()
            if node.state == NodeState.FOLLOWER and node.load < 0.8
        ]
        
        if not available_nodes:
            # Assign to least loaded node
            available_nodes = list(self.nodes.values())
        
        # Sort by load
        available_nodes.sort(key=lambda x: x.load)
        
        # Assign to least loaded node
        assigned_node = available_nodes[0].node_id
        self.task_assignments[task_data["task_id"]] = assigned_node
        
        logger.info(f"Assigned task {task_data['task_id']} to node {assigned_node}")
        return assigned_node
    
    async def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """Get result of a completed task"""
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        
        # Try to get from Redis
        try:
            result_data = self.redis_client.hget("cluster:tasks", task_id)
            if result_data:
                data = pickle.loads(result_data)
                return data["result"]
        except RedisError as e:
            logger.error(f"Failed to get task result: {e}")
        
        return None
    
    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get current cluster status"""
        return {
            "node_id": self.node_id,
            "state": self.state.value,
            "current_term": self.current_term,
            "leader_id": self.leader_id,
            "nodes": len(self.nodes),
            "log_length": len(self.log),
            "commit_index": self.commit_index,
            "metrics": self.metrics
        }

class DistributedExecutor:
    """
    Distributed executor that uses consensus layer for coordination.
    Extends the existing single-machine executor.
    """
    
    def __init__(self, node_id: str, redis_host: str = "localhost", redis_port: int = 6379):
        self.consensus = DistributedConsensus(node_id, redis_host, redis_port)
        self.node_id = node_id
        self.prompt_optimizer = PromptOptimizer()
        
        # Task routing
        self.task_handlers = {
            "agent_execution": self._handle_agent_task,
            "prompt_optimization": self._handle_prompt_optimization,
            "system_command": self._handle_system_command
        }
        
        logger.info(f"Distributed executor initialized: {node_id}")
    
    async def start(self, grpc_port: int = 50051):
        """Start the distributed executor"""
        await self.consensus.start(grpc_port)
        logger.info(f"Distributed executor started on port {grpc_port}")
    
    async def stop(self):
        """Stop the distributed executor"""
        await self.consensus.stop()
        logger.info("Distributed executor stopped")
    
    async def execute_distributed(self, task_type: str, task_data: Dict[str, Any]) -> str:
        """Execute a task across the distributed cluster"""
        if task_type not in self.task_handlers:
            raise ValueError(f"Unknown task type: {task_type}")
        
        # Submit task to consensus layer
        task_id = await self.consensus.submit_task({
            "type": task_type,
            "data": task_data,
            "submitted_by": self.node_id,
            "timestamp": time.time()
        })
        
        logger.info(f"Submitted distributed task: {task_id}")
        return task_id
    
    async def _handle_agent_task(self, task_data: Dict[str, Any]) -> TaskResult:
        """Handle agent execution task"""
        # This would integrate with the existing AgentExecutor
        # For now, simulate task execution
        await asyncio.sleep(0.1)  # Simulate work
        
        return TaskResult(
            success=True,
            data={"result": "Agent task completed"},
            duration=0.1,
            node_id=self.node_id
        )
    
    async def _handle_prompt_optimization(self, task_data: Dict[str, Any]) -> TaskResult:
        """Handle prompt optimization task"""
        prompt = task_data.get("prompt", "")
        context = task_data.get("context", {})
        
        # Use the prompt optimizer
        optimized = self.prompt_optimizer.optimize_prompt(prompt, context)
        
        return TaskResult(
            success=True,
            data={"optimized_prompt": optimized},
            duration=0.05,
            node_id=self.node_id
        )
    
    async def _handle_system_command(self, task_data: Dict[str, Any]) -> TaskResult:
        """Handle system command task"""
        command = task_data.get("command", "")
        
        # In production, this would execute system commands safely
        # For now, simulate
        await asyncio.sleep(0.2)
        
        return TaskResult(
            success=True,
            data={"command_output": f"Executed: {command}"},
            duration=0.2,
            node_id=self.node_id
        )
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a distributed task"""
        result = await self.consensus.get_task_result(task_id)
        
        if result:
            return {
                "task_id": task_id,
                "status": "completed",
                "success": result.success,
                "result": result.data,
                "duration": result.duration,
                "node_id": result.node_id
            }
        
        # Check if task is still queued
        try:
            task_data = self.consensus.redis_client.hget("cluster:task_queue", task_id)
            if task_data:
                data = pickle.loads(task_data)
                return {
                    "task_id": task_id,
                    "status": "queued",
                    "submitted_at": data["submitted_at"]
                }
        except RedisError:
            pass
        
        return {
            "task_id": task_id,
            "status": "unknown"
        }
    
    async def scale_cluster(self, desired_nodes: int):
        """Scale the cluster to desired number of nodes"""
        current_nodes = len(self.consensus.nodes)
        
        if desired_nodes > current_nodes:
            logger.info(f"Scaling up from {current_nodes} to {desired_nodes} nodes")
            # In production, this would launch new nodes
            # For now, just log
        elif desired_nodes < current_nodes:
            logger.info(f"Scaling down from {current_nodes} to {desired_nodes} nodes")
            # In production, this would gracefully remove nodes
            # For now, just log

# Factory function for creating distributed executors
def create_distributed_executor(node_id: str = None, redis_host: str = "localhost", redis_port: int = 6379) -> DistributedExecutor:
    """Create a new distributed executor instance"""
    if node_id is None:
        node_id = f"node-{uuid.uuid4().hex[:8]}"
    
    return DistributedExecutor(node_id, redis_host, redis_port)

# Example usage
async def example_usage():
    """Example of using the distributed executor"""
    # Create executor
    executor = create_distributed_executor()
    
    try:
        # Start executor
        await executor.start()
        
        # Submit some tasks
        task_ids = []
        for i in range(5):
            task_id = await executor.execute_distributed(
                "prompt_optimization",
                {"prompt": f"Optimize this prompt: {i}", "context": {"domain": "code"}}
            )
            task_ids.append(task_id)
        
        # Wait for tasks to complete
        await asyncio.sleep(2)
        
        # Check results
        for task_id in task_ids:
            status = await executor.get_task_status(task_id)
            print(f"Task {task_id}: {status}")
        
        # Get cluster status
        cluster_status = await executor.consensus.get_cluster_status()
        print(f"Cluster status: {cluster_status}")
        
    finally:
        # Stop executor
        await executor.stop()

if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())