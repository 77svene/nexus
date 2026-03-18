# core/distributed/state_manager.py

import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor
import hashlib

import aioredis
import grpc
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Struct

# Import from existing modules
from .consensus import RaftNode, RaftState, LogEntry
from .executor import AgentExecutor, AgentTask, AgentStatus

# Configure logging
logger = logging.getLogger(__name__)

# Protobuf message definitions (would normally be in separate .proto files)
# For this implementation, we'll define them inline
class StateChangeType(Enum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    HEARTBEAT = "heartbeat"
    NODE_JOIN = "node_join"
    NODE_LEAVE = "node_leave"

@dataclass
class AgentState:
    agent_id: str
    task_id: str
    node_id: str
    status: AgentStatus
    data: Dict[str, Any]
    version: int
    created_at: float
    updated_at: float
    lease_expires_at: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentState':
        return cls(**data)

@dataclass
class NodeInfo:
    node_id: str
    address: str
    port: int
    capacity: int
    current_load: int
    last_heartbeat: float
    is_leader: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class DistributedStateManager:
    """
    Manages distributed state for nexus across multiple nodes using Raft consensus,
    Redis for state storage, and gRPC for communication.
    """
    
    def __init__(self, 
                 node_id: str,
                 redis_url: str = "redis://localhost:6379",
                 raft_port: int = 50051,
                 grpc_port: int = 50052,
                 cluster_nodes: Optional[List[str]] = None):
        """
        Initialize the distributed state manager.
        
        Args:
            node_id: Unique identifier for this node
            redis_url: Redis connection URL
            raft_port: Port for Raft consensus communication
            grpc_port: Port for gRPC communication
            cluster_nodes: List of other nodes in the format "host:port"
        """
        self.node_id = node_id
        self.redis_url = redis_url
        self.raft_port = raft_port
        self.grpc_port = grpc_port
        self.cluster_nodes = cluster_nodes or []
        
        # State management
        self.agent_states: Dict[str, AgentState] = {}
        self.node_info: Dict[str, NodeInfo] = {}
        self.lease_timers: Dict[str, asyncio.Task] = {}
        
        # Connections
        self.redis: Optional[aioredis.Redis] = None
        self.raft_node: Optional[RaftNode] = None
        self.grpc_server: Optional[grpc.aio.Server] = None
        
        # Thread pool for blocking operations
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Callbacks
        self.state_change_callbacks: List[Callable[[str, AgentState, StateChangeType], None]] = []
        self.node_change_callbacks: List[Callable[[str, NodeInfo, StateChangeType], None]] = []
        
        # Internal state
        self._running = False
        self._lock = asyncio.Lock()
        self._state_version = 0
        
    async def initialize(self) -> None:
        """Initialize all components of the state manager."""
        try:
            # Connect to Redis
            self.redis = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            logger.info(f"Connected to Redis at {self.redis_url}")
            
            # Initialize Raft consensus
            self.raft_node = RaftNode(
                node_id=self.node_id,
                port=self.raft_port,
                peers=self.cluster_nodes
            )
            await self.raft_node.start()
            logger.info(f"Raft node started on port {self.raft_port}")
            
            # Register Raft callbacks
            self.raft_node.on_commit(self._handle_raft_commit)
            self.raft_node.on_leader_change(self._handle_leader_change)
            
            # Start gRPC server
            await self._start_grpc_server()
            
            # Load existing state from Redis
            await self._load_state_from_redis()
            
            # Register this node
            await self._register_node()
            
            # Start background tasks
            self._running = True
            asyncio.create_task(self._heartbeat_loop())
            asyncio.create_task(self._lease_monitor_loop())
            asyncio.create_task(self._state_sync_loop())
            
            logger.info(f"Distributed state manager initialized on node {self.node_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize state manager: {e}")
            await self.shutdown()
            raise
    
    async def _start_grpc_server(self) -> None:
        """Start the gRPC server for node-to-node communication."""
        from . import state_manager_pb2_grpc  # Would be imported from generated protobuf
        
        self.grpc_server = grpc.aio.server()
        state_manager_pb2_grpc.add_StateManagerServicer_to_server(
            StateManagerServicer(self), 
            self.grpc_server
        )
        
        listen_addr = f'[::]:{self.grpc_port}'
        self.grpc_server.add_insecure_port(listen_addr)
        await self.grpc_server.start()
        logger.info(f"gRPC server started on {listen_addr}")
    
    async def _load_state_from_redis(self) -> None:
        """Load existing agent states and node info from Redis."""
        try:
            # Load agent states
            agent_keys = await self.redis.keys("agent:state:*")
            for key in agent_keys:
                data = await self.redis.get(key)
                if data:
                    state_dict = json.loads(data)
                    state = AgentState.from_dict(state_dict)
                    self.agent_states[state.agent_id] = state
            
            # Load node info
            node_keys = await self.redis.keys("node:info:*")
            for key in node_keys:
                data = await self.redis.get(key)
                if data:
                    node_dict = json.loads(data)
                    node = NodeInfo(**node_dict)
                    self.node_info[node.node_id] = node
            
            logger.info(f"Loaded {len(self.agent_states)} agent states and {len(self.node_info)} nodes from Redis")
            
        except Exception as e:
            logger.error(f"Failed to load state from Redis: {e}")
            raise
    
    async def _register_node(self) -> None:
        """Register this node in the cluster."""
        node_info = NodeInfo(
            node_id=self.node_id,
            address="localhost",  # Would be configurable
            port=self.grpc_port,
            capacity=100,  # Would be based on system resources
            current_load=0,
            last_heartbeat=time.time(),
            is_leader=self.raft_node.is_leader if self.raft_node else False
        )
        
        await self._update_node_info(node_info)
        
        # Announce node join via Raft
        if self.raft_node and self.raft_node.is_leader:
            await self._propose_state_change(
                change_type=StateChangeType.NODE_JOIN,
                key=f"node:{self.node_id}",
                value=node_info.to_dict()
            )
    
    async def _update_node_info(self, node_info: NodeInfo) -> None:
        """Update node information in memory and Redis."""
        async with self._lock:
            self.node_info[node_info.node_id] = node_info
            
            # Update in Redis
            key = f"node:info:{node_info.node_id}"
            await self.redis.set(key, json.dumps(node_info.to_dict()))
            
            # Notify callbacks
            for callback in self.node_change_callbacks:
                try:
                    callback(node_info.node_id, node_info, StateChangeType.UPDATE)
                except Exception as e:
                    logger.error(f"Error in node change callback: {e}")
    
    async def register_agent(self, 
                            agent_id: str, 
                            task_id: str, 
                            initial_data: Dict[str, Any],
                            lease_duration: float = 300.0) -> AgentState:
        """
        Register a new agent in the distributed system.
        
        Args:
            agent_id: Unique identifier for the agent
            task_id: Task identifier
            initial_data: Initial agent data
            lease_duration: Lease duration in seconds
            
        Returns:
            The created AgentState
        """
        if not self.raft_node or not self.raft_node.is_leader:
            # Forward to leader if we're not the leader
            leader_info = self._get_leader_info()
            if leader_info:
                return await self._forward_to_leader(
                    "register_agent", 
                    agent_id=agent_id, 
                    task_id=task_id,
                    initial_data=initial_data,
                    lease_duration=lease_duration
                )
            raise RuntimeError("No leader available")
        
        # Create agent state
        now = time.time()
        state = AgentState(
            agent_id=agent_id,
            task_id=task_id,
            node_id=self.node_id,
            status=AgentStatus.PENDING,
            data=initial_data,
            version=1,
            created_at=now,
            updated_at=now,
            lease_expires_at=now + lease_duration
        )
        
        # Propose state change via Raft
        await self._propose_state_change(
            change_type=StateChangeType.CREATE,
            key=f"agent:{agent_id}",
            value=state.to_dict()
        )
        
        # Start lease timer
        self._start_lease_timer(agent_id, lease_duration)
        
        return state
    
    async def update_agent_state(self, 
                                agent_id: str, 
                                updates: Dict[str, Any],
                                expected_version: Optional[int] = None) -> AgentState:
        """
        Update an agent's state with optimistic concurrency control.
        
        Args:
            agent_id: Agent identifier
            updates: Dictionary of updates to apply
            expected_version: Expected current version for optimistic locking
            
        Returns:
            Updated AgentState
        """
        async with self._lock:
            current_state = self.agent_states.get(agent_id)
            if not current_state:
                raise ValueError(f"Agent {agent_id} not found")
            
            # Check version for optimistic locking
            if expected_version is not None and current_state.version != expected_version:
                raise ValueError(
                    f"Version mismatch: expected {expected_version}, "
                    f"got {current_state.version}"
                )
            
            # Apply updates
            updated_data = {**current_state.data, **updates}
            now = time.time()
            
            updated_state = AgentState(
                agent_id=current_state.agent_id,
                task_id=current_state.task_id,
                node_id=current_state.node_id,
                status=current_state.status,
                data=updated_data,
                version=current_state.version + 1,
                created_at=current_state.created_at,
                updated_at=now,
                lease_expires_at=current_state.lease_expires_at
            )
            
            # Propose state change via Raft
            await self._propose_state_change(
                change_type=StateChangeType.UPDATE,
                key=f"agent:{agent_id}",
                value=updated_state.to_dict()
            )
            
            return updated_state
    
    async def move_agent(self, 
                        agent_id: str, 
                        target_node_id: str,
                        reason: str = "load_balancing") -> bool:
        """
        Move an agent to a different node.
        
        Args:
            agent_id: Agent identifier
            target_node_id: Target node identifier
            reason: Reason for the move
            
        Returns:
            True if move was successful
        """
        if not self.raft_node or not self.raft_node.is_leader:
            raise RuntimeError("Only leader can move nexus")
        
        async with self._lock:
            current_state = self.agent_states.get(agent_id)
            if not current_state:
                raise ValueError(f"Agent {agent_id} not found")
            
            target_node = self.node_info.get(target_node_id)
            if not target_node:
                raise ValueError(f"Target node {target_node_id} not found")
            
            # Update agent state to reflect move
            now = time.time()
            updated_state = AgentState(
                agent_id=current_state.agent_id,
                task_id=current_state.task_id,
                node_id=target_node_id,
                status=AgentStatus.MIGRATING,
                data={**current_state.data, "migration_reason": reason},
                version=current_state.version + 1,
                created_at=current_state.created_at,
                updated_at=now,
                lease_expires_at=current_state.lease_expires_at
            )
            
            # Propose state change
            await self._propose_state_change(
                change_type=StateChangeType.UPDATE,
                key=f"agent:{agent_id}",
                value=updated_state.to_dict()
            )
            
            # Update node loads
            source_node = self.node_info.get(current_state.node_id)
            if source_node:
                source_node.current_load = max(0, source_node.current_load - 1)
                await self._update_node_info(source_node)
            
            target_node.current_load += 1
            await self._update_node_info(target_node)
            
            logger.info(f"Moved agent {agent_id} from {current_state.node_id} to {target_node_id}")
            return True
    
    async def get_agent_state(self, agent_id: str) -> Optional[AgentState]:
        """Get the current state of an agent."""
        return self.agent_states.get(agent_id)
    
    async def get_all_agent_states(self, 
                                  node_id: Optional[str] = None,
                                  status: Optional[AgentStatus] = None) -> List[AgentState]:
        """Get all agent states, optionally filtered by node or status."""
        states = list(self.agent_states.values())
        
        if node_id:
            states = [s for s in states if s.node_id == node_id]
        
        if status:
            states = [s for s in states if s.status == status]
        
        return states
    
    async def get_node_info(self, node_id: Optional[str] = None) -> Dict[str, NodeInfo]:
        """Get information about nodes in the cluster."""
        if node_id:
            node = self.node_info.get(node_id)
            return {node_id: node} if node else {}
        return self.node_info.copy()
    
    async def get_cluster_load(self) -> Dict[str, Any]:
        """Get cluster load statistics."""
        total_nexus = len(self.agent_states)
        total_capacity = sum(n.capacity for n in self.node_info.values())
        total_load = sum(n.current_load for n in self.node_info.values())
        
        node_stats = {}
        for node_id, node in self.node_info.items():
            node_nexus = len([a for a in self.agent_states.values() if a.node_id == node_id])
            node_stats[node_id] = {
                "capacity": node.capacity,
                "current_load": node.current_load,
                "agent_count": node_nexus,
                "utilization": node.current_load / node.capacity if node.capacity > 0 else 0
            }
        
        return {
            "total_nexus": total_nexus,
            "total_capacity": total_capacity,
            "total_load": total_load,
            "overall_utilization": total_load / total_capacity if total_capacity > 0 else 0,
            "nodes": node_stats
        }
    
    async def rebalance_nexus(self, threshold: float = 0.8) -> Dict[str, List[str]]:
        """
        Automatically rebalance nexus across nodes based on load.
        
        Args:
            threshold: Load threshold above which rebalancing is triggered
            
        Returns:
            Dictionary mapping agent_id to new node_id for moved nexus
        """
        if not self.raft_node or not self.raft_node.is_leader:
            raise RuntimeError("Only leader can rebalance nexus")
        
        moves = {}
        cluster_load = await self.get_cluster_load()
        
        # Find overloaded and underloaded nodes
        overloaded = []
        underloaded = []
        
        for node_id, stats in cluster_load["nodes"].items():
            if stats["utilization"] > threshold:
                overloaded.append((node_id, stats["utilization"]))
            elif stats["utilization"] < threshold * 0.5:  # Underloaded if less than half threshold
                underloaded.append((node_id, stats["utilization"]))
        
        # Sort by load
        overloaded.sort(key=lambda x: x[1], reverse=True)
        underloaded.sort(key=lambda x: x[1])
        
        # Move nexus from overloaded to underloaded nodes
        for overloaded_node, _ in overloaded:
            if not underloaded:
                break
            
            # Get nexus on overloaded node
            nexus = await self.get_all_agent_states(node_id=overloaded_node)
            
            for agent in nexus:
                if not underloaded:
                    break
                
                target_node, _ = underloaded[0]
                
                try:
                    success = await self.move_agent(
                        agent.agent_id, 
                        target_node,
                        reason="auto_rebalance"
                    )
                    
                    if success:
                        moves[agent.agent_id] = target_node
                        
                        # Update underloaded list
                        target_stats = cluster_load["nodes"][target_node]
                        target_stats["current_load"] += 1
                        target_stats["utilization"] = (
                            target_stats["current_load"] / target_stats["capacity"]
                        )
                        
                        if target_stats["utilization"] >= threshold * 0.7:
                            underloaded.pop(0)
                
                except Exception as e:
                    logger.error(f"Failed to move agent {agent.agent_id}: {e}")
        
        logger.info(f"Rebalanced {len(moves)} nexus")
        return moves
    
    async def _propose_state_change(self, 
                                   change_type: StateChangeType,
                                   key: str,
                                   value: Dict[str, Any]) -> None:
        """Propose a state change through Raft consensus."""
        if not self.raft_node:
            raise RuntimeError("Raft node not initialized")
        
        # Create log entry
        entry = LogEntry(
            term=self.raft_node.current_term,
            index=self.raft_node.log.last_index + 1,
            data={
                "type": change_type.value,
                "key": key,
                "value": value,
                "timestamp": time.time(),
                "node_id": self.node_id
            }
        )
        
        # Propose to Raft
        success = await self.raft_node.propose(entry)
        if not success:
            raise RuntimeError("Failed to propose state change")
    
    async def _handle_raft_commit(self, entry: LogEntry) -> None:
        """Handle committed log entries from Raft."""
        try:
            data = entry.data
            change_type = StateChangeType(data["type"])
            key = data["key"]
            value = data["value"]
            
            async with self._lock:
                if change_type == StateChangeType.CREATE:
                    state = AgentState.from_dict(value)
                    self.agent_states[state.agent_id] = state
                    await self._persist_agent_state(state)
                    
                    # Notify callbacks
                    for callback in self.state_change_callbacks:
                        try:
                            callback(state.agent_id, state, change_type)
                        except Exception as e:
                            logger.error(f"Error in state change callback: {e}")
                
                elif change_type == StateChangeType.UPDATE:
                    state = AgentState.from_dict(value)
                    self.agent_states[state.agent_id] = state
                    await self._persist_agent_state(state)
                    
                    # Notify callbacks
                    for callback in self.state_change_callbacks:
                        try:
                            callback(state.agent_id, state, change_type)
                        except Exception as e:
                            logger.error(f"Error in state change callback: {e}")
                
                elif change_type == StateChangeType.DELETE:
                    agent_id = key.split(":")[-1]
                    if agent_id in self.agent_states:
                        del self.agent_states[agent_id]
                        await self.redis.delete(f"agent:state:{agent_id}")
                
                elif change_type == StateChangeType.NODE_JOIN:
                    node_info = NodeInfo(**value)
                    self.node_info[node_info.node_id] = node_info
                    await self._update_node_info(node_info)
                
                elif change_type == StateChangeType.NODE_LEAVE:
                    node_id = key.split(":")[-1]
                    if node_id in self.node_info:
                        del self.node_info[node_id]
                        await self.redis.delete(f"node:info:{node_id}")
                
                # Update version counter
                self._state_version += 1
                
        except Exception as e:
            logger.error(f"Error handling Raft commit: {e}")
    
    async def _persist_agent_state(self, state: AgentState) -> None:
        """Persist agent state to Redis."""
        key = f"agent:state:{state.agent_id}"
        await self.redis.set(key, json.dumps(state.to_dict()))
    
    async def _handle_leader_change(self, leader_id: Optional[str]) -> None:
        """Handle leader change events from Raft."""
        logger.info(f"Leader changed to {leader_id}")
        
        # Update node info for all nodes
        for node_id, node in self.node_info.items():
            node.is_leader = (node_id == leader_id)
            await self._update_node_info(node)
    
    def _get_leader_info(self) -> Optional[NodeInfo]:
        """Get information about the current leader."""
        for node in self.node_info.values():
            if node.is_leader:
                return node
        return None
    
    async def _forward_to_leader(self, method: str, **kwargs) -> Any:
        """Forward a request to the leader node."""
        leader_info = self._get_leader_info()
        if not leader_info:
            raise RuntimeError("No leader available")
        
        # In a real implementation, this would make a gRPC call to the leader
        # For now, we'll simulate it
        logger.info(f"Forwarding {method} to leader at {leader_info.address}:{leader_info.port}")
        
        # This would be implemented with gRPC client calls
        raise NotImplementedError("Leader forwarding not yet implemented")
    
    def _start_lease_timer(self, agent_id: str, duration: float) -> None:
        """Start a timer for agent lease expiration."""
        if agent_id in self.lease_timers:
            self.lease_timers[agent_id].cancel()
        
        async def lease_expire():
            await asyncio.sleep(duration)
            await self._handle_lease_expiration(agent_id)
        
        self.lease_timers[agent_id] = asyncio.create_task(lease_expire())
    
    async def _handle_lease_expiration(self, agent_id: str) -> None:
        """Handle agent lease expiration."""
        async with self._lock:
            state = self.agent_states.get(agent_id)
            if not state:
                return
            
            # Check if lease has actually expired
            if state.lease_expires_at > time.time():
                return
            
            logger.warning(f"Agent {agent_id} lease expired")
            
            # Update agent status
            state.status = AgentStatus.FAILED
            state.data["failure_reason"] = "lease_expired"
            state.version += 1
            state.updated_at = time.time()
            
            # Propose state update
            await self._propose_state_change(
                change_type=StateChangeType.UPDATE,
                key=f"agent:{agent_id}",
                value=state.to_dict()
            )
    
    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to maintain node liveness."""
        while self._running:
            try:
                # Update node heartbeat
                if self.node_id in self.node_info:
                    node = self.node_info[self.node_id]
                    node.last_heartbeat = time.time()
                    await self._update_node_info(node)
                
                # Check for dead nodes
                await self._check_dead_nodes()
                
                await asyncio.sleep(5)  # Heartbeat every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(5)
    
    async def _check_dead_nodes(self) -> None:
        """Check for nodes that haven't sent heartbeats and mark them as dead."""
        now = time.time()
        dead_nodes = []
        
        for node_id, node in self.node_info.items():
            if node_id == self.node_id:
                continue  # Don't check ourselves
            
            if now - node.last_heartbeat > 30:  # 30 second timeout
                dead_nodes.append(node_id)
        
        for node_id in dead_nodes:
            logger.warning(f"Node {node_id} appears to be dead")
            
            # Reassign nexus from dead node
            nexus = await self.get_all_agent_states(node_id=node_id)
            for agent in nexus:
                # Find a new node for the agent
                available_nodes = [
                    n for n in self.node_info.values()
                    if n.node_id != node_id and n.current_load < n.capacity
                ]
                
                if available_nodes:
                    # Choose least loaded node
                    target_node = min(available_nodes, key=lambda n: n.current_load / n.capacity)
                    try:
                        await self.move_agent(agent.agent_id, target_node.node_id, "node_failure")
                    except Exception as e:
                        logger.error(f"Failed to reassign agent {agent.agent_id}: {e}")
            
            # Remove dead node
            await self._propose_state_change(
                change_type=StateChangeType.NODE_LEAVE,
                key=f"node:{node_id}",
                value={}
            )
    
    async def _lease_monitor_loop(self) -> None:
        """Monitor and renew leases for nexus on this node."""
        while self._running:
            try:
                now = time.time()
                nexus_to_renew = []
                
                # Find nexus with leases expiring soon
                for agent_id, state in self.agent_states.items():
                    if state.node_id == self.node_id:
                        time_until_expiry = state.lease_expires_at - now
                        if time_until_expiry < 60:  # Renew if less than 60 seconds left
                            nexus_to_renew.append(agent_id)
                
                # Renew leases
                for agent_id in nexus_to_renew:
                    await self._renew_lease(agent_id)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in lease monitor loop: {e}")
                await asyncio.sleep(30)
    
    async def _renew_lease(self, agent_id: str) -> None:
        """Renew an agent's lease."""
        async with self._lock:
            state = self.agent_states.get(agent_id)
            if not state or state.node_id != self.node_id:
                return
            
            # Renew lease for 5 minutes
            new_expiry = time.time() + 300
            state.lease_expires_at = new_expiry
            state.version += 1
            state.updated_at = time.time()
            
            # Propose state update
            await self._propose_state_change(
                change_type=StateChangeType.UPDATE,
                key=f"agent:{agent_id}",
                value=state.to_dict()
            )
            
            # Restart lease timer
            self._start_lease_timer(agent_id, 300)
    
    async def _state_sync_loop(self) -> None:
        """Periodically sync state with Redis for consistency."""
        while self._running:
            try:
                # Sync agent states
                for agent_id, state in self.agent_states.items():
                    key = f"agent:state:{agent_id}"
                    redis_data = await self.redis.get(key)
                    
                    if redis_data:
                        redis_state = AgentState.from_dict(json.loads(redis_data))
                        if redis_state.version > state.version:
                            # Redis has newer version, update local
                            self.agent_states[agent_id] = redis_state
                
                await asyncio.sleep(60)  # Sync every minute
                
            except Exception as e:
                logger.error(f"Error in state sync loop: {e}")
                await asyncio.sleep(60)
    
    def register_state_change_callback(self, 
                                      callback: Callable[[str, AgentState, StateChangeType], None]) -> None:
        """Register a callback for state changes."""
        self.state_change_callbacks.append(callback)
    
    def register_node_change_callback(self,
                                     callback: Callable[[str, NodeInfo, StateChangeType], None]) -> None:
        """Register a callback for node changes."""
        self.node_change_callbacks.append(callback)
    
    async def shutdown(self) -> None:
        """Shutdown the state manager gracefully."""
        self._running = False
        
        # Cancel all timers
        for timer in self.lease_timers.values():
            timer.cancel()
        
        # Stop gRPC server
        if self.grpc_server:
            await self.grpc_server.stop(0)
        
        # Stop Raft node
        if self.raft_node:
            await self.raft_node.stop()
        
        # Close Redis connection
        if self.redis:
            await self.redis.close()
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        logger.info("Distributed state manager shutdown complete")

# gRPC Service Implementation
# In a real implementation, this would be generated from a .proto file
class StateManagerServicer:
    """gRPC servicer for state manager communication."""
    
    def __init__(self, state_manager: DistributedStateManager):
        self.state_manager = state_manager
    
    async def ProposeStateChange(self, request, context):
        """Handle state change proposals from other nodes."""
        try:
            # Forward to Raft consensus
            entry = LogEntry(
                term=request.term,
                index=request.index,
                data=json_format.MessageToDict(request.data)
            )
            
            success = await self.state_manager.raft_node.propose(entry)
            
            # Return response
            response = StateChangeResponse()
            response.success = success
            if not success:
                response.error = "Failed to propose state change"
            return response
            
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return StateChangeResponse(success=False, error=str(e))
    
    async def QueryState(self, request, context):
        """Handle state queries from other nodes."""
        try:
            agent_id = request.agent_id
            state = await self.state_manager.get_agent_state(agent_id)
            
            response = QueryResponse()
            if state:
                response.found = True
                json_format.ParseDict(state.to_dict(), response.state)
            else:
                response.found = False
            
            return response
            
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return QueryResponse(found=False, error=str(e))
    
    async def GetNodeInfo(self, request, context):
        """Get information about nodes in the cluster."""
        try:
            node_id = request.node_id if request.node_id else None
            nodes = await self.state_manager.get_node_info(node_id)
            
            response = NodeInfoResponse()
            for node_id, node_info in nodes.items():
                node_proto = response.nodes.add()
                json_format.ParseDict(node_info.to_dict(), node_proto)
            
            return response
            
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return NodeInfoResponse(error=str(e))
    
    async def Heartbeat(self, request, context):
        """Handle heartbeat from other nodes."""
        try:
            node_id = request.node_id
            if node_id in self.state_manager.node_info:
                node = self.state_manager.node_info[node_id]
                node.last_heartbeat = time.time()
                node.current_load = request.current_load
                await self.state_manager._update_node_info(node)
            
            response = HeartbeatResponse()
            response.success = True
            return response
            
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return HeartbeatResponse(success=False, error=str(e))

# Protobuf message classes (would normally be generated)
# These are simplified representations
class StateChangeRequest:
    def __init__(self):
        self.term = 0
        self.index = 0
        self.data = Struct()

class StateChangeResponse:
    def __init__(self):
        self.success = False
        self.error = ""

class QueryRequest:
    def __init__(self):
        self.agent_id = ""

class QueryResponse:
    def __init__(self):
        self.found = False
        self.state = Struct()
        self.error = ""

class NodeInfoRequest:
    def __init__(self):
        self.node_id = ""

class NodeInfoResponse:
    def __init__(self):
        self.nodes = []
        self.error = ""

class HeartbeatRequest:
    def __init__(self):
        self.node_id = ""
        self.current_load = 0

class HeartbeatResponse:
    def __init__(self):
        self.success = False
        self.error = ""

# Factory function for easy initialization
async def create_state_manager(
    node_id: str,
    redis_url: str = "redis://localhost:6379",
    raft_port: int = 50051,
    grpc_port: int = 50052,
    cluster_nodes: Optional[List[str]] = None
) -> DistributedStateManager:
    """
    Factory function to create and initialize a distributed state manager.
    
    Args:
        node_id: Unique identifier for this node
        redis_url: Redis connection URL
        raft_port: Port for Raft consensus
        grpc_port: Port for gRPC communication
        cluster_nodes: List of other nodes
        
    Returns:
        Initialized DistributedStateManager
    """
    manager = DistributedStateManager(
        node_id=node_id,
        redis_url=redis_url,
        raft_port=raft_port,
        grpc_port=grpc_port,
        cluster_nodes=cluster_nodes
    )
    
    await manager.initialize()
    return manager

# Integration with existing executor
class DistributedAgentExecutor(AgentExecutor):
    """
    Extended agent executor that works with distributed state manager.
    """
    
    def __init__(self, state_manager: DistributedStateManager):
        super().__init__()
        self.state_manager = state_manager
        self.state_manager.register_state_change_callback(self._on_state_change)
    
    async def execute_task(self, task: AgentTask) -> None:
        """Execute a task using distributed state management."""
        # Register agent in distributed state
        agent_state = await self.state_manager.register_agent(
            agent_id=task.agent_id,
            task_id=task.task_id,
            initial_data=task.data
        )
        
        # Execute task
        try:
            result = await super().execute_task(task)
            
            # Update state with result
            await self.state_manager.update_agent_state(
                agent_id=task.agent_id,
                updates={"result": result, "status": "completed"}
            )
            
        except Exception as e:
            # Update state with error
            await self.state_manager.update_agent_state(
                agent_id=task.agent_id,
                updates={"error": str(e), "status": "failed"}
            )
            raise
    
    def _on_state_change(self, agent_id: str, state: AgentState, change_type: StateChangeType) -> None:
        """Handle state changes from the distributed state manager."""
        if change_type == StateChangeType.UPDATE:
            if state.status == AgentStatus.MIGRATING:
                # Agent is being migrated to this node
                if state.node_id == self.state_manager.node_id:
                    asyncio.create_task(self._handle_agent_migration(agent_id, state))
    
    async def _handle_agent_migration(self, agent_id: str, state: AgentState) -> None:
        """Handle an agent being migrated to this node."""
        logger.info(f"Handling migration of agent {agent_id} to this node")
        
        # Recreate the task from state
        task = AgentTask(
            agent_id=state.agent_id,
            task_id=state.task_id,
            data=state.data
        )
        
        # Resume execution
        await self.execute_task(task)