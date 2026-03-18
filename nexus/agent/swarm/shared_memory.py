import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    NAVIGATOR = "navigator"
    EXTRACTOR = "extractor"
    VALIDATOR = "validator"
    COORDINATOR = "coordinator"


class ConflictResolutionStrategy(Enum):
    FIRST_WINS = "first_wins"
    MERGE = "merge"
    ABORT_NEWEST = "abort_newest"
    PRIORITY_BASED = "priority_based"


@dataclass
class DOMSnapshot:
    """Represents a snapshot of the browser state for optimistic concurrency."""
    timestamp: float
    version: int
    url: str
    title: str
    scroll_x: int = 0
    scroll_y: int = 0
    active_elements: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_fresh(self, current_version: int) -> bool:
        return self.version == current_version


@dataclass
class LockRecord:
    """Tracks locks for specific DOM elements or tabs to prevent race conditions."""
    owner_agent_id: str
    acquired_at: float
    operation_type: str  # 'click', 'scroll', 'input', 'read'
    is_locked: bool = True

    def release(self):
        self.is_locked = False


@dataclass
class Subtask:
    """Represents a decomposed subtask within the swarm."""
    id: str
    agent_role: AgentRole
    target_tab_id: str
    action: str
    parameters: Dict[str, Any]
    priority: int = 0
    timeout: float = 30.0
    dependencies: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "agent_role": self.agent_role.value,
            "target_tab_id": self.target_tab_id,
            "action": self.action,
            "parameters": self.parameters,
            "priority": self.priority,
            "timeout": self.timeout,
            "dependencies": self.dependencies,
        }


@dataclass
class TaskResult:
    """Result of a subtask execution."""
    subtask_id: str
    success: bool
    output: Any
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    updated_version: int = 0


class SharedMemory:
    """
    Shared state management and orchestration for Parallel Agent Orchestration (Swarm).
    Handles DOM state synchronization, optimistic concurrency control, and task distribution.
    """

    def __init__(
        self,
        default_timeout: float = 60.0,
        max_retries: int = 3,
        conflict_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.FIRST_WINS,
        tab_manager: Optional[Any] = None,  # Reference to PageManager or similar
    ):
        self.default_timeout = default_timeout
        self.max_retries = max_retries
        self.conflict_strategy = conflict_strategy

        # Core State
        self.state_version = 0
        self.dom_snapshots: Dict[str, DOMSnapshot] = field(default_factory=dict)
        self.lock_records: Dict[str, LockRecord] = field(default_factory=dict)
        self.tab_state: Dict[str, Dict[str, Any]] = field(default_factory=dict)
        self.agent_roles: Dict[str, AgentRole] = field(default_factory=dict)
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.results_buffer: Dict[str, TaskResult] = field(default_factory=dict)

        # Internal locks for state consistency
        self._state_lock = asyncio.Lock()
        self._version_counter = 0

        # Optional external dependency
        self._tab_manager = tab_manager

        # Metrics
        self.concurrency_level = 0
        self.total_tasks_completed = 0

    async def initialize_session(self, page_id: str, url: str):
        """Initialize the shared memory for a new browser session/tab."""
        async with self._state_lock:
            self.state_version += 1
            snapshot = DOMSnapshot(
                timestamp=time.time(),
                version=self.state_version,
                url=url,
                title="Untitled",
            )
            self.dom_snapshots[page_id] = snapshot
            self.tab_state[page_id] = {"url": url, "history": []}
            logger.info(f"Initialized session {page_id} with version {self.state_version}")

    async def register_agent(self, agent_id: str, role: AgentRole):
        """Register a specialized agent to the swarm."""
        async with self._state_lock:
            self.agent_roles[agent_id] = role
            self.concurrency_level += 1
            logger.info(f"Registered agent {agent_id} as {role.value}")

    async def unregister_agent(self, agent_id: str):
        """Remove an agent from the swarm."""
        async with self._state_lock:
            if agent_id in self.agent_roles:
                del self.agent_roles[agent_id]
                self.concurrency_level -= 1
            logger.info(f"Unregistered agent {agent_id}")

    async def acquire_lock(
        self,
        agent_id: str,
        target_id: str,
        operation: str
    ) -> bool:
        """
        Acquire an optimistic lock on a target (element or tab).
        Returns True if lock acquired, False if conflict detected or timeout.
        """
        key = f"{target_id}:{operation}"
        async with self._state_lock:
            record = self.lock_records.get(key)
            if not record:
                record = LockRecord(
                    owner_agent_id=agent_id,
                    acquired_at=time.time(),
                    operation_type=operation
                )
                self.lock_records[key] = record
                return True
            if record.owner_agent_id == agent_id:
                return True
            # Check for timeout on previous owner
            age = time.time() - record.acquired_at
            if age > self.default_timeout:
                record.owner_agent_id = agent_id
                record.acquired_at = time.time()
                return True
            return False

    async def release_lock(self, target_id: str, operation: str):
        """Release the lock on a target."""
        key = f"{target_id}:{operation}"
        if key in self.lock_records:
            del self.lock_records[key]
            logger.debug(f"Released lock on {key}")

    async def get_current_snapshot(self, page_id: str) -> Optional[DOMSnapshot]:
        """Get the latest DOM snapshot for a page."""
        async with self._state_lock:
            return self.dom_snapshots.get(page_id)

    async def update_snapshot(
        self,
        page_id: str,
        url: str,
        title: str,
        scroll_x: int = 0,
        scroll_y: int = 0,
        elements: Optional[Dict[str, Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Update the shared DOM state.
        Increments version to trigger optimistic concurrency checks.
        """
        async with self._state_lock:
            current_snapshot = self.dom_snapshots.get(page_id)
            if current_snapshot:
                new_version = current_snapshot.version + 1
            else:
                new_version = 1

            self.state_version = max(self.state_version, new_version)
            
            snapshot = DOMSnapshot(
                timestamp=time.time(),
                version=new_version,
                url=url,
                title=title,
                scroll_x=scroll_x,
                scroll_y=scroll_y,
                active_elements=elements or {},
                metadata=metadata or {}
            )
            self.dom_snapshots[page_id] = snapshot
            logger.debug(f"Updated snapshot for {page_id} to version {new_version}")
            return snapshot

    async def validate_concurrency(self, page_id: str) -> Tuple[bool, Optional[DOMSnapshot]]:
        """
        Check if the current agent's state is still valid (Optimistic Concurrency Control).
        Returns (is_valid, snapshot).
        """
        async with self._state_lock:
            snapshot = self.dom_snapshots.get(page_id)
            if not snapshot:
                return False, None
            # In a real implementation, agents would compare their local cached version
            # with the shared version here. For simplicity, we return the latest.
            return True, snapshot

    async def decompose_task(
        self,
        goal: str,
        page_id: str,
        agent_pool: List[str]
    ) -> List[Subtask]:
        """
        Breaks a high-level goal into subtasks for specialized agents.
        Example: Goal -> [Navigate, Extract, Validate]
        """
        subtasks = []
        
        # Heuristic decomposition
        if "navigate" in goal.lower() or "go to" in goal.lower():
            subtasks.append(Subtask(
                id=str(uuid.uuid4()),
                agent_role=AgentRole.NAVIGATOR,
                target_tab_id=page_id,
                action="navigate",
                parameters={"url": goal},
                priority=1
            ))
        if "extract" in goal.lower() or "get" in goal.lower():
            subtasks.append(Subtask(
                id=str(uuid.uuid4()),
                agent_role=AgentRole.EXTRACTOR,
                target_tab_id=page_id,
                action="extract",
                parameters