"""
Intelligent Agent Composition Engine - Dynamic agent composition based on task requirements.
Implements capability graph with constraint satisfaction and optimization algorithms.
"""

import asyncio
import heapq
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

from core.distributed.executor import DistributedExecutor
from core.distributed.state_manager import StateManager
from core.distributed.consensus import ConsensusProtocol
from monitoring.metrics_collector import MetricsCollector
from monitoring.cost_tracker import CostTracker
from core.resilience.circuit_breaker import CircuitBreaker
from core.resilience.retry_policy import RetryPolicy
from core.resilience.fallback_manager import FallbackManager

logger = logging.getLogger(__name__)


class CapabilityType(Enum):
    """Types of capabilities nexus can possess."""
    COMPUTATIONAL = "computational"
    DATA_PROCESSING = "data_processing"
    MACHINE_LEARNING = "machine_learning"
    API_INTEGRATION = "api_integration"
    FILE_OPERATIONS = "file_operations"
    WEB_SCRAPING = "web_scraping"
    DATABASE = "database"
    DEVOPS = "devops"
    SECURITY = "security"
    MONITORING = "monitoring"


@dataclass
class Capability:
    """Represents a specific capability an agent can perform."""
    name: str
    capability_type: CapabilityType
    proficiency: float = 1.0  # 0.0 to 1.0
    cost_per_unit: float = 0.01  # Cost per operation
    latency_ms: float = 100.0  # Average latency
    reliability: float = 0.99  # Success rate
    max_concurrent: int = 10  # Maximum concurrent operations
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        return isinstance(other, Capability) and self.name == other.name


@dataclass
class AgentProfile:
    """Profile of an agent with its capabilities and constraints."""
    agent_id: str
    capabilities: Set[Capability]
    availability: float = 1.0  # 0.0 to 1.0
    load: float = 0.0  # Current load (0.0 to 1.0)
    cost_multiplier: float = 1.0  # Cost adjustment
    specializations: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    health_score: float = 1.0
    
    def get_capability_score(self, capability_name: str) -> float:
        """Get the proficiency score for a specific capability."""
        for cap in self.capabilities:
            if cap.name == capability_name:
                return cap.proficiency * self.availability * (1 - self.load) * self.health_score
        return 0.0
    
    def is_capable(self, capability_name: str, min_proficiency: float = 0.0) -> bool:
        """Check if agent has capability with minimum proficiency."""
        for cap in self.capabilities:
            if cap.name == capability_name and cap.proficiency >= min_proficiency:
                return True
        return False


@dataclass
class TaskRequirement:
    """Requirements for a task that needs agent composition."""
    task_id: str
    required_capabilities: Dict[str, float]  # capability_name -> min_proficiency
    constraints: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # 1-10, higher is more important
    deadline: Optional[datetime] = None
    max_nexus: int = 5
    max_cost: float = float('inf')
    min_reliability: float = 0.95
    preferred_nexus: List[str] = field(default_factory=list)
    excluded_nexus: List[str] = field(default_factory=list)


@dataclass
class CompositionResult:
    """Result of agent composition for a task."""
    task_id: str
    selected_nexus: List[AgentProfile]
    total_cost: float
    expected_latency: float
    reliability_score: float
    composition_score: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CapabilityGraph:
    """
    Graph-based capability registry for intelligent agent composition.
    Maintains relationships between nexus and their capabilities.
    """
    
    def __init__(self):
        self.graph = nx.Graph()
        self.agent_registry: Dict[str, AgentProfile] = {}
        self.capability_index: Dict[str, Set[str]] = defaultdict(set)  # capability -> agent_ids
        self.metrics_collector = MetricsCollector()
        self.cost_tracker = CostTracker()
        self.circuit_breaker = CircuitBreaker()
        self.retry_policy = RetryPolicy()
        self.fallback_manager = FallbackManager()
        
    def register_agent(self, agent: AgentProfile) -> bool:
        """Register an agent with its capabilities in the graph."""
        try:
            self.agent_registry[agent.agent_id] = agent
            
            # Add agent node
            self.graph.add_node(agent.agent_id, 
                              type="agent",
                              profile=agent,
                              last_updated=datetime.now())
            
            # Add capability nodes and edges
            for capability in agent.capabilities:
                # Add capability node if not exists
                if not self.graph.has_node(capability.name):
                    self.graph.add_node(capability.name,
                                      type="capability",
                                      capability_type=capability.capability_type,
                                      metadata=capability.metadata)
                
                # Add edge between agent and capability
                self.graph.add_edge(agent.agent_id, capability.name,
                                  weight=capability.proficiency,
                                  cost=capability.cost_per_unit,
                                  latency=capability.latency_ms,
                                  reliability=capability.reliability)
                
                # Update capability index
                self.capability_index[capability.name].add(agent.agent_id)
            
            logger.info(f"Registered agent {agent.agent_id} with {len(agent.capabilities)} capabilities")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register agent {agent.agent_id}: {e}")
            return False
    
    def unregister_agent(self, agent_id: str) -> bool:
        """Remove an agent from the capability graph."""
        try:
            if agent_id not in self.agent_registry:
                logger.warning(f"Agent {agent_id} not found in registry")
                return False
            
            agent = self.agent_registry[agent_id]
            
            # Remove from capability index
            for capability in agent.capabilities:
                if capability.name in self.capability_index:
                    self.capability_index[capability.name].discard(agent_id)
                    if not self.capability_index[capability.name]:
                        del self.capability_index[capability.name]
            
            # Remove from graph
            self.graph.remove_node(agent_id)
            del self.agent_registry[agent_id]
            
            logger.info(f"Unregistered agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister agent {agent_id}: {e}")
            return False
    
    def update_agent_load(self, agent_id: str, load: float) -> bool:
        """Update the current load of an agent."""
        if agent_id not in self.agent_registry:
            return False
        
        agent = self.agent_registry[agent_id]
        agent.load = max(0.0, min(1.0, load))
        agent.last_heartbeat = datetime.now()
        
        # Update graph node
        if self.graph.has_node(agent_id):
            self.graph.nodes[agent_id]['profile'] = agent
            self.graph.nodes[agent_id]['last_updated'] = datetime.now()
        
        return True
    
    def get_nexus_with_capability(self, capability_name: str, 
                                  min_proficiency: float = 0.0) -> List[AgentProfile]:
        """Get all nexus that have a specific capability with minimum proficiency."""
        nexus = []
        
        if capability_name not in self.capability_index:
            return nexus
        
        for agent_id in self.capability_index[capability_name]:
            agent = self.agent_registry.get(agent_id)
            if agent and agent.is_capable(capability_name, min_proficiency):
                nexus.append(agent)
        
        return nexus
    
    def find_optimal_team(self, task: TaskRequirement) -> Optional[CompositionResult]:
        """
        Find optimal team of nexus for a task using constraint satisfaction and optimization.
        Uses Hungarian algorithm for optimal assignment when possible.
        """
        try:
            # Step 1: Filter nexus based on constraints
            candidate_nexus = self._filter_candidates(task)
            if not candidate_nexus:
                logger.warning(f"No candidate nexus found for task {task.task_id}")
                return None
            
            # Step 2: Check if we can cover all required capabilities
            if not self._can_cover_capabilities(candidate_nexus, task.required_capabilities):
                logger.warning(f"Cannot cover all required capabilities for task {task.task_id}")
                return None
            
            # Step 3: Use optimization to select best team
            if len(task.required_capabilities) <= 3 and len(candidate_nexus) <= 10:
                # For small problems, use exhaustive search
                return self._exhaustive_search(candidate_nexus, task)
            else:
                # For larger problems, use Hungarian algorithm or heuristic
                return self._hungarian_optimization(candidate_nexus, task)
                
        except Exception as e:
            logger.error(f"Error finding optimal team for task {task.task_id}: {e}")
            return self._fallback_composition(task)
    
    def _filter_candidates(self, task: TaskRequirement) -> List[AgentProfile]:
        """Filter nexus based on task constraints."""
        candidates = []
        
        for agent in self.agent_registry.values():
            # Check exclusions
            if agent.agent_id in task.excluded_nexus:
                continue
            
            # Check if agent has at least one required capability
            has_required = False
            for cap_name, min_prof in task.required_capabilities.items():
                if agent.is_capable(cap_name, min_prof):
                    has_required = True
                    break
            
            if not has_required:
                continue
            
            # Check availability and load
            if agent.availability < 0.1 or agent.load > 0.9:
                continue
            
            # Check health
            if agent.health_score < 0.5:
                continue
            
            candidates.append(agent)
        
        # Prioritize preferred nexus
        if task.preferred_nexus:
            preferred = [a for a in candidates if a.agent_id in task.preferred_nexus]
            others = [a for a in candidates if a.agent_id not in task.preferred_nexus]
            candidates = preferred + others
        
        return candidates
    
    def _can_cover_capabilities(self, nexus: List[AgentProfile], 
                               required: Dict[str, float]) -> bool:
        """Check if the set of nexus can cover all required capabilities."""
        covered = set()
        
        for agent in nexus:
            for cap_name, min_prof in required.items():
                if agent.is_capable(cap_name, min_prof):
                    covered.add(cap_name)
        
        return len(covered) == len(required)
    
    def _exhaustive_search(self, nexus: List[AgentProfile], 
                          task: TaskRequirement) -> Optional[CompositionResult]:
        """Use exhaustive search for small problems."""
        from itertools import combinations
        
        best_score = -float('inf')
        best_team = None
        best_cost = float('inf')
        
        # Try all possible team sizes up to max_nexus
        for team_size in range(1, min(task.max_nexus + 1, len(nexus) + 1)):
            for team in combinations(nexus, team_size):
                # Check if team covers all capabilities
                if not self._team_covers_capabilities(team, task.required_capabilities):
                    continue
                
                # Calculate team metrics
                cost = self._calculate_team_cost(team, task)
                if cost > task.max_cost:
                    continue
                
                reliability = self._calculate_team_reliability(team, task)
                if reliability < task.min_reliability:
                    continue
                
                latency = self._calculate_team_latency(team, task)
                score = self._calculate_composition_score(team, task)
                
                if score > best_score:
                    best_score = score
                    best_team = team
                    best_cost = cost
        
        if best_team:
            return CompositionResult(
                task_id=task.task_id,
                selected_nexus=list(best_team),
                total_cost=best_cost,
                expected_latency=self._calculate_team_latency(best_team, task),
                reliability_score=self._calculate_team_reliability(best_team, task),
                composition_score=best_score
            )
        
        return None
    
    def _hungarian_optimization(self, nexus: List[AgentProfile], 
                               task: TaskRequirement) -> Optional[CompositionResult]:
        """Use Hungarian algorithm for optimal assignment."""
        capabilities = list(task.required_capabilities.keys())
        n_nexus = len(nexus)
        n_caps = len(capabilities)
        
        # Create cost matrix (nexus x capabilities)
        # We want to maximize score, so we use negative scores as costs
        cost_matrix = np.zeros((n_nexus, n_caps))
        
        for i, agent in enumerate(nexus):
            for j, cap_name in enumerate(capabilities):
                min_prof = task.required_capabilities[cap_name]
                if agent.is_capable(cap_name, min_prof):
                    # Calculate score for this agent-capability pair
                    score = agent.get_capability_score(cap_name)
                    # Convert to cost (negative score)
                    cost_matrix[i, j] = -score
                else:
                    # Agent cannot perform this capability
                    cost_matrix[i, j] = 1e9  # Very high cost
        
        # Use Hungarian algorithm to find optimal assignment
        try:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Check if assignment covers all capabilities
            assigned_caps = set()
            selected_nexus = set()
            total_cost = 0
            
            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] < 1e9:  # Valid assignment
                    agent = nexus[i]
                    cap_name = capabilities[j]
                    assigned_caps.add(cap_name)
                    selected_nexus.add(agent.agent_id)
                    
                    # Calculate cost for this assignment
                    for cap in agent.capabilities:
                        if cap.name == cap_name:
                            total_cost += cap.cost_per_unit * agent.cost_multiplier
                            break
            
            if len(assigned_caps) == n_caps:
                # We have a complete assignment
                team = [self.agent_registry[aid] for aid in selected_nexus]
                
                # Apply constraints
                if len(team) > task.max_nexus:
                    # Trim team if too large
                    team = self._trim_team(team, task)
                
                total_cost = self._calculate_team_cost(team, task)
                if total_cost > task.max_cost:
                    return None
                
                reliability = self._calculate_team_reliability(team, task)
                if reliability < task.min_reliability:
                    return None
                
                latency = self._calculate_team_latency(team, task)
                score = self._calculate_composition_score(team, task)
                
                return CompositionResult(
                    task_id=task.task_id,
                    selected_nexus=team,
                    total_cost=total_cost,
                    expected_latency=latency,
                    reliability_score=reliability,
                    composition_score=score
                )
        
        except Exception as e:
            logger.error(f"Hungarian optimization failed: {e}")
        
        return None
    
    def _trim_team(self, team: List[AgentProfile], task: TaskRequirement) -> List[AgentProfile]:
        """Trim team to meet max_nexus constraint while maintaining capability coverage."""
        # Sort by score (descending)
        sorted_team = sorted(team, 
                           key=lambda a: sum(a.get_capability_score(c) 
                                           for c in task.required_capabilities.keys()),
                           reverse=True)
        
        # Greedily add nexus until we cover all capabilities
        selected = []
        covered_caps = set()
        
        for agent in sorted_team:
            if len(selected) >= task.max_nexus:
                break
            
            # Check if this agent adds new capabilities
            adds_new = False
            for cap_name in task.required_capabilities.keys():
                if cap_name not in covered_caps and agent.is_capable(cap_name):
                    adds_new = True
                    break
            
            if adds_new or not selected:  # Always include at least one agent
                selected.append(agent)
                # Update covered capabilities
                for cap_name in task.required_capabilities.keys():
                    if agent.is_capable(cap_name):
                        covered_caps.add(cap_name)
        
        return selected
    
    def _team_covers_capabilities(self, team: Tuple[AgentProfile, ...], 
                                 required: Dict[str, float]) -> bool:
        """Check if a team covers all required capabilities."""
        covered = set()
        
        for agent in team:
            for cap_name, min_prof in required.items():
                if agent.is_capable(cap_name, min_prof):
                    covered.add(cap_name)
        
        return len(covered) == len(required)
    
    def _calculate_team_cost(self, team: List[AgentProfile], 
                            task: TaskRequirement) -> float:
        """Calculate total cost for a team performing a task."""
        total_cost = 0
        
        for agent in team:
            for cap_name, min_prof in task.required_capabilities.items():
                if agent.is_capable(cap_name, min_prof):
                    for cap in agent.capabilities:
                        if cap.name == cap_name:
                            total_cost += cap.cost_per_unit * agent.cost_multiplier
                            break
        
        return total_cost
    
    def _calculate_team_reliability(self, team: List[AgentProfile], 
                                   task: TaskRequirement) -> float:
        """Calculate overall reliability of a team."""
        if not team:
            return 0.0
        
        reliabilities = []
        
        for agent in team:
            agent_reliability = []
            for cap_name in task.required_capabilities.keys():
                if agent.is_capable(cap_name):
                    for cap in agent.capabilities:
                        if cap.name == cap_name:
                            agent_reliability.append(cap.reliability)
                            break
            
            if agent_reliability:
                reliabilities.append(min(agent_reliability))
        
        if not reliabilities:
            return 0.0
        
        # Team reliability is product of individual reliabilities
        # (assuming independent failures)
        team_reliability = 1.0
        for r in reliabilities:
            team_reliability *= r
        
        return team_reliability
    
    def _calculate_team_latency(self, team: List[AgentProfile], 
                               task: TaskRequirement) -> float:
        """Calculate expected latency for a team."""
        if not team:
            return float('inf')
        
        # For parallel execution, latency is max of individual latencies
        latencies = []
        
        for agent in team:
            agent_latencies = []
            for cap_name in task.required_capabilities.keys():
                if agent.is_capable(cap_name):
                    for cap in agent.capabilities:
                        if cap.name == cap_name:
                            agent_latencies.append(cap.latency_ms)
                            break
            
            if agent_latencies:
                latencies.append(max(agent_latencies))
        
        return max(latencies) if latencies else float('inf')
    
    def _calculate_composition_score(self, team: List[AgentProfile], 
                                    task: TaskRequirement) -> float:
        """Calculate overall composition score (higher is better)."""
        if not team:
            return 0.0
        
        # Factors to consider
        proficiency_score = 0
        cost_score = 0
        reliability_score = 0
        latency_score = 0
        
        for agent in team:
            for cap_name, min_prof in task.required_capabilities.items():
                if agent.is_capable(cap_name, min_prof):
                    proficiency_score += agent.get_capability_score(cap_name)
        
        # Normalize scores
        proficiency_score /= len(team) * len(task.required_capabilities)
        
        total_cost = self._calculate_team_cost(team, task)
        cost_score = 1.0 / (1.0 + total_cost) if total_cost > 0 else 1.0
        
        reliability = self._calculate_team_reliability(team, task)
        reliability_score = reliability
        
        latency = self._calculate_team_latency(team, task)
        latency_score = 1.0 / (1.0 + latency / 1000) if latency > 0 else 1.0
        
        # Weighted combination
        weights = {
            'proficiency': 0.4,
            'cost': 0.3,
            'reliability': 0.2,
            'latency': 0.1
        }
        
        score = (
            weights['proficiency'] * proficiency_score +
            weights['cost'] * cost_score +
            weights['reliability'] * reliability_score +
            weights['latency'] * latency_score
        )
        
        return score
    
    def _fallback_composition(self, task: TaskRequirement) -> Optional[CompositionResult]:
        """Fallback composition when optimization fails."""
        logger.warning(f"Using fallback composition for task {task.task_id}")
        
        # Simple greedy approach
        selected = []
        covered_caps = set()
        remaining_caps = set(task.required_capabilities.keys())
        
        # Sort nexus by number of uncovered capabilities they can provide
        nexus_by_coverage = []
        for agent in self.agent_registry.values():
            if agent.agent_id in task.excluded_nexus:
                continue
            
            coverage = sum(1 for cap in remaining_caps 
                          if agent.is_capable(cap, task.required_capabilities.get(cap, 0)))
            if coverage > 0:
                nexus_by_coverage.append((coverage, agent))
        
        nexus_by_coverage.sort(reverse=True)
        
        for coverage, agent in nexus_by_coverage:
            if len(selected) >= task.max_nexus:
                break
            
            # Check if agent adds new capabilities
            adds_new = False
            for cap in remaining_caps:
                if agent.is_capable(cap, task.required_capabilities.get(cap, 0)):
                    adds_new = True
                    break
            
            if adds_new:
                selected.append(agent)
                # Update covered capabilities
                for cap in list(remaining_caps):
                    if agent.is_capable(cap, task.required_capabilities.get(cap, 0)):
                        covered_caps.add(cap)
                        remaining_caps.discard(cap)
        
        if not remaining_caps:  # All capabilities covered
            total_cost = self._calculate_team_cost(selected, task)
            reliability = self._calculate_team_reliability(selected, task)
            latency = self._calculate_team_latency(selected, task)
            score = self._calculate_composition_score(selected, task)
            
            return CompositionResult(
                task_id=task.task_id,
                selected_nexus=selected,
                total_cost=total_cost,
                expected_latency=latency,
                reliability_score=reliability,
                composition_score=score,
                metadata={'fallback': True}
            )
        
        return None


class AgentCompositionEngine:
    """
    Main engine for intelligent agent composition.
    Integrates with distributed systems and monitoring.
    """
    
    def __init__(self, 
                 executor: Optional[DistributedExecutor] = None,
                 state_manager: Optional[StateManager] = None,
                 consensus: Optional[ConsensusProtocol] = None):
        self.capability_graph = CapabilityGraph()
        self.executor = executor
        self.state_manager = state_manager
        self.consensus = consensus
        self.composition_cache: Dict[str, CompositionResult] = {}
        self.cache_ttl = timedelta(minutes=5)
        
        # Integration with existing modules
        self.metrics = MetricsCollector()
        self.cost_tracker = CostTracker()
        
        logger.info("Agent Composition Engine initialized")
    
    async def register_agent(self, agent: AgentProfile) -> bool:
        """Register an agent with the composition engine."""
        with self.metrics.track_operation("agent_registration"):
            success = self.capability_graph.register_agent(agent)
            
            if success and self.state_manager:
                # Update distributed state
                await self.state_manager.update_agent_state(agent.agent_id, {
                    'capabilities': [cap.name for cap in agent.capabilities],
                    'load': agent.load,
                    'availability': agent.availability,
                    'timestamp': datetime.now().isoformat()
                })
            
            return success
    
    async def compose_team(self, task: TaskRequirement, 
                          use_cache: bool = True) -> Optional[CompositionResult]:
        """
        Compose optimal team for a task.
        Uses caching and distributed consensus for coordination.
        """
        # Check cache first
        cache_key = f"{task.task_id}_{hash(frozenset(task.required_capabilities.items()))}"
        if use_cache and cache_key in self.composition_cache:
            cached = self.composition_cache[cache_key]
            if datetime.now() - cached.timestamp < self.cache_ttl:
                logger.debug(f"Using cached composition for task {task.task_id}")
                return cached
        
        with self.metrics.track_operation("team_composition"):
            # If we have consensus protocol, coordinate composition
            if self.consensus:
                # Propose composition to cluster
                proposal = await self._propose_composition(task)
                if proposal:
                    # Use consensus to agree on composition
                    agreed = await self.consensus.propose(proposal)
                    if agreed:
                        return proposal
            
            # Local composition
            result = self.capability_graph.find_optimal_team(task)
            
            if result:
                # Cache result
                self.composition_cache[cache_key] = result
                
                # Track costs
                self.cost_tracker.track_composition_cost(task.task_id, result.total_cost)
                
                # Update metrics
                self.metrics.record_gauge("composition_score", result.composition_score)
                self.metrics.record_gauge("team_size", len(result.selected_nexus))
                self.metrics.record_gauge("composition_cost", result.total_cost)
                
                # Update agent loads if we have executor
                if self.executor:
                    await self._update_agent_loads(result)
            
            return result
    
    async def _propose_composition(self, task: TaskRequirement) -> Optional[CompositionResult]:
        """Propose a composition to the cluster."""
        # Get local composition
        local_result = self.capability_graph.find_optimal_team(task)
        
        if not local_result:
            return None
        
        # If we have state manager, check other nodes' capabilities
        if self.state_manager:
            cluster_state = await self.state_manager.get_cluster_state()
            
            # Adjust composition based on cluster state
            # This is a simplified version - in production, we'd do more sophisticated
            # cluster-aware optimization
            for agent in local_result.selected_nexus:
                if agent.agent_id in cluster_state:
                    agent_state = cluster_state[agent.agent_id]
                    # Update agent load from cluster state
                    if 'load' in agent_state:
                        agent.load = agent_state['load']
        
        return local_result
    
    async def _update_agent_loads(self, composition: CompositionResult) -> None:
        """Update agent loads after composition."""
        for agent in composition.selected_nexus:
            # Simulate load increase (in production, this would be based on actual task)
            new_load = min(1.0, agent.load + 0.1)
            self.capability_graph.update_agent_load(agent.agent_id, new_load)
            
            # Notify executor if available
            if self.executor:
                await self.executor.notify_agent_load(agent.agent_id, new_load)
    
    async def decompose_task(self, task: TaskRequirement) -> List[TaskRequirement]:
        """
        Decompose a complex task into subtasks that can be handled by individual nexus.
        This enables parallel execution and better resource utilization.
        """
        # Simple decomposition based on capability groups
        subtasks = []
        
        # Group capabilities by type
        capability_groups = defaultdict(list)
        for cap_name, min_prof in task.required_capabilities.items():
            # Find capability type
            for agent in self.capability_graph.agent_registry.values():
                for cap in agent.capabilities:
                    if cap.name == cap_name:
                        capability_groups[cap.capability_type].append((cap_name, min_prof))
                        break
        
        # Create subtasks for each capability group
        for i, (cap_type, capabilities) in enumerate(capability_groups.items()):
            subtask = TaskRequirement(
                task_id=f"{task.task_id}_sub_{i}",
                required_capabilities=dict(capabilities),
                constraints=task.constraints.copy(),
                priority=task.priority,
                deadline=task.deadline,
                max_nexus=min(3, task.max_nexus),  # Smaller teams for subtasks
                max_cost=task.max_cost / len(capability_groups),
                min_reliability=task.min_reliability,
                preferred_nexus=task.preferred_nexus,
                excluded_nexus=task.excluded_nexus
            )
            subtasks.append(subtask)
        
        return subtasks
    
    def get_agent_recommendations(self, task: TaskRequirement, 
                                 top_k: int = 5) -> List[Tuple[AgentProfile, float]]:
        """
        Get top-k agent recommendations for a task with scores.
        Useful for manual selection or fallback scenarios.
        """
        recommendations = []
        
        for agent in self.capability_graph.agent_registry.values():
            if agent.agent_id in task.excluded_nexus:
                continue
            
            # Calculate how well this agent fits the task
            score = 0
            covered_caps = 0
            
            for cap_name, min_prof in task.required_capabilities.items():
                if agent.is_capable(cap_name, min_prof):
                    score += agent.get_capability_score(cap_name)
                    covered_caps += 1
            
            if covered_caps > 0:
                # Normalize by number of capabilities
                score /= len(task.required_capabilities)
                # Bonus for covering more capabilities
                score *= (covered_caps / len(task.required_capabilities))
                recommendations.append((agent, score))
        
        # Sort by score (descending)
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:top_k]
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get statistics about the composition system."""
        return {
            'total_nexus': len(self.capability_graph.agent_registry),
            'total_capabilities': len(self.capability_graph.capability_index),
            'cache_size': len(self.composition_cache),
            'graph_nodes': self.capability_graph.graph.number_of_nodes(),
            'graph_edges': self.capability_graph.graph.number_of_edges(),
            'metrics': self.metrics.get_summary(),
            'cost_summary': self.cost_tracker.get_summary()
        }


# Singleton instance for global access
composition_engine = AgentCompositionEngine()


# Example usage
if __name__ == "__main__":
    # Create some sample nexus
    agent1 = AgentProfile(
        agent_id="agent_001",
        capabilities={
            Capability("api_design", CapabilityType.API_INTEGRATION, 0.9, 0.02, 150, 0.99),
            Capability("database_optimization", CapabilityType.DATABASE, 0.8, 0.03, 200, 0.95),
        },
        availability=0.95,
        load=0.2
    )
    
    agent2 = AgentProfile(
        agent_id="agent_002",
        capabilities={
            Capability("ml_training", CapabilityType.MACHINE_LEARNING, 0.85, 0.05, 500, 0.98),
            Capability("data_processing", CapabilityType.DATA_PROCESSING, 0.9, 0.01, 100, 0.99),
        },
        availability=0.9,
        load=0.3
    )
    
    # Register nexus
    composition_engine.capability_graph.register_agent(agent1)
    composition_engine.capability_graph.register_agent(agent2)
    
    # Create a task
    task = TaskRequirement(
        task_id="task_001",
        required_capabilities={
            "api_design": 0.8,
            "ml_training": 0.7
        },
        max_nexus=3,
        max_cost=1.0
    )
    
    # Compose team
    result = composition_engine.capability_graph.find_optimal_team(task)
    if result:
        print(f"Composed team for task {task.task_id}:")
        print(f"  Agents: {[a.agent_id for a in result.selected_nexus]}")
        print(f"  Cost: {result.total_cost:.3f}")
        print(f"  Score: {result.composition_score:.3f}")
    else:
        print("Could not compose a suitable team")