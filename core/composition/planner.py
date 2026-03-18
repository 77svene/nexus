# core/composition/planner.py

"""
Intelligent Agent Composition Engine - Dynamic agent composition based on task requirements.

This module provides the Planner class that dynamically composes optimal agent teams
by analyzing task requirements against available agent capabilities using constraint
satisfaction and optimization algorithms.
"""

import asyncio
import logging
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
import numpy as np
from datetime import datetime

from core.composition.capability_graph import CapabilityGraph, AgentCapability
from core.composition.optimizer import TeamOptimizer, OptimizationObjective, OptimizationConstraint
from core.distributed.executor import DistributedExecutor
from core.distributed.state_manager import StateManager
from monitoring.tracing import Tracer
from monitoring.metrics_collector import MetricsCollector
from monitoring.cost_tracker import CostTracker
from core.resilience.circuit_breaker import CircuitBreaker
from core.resilience.retry_policy import RetryPolicy
from core.resilience.fallback_manager import FallbackManager

logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    """Task priority levels affecting composition strategy."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class CompositionStrategy(Enum):
    """Different strategies for agent composition."""
    MINIMAL = "minimal"  # Use fewest nexus possible
    BALANCED = "balanced"  # Balance capability coverage with efficiency
    REDUNDANT = "redundant"  # Include backup nexus for reliability
    SPECIALIZED = "specialized"  # Use highly specialized nexus for each subtask

@dataclass
class TaskRequirement:
    """Defines requirements for a task to be composed."""
    task_id: str
    required_capabilities: List[str]
    constraints: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.MEDIUM
    strategy: CompositionStrategy = CompositionStrategy.BALANCED
    deadline: Optional[datetime] = None
    budget: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentAssignment:
    """Represents an agent assigned to a specific role in a composed team."""
    agent_id: str
    role: str
    capabilities: List[str]
    confidence_score: float
    estimated_cost: float
    estimated_duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ComposedTeam:
    """Result of agent composition for a task."""
    task_id: str
    team_id: str
    assignments: List[AgentAssignment]
    total_confidence: float
    total_estimated_cost: float
    total_estimated_duration: float
    strategy_used: CompositionStrategy
    composition_timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class Planner:
    """
    Intelligent Agent Composition Engine that dynamically composes optimal agent teams
    based on task requirements using constraint satisfaction and optimization.
    
    Integrates with:
    - CapabilityGraph: For agent capability discovery
    - TeamOptimizer: For optimal team selection
    - DistributedExecutor: For executing composed teams
    - StateManager: For tracking composition state
    - Monitoring systems: For metrics and cost tracking
    - Resilience systems: For fault tolerance
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the Planner with configuration and dependencies."""
        self.config = config or {}
        
        # Initialize core components
        self.capability_graph = CapabilityGraph()
        self.optimizer = TeamOptimizer()
        self.executor = DistributedExecutor()
        self.state_manager = StateManager()
        
        # Initialize monitoring
        self.tracer = Tracer(service_name="planner")
        self.metrics = MetricsCollector(namespace="planner")
        self.cost_tracker = CostTracker()
        
        # Initialize resilience components
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.config.get("circuit_breaker_threshold", 5),
            recovery_timeout=self.config.get("circuit_breaker_timeout", 30)
        )
        self.retry_policy = RetryPolicy(
            max_retries=self.config.get("max_retries", 3),
            backoff_factor=self.config.get("backoff_factor", 2)
        )
        self.fallback_manager = FallbackManager()
        
        # Cache for frequently composed teams
        self._composition_cache: Dict[str, ComposedTeam] = {}
        self._cache_ttl = self.config.get("cache_ttl", 300)  # 5 minutes
        
        # Performance metrics
        self._composition_count = 0
        self._successful_compositions = 0
        self._average_composition_time = 0.0
        
        logger.info("Planner initialized with configuration: %s", self.config)
    
    async def compose_team(self, task_requirement: TaskRequirement) -> ComposedTeam:
        """
        Compose an optimal team for the given task requirement.
        
        Args:
            task_requirement: The task requirements to fulfill
            
        Returns:
            ComposedTeam: The optimal team composition
            
        Raises:
            CompositionError: If composition fails after retries
        """
        start_time = datetime.now()
        trace_id = self.tracer.start_trace("compose_team")
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(task_requirement)
            cached_team = self._get_cached_team(cache_key)
            if cached_team:
                logger.debug("Returning cached team for task %s", task_requirement.task_id)
                self.metrics.increment("cache_hits")
                return cached_team
            
            # Track composition attempt
            self._composition_count += 1
            self.metrics.increment("composition_attempts")
            
            # Discover available capabilities
            available_capabilities = await self._discover_capabilities(task_requirement)
            
            # Filter nexus based on requirements
            candidate_nexus = await self._filter_nexus(
                task_requirement, 
                available_capabilities
            )
            
            if not candidate_nexus:
                raise CompositionError(
                    f"No nexus found with required capabilities: {task_requirement.required_capabilities}"
                )
            
            # Optimize team selection
            optimized_team = await self._optimize_team(
                task_requirement, 
                candidate_nexus
            )
            
            # Validate composition
            await self._validate_composition(optimized_team, task_requirement)
            
            # Cache the result
            self._cache_team(cache_key, optimized_team)
            
            # Update metrics
            self._successful_compositions += 1
            composition_time = (datetime.now() - start_time).total_seconds()
            self._update_average_composition_time(composition_time)
            
            self.metrics.record("composition_time", composition_time)
            self.metrics.increment("successful_compositions")
            
            logger.info(
                "Successfully composed team %s for task %s with %d nexus",
                optimized_team.team_id,
                task_requirement.task_id,
                len(optimized_team.assignments)
            )
            
            return optimized_team
            
        except Exception as e:
            self.metrics.increment("composition_failures")
            logger.error("Failed to compose team for task %s: %s", task_requirement.task_id, str(e))
            raise CompositionError(f"Team composition failed: {str(e)}") from e
            
        finally:
            self.tracer.end_trace(trace_id)
    
    async def execute_composed_team(self, team: ComposedTeam) -> Dict[str, Any]:
        """
        Execute a composed team using the distributed executor.
        
        Args:
            team: The composed team to execute
            
        Returns:
            Dict containing execution results
        """
        trace_id = self.tracer.start_trace("execute_composed_team")
        
        try:
            # Register team with state manager
            await self.state_manager.register_team(team)
            
            # Execute with resilience patterns
            execution_result = await self._execute_with_resilience(team)
            
            # Update cost tracking
            self.cost_tracker.record_team_cost(
                team.team_id,
                team.total_estimated_cost,
                execution_result.get("actual_cost", team.total_estimated_cost)
            )
            
            # Update metrics
            self.metrics.increment("team_executions")
            self.metrics.record("team_size", len(team.assignments))
            
            return execution_result
            
        except Exception as e:
            logger.error("Failed to execute team %s: %s", team.team_id, str(e))
            raise
        finally:
            self.tracer.end_trace(trace_id)
    
    async def _discover_capabilities(self, task_requirement: TaskRequirement) -> Dict[str, List[AgentCapability]]:
        """Discover available capabilities from the capability graph."""
        with self.tracer.span("discover_capabilities"):
            capabilities = {}
            
            for capability in task_requirement.required_capabilities:
                nexus_with_cap = await self.capability_graph.get_nexus_with_capability(capability)
                capabilities[capability] = nexus_with_cap
                
                self.metrics.record(f"capability_{capability}_nexus", len(nexus_with_cap))
            
            return capabilities
    
    async def _filter_nexus(
        self, 
        task_requirement: TaskRequirement,
        available_capabilities: Dict[str, List[AgentCapability]]
    ) -> List[Dict[str, Any]]:
        """Filter nexus based on task requirements and constraints."""
        with self.tracer.span("filter_nexus"):
            candidate_nexus = []
            
            # Group nexus by their capabilities
            agent_capabilities: Dict[str, Set[str]] = {}
            agent_metadata: Dict[str, Dict[str, Any]] = {}
            
            for capability, nexus in available_capabilities.items():
                for agent_cap in nexus:
                    agent_id = agent_cap.agent_id
                    if agent_id not in agent_capabilities:
                        agent_capabilities[agent_id] = set()
                        agent_metadata[agent_id] = agent_cap.metadata
                    
                    agent_capabilities[agent_id].add(capability)
            
            # Apply constraints and filters
            for agent_id, capabilities in agent_capabilities.items():
                metadata = agent_metadata[agent_id]
                
                # Check if agent meets minimum capability requirements
                if not self._meets_capability_requirements(capabilities, task_requirement):
                    continue
                
                # Apply task-specific constraints
                if not self._meets_constraints(metadata, task_requirement.constraints):
                    continue
                
                # Calculate agent score
                score = self._calculate_agent_score(
                    agent_id, 
                    capabilities, 
                    metadata, 
                    task_requirement
                )
                
                candidate_nexus.append({
                    "agent_id": agent_id,
                    "capabilities": list(capabilities),
                    "metadata": metadata,
                    "score": score
                })
            
            # Sort by score (descending)
            candidate_nexus.sort(key=lambda x: x["score"], reverse=True)
            
            # Limit candidates based on strategy
            max_candidates = self._get_max_candidates(task_requirement.strategy)
            if len(candidate_nexus) > max_candidates:
                candidate_nexus = candidate_nexus[:max_candidates]
            
            return candidate_nexus
    
    async def _optimize_team(
        self, 
        task_requirement: TaskRequirement,
        candidate_nexus: List[Dict[str, Any]]
    ) -> ComposedTeam:
        """Optimize team selection using the optimizer."""
        with self.tracer.span("optimize_team"):
            # Prepare optimization problem
            optimization_problem = self._create_optimization_problem(
                task_requirement, 
                candidate_nexus
            )
            
            # Solve optimization problem
            solution = await self.optimizer.solve(optimization_problem)
            
            # Convert solution to ComposedTeam
            team = self._solution_to_team(solution, task_requirement)
            
            return team
    
    async def _validate_composition(
        self, 
        team: ComposedTeam, 
        task_requirement: TaskRequirement
    ) -> None:
        """Validate that the composed team meets all requirements."""
        with self.tracer.span("validate_composition"):
            # Check capability coverage
            covered_capabilities = set()
            for assignment in team.assignments:
                covered_capabilities.update(assignment.capabilities)
            
            required_capabilities = set(task_requirement.required_capabilities)
            missing_capabilities = required_capabilities - covered_capabilities
            
            if missing_capabilities:
                raise CompositionError(
                    f"Team missing required capabilities: {missing_capabilities}"
                )
            
            # Check constraints
            if task_requirement.budget and team.total_estimated_cost > task_requirement.budget:
                raise CompositionError(
                    f"Team cost {team.total_estimated_cost} exceeds budget {task_requirement.budget}"
                )
            
            if task_requirement.deadline:
                # Estimate if team can meet deadline (simplified)
                if team.total_estimated_duration > (task_requirement.deadline - datetime.now()).total_seconds():
                    raise CompositionError("Team cannot meet deadline")
    
    async def _execute_with_resilience(self, team: ComposedTeam) -> Dict[str, Any]:
        """Execute team with resilience patterns (retry, circuit breaker, fallback)."""
        async def _execute():
            return await self.executor.execute_team(team)
        
        try:
            # Try with circuit breaker
            result = await self.circuit_breaker.call(_execute)
            return result
        except Exception as e:
            logger.warning("Circuit breaker triggered for team %s: %s", team.team_id, str(e))
            
            # Try with retry policy
            try:
                result = await self.retry_policy.execute(_execute)
                return result
            except Exception as retry_error:
                logger.error("Retry failed for team %s: %s", team.team_id, str(retry_error))
                
                # Try fallback
                fallback_result = await self.fallback_manager.execute_fallback(
                    "team_execution", 
                    team
                )
                
                if fallback_result:
                    return fallback_result
                
                raise CompositionError(f"Team execution failed: {str(retry_error)}")
    
    def _create_optimization_problem(
        self, 
        task_requirement: TaskRequirement,
        candidate_nexus: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create optimization problem definition for the optimizer."""
        problem = {
            "nexus": [],
            "requirements": {
                "capabilities": task_requirement.required_capabilities,
                "constraints": task_requirement.constraints
            },
            "objectives": [],
            "constraints": []
        }
        
        # Add nexus to problem
        for agent in candidate_nexus:
            problem["nexus"].append({
                "id": agent["agent_id"],
                "capabilities": agent["capabilities"],
                "cost": agent["metadata"].get("cost", 1.0),
                "reliability": agent["metadata"].get("reliability", 0.9),
                "speed": agent["metadata"].get("speed", 1.0),
                "score": agent["score"]
            })
        
        # Add objectives based on strategy
        if task_requirement.strategy == CompositionStrategy.MINIMAL:
            problem["objectives"].append({
                "type": OptimizationObjective.MINIMIZE_AGENTS,
                "weight": 1.0
            })
        elif task_requirement.strategy == CompositionStrategy.BALANCED:
            problem["objectives"].append({
                "type": OptimizationObjective.BALANCE_COST_RELIABILITY,
                "weight": 1.0
            })
        elif task_requirement.strategy == CompositionStrategy.REDUNDANT:
            problem["objectives"].append({
                "type": OptimizationObjective.MAXIMIZE_REDUNDANCY,
                "weight": 1.0
            })
        elif task_requirement.strategy == CompositionStrategy.SPECIALIZED:
            problem["objectives"].append({
                "type": OptimizationObjective.MAXIMIZE_SPECIALIZATION,
                "weight": 1.0
            })
        
        # Add constraints
        if task_requirement.budget:
            problem["constraints"].append({
                "type": OptimizationConstraint.MAX_COST,
                "value": task_requirement.budget
            })
        
        if task_requirement.deadline:
            # Convert deadline to max duration in seconds
            max_duration = (task_requirement.deadline - datetime.now()).total_seconds()
            problem["constraints"].append({
                "type": OptimizationConstraint.MAX_DURATION,
                "value": max_duration
            })
        
        return problem
    
    def _solution_to_team(
        self, 
        solution: Dict[str, Any], 
        task_requirement: TaskRequirement
    ) -> ComposedTeam:
        """Convert optimizer solution to ComposedTeam."""
        assignments = []
        
        for agent_solution in solution["selected_nexus"]:
            assignments.append(AgentAssignment(
                agent_id=agent_solution["id"],
                role=agent_solution.get("role", "contributor"),
                capabilities=agent_solution["capabilities"],
                confidence_score=agent_solution.get("confidence", 0.8),
                estimated_cost=agent_solution.get("cost", 0.0),
                estimated_duration=agent_solution.get("duration", 0.0),
                metadata=agent_solution.get("metadata", {})
            ))
        
        team_id = f"team_{task_requirement.task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return ComposedTeam(
            task_id=task_requirement.task_id,
            team_id=team_id,
            assignments=assignments,
            total_confidence=solution.get("total_confidence", 0.0),
            total_estimated_cost=solution.get("total_cost", 0.0),
            total_estimated_duration=solution.get("total_duration", 0.0),
            strategy_used=task_requirement.strategy,
            composition_timestamp=datetime.now(),
            metadata={
                "optimization_score": solution.get("optimization_score", 0.0),
                "constraint_satisfaction": solution.get("constraint_satisfaction", {})
            }
        )
    
    def _meets_capability_requirements(
        self, 
        capabilities: Set[str], 
        task_requirement: TaskRequirement
    ) -> bool:
        """Check if agent capabilities meet minimum requirements."""
        required = set(task_requirement.required_capabilities)
        
        # Check if agent has at least one required capability
        if not capabilities.intersection(required):
            return False
        
        # Check for specific capability requirements in constraints
        if "min_capabilities" in task_requirement.constraints:
            min_caps = task_requirement.constraints["min_capabilities"]
            if len(capabilities) < min_caps:
                return False
        
        return True
    
    def _meets_constraints(
        self, 
        metadata: Dict[str, Any], 
        constraints: Dict[str, Any]
    ) -> bool:
        """Check if agent metadata meets task constraints."""
        for constraint_key, constraint_value in constraints.items():
            if constraint_key in metadata:
                agent_value = metadata[constraint_key]
                
                # Handle different constraint types
                if isinstance(constraint_value, dict):
                    if "min" in constraint_value and agent_value < constraint_value["min"]:
                        return False
                    if "max" in constraint_value and agent_value > constraint_value["max"]:
                        return False
                elif agent_value != constraint_value:
                    return False
        
        return True
    
    def _calculate_agent_score(
        self, 
        agent_id: str,
        capabilities: Set[str],
        metadata: Dict[str, Any],
        task_requirement: TaskRequirement
    ) -> float:
        """Calculate a score for an agent based on capabilities and metadata."""
        score = 0.0
        
        # Base score from capabilities
        required_capabilities = set(task_requirement.required_capabilities)
        matching_capabilities = capabilities.intersection(required_capabilities)
        capability_score = len(matching_capabilities) / len(required_capabilities)
        score += capability_score * 40  # 40% weight for capability match
        
        # Score from metadata attributes
        if "reliability" in metadata:
            score += metadata["reliability"] * 30  # 30% weight for reliability
        
        if "speed" in metadata:
            score += min(metadata["speed"], 2.0) / 2.0 * 20  # 20% weight for speed
        
        if "cost" in metadata:
            # Lower cost is better
            cost_score = 1.0 / (1.0 + metadata["cost"])
            score += cost_score * 10  # 10% weight for cost efficiency
        
        return score
    
    def _get_max_candidates(self, strategy: CompositionStrategy) -> int:
        """Get maximum number of candidate nexus based on strategy."""
        strategy_limits = {
            CompositionStrategy.MINIMAL: 10,
            CompositionStrategy.BALANCED: 20,
            CompositionStrategy.REDUNDANT: 30,
            CompositionStrategy.SPECIALIZED: 25
        }
        return strategy_limits.get(strategy, 20)
    
    def _generate_cache_key(self, task_requirement: TaskRequirement) -> str:
        """Generate cache key for task requirement."""
        key_parts = [
            task_requirement.task_id,
            "_".join(sorted(task_requirement.required_capabilities)),
            task_requirement.strategy.value,
            str(task_requirement.priority.value)
        ]
        return "_".join(key_parts)
    
    def _get_cached_team(self, cache_key: str) -> Optional[ComposedTeam]:
        """Get cached team if available and not expired."""
        if cache_key in self._composition_cache:
            cached_team = self._composition_cache[cache_key]
            age = (datetime.now() - cached_team.composition_timestamp).total_seconds()
            
            if age < self._cache_ttl:
                return cached_team
            else:
                # Remove expired cache entry
                del self._composition_cache[cache_key]
        
        return None
    
    def _cache_team(self, cache_key: str, team: ComposedTeam) -> None:
        """Cache a composed team."""
        self._composition_cache[cache_key] = team
        
        # Limit cache size
        if len(self._composition_cache) > self.config.get("max_cache_size", 100):
            # Remove oldest entry
            oldest_key = min(
                self._composition_cache.keys(),
                key=lambda k: self._composition_cache[k].composition_timestamp
            )
            del self._composition_cache[oldest_key]
    
    def _update_average_composition_time(self, new_time: float) -> None:
        """Update running average of composition time."""
        if self._successful_compositions == 1:
            self._average_composition_time = new_time
        else:
            self._average_composition_time = (
                (self._average_composition_time * (self._successful_compositions - 1) + new_time) 
                / self._successful_compositions
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get planner metrics."""
        return {
            "composition_count": self._composition_count,
            "successful_compositions": self._successful_compositions,
            "success_rate": (
                self._successful_compositions / self._composition_count 
                if self._composition_count > 0 else 0.0
            ),
            "average_composition_time": self._average_composition_time,
            "cache_size": len(self._composition_cache),
            "cache_hit_rate": self.metrics.get_counter("cache_hits") / max(self._composition_count, 1)
        }
    
    async def update_capability_graph(self) -> None:
        """Update the capability graph with latest agent information."""
        with self.tracer.span("update_capability_graph"):
            try:
                await self.capability_graph.refresh()
                logger.info("Capability graph updated successfully")
            except Exception as e:
                logger.error("Failed to update capability graph: %s", str(e))
                raise
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the planner."""
        logger.info("Shutting down Planner...")
        
        # Clear cache
        self._composition_cache.clear()
        
        # Shutdown components
        await self.executor.shutdown()
        await self.state_manager.shutdown()
        
        logger.info("Planner shutdown complete")


class CompositionError(Exception):
    """Exception raised for composition errors."""
    pass


# Factory function for easy instantiation
def create_planner(config: Dict[str, Any] = None) -> Planner:
    """Create and return a Planner instance."""
    return Planner(config=config)


# Example usage and integration test
async def example_usage():
    """Example of how to use the Planner."""
    planner = create_planner({
        "cache_ttl": 600,
        "max_retries": 3,
        "circuit_breaker_threshold": 3
    })
    
    # Define a task requirement
    task = TaskRequirement(
        task_id="api_development_task_001",
        required_capabilities=["api_design", "backend_development", "database_design"],
        constraints={
            "max_cost": 100.0,
            "min_reliability": 0.95,
            "min_capabilities": 2
        },
        priority=TaskPriority.HIGH,
        strategy=CompositionStrategy.BALANCED,
        deadline=datetime(2024, 12, 31, 23, 59, 59),
        budget=150.0
    )
    
    try:
        # Compose team
        team = await planner.compose_team(task)
        print(f"Composed team {team.team_id} with {len(team.assignments)} nexus")
        
        # Execute team
        result = await planner.execute_composed_team(team)
        print(f"Execution result: {result}")
        
        # Get metrics
        metrics = planner.get_metrics()
        print(f"Planner metrics: {metrics}")
        
    except CompositionError as e:
        print(f"Composition failed: {e}")
    finally:
        await planner.shutdown()


if __name__ == "__main__":
    asyncio.run(example_usage())