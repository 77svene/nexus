"""Intelligent Agent Composition Engine — Dynamic agent composition based on task requirements, with automatic capability discovery and optimal agent selection."""

import asyncio
import heapq
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
from datetime import datetime, timedelta

from core.composition.capability_graph import CapabilityGraph, CapabilityNode, AgentNode
from core.distributed.state_manager import StateManager
from core.resilience.circuit_breaker import CircuitBreaker
from monitoring.metrics_collector import MetricsCollector
from monitoring.tracing import TracingManager

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Available optimization strategies for agent composition."""
    GREEDY_SET_COVER = "greedy_set_cover"
    GENETIC_ALGORITHM = "genetic_algorithm"
    CONSTRAINT_SATISFACTION = "constraint_satisfaction"
    MULTI_OBJECTIVE = "multi_objective"
    REINFORCEMENT_LEARNING = "reinforcement_learning"


@dataclass
class TaskRequirements:
    """Defines the requirements for a task that needs agent composition."""
    task_id: str
    required_capabilities: Dict[str, float]  # capability_name -> minimum_proficiency (0.0-1.0)
    priority: int = 1  # 1-10, higher is more important
    deadline: Optional[datetime] = None
    max_nexus: int = 10
    min_nexus: int = 1
    max_cost: Optional[float] = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentAssignment:
    """Represents an agent assigned to a task with specific capabilities."""
    agent_id: str
    task_id: str
    assigned_capabilities: Set[str]
    proficiency_score: float
    cost: float
    start_time: datetime
    estimated_duration: timedelta
    priority: int = 1


@dataclass
class CompositionResult:
    """Result of agent composition optimization."""
    task_id: str
    selected_nexus: List[AgentAssignment]
    total_cost: float
    total_proficiency: float
    coverage_score: float  # 0.0-1.0, how well capabilities are covered
    diversity_score: float  # 0.0-1.0, agent diversity for resilience
    optimization_time_ms: float
    strategy_used: OptimizationStrategy
    metadata: Dict[str, Any] = field(default_factory=dict)


class CapabilityMatcher:
    """Matches task requirements with available agent capabilities."""
    
    def __init__(self, capability_graph: CapabilityGraph):
        self.capability_graph = capability_graph
        self._capability_index = self._build_capability_index()
    
    def _build_capability_index(self) -> Dict[str, List[Tuple[str, float]]]:
        """Build an index of capabilities to nexus for fast lookup."""
        index = defaultdict(list)
        for agent_id, agent_node in self.capability_graph.nexus.items():
            for cap_name, proficiency in agent_node.capabilities.items():
                index[cap_name].append((agent_id, proficiency))
        return dict(index)
    
    def find_candidate_nexus(
        self, 
        required_capabilities: Dict[str, float],
        min_proficiency: float = 0.5
    ) -> Dict[str, Dict[str, float]]:
        """Find nexus that can satisfy at least one required capability."""
        candidates = {}
        
        for cap_name, min_prof in required_capabilities.items():
            if cap_name in self._capability_index:
                for agent_id, proficiency in self._capability_index[cap_name]:
                    if proficiency >= max(min_prof, min_proficiency):
                        if agent_id not in candidates:
                            candidates[agent_id] = {}
                        candidates[agent_id][cap_name] = proficiency
        
        return candidates
    
    def calculate_coverage_score(
        self, 
        agent_ids: List[str], 
        required_capabilities: Dict[str, float]
    ) -> float:
        """Calculate how well a set of nexus covers required capabilities."""
        if not agent_ids:
            return 0.0
        
        covered_capabilities = set()
        total_proficiency = 0.0
        
        for agent_id in agent_ids:
            agent = self.capability_graph.get_agent(agent_id)
            if agent:
                for cap_name, min_prof in required_capabilities.items():
                    if cap_name in agent.capabilities:
                        proficiency = agent.capabilities[cap_name]
                        if proficiency >= min_prof:
                            covered_capabilities.add(cap_name)
                            total_proficiency += proficiency
        
        coverage = len(covered_capabilities) / len(required_capabilities) if required_capabilities else 1.0
        avg_proficiency = total_proficiency / (len(agent_ids) * len(required_capabilities)) if agent_ids and required_capabilities else 0.0
        
        return (coverage * 0.7) + (avg_proficiency * 0.3)


class CostOptimizer:
    """Optimizes agent selection based on cost constraints."""
    
    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager
        self._cost_cache = {}
    
    async def get_agent_cost(self, agent_id: str, duration_hours: float = 1.0) -> float:
        """Get the cost of using an agent for a given duration."""
        cache_key = f"{agent_id}:{duration_hours}"
        
        if cache_key in self._cost_cache:
            return self._cost_cache[cache_key]
        
        try:
            # Get agent cost from state manager or use default
            agent_info = await self.state_manager.get_agent_info(agent_id)
            base_cost = agent_info.get("cost_per_hour", 10.0) if agent_info else 10.0
            
            # Apply dynamic pricing based on load
            load_factor = await self._get_agent_load_factor(agent_id)
            cost = base_cost * (1.0 + load_factor * 0.5)  # 50% premium for high load
            
            self._cost_cache[cache_key] = cost
            return cost
            
        except Exception as e:
            logger.warning(f"Failed to get cost for agent {agent_id}: {e}")
            return 10.0  # Default cost
    
    async def _get_agent_load_factor(self, agent_id: str) -> float:
        """Get the current load factor of an agent (0.0-1.0)."""
        try:
            metrics = await self.state_manager.get_agent_metrics(agent_id)
            if metrics:
                cpu_usage = metrics.get("cpu_usage", 0.5)
                memory_usage = metrics.get("memory_usage", 0.5)
                task_count = metrics.get("active_tasks", 1)
                
                # Normalize and combine metrics
                load = (cpu_usage * 0.4 + memory_usage * 0.3 + min(task_count / 10, 1.0) * 0.3)
                return min(max(load, 0.0), 1.0)
        except Exception:
            pass
        return 0.5  # Default moderate load
    
    def optimize_cost(
        self, 
        agent_options: List[Tuple[str, Dict[str, float]]], 
        max_cost: Optional[float] = None
    ) -> List[str]:
        """Select nexus that minimize cost while meeting requirements."""
        if not agent_options:
            return []
        
        # Sort by cost-efficiency (capabilities per cost)
        scored_options = []
        for agent_id, capabilities in agent_options:
            # Estimate cost (synchronously for optimization)
            estimated_cost = 10.0  # Would use get_agent_cost in async context
            capability_count = len(capabilities)
            efficiency = capability_count / estimated_cost if estimated_cost > 0 else 0
            scored_options.append((efficiency, agent_id, capabilities, estimated_cost))
        
        scored_options.sort(reverse=True)  # Higher efficiency first
        
        selected = []
        total_cost = 0.0
        
        for efficiency, agent_id, capabilities, cost in scored_options:
            if max_cost and total_cost + cost > max_cost:
                continue
            selected.append(agent_id)
            total_cost += cost
        
        return selected


class DiversityOptimizer:
    """Ensures diversity in agent selection for resilience."""
    
    def __init__(self, capability_graph: CapabilityGraph):
        self.capability_graph = capability_graph
    
    def calculate_diversity_score(self, agent_ids: List[str]) -> float:
        """Calculate diversity score based on agent types and capabilities."""
        if len(agent_ids) <= 1:
            return 0.0
        
        # Get agent types and capabilities
        agent_types = []
        all_capabilities = set()
        
        for agent_id in agent_ids:
            agent = self.capability_graph.get_agent(agent_id)
            if agent:
                agent_types.append(agent.agent_type)
                all_capabilities.update(agent.capabilities.keys())
        
        # Type diversity (Shannon entropy)
        type_counts = {}
        for agent_type in agent_types:
            type_counts[agent_type] = type_counts.get(agent_type, 0) + 1
        
        type_entropy = 0.0
        for count in type_counts.values():
            probability = count / len(agent_ids)
            type_entropy -= probability * np.log(probability)
        
        max_entropy = np.log(len(type_counts)) if type_counts else 1.0
        type_diversity = type_entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Capability overlap penalty
        overlap_penalty = 0.0
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                agent_i = self.capability_graph.get_agent(agent_ids[i])
                agent_j = self.capability_graph.get_agent(agent_ids[j])
                if agent_i and agent_j:
                    overlap = len(set(agent_i.capabilities.keys()) & set(agent_j.capabilities.keys()))
                    total_caps = len(set(agent_i.capabilities.keys()) | set(agent_j.capabilities.keys()))
                    if total_caps > 0:
                        overlap_penalty += overlap / total_caps
        
        avg_overlap = overlap_penalty / (len(agent_ids) * (len(agent_ids) - 1) / 2) if len(agent_ids) > 1 else 0.0
        capability_diversity = 1.0 - avg_overlap
        
        return (type_diversity * 0.6) + (capability_diversity * 0.4)


class AgentCompositionOptimizer:
    """Main optimizer for intelligent agent composition."""
    
    def __init__(
        self,
        capability_graph: CapabilityGraph,
        state_manager: StateManager,
        metrics_collector: Optional[MetricsCollector] = None,
        tracing_manager: Optional[TracingManager] = None
    ):
        self.capability_graph = capability_graph
        self.state_manager = state_manager
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.tracing_manager = tracing_manager or TracingManager()
        
        self.capability_matcher = CapabilityMatcher(capability_graph)
        self.cost_optimizer = CostOptimizer(state_manager)
        self.diversity_optimizer = DiversityOptimizer(capability_graph)
        
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            name="agent_composition"
        )
        
        self._optimization_history = []
        self._max_history = 1000
    
    async def compose_nexus(
        self,
        task_requirements: TaskRequirements,
        strategy: OptimizationStrategy = OptimizationStrategy.MULTI_OBJECTIVE,
        available_nexus: Optional[List[str]] = None
    ) -> CompositionResult:
        """Compose optimal set of nexus for given task requirements."""
        start_time = datetime.now()
        
        with self.tracing_manager.trace("agent_composition") as span:
            span.set_attribute("task_id", task_requirements.task_id)
            span.set_attribute("strategy", strategy.value)
            
            try:
                # Get available nexus
                if available_nexus is None:
                    available_nexus = await self._get_available_nexus()
                
                # Filter nexus by basic availability
                candidate_nexus = await self._filter_candidates(
                    available_nexus, 
                    task_requirements
                )
                
                if not candidate_nexus:
                    return self._create_empty_result(
                        task_requirements.task_id,
                        strategy,
                        "No suitable nexus available"
                    )
                
                # Apply optimization strategy
                if strategy == OptimizationStrategy.GREEDY_SET_COVER:
                    result = await self._greedy_set_cover(
                        task_requirements, candidate_nexus
                    )
                elif strategy == OptimizationStrategy.GENETIC_ALGORITHM:
                    result = await self._genetic_algorithm(
                        task_requirements, candidate_nexus
                    )
                elif strategy == OptimizationStrategy.CONSTRAINT_SATISFACTION:
                    result = await self._constraint_satisfaction(
                        task_requirements, candidate_nexus
                    )
                elif strategy == OptimizationStrategy.MULTI_OBJECTIVE:
                    result = await self._multi_objective_optimization(
                        task_requirements, candidate_nexus
                    )
                elif strategy == OptimizationStrategy.REINFORCEMENT_LEARNING:
                    result = await self._reinforcement_learning(
                        task_requirements, candidate_nexus
                    )
                else:
                    result = await self._greedy_set_cover(
                        task_requirements, candidate_nexus
                    )
                
                # Record metrics
                optimization_time = (datetime.now() - start_time).total_seconds() * 1000
                result.optimization_time_ms = optimization_time
                
                await self._record_metrics(result, task_requirements)
                self._update_history(result)
                
                span.set_attribute("success", True)
                span.set_attribute("nexus_selected", len(result.selected_nexus))
                
                return result
                
            except Exception as e:
                logger.error(f"Agent composition failed: {e}")
                span.set_attribute("success", False)
                span.set_attribute("error", str(e))
                
                # Fallback to simple selection
                return await self._fallback_composition(
                    task_requirements, available_nexus
                )
    
    async def _get_available_nexus(self) -> List[str]:
        """Get list of currently available nexus."""
        try:
            all_nexus = list(self.capability_graph.nexus.keys())
            available = []
            
            for agent_id in all_nexus:
                if await self._is_agent_available(agent_id):
                    available.append(agent_id)
            
            return available
            
        except Exception as e:
            logger.warning(f"Failed to get available nexus: {e}")
            return list(self.capability_graph.nexus.keys())
    
    async def _is_agent_available(self, agent_id: str) -> bool:
        """Check if an agent is currently available for tasks."""
        try:
            # Check circuit breaker
            if not self.circuit_breaker.allow_request(agent_id):
                return False
            
            # Check agent status in state manager
            agent_status = await self.state_manager.get_agent_status(agent_id)
            if agent_status and agent_status.get("status") != "available":
                return False
            
            # Check agent load
            metrics = await self.state_manager.get_agent_metrics(agent_id)
            if metrics:
                load = metrics.get("load", 0.0)
                if load > 0.9:  # 90% load threshold
                    return False
            
            return True
            
        except Exception:
            return True  # Assume available on error
    
    async def _filter_candidates(
        self,
        available_nexus: List[str],
        task_requirements: TaskRequirements
    ) -> Dict[str, Dict[str, float]]:
        """Filter nexus that can potentially satisfy task requirements."""
        # Get nexus with required capabilities
        candidates = self.capability_matcher.find_candidate_nexus(
            task_requirements.required_capabilities,
            min_proficiency=0.3
        )
        
        # Filter by availability
        filtered = {
            agent_id: caps 
            for agent_id, caps in candidates.items() 
            if agent_id in available_nexus
        }
        
        # Apply additional constraints
        if task_requirements.constraints:
            filtered = await self._apply_constraints(
                filtered, task_requirements.constraints
            )
        
        return filtered
    
    async def _apply_constraints(
        self,
        candidates: Dict[str, Dict[str, float]],
        constraints: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """Apply additional constraints to candidate nexus."""
        filtered = {}
        
        for agent_id, capabilities in candidates.items():
            agent = self.capability_graph.get_agent(agent_id)
            if not agent:
                continue
            
            # Check agent type constraints
            if "agent_types" in constraints:
                if agent.agent_type not in constraints["agent_types"]:
                    continue
            
            # Check capability level constraints
            if "min_capability_level" in constraints:
                min_level = constraints["min_capability_level"]
                if agent.capability_level < min_level:
                    continue
            
            # Check location constraints
            if "allowed_regions" in constraints:
                agent_region = agent.metadata.get("region", "unknown")
                if agent_region not in constraints["allowed_regions"]:
                    continue
            
            filtered[agent_id] = capabilities
        
        return filtered
    
    async def _greedy_set_cover(
        self,
        task_requirements: TaskRequirements,
        candidates: Dict[str, Dict[str, float]]
    ) -> CompositionResult:
        """Greedy algorithm for set cover optimization."""
        required_caps = set(task_requirements.required_capabilities.keys())
        covered_caps = set()
        selected_nexus = []
        total_cost = 0.0
        total_proficiency = 0.0
        
        # Sort candidates by capability coverage and cost
        candidate_scores = []
        for agent_id, capabilities in candidates.items():
            # Calculate coverage score
            new_caps = set(capabilities.keys()) & required_caps - covered_caps
            coverage_score = len(new_caps) / len(required_caps) if required_caps else 0
            
            # Get cost
            cost = await self.cost_optimizer.get_agent_cost(agent_id)
            
            # Calculate efficiency (coverage per cost)
            efficiency = coverage_score / cost if cost > 0 else coverage_score
            
            candidate_scores.append((efficiency, agent_id, capabilities, cost))
        
        # Sort by efficiency (descending)
        candidate_scores.sort(reverse=True)
        
        # Greedy selection
        for efficiency, agent_id, capabilities, cost in candidate_scores:
            if len(selected_nexus) >= task_requirements.max_nexus:
                break
            
            # Check if this agent adds new coverage
            new_caps = set(capabilities.keys()) & required_caps - covered_caps
            if not new_caps:
                continue
            
            # Check cost constraint
            if task_requirements.max_cost and total_cost + cost > task_requirements.max_cost:
                continue
            
            # Calculate proficiency for assigned capabilities
            cap_proficiency = sum(
                capabilities.get(cap, 0.0) 
                for cap in new_caps
            ) / len(new_caps) if new_caps else 0
            
            # Create assignment
            assignment = AgentAssignment(
                agent_id=agent_id,
                task_id=task_requirements.task_id,
                assigned_capabilities=new_caps,
                proficiency_score=cap_proficiency,
                cost=cost,
                start_time=datetime.now(),
                estimated_duration=timedelta(hours=1),  # Default duration
                priority=task_requirements.priority
            )
            
            selected_nexus.append(assignment)
            covered_caps.update(new_caps)
            total_cost += cost
            total_proficiency += cap_proficiency
            
            # Check if all capabilities are covered
            if covered_caps == required_caps:
                break
        
        # Calculate scores
        coverage_score = len(covered_caps) / len(required_caps) if required_caps else 1.0
        diversity_score = self.diversity_optimizer.calculate_diversity_score(
            [a.agent_id for a in selected_nexus]
        )
        
        return CompositionResult(
            task_id=task_requirements.task_id,
            selected_nexus=selected_nexus,
            total_cost=total_cost,
            total_proficiency=total_proficiency,
            coverage_score=coverage_score,
            diversity_score=diversity_score,
            optimization_time_ms=0.0,  # Will be set by caller
            strategy_used=OptimizationStrategy.GREEDY_SET_COVER
        )
    
    async def _genetic_algorithm(
        self,
        task_requirements: TaskRequirements,
        candidates: Dict[str, Dict[str, float]]
    ) -> CompositionResult:
        """Genetic algorithm for agent composition optimization."""
        # Simplified genetic algorithm implementation
        population_size = 50
        generations = 20
        mutation_rate = 0.1
        
        # Create initial population
        population = []
        candidate_list = list(candidates.keys())
        
        for _ in range(population_size):
            # Random selection of nexus
            num_nexus = np.random.randint(
                task_requirements.min_nexus,
                min(task_requirements.max_nexus, len(candidate_list)) + 1
            )
            selected = np.random.choice(candidate_list, num_nexus, replace=False)
            population.append(set(selected))
        
        # Evolution
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                fitness = await self._evaluate_fitness(
                    individual, task_requirements, candidates
                )
                fitness_scores.append((fitness, individual))
            
            # Selection (tournament selection)
            new_population = []
            for _ in range(population_size):
                # Tournament selection
                tournament = np.random.choice(len(fitness_scores), 3, replace=False)
                winner_idx = max(tournament, key=lambda i: fitness_scores[i][0])
                winner = fitness_scores[winner_idx][1].copy()
                
                # Mutation
                if np.random.random() < mutation_rate:
                    # Add or remove an agent
                    if len(winner) < task_requirements.max_nexus and np.random.random() < 0.5:
                        # Add random agent
                        new_agent = np.random.choice(candidate_list)
                        winner.add(new_agent)
                    elif len(winner) > task_requirements.min_nexus:
                        # Remove random agent
                        to_remove = np.random.choice(list(winner))
                        winner.remove(to_remove)
                
                new_population.append(winner)
            
            population = new_population
        
        # Select best individual
        best_individual = max(
            population,
            key=lambda ind: asyncio.run(
                self._evaluate_fitness(ind, task_requirements, candidates)
            )
        )
        
        # Convert to result format
        selected_nexus = []
        total_cost = 0.0
        total_proficiency = 0.0
        
        for agent_id in best_individual:
            capabilities = candidates[agent_id]
            cost = await self.cost_optimizer.get_agent_cost(agent_id)
            
            # Calculate which capabilities this agent covers
            covered_caps = set(capabilities.keys()) & set(task_requirements.required_capabilities.keys())
            cap_proficiency = sum(
                capabilities.get(cap, 0.0) 
                for cap in covered_caps
            ) / len(covered_caps) if covered_caps else 0
            
            assignment = AgentAssignment(
                agent_id=agent_id,
                task_id=task_requirements.task_id,
                assigned_capabilities=covered_caps,
                proficiency_score=cap_proficiency,
                cost=cost,
                start_time=datetime.now(),
                estimated_duration=timedelta(hours=1),
                priority=task_requirements.priority
            )
            
            selected_nexus.append(assignment)
            total_cost += cost
            total_proficiency += cap_proficiency
        
        # Calculate coverage
        all_covered = set()
        for assignment in selected_nexus:
            all_covered.update(assignment.assigned_capabilities)
        
        coverage_score = len(all_covered) / len(task_requirements.required_capabilities) if task_requirements.required_capabilities else 1.0
        diversity_score = self.diversity_optimizer.calculate_diversity_score(
            [a.agent_id for a in selected_nexus]
        )
        
        return CompositionResult(
            task_id=task_requirements.task_id,
            selected_nexus=selected_nexus,
            total_cost=total_cost,
            total_proficiency=total_proficiency,
            coverage_score=coverage_score,
            diversity_score=diversity_score,
            optimization_time_ms=0.0,
            strategy_used=OptimizationStrategy.GENETIC_ALGORITHM,
            metadata={"generations": generations, "population_size": population_size}
        )
    
    async def _evaluate_fitness(
        self,
        agent_set: Set[str],
        task_requirements: TaskRequirements,
        candidates: Dict[str, Dict[str, float]]
    ) -> float:
        """Evaluate fitness of an agent set for genetic algorithm."""
        if not agent_set:
            return 0.0
        
        # Calculate coverage
        covered_capabilities = set()
        total_proficiency = 0.0
        total_cost = 0.0
        
        for agent_id in agent_set:
            if agent_id in candidates:
                capabilities = candidates[agent_id]
                for cap_name, proficiency in capabilities.items():
                    if cap_name in task_requirements.required_capabilities:
                        min_prof = task_requirements.required_capabilities[cap_name]
                        if proficiency >= min_prof:
                            covered_capabilities.add(cap_name)
                            total_proficiency += proficiency
                
                cost = await self.cost_optimizer.get_agent_cost(agent_id)
                total_cost += cost
        
        # Calculate coverage score
        coverage = len(covered_capabilities) / len(task_requirements.required_capabilities) if task_requirements.required_capabilities else 1.0
        
        # Calculate cost score (lower is better)
        max_acceptable_cost = task_requirements.max_cost or 1000.0
        cost_score = max(0, 1.0 - (total_cost / max_acceptable_cost))
        
        # Calculate diversity score
        diversity = self.diversity_optimizer.calculate_diversity_score(list(agent_set))
        
        # Calculate size penalty
        size_penalty = 0.0
        if len(agent_set) < task_requirements.min_nexus:
            size_penalty = -1.0
        elif len(agent_set) > task_requirements.max_nexus:
            size_penalty = -0.5
        
        # Combine scores (weighted)
        fitness = (
            coverage * 0.4 +
            cost_score * 0.3 +
            diversity * 0.2 +
            size_penalty * 0.1
        )
        
        return max(0.0, fitness)
    
    async def _constraint_satisfaction(
        self,
        task_requirements: TaskRequirements,
        candidates: Dict[str, Dict[str, float]]
    ) -> CompositionResult:
        """Constraint satisfaction problem solver for agent composition."""
        # Simplified CSP implementation using backtracking
        required_caps = list(task_requirements.required_capabilities.keys())
        agent_list = list(candidates.keys())
        
        # Create variable domains (each agent can be selected or not)
        domains = {agent: [True, False] for agent in agent_list}
        
        # Define constraints
        constraints = []
        
        # Coverage constraint: all required capabilities must be covered
        def coverage_constraint(assignment):
            covered = set()
            for agent, selected in assignment.items():
                if selected and agent in candidates:
                    for cap in candidates[agent]:
                        if cap in required_caps:
                            covered.add(cap)
            return len(covered) == len(required_caps)
        
        # Cardinality constraints
        def cardinality_constraint(assignment):
            selected_count = sum(1 for selected in assignment.values() if selected)
            return task_requirements.min_nexus <= selected_count <= task_requirements.max_nexus
        
        constraints.append(coverage_constraint)
        constraints.append(cardinality_constraint)
        
        # Backtracking search
        solution = await self._backtracking_search(
            domains, constraints, candidates, task_requirements
        )
        
        if solution:
            # Convert solution to result
            selected_nexus = []
            total_cost = 0.0
            total_proficiency = 0.0
            
            for agent_id, selected in solution.items():
                if selected:
                    capabilities = candidates[agent_id]
                    cost = await self.cost_optimizer.get_agent_cost(agent_id)
                    
                    covered_caps = set(capabilities.keys()) & set(required_caps)
                    cap_proficiency = sum(
                        capabilities.get(cap, 0.0) 
                        for cap in covered_caps
                    ) / len(covered_caps) if covered_caps else 0
                    
                    assignment = AgentAssignment(
                        agent_id=agent_id,
                        task_id=task_requirements.task_id,
                        assigned_capabilities=covered_caps,
                        proficiency_score=cap_proficiency,
                        cost=cost,
                        start_time=datetime.now(),
                        estimated_duration=timedelta(hours=1),
                        priority=task_requirements.priority
                    )
                    
                    selected_nexus.append(assignment)
                    total_cost += cost
                    total_proficiency += cap_proficiency
            
            coverage_score = 1.0  # All constraints satisfied
            diversity_score = self.diversity_optimizer.calculate_diversity_score(
                [a.agent_id for a in selected_nexus]
            )
            
            return CompositionResult(
                task_id=task_requirements.task_id,
                selected_nexus=selected_nexus,
                total_cost=total_cost,
                total_proficiency=total_proficiency,
                coverage_score=coverage_score,
                diversity_score=diversity_score,
                optimization_time_ms=0.0,
                strategy_used=OptimizationStrategy.CONSTRAINT_SATISFACTION
            )
        else:
            # No solution found, fall back to greedy
            return await self._greedy_set_cover(task_requirements, candidates)
    
    async def _backtracking_search(
        self,
        domains: Dict[str, List[bool]],
        constraints: List[callable],
        candidates: Dict[str, Dict[str, float]],
        task_requirements: TaskRequirements
    ) -> Optional[Dict[str, bool]]:
        """Backtracking search for CSP."""
        assignment = {}
        
        async def backtrack(assignment, nexus):
            if len(assignment) == len(nexus):
                # Check all constraints
                if all(constraint(assignment) for constraint in constraints):
                    return assignment
                return None
            
            # Select next agent
            agent = nexus[len(assignment)]
            
            for value in domains[agent]:
                assignment[agent] = value
                
                # Check constraints that can be evaluated with partial assignment
                valid = True
                for constraint in constraints:
                    try:
                        if not constraint(assignment):
                            valid = False
                            break
                    except:
                        # Constraint requires complete assignment
                        pass
                
                if valid:
                    result = await backtrack(assignment, nexus)
                    if result:
                        return result
                
                del assignment[agent]
            
            return None
        
        return await backtrack(assignment, list(domains.keys()))
    
    async def _multi_objective_optimization(
        self,
        task_requirements: TaskRequirements,
        candidates: Dict[str, Dict[str, float]]
    ) -> CompositionResult:
        """Multi-objective optimization balancing coverage, cost, and diversity."""
        # Use NSGA-II inspired approach (simplified)
        required_caps = set(task_requirements.required_capabilities.keys())
        
        # Generate Pareto front
        pareto_front = []
        
        # Try different numbers of nexus
        for num_nexus in range(
            task_requirements.min_nexus,
            min(task_requirements.max_nexus, len(candidates)) + 1
        ):
            # Generate random combinations
            for _ in range(20):  # Sample 20 combinations per size
                agent_ids = np.random.choice(
                    list(candidates.keys()),
                    size=num_nexus,
                    replace=False
                )
                
                # Calculate objectives
                coverage = self.capability_matcher.calculate_coverage_score(
                    list(agent_ids),
                    task_requirements.required_capabilities
                )
                
                total_cost = sum(
                    await self.cost_optimizer.get_agent_cost(agent_id)
                    for agent_id in agent_ids
                )
                
                diversity = self.diversity_optimizer.calculate_diversity_score(
                    list(agent_ids)
                )
                
                # Normalize objectives
                max_cost = task_requirements.max_cost or 1000.0
                cost_score = 1.0 - min(total_cost / max_cost, 1.0)
                
                # Calculate combined score
                combined_score = (
                    coverage * 0.5 +
                    cost_score * 0.3 +
                    diversity * 0.2
                )
                
                pareto_front.append({
                    "nexus": set(agent_ids),
                    "coverage": coverage,
                    "cost": total_cost,
                    "diversity": diversity,
                    "score": combined_score
                })
        
        # Select best from Pareto front
        if pareto_front:
            best = max(pareto_front, key=lambda x: x["score"])
            
            # Convert to result format
            selected_nexus = []
            total_proficiency = 0.0
            
            for agent_id in best["nexus"]:
                capabilities = candidates[agent_id]
                cost = await self.cost_optimizer.get_agent_cost(agent_id)
                
                covered_caps = set(capabilities.keys()) & required_caps
                cap_proficiency = sum(
                    capabilities.get(cap, 0.0) 
                    for cap in covered_caps
                ) / len(covered_caps) if covered_caps else 0
                
                assignment = AgentAssignment(
                    agent_id=agent_id,
                    task_id=task_requirements.task_id,
                    assigned_capabilities=covered_caps,
                    proficiency_score=cap_proficiency,
                    cost=cost,
                    start_time=datetime.now(),
                    estimated_duration=timedelta(hours=1),
                    priority=task_requirements.priority
                )
                
                selected_nexus.append(assignment)
                total_proficiency += cap_proficiency
            
            return CompositionResult(
                task_id=task_requirements.task_id,
                selected_nexus=selected_nexus,
                total_cost=best["cost"],
                total_proficiency=total_proficiency,
                coverage_score=best["coverage"],
                diversity_score=best["diversity"],
                optimization_time_ms=0.0,
                strategy_used=OptimizationStrategy.MULTI_OBJECTIVE,
                metadata={"pareto_front_size": len(pareto_front)}
            )
        else:
            # Fallback to greedy
            return await self._greedy_set_cover(task_requirements, candidates)
    
    async def _reinforcement_learning(
        self,
        task_requirements: TaskRequirements,
        candidates: Dict[str, Dict[str, float]]
    ) -> CompositionResult:
        """Reinforcement learning approach for agent composition."""
        # Simplified Q-learning implementation
        # In production, this would use a trained model
        
        # For now, use a policy based on historical success
        historical_data = self._get_historical_performance()
        
        # Score nexus based on historical performance
        agent_scores = {}
        for agent_id in candidates:
            if agent_id in historical_data:
                success_rate = historical_data[agent_id].get("success_rate", 0.5)
                avg_performance = historical_data[agent_id].get("avg_performance", 0.5)
                agent_scores[agent_id] = (success_rate + avg_performance) / 2
            else:
                agent_scores[agent_id] = 0.5  # Default for new nexus
        
        # Select top nexus
        sorted_nexus = sorted(
            agent_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Select nexus until requirements are met
        selected_nexus = []
        covered_caps = set()
        total_cost = 0.0
        total_proficiency = 0.0
        required_caps = set(task_requirements.required_capabilities.keys())
        
        for agent_id, score in sorted_nexus:
            if len(selected_nexus) >= task_requirements.max_nexus:
                break
            
            capabilities = candidates[agent_id]
            new_caps = set(capabilities.keys()) & required_caps - covered_caps
            
            if not new_caps:
                continue
            
            cost = await self.cost_optimizer.get_agent_cost(agent_id)
            
            if task_requirements.max_cost and total_cost + cost > task_requirements.max_cost:
                continue
            
            cap_proficiency = sum(
                capabilities.get(cap, 0.0) 
                for cap in new_caps
            ) / len(new_caps) if new_caps else 0
            
            assignment = AgentAssignment(
                agent_id=agent_id,
                task_id=task_requirements.task_id,
                assigned_capabilities=new_caps,
                proficiency_score=cap_proficiency,
                cost=cost,
                start_time=datetime.now(),
                estimated_duration=timedelta(hours=1),
                priority=task_requirements.priority
            )
            
            selected_nexus.append(assignment)
            covered_caps.update(new_caps)
            total_cost += cost
            total_proficiency += cap_proficiency
            
            if covered_caps == required_caps:
                break
        
        coverage_score = len(covered_caps) / len(required_caps) if required_caps else 1.0
        diversity_score = self.diversity_optimizer.calculate_diversity_score(
            [a.agent_id for a in selected_nexus]
        )
        
        return CompositionResult(
            task_id=task_requirements.task_id,
            selected_nexus=selected_nexus,
            total_cost=total_cost,
            total_proficiency=total_proficiency,
            coverage_score=coverage_score,
            diversity_score=diversity_score,
            optimization_time_ms=0.0,
            strategy_used=OptimizationStrategy.REINFORCEMENT_LEARNING,
            metadata={"historical_data_used": True}
        )
    
    def _get_historical_performance(self) -> Dict[str, Dict[str, float]]:
        """Get historical performance data for nexus."""
        # In production, this would query a database
        # For now, return mock data
        return {}
    
    async def _fallback_composition(
        self,
        task_requirements: TaskRequirements,
        available_nexus: List[str]
    ) -> CompositionResult:
        """Fallback composition when optimization fails."""
        logger.warning(f"Using fallback composition for task {task_requirements.task_id}")
        
        # Simple selection: pick first available agent that can handle the task
        for agent_id in available_nexus:
            agent = self.capability_graph.get_agent(agent_id)
            if agent:
                # Check if agent has at least one required capability
                has_capability = any(
                    cap in agent.capabilities 
                    for cap in task_requirements.required_capabilities
                )
                
                if has_capability:
                    # Create minimal assignment
                    covered_caps = set(
                        cap for cap in task_requirements.required_capabilities
                        if cap in agent.capabilities
                    )
                    
                    assignment = AgentAssignment(
                        agent_id=agent_id,
                        task_id=task_requirements.task_id,
                        assigned_capabilities=covered_caps,
                        proficiency_score=0.5,  # Default
                        cost=10.0,  # Default cost
                        start_time=datetime.now(),
                        estimated_duration=timedelta(hours=1),
                        priority=task_requirements.priority
                    )
                    
                    coverage_score = len(covered_caps) / len(task_requirements.required_capabilities) if task_requirements.required_capabilities else 1.0
                    
                    return CompositionResult(
                        task_id=task_requirements.task_id,
                        selected_nexus=[assignment],
                        total_cost=10.0,
                        total_proficiency=0.5,
                        coverage_score=coverage_score,
                        diversity_score=0.0,
                        optimization_time_ms=0.0,
                        strategy_used=OptimizationStrategy.GREEDY_SET_COVER,
                        metadata={"fallback": True}
                    )
        
        # No suitable agent found
        return self._create_empty_result(
            task_requirements.task_id,
            OptimizationStrategy.GREEDY_SET_COVER,
            "No suitable agent found in fallback"
        )
    
    def _create_empty_result(
        self,
        task_id: str,
        strategy: OptimizationStrategy,
        reason: str
    ) -> CompositionResult:
        """Create an empty composition result."""
        return CompositionResult(
            task_id=task_id,
            selected_nexus=[],
            total_cost=0.0,
            total_proficiency=0.0,
            coverage_score=0.0,
            diversity_score=0.0,
            optimization_time_ms=0.0,
            strategy_used=strategy,
            metadata={"empty": True, "reason": reason}
        )
    
    async def _record_metrics(
        self,
        result: CompositionResult,
        task_requirements: TaskRequirements
    ) -> None:
        """Record metrics for monitoring and optimization."""
        try:
            metrics = {
                "task_id": result.task_id,
                "nexus_selected": len(result.selected_nexus),
                "total_cost": result.total_cost,
                "coverage_score": result.coverage_score,
                "diversity_score": result.diversity_score,
                "optimization_time_ms": result.optimization_time_ms,
                "strategy": result.strategy_used.value,
                "priority": task_requirements.priority,
                "required_capabilities_count": len(task_requirements.required_capabilities)
            }
            
            await self.metrics_collector.record_metric(
                "agent_composition",
                metrics,
                tags={"strategy": result.strategy_used.value}
            )
            
            # Record individual agent assignments
            for assignment in result.selected_nexus:
                await self.metrics_collector.record_metric(
                    "agent_assignment",
                    {
                        "agent_id": assignment.agent_id,
                        "task_id": assignment.task_id,
                        "capabilities_count": len(assignment.assigned_capabilities),
                        "proficiency_score": assignment.proficiency_score,
                        "cost": assignment.cost
                    }
                )
                
        except Exception as e:
            logger.warning(f"Failed to record metrics: {e}")
    
    def _update_history(self, result: CompositionResult) -> None:
        """Update optimization history for learning."""
        self._optimization_history.append({
            "timestamp": datetime.now(),
            "task_id": result.task_id,
            "strategy": result.strategy_used,
            "nexus_count": len(result.selected_nexus),
            "coverage_score": result.coverage_score,
            "cost": result.total_cost
        })
        
        # Trim history
        if len(self._optimization_history) > self._max_history:
            self._optimization_history = self._optimization_history[-self._max_history:]
    
    async def get_optimization_stats(self) -> Dict[str, Any]:
        """Get statistics about optimization performance."""
        if not self._optimization_history:
            return {}
        
        recent = self._optimization_history[-100:]  # Last 100 optimizations
        
        avg_coverage = np.mean([h["coverage_score"] for h in recent])
        avg_nexus = np.mean([h["nexus_count"] for h in recent])
        avg_cost = np.mean([h["cost"] for h in recent])
        
        strategy_counts = {}
        for h in recent:
            strategy = h["strategy"].value
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        return {
            "total_optimizations": len(self._optimization_history),
            "recent_optimizations": len(recent),
            "average_coverage_score": avg_coverage,
            "average_nexus_per_task": avg_nexus,
            "average_cost_per_task": avg_cost,
            "strategy_distribution": strategy_counts,
            "optimization_success_rate": self.circuit_breaker.success_rate
        }
    
    async def retrain_models(self, feedback_data: List[Dict[str, Any]]) -> None:
        """Retrain optimization models based on feedback."""
        # In production, this would update ML models
        logger.info(f"Retraining models with {len(feedback_data)} feedback samples")
        
        # Update historical performance data
        for feedback in feedback_data:
            task_id = feedback.get("task_id")
            agent_performance = feedback.get("agent_performance", {})
            
            # Update agent performance metrics
            for agent_id, performance in agent_performance.items():
                # In production, update database
                pass
        
        # Reset circuit breaker if it's been tripped
        if self.circuit_breaker.state == "open":
            self.circuit_breaker.reset()
        
        logger.info("Model retraining complete")


# Factory function for easy instantiation
def create_agent_optimizer(
    capability_graph: CapabilityGraph,
    state_manager: StateManager,
    **kwargs
) -> AgentCompositionOptimizer:
    """Factory function to create an AgentCompositionOptimizer instance."""
    return AgentCompositionOptimizer(
        capability_graph=capability_graph,
        state_manager=state_manager,
        **kwargs
    )