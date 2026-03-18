"""core/routing/model_router.py - Multi-Model Agent Routing System"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta

# Import existing modules
from monitoring.cost_tracker import CostTracker
from monitoring.metrics_collector import MetricsCollector
from core.resilience.circuit_breaker import CircuitBreaker
from core.resilience.retry_policy import RetryPolicy
from core.composition.capability_graph import CapabilityGraph

logger = logging.getLogger(__name__)


class TaskComplexity(Enum):
    """Task complexity levels for routing decisions"""
    SIMPLE = 1      # Simple queries, classification, extraction
    MODERATE = 2    # Summarization, basic reasoning, transformation
    COMPLEX = 3     # Multi-step reasoning, code generation, creative writing
    CRITICAL = 4    # Safety-critical, high-precision tasks


class ModelTier(Enum):
    """Model performance/cost tiers"""
    ECONOMY = "economy"      # Fastest, cheapest, lower quality
    STANDARD = "standard"    # Balanced performance and cost
    PREMIUM = "premium"      # Highest quality, most expensive
    SPECIALIZED = "specialized"  # Domain-specific models


@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    model_id: str
    provider: str  # e.g., "openai", "anthropic", "local"
    tier: ModelTier
    capabilities: List[str]
    cost_per_1k_tokens: float
    max_tokens: int
    latency_ms: float  # Average response latency
    accuracy_score: float  # 0.0 to 1.0
    rate_limit_rpm: int  # Requests per minute
    current_load: int = 0
    circuit_breaker: Optional[CircuitBreaker] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskRequirements:
    """Requirements for a specific task"""
    task_id: str
    complexity: TaskComplexity
    required_capabilities: List[str]
    max_cost: Optional[float] = None
    max_latency_ms: Optional[float] = None
    min_accuracy: float = 0.7
    priority: int = 1  # 1-10, higher is more important
    context_size: int = 0  # Estimated context tokens
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    """Result of routing decision"""
    task_id: str
    selected_model: ModelConfig
    confidence: float  # 0.0 to 1.0
    estimated_cost: float
    estimated_latency_ms: float
    fallback_models: List[ModelConfig]
    routing_metadata: Dict[str, Any] = field(default_factory=dict)


class ModelRouter:
    """Intelligent multi-model agent routing system"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.models: Dict[str, ModelConfig] = {}
        self.cost_tracker = CostTracker()
        self.metrics_collector = MetricsCollector()
        self.capability_graph = CapabilityGraph()
        self.retry_policy = RetryPolicy()
        
        # Routing weights (can be tuned)
        self.weights = {
            "cost": 0.3,
            "latency": 0.25,
            "accuracy": 0.35,
            "load": 0.1
        }
        
        # Historical performance tracking
        self.performance_history: Dict[str, List[Dict]] = {}
        self.routing_decisions: List[RoutingDecision] = []
        
        # Load balancing
        self.load_balancer_window = timedelta(minutes=5)
        self.load_history: Dict[str, List[Tuple[datetime, int]]] = {}
        
        logger.info("ModelRouter initialized")
    
    def register_model(self, model_config: ModelConfig) -> None:
        """Register a new model with the router"""
        self.models[model_config.model_id] = model_config
        self.load_history[model_config.model_id] = []
        
        if model_config.circuit_breaker is None:
            model_config.circuit_breaker = CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=60,
                name=f"model_{model_config.model_id}"
            )
        
        # Update capability graph
        for capability in model_config.capabilities:
            self.capability_graph.add_capability_node(
                capability, 
                model_config.model_id,
                weight=model_config.accuracy_score
            )
        
        logger.info(f"Registered model: {model_config.model_id} ({model_config.tier.value})")
    
    def update_model_load(self, model_id: str, delta: int) -> None:
        """Update current load for a model"""
        if model_id in self.models:
            self.models[model_id].current_load += delta
            self.load_history[model_id].append((datetime.now(), self.models[model_id].current_load))
            
            # Clean old load history
            cutoff = datetime.now() - self.load_balancer_window
            self.load_history[model_id] = [
                (ts, load) for ts, load in self.load_history[model_id]
                if ts > cutoff
            ]
    
    def get_average_load(self, model_id: str) -> float:
        """Get average load over the window period"""
        if model_id not in self.load_history or not self.load_history[model_id]:
            return 0.0
        
        loads = [load for _, load in self.load_history[model_id]]
        return sum(loads) / len(loads)
    
    def _calculate_model_score(
        self, 
        model: ModelConfig, 
        requirements: TaskRequirements
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate a score for a model based on requirements"""
        scores = {}
        
        # Cost score (lower cost is better)
        estimated_tokens = requirements.context_size + 1000  # Estimate output tokens
        estimated_cost = (estimated_tokens / 1000) * model.cost_per_1k_tokens
        
        if requirements.max_cost and estimated_cost > requirements.max_cost:
            cost_score = 0.0
        else:
            # Normalize cost (assuming max reasonable cost is $10 per 1k tokens)
            cost_score = max(0, 1.0 - (model.cost_per_1k_tokens / 10.0))
        scores["cost"] = cost_score
        
        # Latency score (lower latency is better)
        if requirements.max_latency_ms and model.latency_ms > requirements.max_latency_ms:
            latency_score = 0.0
        else:
            # Normalize latency (assuming max reasonable latency is 10 seconds)
            latency_score = max(0, 1.0 - (model.latency_ms / 10000.0))
        scores["latency"] = latency_score
        
        # Accuracy score
        if model.accuracy_score < requirements.min_accuracy:
            accuracy_score = 0.0
        else:
            accuracy_score = model.accuracy_score
        scores["accuracy"] = accuracy_score
        
        # Load score (lower load is better)
        load_ratio = model.current_load / model.rate_limit_rpm if model.rate_limit_rpm > 0 else 0
        load_score = max(0, 1.0 - load_ratio)
        scores["load"] = load_score
        
        # Capability match score
        required_set = set(requirements.required_capabilities)
        model_set = set(model.capabilities)
        capability_match = len(required_set.intersection(model_set)) / len(required_set) if required_set else 1.0
        scores["capability"] = capability_match
        
        # Complexity match score
        complexity_match = 1.0
        if requirements.complexity == TaskComplexity.CRITICAL and model.tier != ModelTier.PREMIUM:
            complexity_match = 0.3
        elif requirements.complexity == TaskComplexity.COMPLEX and model.tier == ModelTier.ECONOMY:
            complexity_match = 0.5
        scores["complexity"] = complexity_match
        
        # Circuit breaker score
        cb_score = 1.0 if model.circuit_breaker and model.circuit_breaker.state == "closed" else 0.0
        scores["circuit_breaker"] = cb_score
        
        # Calculate weighted total score
        total_score = (
            self.weights["cost"] * scores["cost"] +
            self.weights["latency"] * scores["latency"] +
            self.weights["accuracy"] * scores["accuracy"] +
            self.weights["load"] * scores["load"] +
            0.1 * scores["capability"] +
            0.1 * scores["complexity"] +
            0.2 * scores["circuit_breaker"]
        )
        
        return total_score, scores
    
    def _get_fallback_models(
        self, 
        primary_model: ModelConfig, 
        requirements: TaskRequirements,
        max_fallbacks: int = 3
    ) -> List[ModelConfig]:
        """Get fallback models for redundancy"""
        candidates = []
        
        for model in self.models.values():
            if model.model_id == primary_model.model_id:
                continue
            
            # Check basic requirements
            if model.accuracy_score < requirements.min_accuracy * 0.9:  # Allow 10% lower for fallback
                continue
            
            # Check capability match
            required_set = set(requirements.required_capabilities)
            model_set = set(model.capabilities)
            if not required_set.issubset(model_set):
                continue
            
            # Check circuit breaker
            if model.circuit_breaker and model.circuit_breaker.state != "closed":
                continue
            
            candidates.append(model)
        
        # Sort by score (simplified)
        candidates.sort(
            key=lambda m: m.accuracy_score * (1.0 / (m.cost_per_1k_tokens + 0.01)),
            reverse=True
        )
        
        return candidates[:max_fallbacks]
    
    async def route_task(self, requirements: TaskRequirements) -> RoutingDecision:
        """Route a task to the optimal model"""
        start_time = time.time()
        
        # Filter models by basic requirements
        eligible_models = []
        for model in self.models.values():
            # Check capability match
            required_set = set(requirements.required_capabilities)
            model_set = set(model.capabilities)
            if not required_set.issubset(model_set):
                continue
            
            # Check circuit breaker
            if model.circuit_breaker and model.circuit_breaker.state == "open":
                continue
            
            eligible_models.append(model)
        
        if not eligible_models:
            raise ValueError(f"No eligible models for requirements: {requirements}")
        
        # Score each model
        scored_models = []
        for model in eligible_models:
            score, score_details = self._calculate_model_score(model, requirements)
            scored_models.append((model, score, score_details))
        
        # Sort by score (descending)
        scored_models.sort(key=lambda x: x[1], reverse=True)
        
        # Select best model
        best_model, best_score, best_scores = scored_models[0]
        
        # Calculate estimates
        estimated_tokens = requirements.context_size + 1000
        estimated_cost = (estimated_tokens / 1000) * best_model.cost_per_1k_tokens
        estimated_latency = best_model.latency_ms
        
        # Get fallback models
        fallback_models = self._get_fallback_models(best_model, requirements)
        
        # Create routing decision
        decision = RoutingDecision(
            task_id=requirements.task_id,
            selected_model=best_model,
            confidence=best_score,
            estimated_cost=estimated_cost,
            estimated_latency_ms=estimated_latency,
            fallback_models=fallback_models,
            routing_metadata={
                "score_details": best_scores,
                "alternatives_considered": len(eligible_models),
                "routing_time_ms": (time.time() - start_time) * 1000
            }
        )
        
        # Record metrics
        self.metrics_collector.record_metric(
            "model_routing.decision_time_ms",
            decision.routing_metadata["routing_time_ms"],
            tags={"task_complexity": requirements.complexity.name}
        )
        
        self.metrics_collector.record_metric(
            "model_routing.selected_model_tier",
            1 if best_model.tier == ModelTier.PREMIUM else 0,
            tags={"model_id": best_model.model_id}
        )
        
        # Store decision
        self.routing_decisions.append(decision)
        
        # Update load
        self.update_model_load(best_model.model_id, 1)
        
        logger.info(
            f"Routed task {requirements.task_id} to {best_model.model_id} "
            f"(score: {best_score:.3f}, tier: {best_model.tier.value})"
        )
        
        return decision
    
    async def execute_with_fallback(
        self, 
        requirements: TaskRequirements,
        execution_func: callable
    ) -> Any:
        """Execute a task with automatic fallback to alternative models"""
        decision = await self.route_task(requirements)
        
        # Try primary model
        try:
            if decision.selected_model.circuit_breaker:
                with decision.selected_model.circuit_breaker:
                    result = await execution_func(
                        decision.selected_model.model_id,
                        requirements
                    )
                    
                    # Record success
                    self._record_success(decision.selected_model.model_id, requirements)
                    return result
            else:
                result = await execution_func(
                    decision.selected_model.model_id,
                    requirements
                )
                self._record_success(decision.selected_model.model_id, requirements)
                return result
                
        except Exception as e:
            logger.warning(
                f"Primary model {decision.selected_model.model_id} failed: {e}"
            )
            self._record_failure(decision.selected_model.model_id, requirements)
            
            # Try fallback models
            for fallback_model in decision.fallback_models:
                try:
                    logger.info(f"Trying fallback model: {fallback_model.model_id}")
                    
                    if fallback_model.circuit_breaker:
                        with fallback_model.circuit_breaker:
                            result = await execution_func(
                                fallback_model.model_id,
                                requirements
                            )
                            self._record_success(fallback_model.model_id, requirements)
                            return result
                    else:
                        result = await execution_func(
                            fallback_model.model_id,
                            requirements
                        )
                        self._record_success(fallback_model.model_id, requirements)
                        return result
                        
                except Exception as fallback_error:
                    logger.warning(
                        f"Fallback model {fallback_model.model_id} failed: {fallback_error}"
                    )
                    self._record_failure(fallback_model.model_id, requirements)
                    continue
            
            # All models failed
            raise Exception(f"All models failed for task {requirements.task_id}")
    
    def _record_success(self, model_id: str, requirements: TaskRequirements) -> None:
        """Record successful execution"""
        if model_id not in self.performance_history:
            self.performance_history[model_id] = []
        
        self.performance_history[model_id].append({
            "timestamp": datetime.now(),
            "success": True,
            "task_complexity": requirements.complexity.name,
            "context_size": requirements.context_size
        })
        
        # Update load
        self.update_model_load(model_id, -1)
        
        # Update metrics
        self.metrics_collector.increment_counter(
            "model_routing.execution.success",
            tags={"model_id": model_id}
        )
    
    def _record_failure(self, model_id: str, requirements: TaskRequirements) -> None:
        """Record failed execution"""
        if model_id not in self.performance_history:
            self.performance_history[model_id] = []
        
        self.performance_history[model_id].append({
            "timestamp": datetime.now(),
            "success": False,
            "task_complexity": requirements.complexity.name,
            "context_size": requirements.context_size
        })
        
        # Update load
        self.update_model_load(model_id, -1)
        
        # Update metrics
        self.metrics_collector.increment_counter(
            "model_routing.execution.failure",
            tags={"model_id": model_id}
        )
    
    def get_model_performance(self, model_id: str) -> Dict[str, Any]:
        """Get performance statistics for a model"""
        if model_id not in self.performance_history:
            return {}
        
        history = self.performance_history[model_id]
        if not history:
            return {}
        
        successes = sum(1 for h in history if h["success"])
        total = len(history)
        
        # Calculate average latency from recent decisions
        recent_decisions = [
            d for d in self.routing_decisions[-100:]  # Last 100 decisions
            if d.selected_model.model_id == model_id
        ]
        
        avg_latency = 0
        if recent_decisions:
            avg_latency = sum(d.estimated_latency_ms for d in recent_decisions) / len(recent_decisions)
        
        return {
            "success_rate": successes / total if total > 0 else 0,
            "total_executions": total,
            "average_latency_ms": avg_latency,
            "current_load": self.models[model_id].current_load if model_id in self.models else 0,
            "circuit_breaker_state": (
                self.models[model_id].circuit_breaker.state 
                if model_id in self.models and self.models[model_id].circuit_breaker 
                else "unknown"
            )
        }
    
    def optimize_weights(self, evaluation_data: List[Dict]) -> None:
        """Optimize routing weights based on historical performance"""
        # This is a placeholder for a more sophisticated optimization algorithm
        # In production, this could use reinforcement learning or Bayesian optimization
        
        logger.info("Optimizing routing weights based on historical data")
        
        # Simple heuristic: increase weight for metrics that correlate with success
        for model_id, history in self.performance_history.items():
            if len(history) < 10:
                continue
            
            success_rate = sum(1 for h in history if h["success"]) / len(history)
            
            # Adjust weights based on model performance
            if success_rate > 0.9:
                # High-performing model: prioritize accuracy and latency
                self.weights["accuracy"] = min(0.4, self.weights["accuracy"] + 0.05)
                self.weights["latency"] = min(0.3, self.weights["latency"] + 0.02)
            elif success_rate < 0.7:
                # Low-performing model: prioritize cost and load balancing
                self.weights["cost"] = min(0.4, self.weights["cost"] + 0.05)
                self.weights["load"] = min(0.2, self.weights["load"] + 0.02)
        
        logger.info(f"Updated routing weights: {self.weights}")
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get overall routing statistics"""
        total_decisions = len(self.routing_decisions)
        
        if total_decisions == 0:
            return {"total_decisions": 0}
        
        # Count by model tier
        tier_counts = {}
        for decision in self.routing_decisions:
            tier = decision.selected_model.tier.value
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        # Calculate average confidence
        avg_confidence = sum(d.confidence for d in self.routing_decisions) / total_decisions
        
        # Calculate average cost
        avg_cost = sum(d.estimated_cost for d in self.routing_decisions) / total_decisions
        
        return {
            "total_decisions": total_decisions,
            "tier_distribution": tier_counts,
            "average_confidence": avg_confidence,
            "average_estimated_cost": avg_cost,
            "current_weights": self.weights,
            "registered_models": len(self.models)
        }


# Example usage and integration
async def example_usage():
    """Example of how to use the ModelRouter"""
    
    # Initialize router
    router = ModelRouter()
    
    # Register models
    router.register_model(ModelConfig(
        model_id="gpt-3.5-turbo",
        provider="openai",
        tier=ModelTier.ECONOMY,
        capabilities=["text_generation", "summarization", "translation"],
        cost_per_1k_tokens=0.002,
        max_tokens=4096,
        latency_ms=500,
        accuracy_score=0.85,
        rate_limit_rpm=3500
    ))
    
    router.register_model(ModelConfig(
        model_id="gpt-4",
        provider="openai",
        tier=ModelTier.PREMIUM,
        capabilities=["text_generation", "reasoning", "code_generation", "complex_analysis"],
        cost_per_1k_tokens=0.03,
        max_tokens=8192,
        latency_ms=2000,
        accuracy_score=0.95,
        rate_limit_rpm=200
    ))
    
    router.register_model(ModelConfig(
        model_id="claude-3-opus",
        provider="anthropic",
        tier=ModelTier.PREMIUM,
        capabilities=["text_generation", "reasoning", "creative_writing", "analysis"],
        cost_per_1k_tokens=0.015,
        max_tokens=100000,
        latency_ms=1500,
        accuracy_score=0.93,
        rate_limit_rpm=100
    ))
    
    # Define task requirements
    requirements = TaskRequirements(
        task_id="task_001",
        complexity=TaskComplexity.COMPLEX,
        required_capabilities=["reasoning", "code_generation"],
        max_cost=0.10,
        max_latency_ms=3000,
        min_accuracy=0.9,
        priority=8,
        context_size=2000
    )
    
    # Route task
    decision = await router.route_task(requirements)
    print(f"Selected model: {decision.selected_model.model_id}")
    print(f"Confidence: {decision.confidence:.2f}")
    print(f"Estimated cost: ${decision.estimated_cost:.4f}")
    
    # Execute with fallback
    async def mock_execution(model_id: str, req: TaskRequirements) -> str:
        """Mock execution function"""
        await asyncio.sleep(0.1)  # Simulate work
        return f"Result from {model_id}"
    
    try:
        result = await router.execute_with_fallback(requirements, mock_execution)
        print(f"Execution result: {result}")
    except Exception as e:
        print(f"Execution failed: {e}")
    
    # Get statistics
    stats = router.get_routing_stats()
    print(f"Routing stats: {stats}")


if __name__ == "__main__":
    asyncio.run(example_usage())