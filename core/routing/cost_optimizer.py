"""core/routing/cost_optimizer.py - Multi-Model Agent Routing System"""

import asyncio
import time
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict
import heapq

from core.distributed.state_manager import StateManager
from core.distributed.executor import DistributedExecutor
from monitoring.cost_tracker import CostTracker
from monitoring.metrics_collector import MetricsCollector
from monitoring.tracing import Tracing
from core.composition.capability_graph import CapabilityGraph
from core.composition.planner import TaskPlanner
from core.resilience.circuit_breaker import CircuitBreaker
from core.resilience.retry_policy import RetryPolicy
from core.resilience.fallback_manager import FallbackManager

logger = logging.getLogger(__name__)


class ModelTier(Enum):
    """Model performance tiers"""
    TIER_1 = "premium"  # GPT-4, Claude 3 Opus - high quality, high cost
    TIER_2 = "balanced"  # GPT-3.5, Claude 3 Sonnet - balanced
    TIER_3 = "efficient"  # Smaller models, fast, cheap
    TIER_4 = "specialized"  # Domain-specific fine-tuned models


class TaskComplexity(Enum):
    """Task complexity levels"""
    TRIVIAL = 1      # Simple queries, formatting
    SIMPLE = 2       # Basic analysis, simple code generation
    MODERATE = 3     # Complex reasoning, multi-step tasks
    COMPLEX = 4      # Research, advanced coding, creative tasks
    EXPERT = 5       # Novel problems, cutting-edge research


class RoutingStrategy(Enum):
    """Routing strategies"""
    COST_OPTIMIZED = "cost_optimized"
    PERFORMANCE_OPTIMIZED = "performance_optimized"
    BALANCED = "balanced"
    LATENCY_SENSITIVE = "latency_sensitive"
    QUALITY_SENSITIVE = "quality_sensitive"


@dataclass
class ModelConfig:
    """Configuration for an LLM model"""
    model_id: str
    provider: str
    tier: ModelTier
    cost_per_1k_tokens: float
    avg_latency_ms: float
    max_tokens: int
    capabilities: List[str]
    reliability_score: float = 0.95
    current_load: float = 0.0
    circuit_breaker: Optional[CircuitBreaker] = None
    
    def to_dict(self) -> Dict:
        return {
            "model_id": self.model_id,
            "provider": self.provider,
            "tier": self.tier.value,
            "cost_per_1k_tokens": self.cost_per_1k_tokens,
            "avg_latency_ms": self.avg_latency_ms,
            "max_tokens": self.max_tokens,
            "capabilities": self.capabilities,
            "reliability_score": self.reliability_score,
            "current_load": self.current_load
        }


@dataclass
class TaskRequirements:
    """Requirements for a specific task"""
    task_id: str
    complexity: TaskComplexity
    required_capabilities: List[str]
    max_cost: float
    max_latency_ms: float
    min_quality_score: float
    estimated_tokens: int
    priority: int = 1  # 1-10, higher is more important
    deadline: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RoutingDecision:
    """Result of routing decision"""
    task_id: str
    selected_model: str
    confidence_score: float
    estimated_cost: float
    estimated_latency_ms: float
    reasoning: List[str]
    alternatives: List[Dict[str, Any]]
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "selected_model": self.selected_model,
            "confidence_score": self.confidence_score,
            "estimated_cost": self.estimated_cost,
            "estimated_latency_ms": self.estimated_latency_ms,
            "reasoning": self.reasoning,
            "alternatives": self.alternatives,
            "timestamp": self.timestamp.isoformat()
        }


class CostOptimizer:
    """
    Intelligent multi-model routing system that optimizes for cost, performance,
    and quality based on task requirements and system constraints.
    """
    
    def __init__(
        self,
        state_manager: StateManager,
        executor: DistributedExecutor,
        cost_tracker: CostTracker,
        metrics_collector: MetricsCollector,
        capability_graph: CapabilityGraph,
        config_path: Optional[str] = None
    ):
        self.state_manager = state_manager
        self.executor = executor
        self.cost_tracker = cost_tracker
        self.metrics_collector = metrics_collector
        self.capability_graph = capability_graph
        self.tracing = Tracing()
        self.fallback_manager = FallbackManager()
        
        # Model registry
        self.models: Dict[str, ModelConfig] = {}
        self.model_performance_history: Dict[str, List[float]] = defaultdict(list)
        
        # Routing configuration
        self.routing_strategy = RoutingStrategy.BALANCED
        self.budget_constraints: Dict[str, float] = {}  # agent_id -> daily budget
        self.performance_thresholds: Dict[str, float] = {}
        
        # Learning and optimization
        self.task_model_mapping: Dict[str, List[str]] = defaultdict(list)  # task_type -> successful models
        self.routing_history: List[RoutingDecision] = []
        self.load_balancer = LoadBalancer()
        
        # Cache for routing decisions
        self.routing_cache: Dict[str, RoutingDecision] = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Initialize models from config
        self._initialize_models(config_path)
        
        # Start background optimization
        self._optimization_task = asyncio.create_task(self._background_optimization())
        
        logger.info(f"CostOptimizer initialized with {len(self.models)} models")
    
    def _initialize_models(self, config_path: Optional[str] = None):
        """Initialize model configurations"""
        default_models = [
            ModelConfig(
                model_id="gpt-4-turbo",
                provider="openai",
                tier=ModelTier.TIER_1,
                cost_per_1k_tokens=0.03,
                avg_latency_ms=2000,
                max_tokens=128000,
                capabilities=["reasoning", "coding", "analysis", "creative"],
                reliability_score=0.98
            ),
            ModelConfig(
                model_id="claude-3-opus",
                provider="anthropic",
                tier=ModelTier.TIER_1,
                cost_per_1k_tokens=0.075,
                avg_latency_ms=2500,
                max_tokens=200000,
                capabilities=["reasoning", "coding", "analysis", "creative", "long_context"],
                reliability_score=0.97
            ),
            ModelConfig(
                model_id="gpt-3.5-turbo",
                provider="openai",
                tier=ModelTier.TIER_2,
                cost_per_1k_tokens=0.002,
                avg_latency_ms=800,
                max_tokens=16000,
                capabilities=["coding", "simple_reasoning", "formatting"],
                reliability_score=0.95
            ),
            ModelConfig(
                model_id="claude-3-sonnet",
                provider="anthropic",
                tier=ModelTier.TIER_2,
                cost_per_1k_tokens=0.015,
                avg_latency_ms=1200,
                max_tokens=200000,
                capabilities=["reasoning", "coding", "analysis", "creative"],
                reliability_score=0.96
            ),
            ModelConfig(
                model_id="mixtral-8x7b",
                provider="together",
                tier=ModelTier.TIER_3,
                cost_per_1k_tokens=0.0006,
                avg_latency_ms=600,
                max_tokens=32000,
                capabilities=["coding", "simple_reasoning", "fast_inference"],
                reliability_score=0.92
            ),
            ModelConfig(
                model_id="code-llama-34b",
                provider="replicate",
                tier=ModelTier.TIER_4,
                cost_per_1k_tokens=0.0008,
                avg_latency_ms=900,
                max_tokens=16000,
                capabilities=["coding", "code_completion", "debugging"],
                reliability_score=0.94
            )
        ]
        
        for model in default_models:
            self.models[model.model_id] = model
            model.circuit_breaker = CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=60,
                name=f"cb_{model.model_id}"
            )
        
        # Load custom models from config if provided
        if config_path:
            self._load_custom_models(config_path)
    
    def _load_custom_models(self, config_path: str):
        """Load custom model configurations from file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                for model_data in config.get('models', []):
                    model = ModelConfig(**model_data)
                    self.models[model.model_id] = model
                    logger.info(f"Loaded custom model: {model.model_id}")
        except Exception as e:
            logger.error(f"Failed to load custom models: {e}")
    
    async def route_task(
        self,
        task_requirements: TaskRequirements,
        agent_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> RoutingDecision:
        """
        Route a task to the optimal model based on requirements and constraints.
        
        Args:
            task_requirements: Task requirements and constraints
            agent_id: ID of the agent requesting routing
            context: Additional context for routing decision
            
        Returns:
            RoutingDecision with selected model and metadata
        """
        start_time = time.time()
        trace_id = self.tracing.start_trace("route_task", {
            "task_id": task_requirements.task_id,
            "agent_id": agent_id
        })
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(task_requirements, agent_id)
            cached_decision = self._get_cached_decision(cache_key)
            if cached_decision:
                logger.debug(f"Using cached routing decision for task {task_requirements.task_id}")
                return cached_decision
            
            # Get eligible models
            eligible_models = await self._get_eligible_models(task_requirements)
            
            if not eligible_models:
                logger.warning(f"No eligible models for task {task_requirements.task_id}")
                return await self._create_fallback_decision(task_requirements)
            
            # Score and rank models
            scored_models = await self._score_models(
                eligible_models,
                task_requirements,
                agent_id,
                context
            )
            
            # Select best model
            selected_model, confidence, reasoning = self._select_best_model(
                scored_models,
                task_requirements
            )
            
            # Create routing decision
            decision = RoutingDecision(
                task_id=task_requirements.task_id,
                selected_model=selected_model.model_id,
                confidence_score=confidence,
                estimated_cost=self._estimate_cost(selected_model, task_requirements),
                estimated_latency_ms=self._estimate_latency(selected_model, task_requirements),
                reasoning=reasoning,
                alternatives=self._create_alternatives_list(scored_models[:5]),
                timestamp=datetime.utcnow()
            )
            
            # Cache decision
            self._cache_decision(cache_key, decision)
            
            # Record decision
            self.routing_history.append(decision)
            if len(self.routing_history) > 1000:
                self.routing_history = self.routing_history[-1000:]
            
            # Update metrics
            self.metrics_collector.record_routing_decision(
                task_id=task_requirements.task_id,
                model_id=selected_model.model_id,
                confidence=confidence,
                estimated_cost=decision.estimated_cost
            )
            
            # Update load balancer
            self.load_balancer.record_assignment(selected_model.model_id)
            
            duration_ms = (time.time() - start_time) * 1000
            self.tracing.end_trace(trace_id, {
                "selected_model": selected_model.model_id,
                "confidence": confidence,
                "duration_ms": duration_ms
            })
            
            logger.info(
                f"Routed task {task_requirements.task_id} to {selected_model.model_id} "
                f"(confidence: {confidence:.2f}, cost: ${decision.estimated_cost:.4f})"
            )
            
            return decision
            
        except Exception as e:
            self.tracing.record_error(trace_id, str(e))
            logger.error(f"Routing failed for task {task_requirements.task_id}: {e}")
            return await self._create_fallback_decision(task_requirements)
    
    async def _get_eligible_models(
        self,
        task_requirements: TaskRequirements
    ) -> List[ModelConfig]:
        """Get models that meet the task requirements"""
        eligible = []
        
        for model in self.models.values():
            # Check circuit breaker
            if model.circuit_breaker and model.circuit_breaker.is_open():
                continue
            
            # Check capabilities
            if not self._has_required_capabilities(model, task_requirements.required_capabilities):
                continue
            
            # Check complexity support
            if not self._supports_complexity(model, task_requirements.complexity):
                continue
            
            # Check token limits
            if task_requirements.estimated_tokens > model.max_tokens:
                continue
            
            # Check reliability threshold
            if model.reliability_score < 0.8:
                continue
            
            eligible.append(model)
        
        return eligible
    
    def _has_required_capabilities(
        self,
        model: ModelConfig,
        required_capabilities: List[str]
    ) -> bool:
        """Check if model has all required capabilities"""
        model_capabilities = set(model.capabilities)
        required = set(required_capabilities)
        return required.issubset(model_capabilities)
    
    def _supports_complexity(
        self,
        model: ModelConfig,
        complexity: TaskComplexity
    ) -> bool:
        """Check if model supports the task complexity"""
        complexity_support = {
            ModelTier.TIER_1: [TaskComplexity.TRIVIAL, TaskComplexity.SIMPLE, 
                              TaskComplexity.MODERATE, TaskComplexity.COMPLEX, TaskComplexity.EXPERT],
            ModelTier.TIER_2: [TaskComplexity.TRIVIAL, TaskComplexity.SIMPLE, TaskComplexity.MODERATE],
            ModelTier.TIER_3: [TaskComplexity.TRIVIAL, TaskComplexity.SIMPLE],
            ModelTier.TIER_4: [TaskComplexity.TRIVIAL, TaskComplexity.SIMPLE, TaskComplexity.MODERATE]
        }
        
        return complexity in complexity_support.get(model.tier, [])
    
    async def _score_models(
        self,
        models: List[ModelConfig],
        task_requirements: TaskRequirements,
        agent_id: str,
        context: Optional[Dict[str, Any]]
    ) -> List[Tuple[ModelConfig, float, List[str]]]:
        """Score models based on multiple criteria"""
        scored_models = []
        
        for model in models:
            score_components = []
            reasoning = []
            
            # 1. Cost score (0-1, higher is better/cheaper)
            cost_score = self._calculate_cost_score(model, task_requirements)
            score_components.append(("cost", cost_score, 0.3))
            reasoning.append(f"Cost score: {cost_score:.2f}")
            
            # 2. Performance score (0-1, higher is better)
            perf_score = self._calculate_performance_score(model, task_requirements)
            score_components.append(("performance", perf_score, 0.25))
            reasoning.append(f"Performance score: {perf_score:.2f}")
            
            # 3. Quality score (0-1, higher is better)
            quality_score = self._calculate_quality_score(model, task_requirements)
            score_components.append(("quality", quality_score, 0.25))
            reasoning.append(f"Quality score: {quality_score:.2f}")
            
            # 4. Load balancing score (0-1, higher is better)
            load_score = self._calculate_load_score(model)
            score_components.append(("load", load_score, 0.1))
            reasoning.append(f"Load score: {load_score:.2f}")
            
            # 5. Historical success score (0-1, higher is better)
            history_score = await self._calculate_history_score(model, task_requirements)
            score_components.append(("history", history_score, 0.1))
            reasoning.append(f"History score: {history_score:.2f}")
            
            # Calculate weighted total score
            total_score = sum(score * weight for _, score, weight in score_components)
            
            # Apply routing strategy weights
            total_score = self._apply_routing_strategy(total_score, score_components, task_requirements)
            
            scored_models.append((model, total_score, reasoning))
        
        # Sort by score descending
        scored_models.sort(key=lambda x: x[1], reverse=True)
        return scored_models
    
    def _calculate_cost_score(
        self,
        model: ModelConfig,
        task_requirements: TaskRequirements
    ) -> float:
        """Calculate cost efficiency score (0-1)"""
        estimated_cost = self._estimate_cost(model, task_requirements)
        
        if estimated_cost > task_requirements.max_cost:
            return 0.0
        
        # Normalize cost relative to budget (lower cost = higher score)
        cost_ratio = estimated_cost / task_requirements.max_cost
        cost_score = max(0.0, 1.0 - cost_ratio)
        
        # Apply tier-based discount for lower tiers
        tier_discount = {
            ModelTier.TIER_1: 1.0,
            ModelTier.TIER_2: 1.1,
            ModelTier.TIER_3: 1.3,
            ModelTier.TIER_4: 1.2
        }
        
        return min(1.0, cost_score * tier_discount.get(model.tier, 1.0))
    
    def _calculate_performance_score(
        self,
        model: ModelConfig,
        task_requirements: TaskRequirements
    ) -> float:
        """Calculate performance score based on latency and throughput"""
        estimated_latency = self._estimate_latency(model, task_requirements)
        
        if estimated_latency > task_requirements.max_latency_ms:
            return 0.0
        
        # Normalize latency (lower latency = higher score)
        latency_ratio = estimated_latency / task_requirements.max_latency_ms
        latency_score = max(0.0, 1.0 - latency_ratio)
        
        # Factor in model's current load
        load_penalty = 1.0 - (model.current_load * 0.5)
        
        return latency_score * load_penalty
    
    def _calculate_quality_score(
        self,
        model: ModelConfig,
        task_requirements: TaskRequirements
    ) -> float:
        """Calculate expected quality score"""
        # Base quality from model tier
        tier_quality = {
            ModelTier.TIER_1: 0.95,
            ModelTier.TIER_2: 0.85,
            ModelTier.TIER_3: 0.75,
            ModelTier.TIER_4: 0.80
        }
        
        base_quality = tier_quality.get(model.tier, 0.7)
        
        # Adjust for complexity match
        complexity_match = self._complexity_match_score(model, task_requirements.complexity)
        
        # Factor in reliability
        reliability_factor = model.reliability_score
        
        return base_quality * complexity_match * reliability_factor
    
    def _complexity_match_score(
        self,
        model: ModelConfig,
        complexity: TaskComplexity
    ) -> float:
        """Score how well model complexity matches task complexity"""
        # Models should not be over or under-powered for the task
        tier_complexity_range = {
            ModelTier.TIER_1: (3, 5),  # Good for complex to expert
            ModelTier.TIER_2: (2, 4),  # Good for simple to complex
            ModelTier.TIER_3: (1, 2),  # Good for trivial to simple
            ModelTier.TIER_4: (1, 3)   # Good for trivial to moderate
        }
        
        min_comp, max_comp = tier_complexity_range.get(model.tier, (1, 5))
        
        if min_comp <= complexity.value <= max_comp:
            return 1.0
        elif complexity.value < min_comp:
            # Model is over-powered (waste of resources)
            return 0.7
        else:
            # Model is under-powered (might struggle)
            return 0.5
    
    def _calculate_load_score(self, model: ModelConfig) -> float:
        """Calculate load balancing score"""
        # Prefer models with lower current load
        load_score = 1.0 - model.current_load
        
        # Factor in load balancer recommendations
        balancer_score = self.load_balancer.get_model_score(model.model_id)
        
        return (load_score * 0.7) + (balancer_score * 0.3)
    
    async def _calculate_history_score(
        self,
        model: ModelConfig,
        task_requirements: TaskRequirements
    ) -> float:
        """Calculate score based on historical performance for similar tasks"""
        task_type = self._extract_task_type(task_requirements)
        
        if task_type in self.task_model_mapping:
            successful_models = self.task_model_mapping[task_type]
            if model.model_id in successful_models:
                # Count how often this model succeeded for this task type
                success_count = successful_models.count(model.model_id)
                total_count = len(successful_models)
                return min(1.0, success_count / total_count * 1.5)  # Boost for proven success
        
        # Check performance history
        if model.model_id in self.model_performance_history:
            history = self.model_performance_history[model.model_id]
            if history:
                avg_performance = np.mean(history[-10:])  # Last 10 performances
                return avg_performance
        
        return 0.5  # Neutral score for unknown
    
    def _extract_task_type(self, task_requirements: TaskRequirements) -> str:
        """Extract task type from requirements for historical matching"""
        # Create a hash of key characteristics
        key_parts = [
            task_requirements.complexity.name,
            "_".join(sorted(task_requirements.required_capabilities[:3])),
            str(task_requirements.estimated_tokens // 1000) + "k_tokens"
        ]
        return hashlib.md5("_".join(key_parts).encode()).hexdigest()[:8]
    
    def _apply_routing_strategy(
        self,
        base_score: float,
        score_components: List[Tuple[str, float, float]],
        task_requirements: TaskRequirements
    ) -> float:
        """Apply routing strategy to adjust scores"""
        if self.routing_strategy == RoutingStrategy.COST_OPTIMIZED:
            # Boost cost score, reduce others
            cost_score = next((s for n, s, _ in score_components if n == "cost"), 0.5)
            return base_score * 0.6 + cost_score * 0.4
        
        elif self.routing_strategy == RoutingStrategy.PERFORMANCE_OPTIMIZED:
            # Boost performance score
            perf_score = next((s for n, s, _ in score_components if n == "performance"), 0.5)
            return base_score * 0.6 + perf_score * 0.4
        
        elif self.routing_strategy == RoutingStrategy.QUALITY_SENSITIVE:
            # Boost quality score
            quality_score = next((s for n, s, _ in score_components if n == "quality"), 0.5)
            return base_score * 0.5 + quality_score * 0.5
        
        elif self.routing_strategy == RoutingStrategy.LATENCY_SENSITIVE:
            # Boost performance score (which includes latency)
            perf_score = next((s for n, s, _ in score_components if n == "performance"), 0.5)
            return base_score * 0.4 + perf_score * 0.6
        
        # Default: BALANCED strategy
        return base_score
    
    def _select_best_model(
        self,
        scored_models: List[Tuple[ModelConfig, float, List[str]]],
        task_requirements: TaskRequirements
    ) -> Tuple[ModelConfig, float, List[str]]:
        """Select the best model from scored list"""
        if not scored_models:
            raise ValueError("No models to select from")
        
        best_model, best_score, best_reasoning = scored_models[0]
        
        # Check if score meets minimum threshold
        min_threshold = 0.3
        if best_score < min_threshold:
            # Try to find any model that meets threshold
            for model, score, reasoning in scored_models:
                if score >= min_threshold:
                    return model, score, reasoning
        
        return best_model, best_score, best_reasoning
    
    def _estimate_cost(self, model: ModelConfig, task_requirements: TaskRequirements) -> float:
        """Estimate cost for task on given model"""
        tokens = task_requirements.estimated_tokens
        cost_per_token = model.cost_per_1k_tokens / 1000
        return tokens * cost_per_token
    
    def _estimate_latency(self, model: ModelConfig, task_requirements: TaskRequirements) -> float:
        """Estimate latency for task on given model"""
        base_latency = model.avg_latency_ms
        
        # Adjust for token count
        token_factor = task_requirements.estimated_tokens / 1000
        latency = base_latency * (1 + token_factor * 0.1)
        
        # Adjust for current load
        load_factor = 1 + (model.current_load * 0.5)
        
        return latency * load_factor
    
    def _create_alternatives_list(
        self,
        scored_models: List[Tuple[ModelConfig, float, List[str]]]
    ) -> List[Dict[str, Any]]:
        """Create list of alternative models"""
        alternatives = []
        
        for model, score, reasoning in scored_models[:5]:  # Top 5 alternatives
            alternatives.append({
                "model_id": model.model_id,
                "score": score,
                "tier": model.tier.value,
                "cost_per_1k_tokens": model.cost_per_1k_tokens,
                "avg_latency_ms": model.avg_latency_ms
            })
        
        return alternatives
    
    def _generate_cache_key(
        self,
        task_requirements: TaskRequirements,
        agent_id: str
    ) -> str:
        """Generate cache key for routing decision"""
        key_parts = [
            task_requirements.task_id,
            agent_id,
            task_requirements.complexity.name,
            str(task_requirements.estimated_tokens),
            str(task_requirements.max_cost),
            str(task_requirements.max_latency_ms),
            "_".join(sorted(task_requirements.required_capabilities))
        ]
        return hashlib.md5("_".join(key_parts).encode()).hexdigest()
    
    def _get_cached_decision(self, cache_key: str) -> Optional[RoutingDecision]:
        """Get cached routing decision if still valid"""
        if cache_key in self.routing_cache:
            decision = self.routing_cache[cache_key]
            if (datetime.utcnow() - decision.timestamp).seconds < self.cache_ttl:
                return decision
            else:
                del self.routing_cache[cache_key]
        return None
    
    def _cache_decision(self, cache_key: str, decision: RoutingDecision):
        """Cache routing decision"""
        self.routing_cache[cache_key] = decision
        
        # Limit cache size
        if len(self.routing_cache) > 1000:
            # Remove oldest entries
            sorted_keys = sorted(
                self.routing_cache.keys(),
                key=lambda k: self.routing_cache[k].timestamp
            )
            for key in sorted_keys[:200]:  # Remove oldest 200
                del self.routing_cache[key]
    
    async def _create_fallback_decision(
        self,
        task_requirements: TaskRequirements
    ) -> RoutingDecision:
        """Create fallback routing decision when no models are eligible"""
        # Use fallback manager to get fallback model
        fallback_model_id = await self.fallback_manager.get_fallback_model(
            task_requirements.required_capabilities
        )
        
        if fallback_model_id and fallback_model_id in self.models:
            model = self.models[fallback_model_id]
        else:
            # Use cheapest available model
            model = min(
                self.models.values(),
                key=lambda m: m.cost_per_1k_tokens
            )
        
        return RoutingDecision(
            task_id=task_requirements.task_id,
            selected_model=model.model_id,
            confidence_score=0.3,
            estimated_cost=self._estimate_cost(model, task_requirements),
            estimated_latency_ms=self._estimate_latency(model, task_requirements) * 1.5,
            reasoning=["Fallback routing: no optimal model found", "Using cheapest available model"],
            alternatives=[],
            timestamp=datetime.utcnow()
        )
    
    async def update_model_performance(
        self,
        model_id: str,
        task_id: str,
        actual_cost: float,
        actual_latency_ms: float,
        quality_score: float,
        success: bool
    ):
        """Update model performance metrics after task completion"""
        if model_id not in self.models:
            return
        
        model = self.models[model_id]
        
        # Update performance history
        performance_score = quality_score * (1.0 if success else 0.0)
        self.model_performance_history[model_id].append(performance_score)
        
        # Keep only last 100 entries
        if len(self.model_performance_history[model_id]) > 100:
            self.model_performance_history[model_id] = self.model_performance_history[model_id][-100:]
        
        # Update circuit breaker
        if model.circuit_breaker:
            if success:
                model.circuit_breaker.record_success()
            else:
                model.circuit_breaker.record_failure()
        
        # Update cost tracker
        self.cost_tracker.record_model_cost(
            model_id=model_id,
            task_id=task_id,
            cost=actual_cost
        )
        
        # Update metrics collector
        self.metrics_collector.record_model_performance(
            model_id=model_id,
            latency_ms=actual_latency_ms,
            quality_score=quality_score,
            success=success
        )
        
        # Update task-model mapping for successful tasks
        if success and task_id:
            task_type = self._extract_task_type_from_id(task_id)
            if task_type:
                self.task_model_mapping[task_type].append(model_id)
                # Keep only last 50 mappings per task type
                if len(self.task_model_mapping[task_type]) > 50:
                    self.task_model_mapping[task_type] = self.task_model_mapping[task_type][-50:]
        
        logger.debug(
            f"Updated performance for {model_id}: "
            f"cost=${actual_cost:.4f}, latency={actual_latency_ms}ms, "
            f"quality={quality_score:.2f}, success={success}"
        )
    
    def _extract_task_type_from_id(self, task_id: str) -> Optional[str]:
        """Extract task type from task ID (if encoded)"""
        # In a real implementation, this would parse the task ID
        # For now, return a hash of the task ID
        if task_id:
            return hashlib.md5(task_id.encode()).hexdigest()[:8]
        return None
    
    async def update_model_load(self, model_id: str, load: float):
        """Update model's current load"""
        if model_id in self.models:
            self.models[model_id].current_load = max(0.0, min(1.0, load))
            
            # Update load balancer
            self.load_balancer.update_load(model_id, load)
    
    async def set_routing_strategy(self, strategy: RoutingStrategy):
        """Set the routing strategy"""
        self.routing_strategy = strategy
        logger.info(f"Routing strategy set to: {strategy.value}")
    
    async def set_budget_constraint(self, agent_id: str, daily_budget: float):
        """Set daily budget constraint for an agent"""
        self.budget_constraints[agent_id] = daily_budget
        logger.info(f"Set daily budget of ${daily_budget:.2f} for agent {agent_id}")
    
    async def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics"""
        total_decisions = len(self.routing_history)
        
        if total_decisions == 0:
            return {"total_decisions": 0}
        
        # Calculate model distribution
        model_distribution = defaultdict(int)
        total_cost = 0.0
        total_confidence = 0.0
        
        for decision in self.routing_history:
            model_distribution[decision.selected_model] += 1
            total_cost += decision.estimated_cost
            total_confidence += decision.confidence_score
        
        avg_confidence = total_confidence / total_decisions
        avg_cost = total_cost / total_decisions
        
        # Calculate strategy distribution
        strategy_counts = defaultdict(int)
        for decision in self.routing_history[-100:]:  # Last 100 decisions
            # Infer strategy from reasoning (simplified)
            if any("cost" in r.lower() for r in decision.reasoning):
                strategy_counts["cost_optimized"] += 1
            elif any("performance" in r.lower() for r in decision.reasoning):
                strategy_counts["performance_optimized"] += 1
            else:
                strategy_counts["balanced"] += 1
        
        return {
            "total_decisions": total_decisions,
            "model_distribution": dict(model_distribution),
            "average_confidence": avg_confidence,
            "average_cost": avg_cost,
            "strategy_distribution": dict(strategy_counts),
            "cache_hit_rate": len(self.routing_cache) / max(total_decisions, 1),
            "models_available": len(self.models),
            "current_strategy": self.routing_strategy.value
        }
    
    async def _background_optimization(self):
        """Background task for continuous optimization"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Clean old cache entries
                self._clean_cache()
                
                # Optimize model weights based on performance
                await self._optimize_model_weights()
                
                # Update load balancer
                self.load_balancer.rebalance()
                
                logger.debug("Background optimization completed")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background optimization error: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    def _clean_cache(self):
        """Clean expired cache entries"""
        current_time = datetime.utcnow()
        expired_keys = []
        
        for key, decision in self.routing_cache.items():
            if (current_time - decision.timestamp).seconds > self.cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.routing_cache[key]
    
    async def _optimize_model_weights(self):
        """Optimize model selection weights based on historical performance"""
        # Calculate success rates per model
        model_success_rates = {}
        
        for model_id, history in self.model_performance_history.items():
            if history:
                success_rate = np.mean(history)
                model_success_rates[model_id] = success_rate
        
        # Adjust model reliability scores based on performance
        for model_id, success_rate in model_success_rates.items():
            if model_id in self.models:
                # Smooth update of reliability score
                current = self.models[model_id].reliability_score
                new_score = current * 0.9 + success_rate * 0.1
                self.models[model_id].reliability_score = new_score
    
    async def shutdown(self):
        """Shutdown the cost optimizer"""
        self._optimization_task.cancel()
        try:
            await self._optimization_task
        except asyncio.CancelledError:
            pass
        
        logger.info("CostOptimizer shutdown complete")


class LoadBalancer:
    """Simple load balancer for model instances"""
    
    def __init__(self):
        self.model_loads: Dict[str, float] = {}
        self.assignment_counts: Dict[str, int] = defaultdict(int)
        self.last_rebalance = datetime.utcnow()
    
    def record_assignment(self, model_id: str):
        """Record a task assignment to a model"""
        self.assignment_counts[model_id] += 1
        
        # Update load estimate (simplified)
        if model_id in self.model_loads:
            self.model_loads[model_id] = min(1.0, self.model_loads[model_id] + 0.1)
    
    def update_load(self, model_id: str, load: float):
        """Update model's current load"""
        self.model_loads[model_id] = load
    
    def get_model_score(self, model_id: str) -> float:
        """Get load balancing score for model (0-1, higher is better)"""
        load = self.model_loads.get(model_id, 0.0)
        count = self.assignment_counts.get(model_id, 0)
        
        # Prefer models with lower load and fewer recent assignments
        load_score = 1.0 - load
        count_penalty = min(1.0, count / 100)  # Normalize by expected max
        
        return load_score * (1.0 - count_penalty * 0.3)
    
    def rebalance(self):
        """Rebalance load estimates"""
        # Decay assignment counts over time
        for model_id in list(self.assignment_counts.keys()):
            self.assignment_counts[model_id] = max(0, self.assignment_counts[model_id] - 10)
            if self.assignment_counts[model_id] == 0:
                del self.assignment_counts[model_id]
        
        # Decay loads slightly
        for model_id in list(self.model_loads.keys()):
            self.model_loads[model_id] = max(0.0, self.model_loads[model_id] * 0.9)
        
        self.last_rebalance = datetime.utcnow()