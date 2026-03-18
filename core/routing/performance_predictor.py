"""
core/routing/performance_predictor.py

Intelligent routing layer for multi-model agent orchestration.
Routes tasks to optimal LLM models based on complexity, cost, performance, and system load.
"""

import asyncio
import json
import time
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
import statistics
import logging
from concurrent.futures import ThreadPoolExecutor

from monitoring.cost_tracker import CostTracker
from monitoring.metrics_collector import MetricsCollector
from monitoring.tracing import TracingManager
from core.resilience.circuit_breaker import CircuitBreakerRegistry
from core.resilience.retry_policy import RetryPolicy
from core.resilience.fallback_manager import FallbackManager
from core.distributed.state_manager import DistributedStateManager
from core.composition.capability_graph import CapabilityGraph

logger = logging.getLogger(__name__)


class TaskComplexity(Enum):
    """Task complexity levels for routing decisions."""
    TRIVIAL = 1      # Simple lookups, formatting
    SIMPLE = 2       # Basic transformations, simple Q&A
    MODERATE = 3     # Analysis, summarization, code generation
    COMPLEX = 4      # Multi-step reasoning, complex problem solving
    EXPERT = 5       # Research, novel solutions, architectural decisions


class ModelTier(Enum):
    """Model tiers based on capability and cost."""
    ECONOMY = "economy"        # Fast, cheap models (GPT-3.5, Claude Instant)
    STANDARD = "standard"      # Balanced models (GPT-4, Claude 2)
    PREMIUM = "premium"        # High-capability models (GPT-4 Turbo, Claude 3 Opus)
    SPECIALIZED = "specialized"  # Domain-specific fine-tuned models


@dataclass
class ModelProfile:
    """Profile of an available model with capabilities and constraints."""
    model_id: str
    tier: ModelTier
    capabilities: Set[str]
    cost_per_1k_tokens: float
    avg_latency_ms: float
    max_context_tokens: int
    quality_score: float  # 0-1 scale
    supports_streaming: bool = True
    supports_functions: bool = False
    rate_limit_rpm: int = 60
    current_load: float = 0.0  # 0-1 scale
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['tier'] = self.tier.value
        data['capabilities'] = list(self.capabilities)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelProfile':
        """Create from dictionary."""
        data['tier'] = ModelTier(data['tier'])
        data['capabilities'] = set(data['capabilities'])
        return cls(**data)


@dataclass
class TaskRequirements:
    """Requirements extracted from a task for routing decisions."""
    task_id: str
    complexity: TaskComplexity
    required_capabilities: Set[str]
    max_cost_usd: Optional[float] = None
    max_latency_ms: Optional[float] = None
    min_quality_score: float = 0.7
    context_length: int = 0
    priority: int = 1  # 1-10 scale
    deadline: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RoutingDecision:
    """Result of routing decision with reasoning."""
    task_id: str
    selected_model: str
    confidence: float
    estimated_cost: float
    estimated_latency_ms: float
    reasoning: List[str]
    alternatives: List[Tuple[str, float]]  # (model_id, score)
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/monitoring."""
        return {
            'task_id': self.task_id,
            'selected_model': self.selected_model,
            'confidence': self.confidence,
            'estimated_cost': self.estimated_cost,
            'estimated_latency_ms': self.estimated_latency_ms,
            'reasoning': self.reasoning,
            'alternatives': self.alternatives,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


class PerformancePredictor:
    """
    Intelligent routing layer that selects optimal LLM models for tasks.
    
    Considers:
    - Task complexity and requirements
    - Model capabilities and performance profiles
    - Current system load and rate limits
    - Cost constraints and budgets
    - Historical performance data
    - Circuit breaker states for resilience
    """
    
    def __init__(
        self,
        cost_tracker: CostTracker,
        metrics_collector: MetricsCollector,
        circuit_breaker_registry: CircuitBreakerRegistry,
        state_manager: DistributedStateManager,
        capability_graph: CapabilityGraph,
        config: Optional[Dict[str, Any]] = None
    ):
        self.cost_tracker = cost_tracker
        self.metrics_collector = metrics_collector
        self.circuit_breaker_registry = circuit_breaker_registry
        self.state_manager = state_manager
        self.capability_graph = capability_graph
        
        self.config = config or self._default_config()
        self.models: Dict[str, ModelProfile] = {}
        self.routing_history: List[RoutingDecision] = []
        self.performance_cache: Dict[str, Dict[str, float]] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize with default models
        self._initialize_default_models()
        
        # Start background tasks
        self._start_background_tasks()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for the performance predictor."""
        return {
            'max_history_size': 1000,
            'performance_window_hours': 24,
            'cost_weight': 0.3,
            'latency_weight': 0.25,
            'quality_weight': 0.35,
            'load_weight': 0.1,
            'enable_learning': True,
            'min_confidence_threshold': 0.6,
            'max_alternatives': 3,
            'refresh_interval_seconds': 300,
            'budget_alert_threshold': 0.8
        }
    
    def _initialize_default_models(self):
        """Initialize with common model profiles."""
        default_models = [
            ModelProfile(
                model_id="gpt-3.5-turbo",
                tier=ModelTier.ECONOMY,
                capabilities={"text", "chat", "simple_reasoning", "code_generation"},
                cost_per_1k_tokens=0.002,
                avg_latency_ms=800,
                max_context_tokens=4096,
                quality_score=0.75,
                supports_functions=True,
                rate_limit_rpm=3500
            ),
            ModelProfile(
                model_id="gpt-4",
                tier=ModelTier.STANDARD,
                capabilities={"text", "chat", "complex_reasoning", "code_generation", "analysis"},
                cost_per_1k_tokens=0.03,
                avg_latency_ms=2000,
                max_context_tokens=8192,
                quality_score=0.92,
                supports_functions=True,
                rate_limit_rpm=200
            ),
            ModelProfile(
                model_id="gpt-4-turbo",
                tier=ModelTier.PREMIUM,
                capabilities={"text", "chat", "complex_reasoning", "code_generation", "analysis", "long_context"},
                cost_per_1k_tokens=0.01,
                avg_latency_ms=1500,
                max_context_tokens=128000,
                quality_score=0.95,
                supports_functions=True,
                rate_limit_rpm=500
            ),
            ModelProfile(
                model_id="claude-instant-1.2",
                tier=ModelTier.ECONOMY,
                capabilities={"text", "chat", "simple_reasoning", "creative_writing"},
                cost_per_1k_tokens=0.0015,
                avg_latency_ms=600,
                max_context_tokens=100000,
                quality_score=0.78,
                supports_streaming=True,
                rate_limit_rpm=1000
            ),
            ModelProfile(
                model_id="claude-2.1",
                tier=ModelTier.STANDARD,
                capabilities={"text", "chat", "complex_reasoning", "analysis", "creative_writing", "long_context"},
                cost_per_1k_tokens=0.011,
                avg_latency_ms=1800,
                max_context_tokens=200000,
                quality_score=0.89,
                supports_streaming=True,
                rate_limit_rpm=300
            )
        ]
        
        for model in default_models:
            self.models[model.model_id] = model
    
    def _start_background_tasks(self):
        """Start background tasks for model updates and performance tracking."""
        asyncio.create_task(self._refresh_model_loads())
        asyncio.create_task(self._update_performance_metrics())
    
    async def _refresh_model_loads(self):
        """Periodically refresh model load information."""
        while True:
            try:
                await self._update_model_loads()
                await asyncio.sleep(self.config['refresh_interval_seconds'])
            except Exception as e:
                logger.error(f"Error refreshing model loads: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _update_model_loads(self):
        """Update current load for all models based on metrics."""
        for model_id, profile in self.models.items():
            try:
                # Get recent request count from metrics
                recent_requests = await self.metrics_collector.get_request_count(
                    model_id=model_id,
                    window_seconds=300
                )
                
                # Calculate load as percentage of rate limit
                max_requests = profile.rate_limit_rpm * 5  # 5-minute window
                profile.current_load = min(1.0, recent_requests / max_requests)
                
                # Update in distributed state
                await self.state_manager.set(
                    f"model_load:{model_id}",
                    profile.current_load,
                    ttl=300
                )
            except Exception as e:
                logger.warning(f"Failed to update load for {model_id}: {e}")
    
    async def _update_performance_metrics(self):
        """Periodically update performance metrics from historical data."""
        while True:
            try:
                await self._calculate_performance_metrics()
                await asyncio.sleep(600)  # Update every 10 minutes
            except Exception as e:
                logger.error(f"Error updating performance metrics: {e}")
                await asyncio.sleep(300)
    
    async def _calculate_performance_metrics(self):
        """Calculate performance metrics for each model."""
        window = timedelta(hours=self.config['performance_window_hours'])
        
        for model_id in self.models:
            try:
                # Get success rate
                success_rate = await self.metrics_collector.get_success_rate(
                    model_id=model_id,
                    window=window
                )
                
                # Get average latency
                avg_latency = await self.metrics_collector.get_average_latency(
                    model_id=model_id,
                    window=window
                )
                
                # Get quality scores from feedback
                quality_scores = await self.metrics_collector.get_quality_scores(
                    model_id=model_id,
                    window=window
                )
                
                if quality_scores:
                    avg_quality = statistics.mean(quality_scores)
                    # Update model quality score with exponential smoothing
                    old_score = self.models[model_id].quality_score
                    self.models[model_id].quality_score = 0.7 * old_score + 0.3 * avg_quality
                
                # Update average latency
                if avg_latency:
                    self.models[model_id].avg_latency_ms = avg_latency
                
                # Cache performance metrics
                self.performance_cache[model_id] = {
                    'success_rate': success_rate or 0.0,
                    'avg_latency': avg_latency or self.models[model_id].avg_latency_ms,
                    'quality_score': self.models[model_id].quality_score,
                    'last_updated': datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                logger.warning(f"Failed to calculate metrics for {model_id}: {e}")
    
    def analyze_task(self, task: Dict[str, Any]) -> TaskRequirements:
        """
        Analyze a task to extract routing requirements.
        
        Args:
            task: Task dictionary with content and metadata
            
        Returns:
            TaskRequirements object with extracted requirements
        """
        # Extract basic information
        task_id = task.get('id', f"task_{int(time.time())}")
        content = task.get('content', '')
        metadata = task.get('metadata', {})
        
        # Determine complexity based on content analysis
        complexity = self._estimate_complexity(content, metadata)
        
        # Extract required capabilities
        required_capabilities = self._extract_required_capabilities(task)
        
        # Get constraints from metadata
        max_cost = metadata.get('max_cost_usd')
        max_latency = metadata.get('max_latency_ms')
        min_quality = metadata.get('min_quality_score', 0.7)
        priority = metadata.get('priority', 1)
        deadline = metadata.get('deadline')
        
        # Estimate context length
        context_length = len(content) // 4  # Rough token estimate
        
        return TaskRequirements(
            task_id=task_id,
            complexity=complexity,
            required_capabilities=required_capabilities,
            max_cost_usd=max_cost,
            max_latency_ms=max_latency,
            min_quality_score=min_quality,
            context_length=context_length,
            priority=priority,
            deadline=deadline,
            metadata=metadata
        )
    
    def _estimate_complexity(self, content: str, metadata: Dict[str, Any]) -> TaskComplexity:
        """Estimate task complexity based on content and metadata."""
        # Check metadata for explicit complexity
        if 'complexity' in metadata:
            try:
                return TaskComplexity[metadata['complexity'].upper()]
            except (KeyError, AttributeError):
                pass
        
        # Simple heuristics based on content
        content_lower = content.lower()
        
        # Check for complexity indicators
        if any(term in content_lower for term in ['analyze', 'design', 'architect', 'optimize', 'research']):
            return TaskComplexity.COMPLEX
        elif any(term in content_lower for term in ['explain', 'compare', 'evaluate', 'implement']):
            return TaskComplexity.MODERATE
        elif any(term in content_lower for term in ['list', 'describe', 'summarize', 'translate']):
            return TaskComplexity.SIMPLE
        elif any(term in content_lower for term in ['format', 'convert', 'extract', 'validate']):
            return TaskComplexity.TRIVIAL
        
        # Default based on length
        word_count = len(content.split())
        if word_count > 500:
            return TaskComplexity.COMPLEX
        elif word_count > 200:
            return TaskComplexity.MODERATE
        elif word_count > 50:
            return TaskComplexity.SIMPLE
        else:
            return TaskComplexity.TRIVIAL
    
    def _extract_required_capabilities(self, task: Dict[str, Any]) -> Set[str]:
        """Extract required capabilities from task content and metadata."""
        capabilities = set()
        content = task.get('content', '').lower()
        metadata = task.get('metadata', {})
        
        # Check metadata for explicit capabilities
        if 'required_capabilities' in metadata:
            capabilities.update(metadata['required_capabilities'])
        
        # Infer capabilities from content
        if any(term in content for term in ['code', 'program', 'function', 'script', 'develop']):
            capabilities.add('code_generation')
        
        if any(term in content for term in ['analyze', 'analysis', 'evaluate', 'assess']):
            capabilities.add('analysis')
        
        if any(term in content for term in ['reason', 'logic', 'deduce', 'infer']):
            capabilities.add('complex_reasoning')
        
        if any(term in content for term in ['write', 'create', 'story', 'poem', 'creative']):
            capabilities.add('creative_writing')
        
        if any(term in content for term in ['long', 'document', 'book', 'extensive']):
            capabilities.add('long_context')
        
        # Default capability
        if not capabilities:
            capabilities.add('text')
        
        return capabilities
    
    def _calculate_model_score(
        self,
        model: ModelProfile,
        requirements: TaskRequirements,
        weights: Dict[str, float]
    ) -> Tuple[float, List[str]]:
        """
        Calculate a score for a model given task requirements.
        
        Returns:
            Tuple of (score, reasoning_list)
        """
        reasoning = []
        scores = {}
        
        # Check circuit breaker
        if not self.circuit_breaker_registry.is_available(model.model_id):
            reasoning.append(f"Model {model.model_id} is unavailable (circuit breaker open)")
            return 0.0, reasoning
        
        # Check capability match
        missing_capabilities = requirements.required_capabilities - model.capabilities
        if missing_capabilities:
            reasoning.append(f"Missing capabilities: {missing_capabilities}")
            return 0.0, reasoning
        
        # Check context length
        if requirements.context_length > model.max_context_tokens:
            reasoning.append(f"Context too long: {requirements.context_length} > {model.max_context_tokens}")
            return 0.0, reasoning
        
        # Check quality threshold
        if model.quality_score < requirements.min_quality_score:
            reasoning.append(f"Quality too low: {model.quality_score:.2f} < {requirements.min_quality_score}")
            return 0.0, reasoning
        
        # Calculate cost score (lower is better, normalize to 0-1)
        estimated_tokens = max(100, requirements.context_length * 2)  # Estimate output tokens
        estimated_cost = (estimated_tokens / 1000) * model.cost_per_1k_tokens
        
        if requirements.max_cost_usd and estimated_cost > requirements.max_cost_usd:
            reasoning.append(f"Cost exceeds budget: ${estimated_cost:.4f} > ${requirements.max_cost_usd:.4f}")
            return 0.0, reasoning
        
        # Normalize cost score (inverse, so lower cost = higher score)
        max_cost = 0.1  # Assume max cost per task
        cost_score = max(0, 1 - (estimated_cost / max_cost))
        scores['cost'] = cost_score
        reasoning.append(f"Cost: ${estimated_cost:.4f} (score: {cost_score:.2f})")
        
        # Calculate latency score (lower is better)
        estimated_latency = model.avg_latency_ms
        
        if requirements.max_latency_ms and estimated_latency > requirements.max_latency_ms:
            reasoning.append(f"Latency too high: {estimated_latency:.0f}ms > {requirements.max_latency_ms:.0f}ms")
            return 0.0, reasoning
        
        # Normalize latency score
        max_latency = 5000  # 5 seconds max
        latency_score = max(0, 1 - (estimated_latency / max_latency))
        scores['latency'] = latency_score
        reasoning.append(f"Latency: {estimated_latency:.0f}ms (score: {latency_score:.2f})")
        
        # Quality score (already normalized 0-1)
        quality_score = model.quality_score
        scores['quality'] = quality_score
        reasoning.append(f"Quality: {quality_score:.2f}")
        
        # Load score (lower load is better)
        load_score = 1 - model.current_load
        scores['load'] = load_score
        reasoning.append(f"Load: {model.current_load:.2f} (score: {load_score:.2f})")
        
        # Complexity match bonus
        complexity_match = self._calculate_complexity_match(model, requirements.complexity)
        scores['complexity'] = complexity_match
        reasoning.append(f"Complexity match: {complexity_match:.2f}")
        
        # Calculate weighted total score
        total_score = (
            weights['cost'] * scores['cost'] +
            weights['latency'] * scores['latency'] +
            weights['quality'] * scores['quality'] +
            weights['load'] * scores['load'] +
            0.1 * scores['complexity']  # Small bonus for complexity match
        )
        
        # Apply priority multiplier
        priority_multiplier = 0.5 + (requirements.priority / 20)  # 0.5 to 1.0
        total_score *= priority_multiplier
        
        reasoning.append(f"Total score: {total_score:.3f} (priority multiplier: {priority_multiplier:.2f})")
        
        return total_score, reasoning
    
    def _calculate_complexity_match(self, model: ModelProfile, complexity: TaskComplexity) -> float:
        """Calculate how well model tier matches task complexity."""
        # Map complexity to ideal tier
        ideal_tiers = {
            TaskComplexity.TRIVIAL: ModelTier.ECONOMY,
            TaskComplexity.SIMPLE: ModelTier.ECONOMY,
            TaskComplexity.MODERATE: ModelTier.STANDARD,
            TaskComplexity.COMPLEX: ModelTier.STANDARD,
            TaskComplexity.EXPERT: ModelTier.PREMIUM
        }
        
        ideal_tier = ideal_tiers[complexity]
        
        # Exact match
        if model.tier == ideal_tier:
            return 1.0
        
        # Adjacent tiers
        tier_order = [ModelTier.ECONOMY, ModelTier.STANDARD, ModelTier.PREMIUM, ModelTier.SPECIALIZED]
        model_idx = tier_order.index(model.tier)
        ideal_idx = tier_order.index(ideal_tier)
        
        distance = abs(model_idx - ideal_idx)
        
        if distance == 1:
            return 0.7  # Close match
        elif distance == 2:
            return 0.4  # Moderate mismatch
        else:
            return 0.1  # Poor match
    
    async def route_task(
        self,
        task: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None
    ) -> RoutingDecision:
        """
        Route a task to the optimal model.
        
        Args:
            task: Task dictionary with content and metadata
            constraints: Additional routing constraints
            
        Returns:
            RoutingDecision with selected model and reasoning
        """
        start_time = time.time()
        
        # Analyze task requirements
        requirements = self.analyze_task(task)
        
        # Merge constraints
        if constraints:
            if 'max_cost_usd' in constraints:
                requirements.max_cost_usd = constraints['max_cost_usd']
            if 'max_latency_ms' in constraints:
                requirements.max_latency_ms = constraints['max_latency_ms']
            if 'min_quality_score' in constraints:
                requirements.min_quality_score = constraints['min_quality_score']
        
        # Get weights from config
        weights = {
            'cost': self.config['cost_weight'],
            'latency': self.config['latency_weight'],
            'quality': self.config['quality_weight'],
            'load': self.config['load_weight']
        }
        
        # Score all models
        model_scores = []
        all_reasoning = {}
        
        for model_id, model in self.models.items():
            score, reasoning = self._calculate_model_score(model, requirements, weights)
            if score > 0:
                model_scores.append((model_id, score))
                all_reasoning[model_id] = reasoning
        
        # Sort by score
        model_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Check if we have any valid models
        if not model_scores:
            # Use fallback strategy
            selected_model = await self._fallback_selection(requirements)
            confidence = 0.3
            reasoning = ["No models meet requirements, using fallback"]
            alternatives = []
        else:
            # Select top model
            selected_model, top_score = model_scores[0]
            confidence = top_score
            
            # Get alternatives
            alternatives = model_scores[1:self.config['max_alternatives'] + 1]
            reasoning = all_reasoning[selected_model]
        
        # Calculate estimated cost and latency
        selected_profile = self.models.get(selected_model)
        estimated_cost = 0.0
        estimated_latency = 0.0
        
        if selected_profile:
            estimated_tokens = max(100, requirements.context_length * 2)
            estimated_cost = (estimated_tokens / 1000) * selected_profile.cost_per_1k_tokens
            estimated_latency = selected_profile.avg_latency_ms
        
        # Create routing decision
        decision = RoutingDecision(
            task_id=requirements.task_id,
            selected_model=selected_model,
            confidence=confidence,
            estimated_cost=estimated_cost,
            estimated_latency_ms=estimated_latency,
            reasoning=reasoning,
            alternatives=alternatives,
            timestamp=datetime.utcnow(),
            metadata={
                'complexity': requirements.complexity.name,
                'required_capabilities': list(requirements.required_capabilities),
                'processing_time_ms': (time.time() - start_time) * 1000
            }
        )
        
        # Store in history
        self.routing_history.append(decision)
        if len(self.routing_history) > self.config['max_history_size']:
            self.routing_history = self.routing_history[-self.config['max_history_size']:]
        
        # Log decision
        logger.info(f"Routed task {requirements.task_id} to {selected_model} "
                   f"(confidence: {confidence:.2f}, cost: ${estimated_cost:.4f})")
        
        # Record metrics
        await self.metrics_collector.record_routing_decision(decision.to_dict())
        
        # Update cost tracker
        await self.cost_tracker.record_estimate(
            task_id=requirements.task_id,
            model_id=selected_model,
            estimated_cost=estimated_cost,
            actual_cost=None  # Will be updated after execution
        )
        
        return decision
    
    async def _fallback_selection(self, requirements: TaskRequirements) -> str:
        """Select a fallback model when no models meet requirements."""
        # Try to find any available model
        for model_id, model in self.models.items():
            if self.circuit_breaker_registry.is_available(model_id):
                # Check if it at least has basic capabilities
                if 'text' in model.capabilities:
                    return model_id
        
        # Last resort: return the first model (even if circuit breaker is open)
        # The circuit breaker will handle the failure appropriately
        return list(self.models.keys())[0] if self.models else "gpt-3.5-turbo"
    
    async def update_model_performance(
        self,
        model_id: str,
        task_id: str,
        actual_latency_ms: float,
        actual_cost: float,
        success: bool,
        quality_score: Optional[float] = None
    ):
        """
        Update model performance based on actual execution results.
        
        Args:
            model_id: Model that executed the task
            task_id: Task that was executed
            actual_latency_ms: Actual latency observed
            actual_cost: Actual cost incurred
            success: Whether the task completed successfully
            quality_score: Quality score from feedback (0-1)
        """
        # Update cost tracker with actual cost
        await self.cost_tracker.record_actual(
            task_id=task_id,
            model_id=model_id,
            actual_cost=actual_cost
        )
        
        # Record metrics
        await self.metrics_collector.record_execution(
            model_id=model_id,
            task_id=task_id,
            latency_ms=actual_latency_ms,
            success=success,
            quality_score=quality_score
        )
        
        # Update model's average latency with exponential smoothing
        if model_id in self.models:
            old_latency = self.models[model_id].avg_latency_ms
            # Weight recent observations more heavily
            self.models[model_id].avg_latency_ms = 0.7 * old_latency + 0.3 * actual_latency_ms
        
        # Check if we should update routing decision in history
        for decision in reversed(self.routing_history):
            if decision.task_id == task_id:
                decision.metadata['actual_latency_ms'] = actual_latency_ms
                decision.metadata['actual_cost'] = actual_cost
                decision.metadata['success'] = success
                decision.metadata['quality_score'] = quality_score
                break
        
        logger.debug(f"Updated performance for {model_id} on task {task_id}: "
                    f"latency={actual_latency_ms:.0f}ms, cost=${actual_cost:.4f}, success={success}")
    
    def get_model_recommendations(
        self,
        task_type: str,
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get model recommendations for a given task type.
        
        Args:
            task_type: Type of task (e.g., 'code_generation', 'analysis')
            constraints: Optional constraints
            
        Returns:
            List of model recommendations with scores
        """
        recommendations = []
        
        for model_id, model in self.models.items():
            # Check if model supports the task type
            if task_type in model.capabilities or 'text' in model.capabilities:
                score = model.quality_score
                
                # Adjust for cost
                cost_factor = 1.0 / (1.0 + model.cost_per_1k_tokens * 100)
                score *= cost_factor
                
                # Adjust for latency
                latency_factor = 1.0 / (1.0 + model.avg_latency_ms / 1000)
                score *= latency_factor
                
                # Adjust for load
                load_factor = 1.0 - model.current_load
                score *= load_factor
                
                recommendations.append({
                    'model_id': model_id,
                    'tier': model.tier.value,
                    'score': score,
                    'cost_per_1k_tokens': model.cost_per_1k_tokens,
                    'avg_latency_ms': model.avg_latency_ms,
                    'quality_score': model.quality_score,
                    'current_load': model.current_load,
                    'capabilities': list(model.capabilities)
                })
        
        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return recommendations
    
    def register_model(self, model_profile: ModelProfile):
        """Register a new model or update an existing one."""
        self.models[model_profile.model_id] = model_profile
        logger.info(f"Registered model: {model_profile.model_id} ({model_profile.tier.value})")
    
    def remove_model(self, model_id: str):
        """Remove a model from routing."""
        if model_id in self.models:
            del self.models[model_id]
            logger.info(f"Removed model: {model_id}")
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics and performance metrics."""
        if not self.routing_history:
            return {}
        
        total_decisions = len(self.routing_history)
        model_counts = {}
        avg_confidence = 0.0
        avg_cost = 0.0
        avg_latency = 0.0
        
        for decision in self.routing_history:
            model_counts[decision.selected_model] = model_counts.get(decision.selected_model, 0) + 1
            avg_confidence += decision.confidence
            avg_cost += decision.estimated_cost
            avg_latency += decision.estimated_latency_ms
        
        avg_confidence /= total_decisions
        avg_cost /= total_decisions
        avg_latency /= total_decisions
        
        # Calculate success rate from recent decisions
        recent_decisions = self.routing_history[-100:] if len(self.routing_history) > 100 else self.routing_history
        success_count = sum(1 for d in recent_decisions if d.metadata.get('success', True))
        success_rate = success_count / len(recent_decisions) if recent_decisions else 0.0
        
        return {
            'total_decisions': total_decisions,
            'model_distribution': model_counts,
            'avg_confidence': avg_confidence,
            'avg_estimated_cost': avg_cost,
            'avg_estimated_latency_ms': avg_latency,
            'success_rate': success_rate,
            'models_registered': len(self.models),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the routing system."""
        healthy_models = []
        unhealthy_models = []
        
        for model_id, model in self.models.items():
            if self.circuit_breaker_registry.is_available(model_id):
                healthy_models.append(model_id)
            else:
                unhealthy_models.append(model_id)
        
        return {
            'status': 'healthy' if healthy_models else 'degraded',
            'healthy_models': healthy_models,
            'unhealthy_models': unhealthy_models,
            'total_models': len(self.models),
            'routing_decisions_today': len([
                d for d in self.routing_history
                if d.timestamp.date() == datetime.utcnow().date()
            ]),
            'timestamp': datetime.utcnow().isoformat()
        }


# Factory function for easy instantiation
def create_performance_predictor(
    cost_tracker: CostTracker,
    metrics_collector: MetricsCollector,
    circuit_breaker_registry: CircuitBreakerRegistry,
    state_manager: DistributedStateManager,
    capability_graph: CapabilityGraph,
    config: Optional[Dict[str, Any]] = None
) -> PerformancePredictor:
    """Create and return a PerformancePredictor instance."""
    return PerformancePredictor(
        cost_tracker=cost_tracker,
        metrics_collector=metrics_collector,
        circuit_breaker_registry=circuit_breaker_registry,
        state_manager=state_manager,
        capability_graph=capability_graph,
        config=config
    )