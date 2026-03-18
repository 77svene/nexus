# monitoring/cost_tracker.py
"""
Real-time Agent Observability Platform — Cost Tracking Module
Comprehensive cost monitoring with LLM token usage tracking, compute time analysis,
and distributed cost attribution across agent executions.
"""

import time
import asyncio
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
from datetime import datetime, timedelta
import json
import hashlib
from collections import defaultdict
import uuid

# OpenTelemetry imports
from opentelemetry import trace, metrics
from opentelemetry.trace import Status, StatusCode, Span
from opentelemetry.metrics import Counter, Histogram, ObservableGauge
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource

# Prometheus client for metrics export
from prometheus_client import Counter as PrometheusCounter, Histogram as PrometheusHistogram, Gauge, generate_latest

# Import existing monitoring modules
from monitoring.tracing import get_tracer, create_span, add_span_event
from monitoring.metrics_collector import MetricsCollector, MetricType

# Import existing distributed modules for integration
from core.distributed.executor import AgentExecutionContext
from core.distributed.state_manager import StateManager

# Cost configuration constants
DEFAULT_COST_PER_1K_TOKENS = {
    "gpt-4": 0.03,
    "gpt-4-32k": 0.06,
    "gpt-3.5-turbo": 0.002,
    "gpt-3.5-turbo-16k": 0.004,
    "claude-3-opus": 0.015,
    "claude-3-sonnet": 0.003,
    "claude-3-haiku": 0.00025,
    "default": 0.01
}

DEFAULT_COST_PER_COMPUTE_SECOND = 0.0001  # $0.0001 per compute second

class CostCategory(Enum):
    """Categories of costs tracked in the system"""
    LLM_TOKENS = "llm_tokens"
    COMPUTE_TIME = "compute_time"
    STORAGE = "storage"
    NETWORK = "network"
    EXTERNAL_API = "external_api"
    MEMORY = "memory"

@dataclass
class TokenUsage:
    """Tracks token usage for LLM calls"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = "default"
    cached_tokens: int = 0
    audio_tokens: int = 0
    image_tokens: int = 0
    
    def __add__(self, other: 'TokenUsage') -> 'TokenUsage':
        """Combine token usage from multiple calls"""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            model=self.model if self.model != "default" else other.model,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            audio_tokens=self.audio_tokens + other.audio_tokens,
            image_tokens=self.image_tokens + other.image_tokens
        )

@dataclass
class CostBreakdown:
    """Detailed breakdown of costs for an execution"""
    llm_cost: float = 0.0
    compute_cost: float = 0.0
    storage_cost: float = 0.0
    network_cost: float = 0.0
    external_api_cost: float = 0.0
    memory_cost: float = 0.0
    total_cost: float = 0.0
    currency: str = "USD"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def calculate_total(self):
        """Calculate total cost from all components"""
        self.total_cost = (
            self.llm_cost +
            self.compute_cost +
            self.storage_cost +
            self.network_cost +
            self.external_api_cost +
            self.memory_cost
        )
        return self.total_cost

@dataclass
class AgentCostProfile:
    """Cost profile for a specific agent or execution"""
    agent_id: str
    execution_id: str
    task_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    parent_execution_id: Optional[str] = None
    
    # Token usage tracking
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    
    # Time tracking
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    compute_time_seconds: float = 0.0
    
    # Cost breakdown
    cost_breakdown: CostBreakdown = field(default_factory=CostBreakdown)
    
    # Metadata
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    span_ids: List[str] = field(default_factory=list)
    
    # Performance metrics
    tokens_per_second: float = 0.0
    cost_per_second: float = 0.0
    
    def finalize(self):
        """Finalize cost calculations when execution completes"""
        if self.end_time is None:
            self.end_time = datetime.utcnow()
        
        # Calculate compute time
        time_diff = (self.end_time - self.start_time).total_seconds()
        self.compute_time_seconds = max(0.001, time_diff)  # Minimum 1ms
        
        # Calculate tokens per second
        if self.compute_time_seconds > 0:
            self.tokens_per_second = self.token_usage.total_tokens / self.compute_time_seconds
        
        # Calculate cost per second
        if self.compute_time_seconds > 0:
            self.cost_per_second = self.cost_breakdown.total_cost / self.compute_time_seconds
        
        return self

class CostTracker:
    """
    Main cost tracking system that integrates with OpenTelemetry and Prometheus
    for real-time observability of agent execution costs.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        state_manager: Optional[StateManager] = None
    ):
        self.config = config or {}
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.state_manager = state_manager
        
        # Initialize OpenTelemetry metrics
        self.tracer = get_tracer("cost_tracker")
        self.meter = metrics.get_meter("cost_tracker")
        
        # Cost configuration
        self.token_costs = {**DEFAULT_COST_PER_1K_TOKENS, **self.config.get("token_costs", {})}
        self.compute_cost_per_second = self.config.get("compute_cost_per_second", DEFAULT_COST_PER_COMPUTE_SECOND)
        
        # Active execution tracking
        self.active_executions: Dict[str, AgentCostProfile] = {}
        self.execution_lock = threading.RLock()
        
        # Historical data
        self.execution_history: List[AgentCostProfile] = []
        self.history_max_size = self.config.get("history_max_size", 10000)
        
        # Cost aggregation by various dimensions
        self.cost_by_agent: Dict[str, float] = defaultdict(float)
        self.cost_by_user: Dict[str, float] = defaultdict(float)
        self.cost_by_task: Dict[str, float] = defaultdict(float)
        self.cost_by_model: Dict[str, float] = defaultdict(float)
        
        # Initialize metrics
        self._init_metrics()
        
        # Start background aggregation
        self._start_background_aggregation()
    
    def _init_metrics(self):
        """Initialize OpenTelemetry and Prometheus metrics"""
        
        # OpenTelemetry metrics
        self.execution_counter = self.meter.create_counter(
            name="agent_execution_count",
            description="Total number of agent executions",
            unit="1"
        )
        
        self.token_counter = self.meter.create_counter(
            name="llm_token_usage",
            description="Total LLM tokens used",
            unit="tokens"
        )
        
        self.cost_histogram = self.meter.create_histogram(
            name="agent_execution_cost",
            description="Cost distribution of agent executions",
            unit="USD"
        )
        
        self.compute_time_histogram = self.meter.create_histogram(
            name="agent_compute_time",
            description="Compute time distribution of agent executions",
            unit="seconds"
        )
        
        # Observable gauge for current active executions
        self.active_executions_gauge = self.meter.create_observable_gauge(
            name="active_agent_executions",
            callbacks=[self._get_active_executions],
            description="Number of currently active agent executions",
            unit="1"
        )
        
        # Prometheus metrics (for direct export)
        self.prometheus_cost_total = PrometheusCounter(
            'sovereign_agent_cost_total',
            'Total cost of agent executions',
            ['agent_id', 'model', 'user_id', 'task_type']
        )
        
        self.prometheus_tokens_total = PrometheusCounter(
            'sovereign_llm_tokens_total',
            'Total LLM tokens used',
            ['agent_id', 'model', 'token_type']
        )
        
        self.prometheus_execution_duration = PrometheusHistogram(
            'sovereign_agent_execution_duration_seconds',
            'Duration of agent executions',
            ['agent_id', 'status'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]
        )
        
        # Custom metrics from metrics collector
        self.metrics_collector.register_metric(
            name="cost_tracker.llm_calls",
            metric_type=MetricType.COUNTER,
            description="Number of LLM API calls tracked"
        )
        
        self.metrics_collector.register_metric(
            name="cost_tracker.execution_errors",
            metric_type=MetricType.COUNTER,
            description="Number of execution errors in cost tracking"
        )
    
    def _get_active_executions(self, options):
        """Callback for active executions gauge"""
        with self.execution_lock:
            return [metrics.Observation(len(self.active_executions), {})]
    
    def _start_background_aggregation(self):
        """Start background thread for periodic aggregation and cleanup"""
        def aggregation_worker():
            while True:
                try:
                    self._aggregate_costs()
                    self._cleanup_old_executions()
                    time.sleep(60)  # Run every minute
                except Exception as e:
                    print(f"Error in cost aggregation worker: {e}")
                    time.sleep(10)
        
        thread = threading.Thread(target=aggregation_worker, daemon=True)
        thread.start()
    
    def _aggregate_costs(self):
        """Aggregate costs from completed executions"""
        with self.execution_lock:
            completed_executions = [
                profile for profile in self.execution_history
                if profile.end_time is not None
            ]
            
            for profile in completed_executions:
                # Update aggregation dictionaries
                self.cost_by_agent[profile.agent_id] += profile.cost_breakdown.total_cost
                
                if profile.user_id:
                    self.cost_by_user[profile.user_id] += profile.cost_breakdown.total_cost
                
                if profile.task_id:
                    self.cost_by_task[profile.task_id] += profile.cost_breakdown.total_cost
                
                self.cost_by_model[profile.token_usage.model] += profile.cost_breakdown.llm_cost
    
    def _cleanup_old_executions(self):
        """Remove old executions from history to prevent memory leaks"""
        with self.execution_lock:
            if len(self.execution_history) > self.history_max_size:
                # Keep only the most recent executions
                self.execution_history = self.execution_history[-self.history_max_size:]
    
    def start_execution(
        self,
        agent_id: str,
        execution_id: Optional[str] = None,
        task_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        parent_execution_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start tracking a new agent execution.
        
        Returns:
            execution_id: Unique identifier for this execution
        """
        execution_id = execution_id or str(uuid.uuid4())
        
        with self.execution_lock:
            if execution_id in self.active_executions:
                raise ValueError(f"Execution {execution_id} is already being tracked")
            
            profile = AgentCostProfile(
                agent_id=agent_id,
                execution_id=execution_id,
                task_id=task_id,
                user_id=user_id,
                session_id=session_id,
                parent_execution_id=parent_execution_id,
                tags=tags or {},
                metadata=metadata or {}
            )
            
            self.active_executions[execution_id] = profile
            
            # Start OpenTelemetry span
            span = self.tracer.start_span(
                name=f"agent_execution:{agent_id}",
                attributes={
                    "agent.id": agent_id,
                    "execution.id": execution_id,
                    "task.id": task_id or "",
                    "user.id": user_id or "",
                    "session.id": session_id or "",
                    "parent.execution.id": parent_execution_id or ""
                }
            )
            
            # Store span context in profile
            profile.metadata["span_id"] = format(span.get_span_context().span_id, "016x")
            profile.metadata["trace_id"] = format(span.get_span_context().trace_id, "032x")
            profile.span_ids.append(profile.metadata["span_id"])
            
            # Record metrics
            self.execution_counter.add(1, {"agent_id": agent_id})
            
            # Add to state manager if available
            if self.state_manager:
                self.state_manager.set(
                    f"cost_tracking:execution:{execution_id}",
                    {
                        "status": "active",
                        "start_time": profile.start_time.isoformat(),
                        "agent_id": agent_id
                    },
                    ttl=3600  # 1 hour TTL
                )
            
            return execution_id
    
    def record_token_usage(
        self,
        execution_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        model: str = "default",
        cached_tokens: int = 0,
        audio_tokens: int = 0,
        image_tokens: int = 0,
        span: Optional[Span] = None
    ):
        """Record LLM token usage for an execution"""
        with self.execution_lock:
            if execution_id not in self.active_executions:
                raise ValueError(f"Execution {execution_id} not found in active executions")
            
            profile = self.active_executions[execution_id]
            
            # Update token usage
            token_usage = TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                model=model,
                cached_tokens=cached_tokens,
                audio_tokens=audio_tokens,
                image_tokens=image_tokens
            )
            
            profile.token_usage += token_usage
            
            # Calculate LLM cost
            cost_per_1k_tokens = self.token_costs.get(model, self.token_costs["default"])
            llm_cost = (token_usage.total_tokens / 1000) * cost_per_1k_tokens
            profile.cost_breakdown.llm_cost += llm_cost
            
            # Record in OpenTelemetry
            if span is None:
                # Try to get current span
                current_span = trace.get_current_span()
                if current_span and current_span.is_recording():
                    span = current_span
            
            if span and span.is_recording():
                span.set_attributes({
                    "llm.model": model,
                    "llm.prompt_tokens": prompt_tokens,
                    "llm.completion_tokens": completion_tokens,
                    "llm.total_tokens": token_usage.total_tokens,
                    "llm.cached_tokens": cached_tokens,
                    "llm.cost": llm_cost
                })
                
                # Add event for token usage
                span.add_event("llm_token_usage", {
                    "model": model,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": token_usage.total_tokens,
                    "cost": llm_cost
                })
            
            # Record Prometheus metrics
            self.prometheus_tokens_total.labels(
                agent_id=profile.agent_id,
                model=model,
                token_type="prompt"
            ).inc(prompt_tokens)
            
            self.prometheus_tokens_total.labels(
                agent_id=profile.agent_id,
                model=model,
                token_type="completion"
            ).inc(completion_tokens)
            
            # Update OpenTelemetry metrics
            self.token_counter.add(
                token_usage.total_tokens,
                {
                    "agent_id": profile.agent_id,
                    "model": model,
                    "token_type": "total"
                }
            )
            
            # Update metrics collector
            self.metrics_collector.increment("cost_tracker.llm_calls")
    
    def record_compute_time(
        self,
        execution_id: str,
        compute_time_seconds: float,
        span: Optional[Span] = None
    ):
        """Record compute time for an execution"""
        with self.execution_lock:
            if execution_id not in self.active_executions:
                raise ValueError(f"Execution {execution_id} not found in active executions")
            
            profile = self.active_executions[execution_id]
            profile.compute_time_seconds += compute_time_seconds
            
            # Calculate compute cost
            compute_cost = compute_time_seconds * self.compute_cost_per_second
            profile.cost_breakdown.compute_cost += compute_cost
            
            # Record in OpenTelemetry
            if span is None:
                current_span = trace.get_current_span()
                if current_span and current_span.is_recording():
                    span = current_span
            
            if span and span.is_recording():
                span.set_attributes({
                    "compute.time_seconds": compute_time_seconds,
                    "compute.cost": compute_cost
                })
    
    def record_external_cost(
        self,
        execution_id: str,
        cost: float,
        category: CostCategory,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        span: Optional[Span] = None
    ):
        """Record external API or other costs"""
        with self.execution_lock:
            if execution_id not in self.active_executions:
                raise ValueError(f"Execution {execution_id} not found in active executions")
            
            profile = self.active_executions[execution_id]
            
            # Add cost to appropriate category
            if category == CostCategory.EXTERNAL_API:
                profile.cost_breakdown.external_api_cost += cost
            elif category == CostCategory.STORAGE:
                profile.cost_breakdown.storage_cost += cost
            elif category == CostCategory.NETWORK:
                profile.cost_breakdown.network_cost += cost
            elif category == CostCategory.MEMORY:
                profile.cost_breakdown.memory_cost += cost
            
            # Record in OpenTelemetry
            if span is None:
                current_span = trace.get_current_span()
                if current_span and current_span.is_recording():
                    span = current_span
            
            if span and span.is_recording():
                span.add_event("external_cost", {
                    "category": category.value,
                    "cost": cost,
                    "description": description,
                    "metadata": json.dumps(metadata or {})
                })
    
    def end_execution(
        self,
        execution_id: str,
        status: str = "completed",
        error: Optional[str] = None,
        final_metadata: Optional[Dict[str, Any]] = None
    ) -> AgentCostProfile:
        """
        End tracking for an execution and calculate final costs.
        
        Returns:
            AgentCostProfile with complete cost breakdown
        """
        with self.execution_lock:
            if execution_id not in self.active_executions:
                raise ValueError(f"Execution {execution_id} not found in active executions")
            
            profile = self.active_executions.pop(execution_id)
            profile.finalize()
            
            # Calculate total cost
            profile.cost_breakdown.calculate_total()
            
            # Update metadata
            if final_metadata:
                profile.metadata.update(final_metadata)
            
            profile.metadata["status"] = status
            profile.metadata["end_time"] = profile.end_time.isoformat()
            
            if error:
                profile.metadata["error"] = error
                self.metrics_collector.increment("cost_tracker.execution_errors")
            
            # End OpenTelemetry span
            span_context = trace.get_current_span()
            if span_context and span_context.is_recording():
                span_context.set_status(Status(StatusCode.OK if status == "completed" else StatusCode.ERROR))
                span_context.set_attributes({
                    "execution.status": status,
                    "execution.total_cost": profile.cost_breakdown.total_cost,
                    "execution.compute_time": profile.compute_time_seconds,
                    "execution.tokens_per_second": profile.tokens_per_second,
                    "execution.cost_per_second": profile.cost_per_second
                })
                span_context.end()
            
            # Record final metrics
            self.cost_histogram.record(
                profile.cost_breakdown.total_cost,
                {
                    "agent_id": profile.agent_id,
                    "status": status,
                    "model": profile.token_usage.model
                }
            )
            
            self.compute_time_histogram.record(
                profile.compute_time_seconds,
                {
                    "agent_id": profile.agent_id,
                    "status": status
                }
            )
            
            self.prometheus_execution_duration.labels(
                agent_id=profile.agent_id,
                status=status
            ).observe(profile.compute_time_seconds)
            
            self.prometheus_cost_total.labels(
                agent_id=profile.agent_id,
                model=profile.token_usage.model,
                user_id=profile.user_id or "anonymous",
                task_type=profile.task_id or "unknown"
            ).inc(profile.cost_breakdown.total_cost)
            
            # Add to history
            self.execution_history.append(profile)
            
            # Update state manager
            if self.state_manager:
                self.state_manager.set(
                    f"cost_tracking:execution:{execution_id}",
                    {
                        "status": status,
                        "total_cost": profile.cost_breakdown.total_cost,
                        "end_time": profile.end_time.isoformat(),
                        "compute_time": profile.compute_time_seconds,
                        "token_usage": profile.token_usage.total_tokens
                    },
                    ttl=86400  # 24 hours TTL
                )
            
            return profile
    
    def get_execution_cost(self, execution_id: str) -> Optional[AgentCostProfile]:
        """Get cost profile for a specific execution"""
        with self.execution_lock:
            # Check active executions
            if execution_id in self.active_executions:
                return self.active_executions[execution_id]
            
            # Check history
            for profile in self.execution_history:
                if profile.execution_id == execution_id:
                    return profile
            
            return None
    
    def get_agent_costs(
        self,
        agent_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[AgentCostProfile]:
        """Get cost profiles for a specific agent within a time range"""
        with self.execution_lock:
            results = []
            
            for profile in self.execution_history:
                if profile.agent_id != agent_id:
                    continue
                
                if start_time and profile.start_time < start_time:
                    continue
                
                if end_time and profile.start_time > end_time:
                    continue
                
                results.append(profile)
            
            return results
    
    def get_cost_summary(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get cost summary across all executions"""
        with self.execution_lock:
            total_cost = 0.0
            total_tokens = 0
            total_compute_time = 0.0
            execution_count = 0
            
            agent_costs = defaultdict(float)
            model_costs = defaultdict(float)
            user_costs = defaultdict(float)
            
            for profile in self.execution_history:
                if start_time and profile.start_time < start_time:
                    continue
                
                if end_time and profile.start_time > end_time:
                    continue
                
                total_cost += profile.cost_breakdown.total_cost
                total_tokens += profile.token_usage.total_tokens
                total_compute_time += profile.compute_time_seconds
                execution_count += 1
                
                agent_costs[profile.agent_id] += profile.cost_breakdown.total_cost
                model_costs[profile.token_usage.model] += profile.cost_breakdown.llm_cost
                
                if profile.user_id:
                    user_costs[profile.user_id] += profile.cost_breakdown.total_cost
            
            return {
                "total_cost": total_cost,
                "total_tokens": total_tokens,
                "total_compute_time": total_compute_time,
                "execution_count": execution_count,
                "average_cost_per_execution": total_cost / max(1, execution_count),
                "average_tokens_per_execution": total_tokens / max(1, execution_count),
                "cost_by_agent": dict(agent_costs),
                "cost_by_model": dict(model_costs),
                "cost_by_user": dict(user_costs),
                "start_time": start_time.isoformat() if start_time else None,
                "end_time": end_time.isoformat() if end_time else None
            }
    
    @contextmanager
    def track_execution(
        self,
        agent_id: str,
        execution_id: Optional[str] = None,
        **kwargs
    ):
        """
        Context manager for tracking an agent execution.
        
        Usage:
            with cost_tracker.track_execution("my_agent") as execution_id:
                # Agent execution code
                cost_tracker.record_token_usage(execution_id, 100, 50, "gpt-4")
        """
        execution_id = self.start_execution(agent_id, execution_id, **kwargs)
        
        try:
            yield execution_id
            self.end_execution(execution_id, status="completed")
        except Exception as e:
            self.end_execution(execution_id, status="failed", error=str(e))
            raise
    
    def export_metrics(self) -> bytes:
        """Export Prometheus metrics"""
        return generate_latest()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the cost tracker"""
        with self.execution_lock:
            return {
                "status": "healthy",
                "active_executions": len(self.active_executions),
                "historical_executions": len(self.execution_history),
                "tracked_nexus": len(self.cost_by_agent),
                "tracked_users": len(self.cost_by_user),
                "memory_usage_mb": self._estimate_memory_usage() / (1024 * 1024),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage in bytes"""
        # Rough estimation
        profile_size = 1000  # bytes per profile (approximate)
        return len(self.active_executions) * profile_size + len(self.execution_history) * profile_size

# Integration with existing distributed executor
class CostAwareExecutor:
    """
    Wrapper around the distributed executor that adds cost tracking.
    Integrates with core/distributed/executor.py
    """
    
    def __init__(
        self,
        executor,  # AgentExecutor from core/distributed/executor.py
        cost_tracker: CostTracker
    ):
        self.executor = executor
        self.cost_tracker = cost_tracker
    
    async def execute_with_cost_tracking(
        self,
        agent_id: str,
        task: Dict[str, Any],
        context: Optional[AgentExecutionContext] = None,
        **kwargs
    ) -> Tuple[Any, AgentCostProfile]:
        """
        Execute an agent with cost tracking.
        
        Returns:
            Tuple of (result, cost_profile)
        """
        execution_id = str(uuid.uuid4())
        
        # Start cost tracking
        self.cost_tracker.start_execution(
            agent_id=agent_id,
            execution_id=execution_id,
            task_id=task.get("task_id"),
            user_id=context.user_id if context else None,
            session_id=context.session_id if context else None,
            tags={"task_type": task.get("type", "unknown")}
        )
        
        start_time = time.time()
        
        try:
            # Execute the agent
            result = await self.executor.execute(agent_id, task, context, **kwargs)
            
            # Calculate compute time
            compute_time = time.time() - start_time
            self.cost_tracker.record_compute_time(execution_id, compute_time)
            
            # If result includes token usage, record it
            if isinstance(result, dict) and "token_usage" in result:
                token_usage = result["token_usage"]
                self.cost_tracker.record_token_usage(
                    execution_id=execution_id,
                    prompt_tokens=token_usage.get("prompt_tokens", 0),
                    completion_tokens=token_usage.get("completion_tokens", 0),
                    model=token_usage.get("model", "default")
                )
            
            # End cost tracking
            cost_profile = self.cost_tracker.end_execution(
                execution_id=execution_id,
                status="completed"
            )
            
            return result, cost_profile
            
        except Exception as e:
            # Record failure
            compute_time = time.time() - start_time
            self.cost_tracker.record_compute_time(execution_id, compute_time)
            
            cost_profile = self.cost_tracker.end_execution(
                execution_id=execution_id,
                status="failed",
                error=str(e)
            )
            
            raise

# Factory function for easy initialization
def create_cost_tracker(
    config: Optional[Dict[str, Any]] = None,
    metrics_collector: Optional[MetricsCollector] = None,
    state_manager: Optional[StateManager] = None
) -> CostTracker:
    """Factory function to create a CostTracker instance"""
    return CostTracker(
        config=config,
        metrics_collector=metrics_collector,
        state_manager=state_manager
    )

# Decorator for automatic cost tracking
def track_cost(
    agent_id: str,
    cost_tracker: Optional[CostTracker] = None,
    **tracker_kwargs
):
    """
    Decorator for automatic cost tracking of functions.
    
    Usage:
        @track_cost("my_agent")
        async def my_agent_function(param1, param2):
            # Function implementation
            return result
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            tracker = cost_tracker or create_cost_tracker(**tracker_kwargs)
            
            with tracker.track_execution(agent_id) as execution_id:
                # Store execution_id in kwargs for the function to use
                kwargs["_cost_tracking_execution_id"] = execution_id
                
                # Execute the function
                result = await func(*args, **kwargs)
                
                # If result includes cost information, record it
                if isinstance(result, dict):
                    if "token_usage" in result:
                        token_usage = result["token_usage"]
                        tracker.record_token_usage(
                            execution_id=execution_id,
                            **token_usage
                        )
                    
                    if "compute_time" in result:
                        tracker.record_compute_time(
                            execution_id=execution_id,
                            compute_time_seconds=result["compute_time"]
                        )
                
                return result
        
        return wrapper
    return decorator

# Example usage and integration
if __name__ == "__main__":
    # Example configuration
    config = {
        "token_costs": {
            "gpt-4": 0.03,
            "gpt-3.5-turbo": 0.002,
            "claude-3-opus": 0.015
        },
        "compute_cost_per_second": 0.0001,
        "history_max_size": 5000
    }
    
    # Create cost tracker
    tracker = create_cost_tracker(config)
    
    # Example: Track an execution
    with tracker.track_execution("example_agent", task_id="task_123") as execution_id:
        # Simulate LLM call
        tracker.record_token_usage(
            execution_id=execution_id,
            prompt_tokens=150,
            completion_tokens=80,
            model="gpt-4"
        )
        
        # Simulate compute time
        time.sleep(0.5)
        tracker.record_compute_time(execution_id, 0.5)
        
        # Simulate external API cost
        tracker.record_external_cost(
            execution_id=execution_id,
            cost=0.02,
            category=CostCategory.EXTERNAL_API,
            description="External API call"
        )
    
    # Get cost summary
    summary = tracker.get_cost_summary()
    print(f"Total cost: ${summary['total_cost']:.4f}")
    print(f"Total tokens: {summary['total_tokens']}")
    print(f"Execution count: {summary['execution_count']}")
    
    # Export metrics
    metrics = tracker.export_metrics()
    print(f"Metrics exported: {len(metrics)} bytes")