"""
Real-time Agent Observability Platform
Comprehensive monitoring with distributed tracing, performance metrics, and cost tracking
"""

import os
import time
import asyncio
import threading
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import contextmanager, asynccontextmanager
from functools import wraps
import uuid
import inspect

# OpenTelemetry imports
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider, SpanProcessor
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.trace import Status, StatusCode
from opentelemetry.metrics import Observation, CallbackOptions

# Prometheus imports
from prometheus_client import start_http_server, Counter, Histogram, Gauge, Summary

# Cost tracking constants
DEFAULT_LLM_COST_PER_1K_TOKENS = {
    "gpt-4": 0.03,
    "gpt-4-32k": 0.06,
    "gpt-3.5-turbo": 0.002,
    "gpt-3.5-turbo-16k": 0.003,
    "claude-2": 0.011,
    "claude-instant": 0.001,
    "default": 0.005
}

DEFAULT_COMPUTE_COST_PER_SECOND = 0.0001  # $0.0001 per second of compute

class SpanType(Enum):
    AGENT_EXECUTION = "agent_execution"
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    PLUGIN_EXECUTION = "plugin_execution"
    DISTRIBUTED_TASK = "distributed_task"
    STATE_SYNC = "state_sync"

@dataclass
class CostBreakdown:
    llm_tokens: int = 0
    llm_cost: float = 0.0
    compute_seconds: float = 0.0
    compute_cost: float = 0.0
    total_cost: float = 0.0
    currency: str = "USD"
    
    def calculate_total(self):
        self.total_cost = self.llm_cost + self.compute_cost
        return self.total_cost

@dataclass
class AgentSpan:
    span_id: str
    trace_id: str
    parent_span_id: Optional[str] = None
    span_type: SpanType = SpanType.AGENT_EXECUTION
    agent_id: str = ""
    agent_name: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration: Optional[float] = None
    status: str = "running"
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    cost_breakdown: CostBreakdown = field(default_factory=CostBreakdown)
    error: Optional[str] = None
    
    def finish(self, status: str = "success", error: Optional[str] = None):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.status = status
        self.error = error
        
    def add_event(self, name: str, attributes: Optional[Dict] = None):
        event = {
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {}
        }
        self.events.append(event)
        
    def to_dict(self) -> Dict:
        result = asdict(self.result)
        result["span_type"] = self.span_type.value
        return result

class MetricsCollector:
    """Collects and exposes metrics for Prometheus"""
    
    def __init__(self, namespace: str = "sovereign"):
        self.namespace = namespace
        
        # Agent execution metrics
        self.agent_executions_total = Counter(
            f"{namespace}_agent_executions_total",
            "Total number of agent executions",
            ["agent_name", "status"]
        )
        
        self.agent_execution_duration = Histogram(
            f"{namespace}_agent_execution_duration_seconds",
            "Agent execution duration in seconds",
            ["agent_name"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]
        )
        
        # LLM metrics
        self.llm_calls_total = Counter(
            f"{namespace}_llm_calls_total",
            "Total number of LLM API calls",
            ["model", "status"]
        )
        
        self.llm_tokens_total = Counter(
            f"{namespace}_llm_tokens_total",
            "Total LLM tokens used",
            ["model", "token_type"]  # token_type: prompt, completion
        )
        
        self.llm_cost_total = Counter(
            f"{namespace}_llm_cost_dollars",
            "Total LLM cost in dollars",
            ["model"]
        )
        
        # Cost metrics
        self.agent_cost_total = Counter(
            f"{namespace}_agent_cost_dollars",
            "Total agent cost in dollars",
            ["agent_name", "cost_type"]  # cost_type: llm, compute, total
        )
        
        # Distributed system metrics
        self.distributed_tasks_total = Counter(
            f"{namespace}_distributed_tasks_total",
            "Total distributed tasks",
            ["task_type", "status"]
        )
        
        self.consensus_rounds_total = Counter(
            f"{namespace}_consensus_rounds_total",
            "Total consensus rounds",
            ["consensus_type"]
        )
        
        # Resource metrics
        self.active_spans = Gauge(
            f"{namespace}_active_spans",
            "Number of active tracing spans"
        )
        
        self.queue_size = Gauge(
            f"{namespace}_queue_size",
            "Size of task queues",
            ["queue_name"]
        )
        
    def record_agent_execution(self, agent_name: str, duration: float, status: str, cost: float):
        self.agent_executions_total.labels(agent_name=agent_name, status=status).inc()
        self.agent_execution_duration.labels(agent_name=agent_name).observe(duration)
        self.agent_cost_total.labels(agent_name=agent_name, cost_type="total").inc(cost)
        
    def record_llm_call(self, model: str, prompt_tokens: int, completion_tokens: int, cost: float, status: str):
        self.llm_calls_total.labels(model=model, status=status).inc()
        self.llm_tokens_total.labels(model=model, token_type="prompt").inc(prompt_tokens)
        self.llm_tokens_total.labels(model=model, token_type="completion").inc(completion_tokens)
        self.llm_cost_total.labels(model=model).inc(cost)

class DistributedTracing:
    """Distributed tracing with OpenTelemetry"""
    
    def __init__(self, service_name: str = "sovereign-agent", otlp_endpoint: Optional[str] = None):
        self.service_name = service_name
        self.otlp_endpoint = otlp_endpoint or os.getenv("OTLP_ENDPOINT", "localhost:4317")
        
        # Initialize tracer
        resource = Resource.create({SERVICE_NAME: service_name})
        tracer_provider = TracerProvider(resource=resource)
        
        # Add span processors
        if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
            otlp_exporter = OTLPSpanExporter(endpoint=self.otlp_endpoint)
            tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        else:
            # Fallback to console exporter for development
            tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        
        trace.set_tracer_provider(tracer_provider)
        self.tracer = trace.get_tracer(__name__)
        
        # Map to store custom spans
        self.active_spans: Dict[str, AgentSpan] = {}
        self._lock = threading.RLock()
        
    @contextmanager
    def start_span(self, 
                   name: str, 
                   span_type: SpanType = SpanType.AGENT_EXECUTION,
                   attributes: Optional[Dict] = None,
                   parent_span_id: Optional[str] = None):
        """Start a new tracing span"""
        
        span_id = str(uuid.uuid4())
        trace_id = str(uuid.uuid4())
        
        # Create custom span for our tracking
        custom_span = AgentSpan(
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            span_type=span_type,
            attributes=attributes or {},
            start_time=time.time()
        )
        
        with self._lock:
            self.active_spans[span_id] = custom_span
        
        # Start OpenTelemetry span
        otel_span = self.tracer.start_span(name)
        otel_span.set_attribute("span_id", span_id)
        otel_span.set_attribute("trace_id", trace_id)
        otel_span.set_attribute("span_type", span_type.value)
        
        if attributes:
            for key, value in attributes.items():
                otel_span.set_attribute(key, str(value))
        
        try:
            yield custom_span, otel_span
            custom_span.finish("success")
            otel_span.set_status(Status(StatusCode.OK))
        except Exception as e:
            custom_span.finish("error", str(e))
            otel_span.set_status(Status(StatusCode.ERROR, str(e)))
            otel_span.record_exception(e)
            raise
        finally:
            otel_span.end()
            with self._lock:
                self.active_spans.pop(span_id, None)
    
    @asynccontextmanager
    async def start_async_span(self, name: str, **kwargs):
        """Async version of start_span"""
        span_id = str(uuid.uuid4())
        trace_id = str(uuid.uuid4())
        
        custom_span = AgentSpan(
            span_id=span_id,
            trace_id=trace_id,
            **kwargs,
            start_time=time.time()
        )
        
        with self._lock:
            self.active_spans[span_id] = custom_span
        
        otel_span = self.tracer.start_span(name)
        
        try:
            yield custom_span, otel_span
            custom_span.finish("success")
            otel_span.set_status(Status(StatusCode.OK))
        except Exception as e:
            custom_span.finish("error", str(e))
            otel_span.set_status(Status(StatusCode.ERROR, str(e)))
            otel_span.record_exception(e)
            raise
        finally:
            otel_span.end()
            with self._lock:
                self.active_spans.pop(span_id, None)

class CostTracker:
    """Tracks costs for LLM usage and compute time"""
    
    def __init__(self, cost_config: Optional[Dict] = None):
        self.cost_config = cost_config or {}
        self.llm_costs = {**DEFAULT_LLM_COST_PER_1K_TOKENS, **self.cost_config.get("llm_costs", {})}
        self.compute_cost_per_second = self.cost_config.get("compute_cost_per_second", DEFAULT_COMPUTE_COST_PER_SECOND)
        
        # Track cumulative costs
        self.total_llm_cost = 0.0
        self.total_compute_cost = 0.0
        self._lock = threading.RLock()
        
    def calculate_llm_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost for LLM API call"""
        cost_per_1k = self.llm_costs.get(model, self.llm_costs["default"])
        total_tokens = prompt_tokens + completion_tokens
        cost = (total_tokens / 1000) * cost_per_1k
        
        with self._lock:
            self.total_llm_cost += cost
            
        return cost
    
    def calculate_compute_cost(self, duration_seconds: float) -> float:
        """Calculate cost for compute time"""
        cost = duration_seconds * self.compute_cost_per_second
        
        with self._lock:
            self.total_compute_cost += cost
            
        return cost
    
    def create_cost_breakdown(self, 
                             model: str, 
                             prompt_tokens: int, 
                             completion_tokens: int,
                             duration_seconds: float) -> CostBreakdown:
        """Create a complete cost breakdown"""
        llm_cost = self.calculate_llm_cost(model, prompt_tokens, completion_tokens)
        compute_cost = self.calculate_compute_cost(duration_seconds)
        
        breakdown = CostBreakdown(
            llm_tokens=prompt_tokens + completion_tokens,
            llm_cost=llm_cost,
            compute_seconds=duration_seconds,
            compute_cost=compute_cost
        )
        breakdown.calculate_total()
        
        return breakdown

class AgentObservability:
    """Main observability platform for nexus"""
    
    def __init__(self, 
                 service_name: str = "sovereign-agent",
                 enable_tracing: bool = True,
                 enable_metrics: bool = True,
                 prometheus_port: int = 9090,
                 cost_config: Optional[Dict] = None):
        
        self.service_name = service_name
        self.enable_tracing = enable_tracing
        self.enable_metrics = enable_metrics
        
        # Initialize components
        self.tracing = DistributedTracing(service_name) if enable_tracing else None
        self.metrics = MetricsCollector() if enable_metrics else None
        self.cost_tracker = CostTracker(cost_config)
        
        # Start Prometheus metrics server
        if enable_metrics:
            try:
                start_http_server(prometheus_port)
                print(f"Prometheus metrics server started on port {prometheus_port}")
            except Exception as e:
                print(f"Failed to start Prometheus server: {e}")
        
        # Integration with existing modules
        self._integrate_with_existing_modules()
        
    def _integrate_with_existing_modules(self):
        """Integrate with existing distributed system modules"""
        try:
            # Import existing modules
            from core.distributed.executor import DistributedExecutor
            from core.distributed.consensus import ConsensusManager
            from core.distributed.state_manager import StateManager
            
            # Monkey patch to add observability
            self._patch_executor(DistributedExecutor)
            self._patch_consensus(ConsensusManager)
            self._patch_state_manager(StateManager)
            
        except ImportError as e:
            print(f"Could not integrate with existing modules: {e}")
    
    def _patch_executor(self, executor_class):
        """Add observability to distributed executor"""
        original_execute = executor_class.execute
        
        @wraps(original_execute)
        def traced_execute(self, task, *args, **kwargs):
            with self.tracing.start_span(
                f"execute_{task.get('type', 'unknown')}",
                span_type=SpanType.DISTRIBUTED_TASK,
                attributes={"task_type": task.get("type"), "task_id": task.get("id")}
            ) as (custom_span, otel_span):
                
                start_time = time.time()
                try:
                    result = original_execute(self, task, *args, **kwargs)
                    duration = time.time() - start_time
                    
                    # Record metrics
                    if self.metrics:
                        self.metrics.distributed_tasks_total.labels(
                            task_type=task.get("type", "unknown"),
                            status="success"
                        ).inc()
                    
                    custom_span.add_event("task_completed", {"duration": duration})
                    return result
                    
                except Exception as e:
                    duration = time.time() - start_time
                    if self.metrics:
                        self.metrics.distributed_tasks_total.labels(
                            task_type=task.get("type", "unknown"),
                            status="error"
                        ).inc()
                    raise
        
        executor_class.execute = traced_execute
    
    def _patch_consensus(self, consensus_class):
        """Add observability to consensus manager"""
        original_run_round = consensus_class.run_consensus_round
        
        @wraps(original_run_round)
        def traced_run_round(self, *args, **kwargs):
            with self.tracing.start_span(
                "consensus_round",
                span_type=SpanType.STATE_SYNC,
                attributes={"consensus_type": self.consensus_type}
            ) as (custom_span, otel_span):
                
                result = original_run_round(self, *args, **kwargs)
                
                if self.metrics:
                    self.metrics.consensus_rounds_total.labels(
                        consensus_type=self.consensus_type
                    ).inc()
                
                return result
        
        consensus_class.run_consensus_round = traced_run_round
    
    def _patch_state_manager(self, state_manager_class):
        """Add observability to state manager"""
        original_sync = state_manager_class.sync_state
        
        @wraps(original_sync)
        def traced_sync(self, *args, **kwargs):
            with self.tracing.start_span(
                "state_sync",
                span_type=SpanType.STATE_SYNC,
                attributes={"state_key": getattr(self, 'state_key', 'unknown')}
            ) as (custom_span, otel_span):
                
                return original_sync(self, *args, **kwargs)
        
        state_manager_class.sync_state = traced_sync
    
    def trace_agent_execution(self, agent_name: str):
        """Decorator to trace agent execution"""
        def decorator(func: Callable):
            if inspect.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    return await self._trace_async_agent_execution(func, agent_name, *args, **kwargs)
                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    return self._trace_sync_agent_execution(func, agent_name, *args, **kwargs)
                return sync_wrapper
        return decorator
    
    def _trace_sync_agent_execution(self, func: Callable, agent_name: str, *args, **kwargs):
        """Trace synchronous agent execution"""
        start_time = time.time()
        
        with self.tracing.start_span(
            f"agent_{agent_name}",
            span_type=SpanType.AGENT_EXECUTION,
            attributes={"agent_name": agent_name}
        ) as (custom_span, otel_span):
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Calculate cost (assuming we can extract token usage from result)
                token_usage = self._extract_token_usage(result)
                cost_breakdown = self.cost_tracker.create_cost_breakdown(
                    model=token_usage.get("model", "default"),
                    prompt_tokens=token_usage.get("prompt_tokens", 0),
                    completion_tokens=token_usage.get("completion_tokens", 0),
                    duration_seconds=duration
                )
                
                custom_span.cost_breakdown = cost_breakdown
                custom_span.attributes["token_usage"] = token_usage
                
                # Record metrics
                if self.metrics:
                    self.metrics.record_agent_execution(
                        agent_name=agent_name,
                        duration=duration,
                        status="success",
                        cost=cost_breakdown.total_cost
                    )
                    
                    if token_usage.get("model"):
                        self.metrics.record_llm_call(
                            model=token_usage["model"],
                            prompt_tokens=token_usage.get("prompt_tokens", 0),
                            completion_tokens=token_usage.get("completion_tokens", 0),
                            cost=cost_breakdown.llm_cost,
                            status="success"
                        )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                if self.metrics:
                    self.metrics.record_agent_execution(
                        agent_name=agent_name,
                        duration=duration,
                        status="error",
                        cost=0
                    )
                
                raise
    
    async def _trace_async_agent_execution(self, func: Callable, agent_name: str, *args, **kwargs):
        """Trace asynchronous agent execution"""
        start_time = time.time()
        
        async with self.tracing.start_async_span(
            f"agent_{agent_name}",
            span_type=SpanType.AGENT_EXECUTION,
            attributes={"agent_name": agent_name}
        ) as (custom_span, otel_span):
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                token_usage = self._extract_token_usage(result)
                cost_breakdown = self.cost_tracker.create_cost_breakdown(
                    model=token_usage.get("model", "default"),
                    prompt_tokens=token_usage.get("prompt_tokens", 0),
                    completion_tokens=token_usage.get("completion_tokens", 0),
                    duration_seconds=duration
                )
                
                custom_span.cost_breakdown = cost_breakdown
                
                if self.metrics:
                    self.metrics.record_agent_execution(
                        agent_name=agent_name,
                        duration=duration,
                        status="success",
                        cost=cost_breakdown.total_cost
                    )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                if self.metrics:
                    self.metrics.record_agent_execution(
                        agent_name=agent_name,
                        duration=duration,
                        status="error",
                        cost=0
                    )
                
                raise
    
    def _extract_token_usage(self, result: Any) -> Dict:
        """Extract token usage from agent result"""
        # This is a placeholder - implement based on your actual result structure
        if hasattr(result, 'usage'):
            return {
                "model": getattr(result, 'model', 'default'),
                "prompt_tokens": getattr(result.usage, 'prompt_tokens', 0),
                "completion_tokens": getattr(result.usage, 'completion_tokens', 0),
                "total_tokens": getattr(result.usage, 'total_tokens', 0)
            }
        elif isinstance(result, dict) and 'usage' in result:
            usage = result['usage']
            return {
                "model": result.get('model', 'default'),
                "prompt_tokens": usage.get('prompt_tokens', 0),
                "completion_tokens": usage.get('completion_tokens', 0),
                "total_tokens": usage.get('total_tokens', 0)
            }
        return {"model": "default", "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    def get_agent_metrics(self, agent_name: str, time_window: timedelta = timedelta(hours=1)) -> Dict:
        """Get metrics for a specific agent"""
        # This would query your metrics backend
        # For now, return placeholder data
        return {
            "agent_name": agent_name,
            "time_window": str(time_window),
            "total_executions": 0,
            "success_rate": 0.0,
            "avg_duration": 0.0,
            "total_cost": 0.0,
            "llm_usage": {
                "total_tokens": 0,
                "total_cost": 0.0
            }
        }
    
    def get_system_health(self) -> Dict:
        """Get overall system health metrics"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "active_spans": len(self.tracing.active_spans) if self.tracing else 0,
            "total_llm_cost": self.cost_tracker.total_llm_cost,
            "total_compute_cost": self.cost_tracker.total_compute_cost,
            "services_healthy": True,
            "metrics_collector_active": self.metrics is not None,
            "tracing_active": self.tracing is not None
        }
    
    def export_traces(self, format: str = "json") -> str:
        """Export active traces"""
        if not self.tracing:
            return "[]"
        
        traces = []
        for span_id, span in self.tracing.active_spans.items():
            traces.append(span.to_dict())
        
        if format == "json":
            return json.dumps(traces, indent=2, default=str)
        return str(traces)

# Global observability instance
_observability_instance: Optional[AgentObservability] = None

def initialize_observability(
    service_name: str = "sovereign-agent",
    enable_tracing: bool = True,
    enable_metrics: bool = True,
    prometheus_port: int = 9090,
    cost_config: Optional[Dict] = None
) -> AgentObservability:
    """Initialize the global observability instance"""
    global _observability_instance
    
    if _observability_instance is None:
        _observability_instance = AgentObservability(
            service_name=service_name,
            enable_tracing=enable_tracing,
            enable_metrics=enable_metrics,
            prometheus_port=prometheus_port,
            cost_config=cost_config
        )
    
    return _observability_instance

def get_observability() -> Optional[AgentObservability]:
    """Get the global observability instance"""
    return _observability_instance

def trace_agent(agent_name: str):
    """Convenience decorator for tracing agent execution"""
    observability = get_observability()
    if observability:
        return observability.trace_agent_execution(agent_name)
    else:
        # Return a no-op decorator if observability is not initialized
        def decorator(func):
            return func
        return decorator

# Context managers for manual tracing
@contextmanager
def trace_span(name: str, span_type: SpanType = SpanType.AGENT_EXECUTION, **attributes):
    """Context manager for manual tracing"""
    observability = get_observability()
    if observability and observability.tracing:
        with observability.tracing.start_span(name, span_type, attributes) as (custom_span, otel_span):
            yield custom_span
    else:
        yield None

@asynccontextmanager
async def trace_async_span(name: str, span_type: SpanType = SpanType.AGENT_EXECUTION, **attributes):
    """Async context manager for manual tracing"""
    observability = get_observability()
    if observability and observability.tracing:
        async with observability.tracing.start_async_span(name, span_type, attributes) as (custom_span, otel_span):
            yield custom_span
    else:
        yield None

# Integration with existing plugins and tools
def integrate_with_prompt_optimization():
    """Integrate with prompt optimization tool"""
    try:
        from plugins.llm_application_dev.skills.prompt_engineering_patterns.scripts.optimize_prompt import optimize_prompt
        
        original_optimize = optimize_prompt
        
        @wraps(original_optimize)
        def traced_optimize(*args, **kwargs):
            with trace_span("prompt_optimization", SpanType.TOOL_CALL, tool="optimize_prompt"):
                return original_optimize(*args, **kwargs)
        
        # Replace the original function
        import sys
        module = sys.modules['plugins.llm_application_dev.skills.prompt_engineering_patterns.scripts.optimize_prompt']
        module.optimize_prompt = traced_optimize
        
    except ImportError:
        pass

def integrate_with_api_design():
    """Integrate with API design tools"""
    try:
        from plugins.backend_development.skills.api_design_principles.assets.rest_api_template import create_api_template
        
        original_create = create_api_template
        
        @wraps(original_create)
        def traced_create(*args, **kwargs):
            with trace_span("api_template_creation", SpanType.TOOL_CALL, tool="create_api_template"):
                return original_create(*args, **kwargs)
        
        import sys
        module = sys.modules['plugins.backend_development.skills.api_design_principles.assets.rest_api_template']
        module.create_api_template = traced_create
        
    except ImportError:
        pass

# Initialize on module import if environment variables are set
if os.getenv("SOVEREIGN_OBSERVABILITY_ENABLED", "true").lower() == "true":
    initialize_observability(
        service_name=os.getenv("SOVEREIGN_SERVICE_NAME", "sovereign-agent"),
        enable_tracing=os.getenv("SOVEREIGN_TRACING_ENABLED", "true").lower() == "true",
        enable_metrics=os.getenv("SOVEREIGN_METRICS_ENABLED", "true").lower() == "true",
        prometheus_port=int(os.getenv("SOVEREIGN_PROMETHEUS_PORT", "9090")),
        cost_config={
            "llm_costs": json.loads(os.getenv("SOVEREIGN_LLM_COSTS", "{}")),
            "compute_cost_per_second": float(os.getenv("SOVEREIGN_COMPUTE_COST_PER_SECOND", "0.0001"))
        }
    )
    
    # Integrate with existing tools
    integrate_with_prompt_optimization()
    integrate_with_api_design()

# Example usage
if __name__ == "__main__":
    # Initialize observability
    obs = initialize_observability(
        service_name="test-agent",
        enable_tracing=True,
        enable_metrics=True,
        prometheus_port=9091
    )
    
    # Example agent function
    @trace_agent("example_agent")
    def example_agent_function(prompt: str) -> str:
        # Simulate LLM call
        time.sleep(0.5)
        return f"Response to: {prompt}"
    
    # Run example
    result = example_agent_function("Hello, world!")
    print(f"Result: {result}")
    
    # Get system health
    health = obs.get_system_health()
    print(f"System health: {health}")
    
    # Export traces
    traces = obs.export_traces()
    print(f"Active traces: {traces}")