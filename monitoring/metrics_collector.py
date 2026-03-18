"""
Real-time Agent Observability Platform
Comprehensive monitoring with distributed tracing, performance metrics, and cost tracking
"""

import time
import json
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import threading
from contextlib import contextmanager
from functools import wraps

# OpenTelemetry imports for distributed tracing
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

# Prometheus metrics
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, 
    generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
)
from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily

# Existing modules integration
from core.distributed.executor import AgentExecutor, ExecutionContext
from core.distributed.state_manager import StateManager
from monitoring.tracing import DistributedTracer

# Cost calculation constants
COST_PER_TOKEN = {
    "gpt-4": 0.00003,
    "gpt-4-turbo": 0.00001,
    "gpt-3.5-turbo": 0.000002,
    "claude-3-opus": 0.000015,
    "claude-3-sonnet": 0.000003,
    "claude-3-haiku": 0.00000025,
    "default": 0.00001
}

COMPUTE_COST_PER_SECOND = 0.0001  # $0.0001 per second of compute


class MetricType(Enum):
    """Types of metrics to collect"""
    EXECUTION_TIME = "execution_time"
    TOKEN_USAGE = "token_usage"
    COST = "cost"
    SUCCESS_RATE = "success_rate"
    ERROR_RATE = "error_rate"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    RESOURCE_USAGE = "resource_usage"


@dataclass
class AgentMetrics:
    """Metrics for a single agent execution"""
    agent_id: str
    execution_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    tokens_used: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    model: str = "default"
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        return data
    
    def calculate_cost(self) -> float:
        """Calculate cost based on token usage and compute time"""
        token_cost = self.tokens_used * COST_PER_TOKEN.get(self.model, COST_PER_TOKEN["default"])
        compute_cost = self.duration_seconds * COMPUTE_COST_PER_SECOND
        return token_cost + compute_cost


@dataclass
class SystemMetrics:
    """System-wide metrics"""
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_tokens_used: int = 0
    total_cost: float = 0.0
    average_execution_time: float = 0.0
    active_nexus: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


class MetricsCollector:
    """
    Real-time Agent Observability Platform
    Collects and exports metrics for agent execution monitoring
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern for metrics collector"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize metrics collector"""
        if self._initialized:
            return
            
        self._initialized = True
        self.registry = CollectorRegistry()
        
        # Initialize OpenTelemetry tracing
        self._init_tracing()
        
        # Initialize Prometheus metrics
        self._init_prometheus_metrics()
        
        # Metrics storage
        self.agent_metrics: Dict[str, List[AgentMetrics]] = {}
        self.system_metrics = SystemMetrics()
        self.active_spans: Dict[str, trace.Span] = {}
        
        # Background tasks
        self._running = False
        self._background_tasks = []
        
        # Integration with existing modules
        self.tracer = DistributedTracer()
        self.state_manager = StateManager()
        
        # Start background collection
        self.start_background_collection()
    
    def _init_tracing(self):
        """Initialize OpenTelemetry distributed tracing"""
        # Create tracer provider
        resource = Resource.create({
            "service.name": "sovereign-agent-monitoring",
            "service.version": "1.0.0"
        })
        
        tracer_provider = TracerProvider(resource=resource)
        
        # Configure exporters based on environment
        try:
            # Try Jaeger first
            jaeger_exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=6831,
            )
            tracer_provider.add_span_processor(
                BatchSpanProcessor(jaeger_exporter)
            )
        except Exception:
            # Fall back to OTLP
            try:
                otlp_exporter = OTLPSpanExporter(
                    endpoint="localhost:4317",
                    insecure=True
                )
                tracer_provider.add_span_processor(
                    BatchSpanProcessor(otlp_exporter)
                )
            except Exception:
                # Console exporter as fallback
                console_exporter = ConsoleSpanExporter()
                tracer_provider.add_span_processor(
                    BatchSpanProcessor(console_exporter)
                )
        
        # Set the tracer provider
        trace.set_tracer_provider(tracer_provider)
        self.tracer = trace.get_tracer(__name__)
        
        # Propagator for distributed tracing
        self.propagator = TraceContextTextMapPropagator()
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        # Agent execution metrics
        self.agent_execution_counter = Counter(
            'agent_executions_total',
            'Total number of agent executions',
            ['agent_id', 'status'],
            registry=self.registry
        )
        
        self.agent_execution_duration = Histogram(
            'agent_execution_duration_seconds',
            'Agent execution duration in seconds',
            ['agent_id'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry
        )
        
        self.agent_token_usage = Counter(
            'agent_token_usage_total',
            'Total token usage by nexus',
            ['agent_id', 'model', 'token_type'],
            registry=self.registry
        )
        
        self.agent_cost = Counter(
            'agent_cost_dollars_total',
            'Total cost in dollars for agent executions',
            ['agent_id', 'model'],
            registry=self.registry
        )
        
        self.active_nexus_gauge = Gauge(
            'active_nexus',
            'Number of currently active nexus',
            registry=self.registry
        )
        
        self.agent_success_rate = Gauge(
            'agent_success_rate',
            'Success rate of agent executions',
            ['agent_id'],
            registry=self.registry
        )
        
        # System-wide metrics
        self.system_throughput = Gauge(
            'system_throughput_executions_per_second',
            'System throughput in executions per second',
            registry=self.registry
        )
        
        self.system_latency = Summary(
            'system_latency_seconds',
            'System latency distribution',
            registry=self.registry
        )
        
        # Resource usage metrics
        self.memory_usage = Gauge(
            'memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        self.cpu_usage = Gauge(
            'cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
    
    def start_background_collection(self):
        """Start background metrics collection"""
        if self._running:
            return
            
        self._running = True
        
        # Start system metrics collection
        loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=self._run_background_tasks,
            args=(loop,),
            daemon=True
        )
        thread.start()
    
    def _run_background_tasks(self, loop):
        """Run background tasks in separate thread"""
        asyncio.set_event_loop(loop)
        
        # Schedule periodic tasks
        tasks = [
            self._collect_system_metrics(),
            self._update_prometheus_metrics(),
            self._cleanup_old_metrics()
        ]
        
        loop.run_until_complete(asyncio.gather(*tasks))
    
    async def _collect_system_metrics(self):
        """Collect system-wide metrics periodically"""
        while self._running:
            try:
                # Update active nexus count
                self.active_nexus_gauge.set(self.system_metrics.active_nexus)
                
                # Calculate throughput
                if self.system_metrics.total_executions > 0:
                    uptime = (datetime.now() - self.system_metrics.timestamp).total_seconds()
                    if uptime > 0:
                        throughput = self.system_metrics.total_executions / uptime
                        self.system_throughput.set(throughput)
                
                # Calculate success rate
                if self.system_metrics.total_executions > 0:
                    success_rate = (
                        self.system_metrics.successful_executions / 
                        self.system_metrics.total_executions
                    )
                    # Update per-agent success rates
                    for agent_id, metrics_list in self.agent_metrics.items():
                        if metrics_list:
                            agent_success = sum(1 for m in metrics_list if m.success)
                            agent_rate = agent_success / len(metrics_list)
                            self.agent_success_rate.labels(agent_id=agent_id).set(agent_rate)
                
                # Update memory and CPU usage (simplified)
                import psutil
                process = psutil.Process()
                self.memory_usage.set(process.memory_info().rss)
                self.cpu_usage.set(process.cpu_percent())
                
            except Exception as e:
                print(f"Error collecting system metrics: {e}")
            
            await asyncio.sleep(5)  # Collect every 5 seconds
    
    async def _update_prometheus_metrics(self):
        """Update Prometheus metrics from collected data"""
        while self._running:
            try:
                # Metrics are updated in real-time, this is for periodic updates
                await asyncio.sleep(10)
            except Exception as e:
                print(f"Error updating Prometheus metrics: {e}")
    
    async def _cleanup_old_metrics(self):
        """Clean up old metrics to prevent memory leaks"""
        while self._running:
            try:
                cutoff = datetime.now() - timedelta(hours=24)
                
                for agent_id in list(self.agent_metrics.keys()):
                    self.agent_metrics[agent_id] = [
                        m for m in self.agent_metrics[agent_id]
                        if m.start_time > cutoff
                    ]
                    
                    if not self.agent_metrics[agent_id]:
                        del self.agent_metrics[agent_id]
                
            except Exception as e:
                print(f"Error cleaning up metrics: {e}")
            
            await asyncio.sleep(3600)  # Clean up every hour
    
    @contextmanager
    def trace_agent_execution(
        self,
        agent_id: str,
        execution_id: str,
        model: str = "default",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for tracing agent execution
        Integrates with OpenTelemetry for distributed tracing
        """
        # Start OpenTelemetry span
        with self.tracer.start_as_current_span(
            f"agent_execution_{agent_id}",
            attributes={
                "agent.id": agent_id,
                "execution.id": execution_id,
                "model": model,
                "service.name": "sovereign-agent"
            }
        ) as span:
            # Create metrics object
            metrics = AgentMetrics(
                agent_id=agent_id,
                execution_id=execution_id,
                start_time=datetime.now(),
                model=model,
                metadata=metadata or {},
                trace_id=format(span.get_span_context().trace_id, '032x'),
                span_id=format(span.get_span_context().span_id, '016x')
            )
            
            # Store span reference
            self.active_spans[execution_id] = span
            
            try:
                yield metrics
                
                # Mark as successful
                metrics.success = True
                
            except Exception as e:
                # Record error
                metrics.success = False
                metrics.error_message = str(e)
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise
                
            finally:
                # Finalize metrics
                metrics.end_time = datetime.now()
                metrics.duration_seconds = (
                    metrics.end_time - metrics.start_time
                ).total_seconds()
                metrics.cost = metrics.calculate_cost()
                
                # Record metrics
                self._record_metrics(metrics)
                
                # Clean up
                if execution_id in self.active_spans:
                    del self.active_spans[execution_id]
    
    def _record_metrics(self, metrics: AgentMetrics):
        """Record metrics to all systems"""
        # Store in memory
        if metrics.agent_id not in self.agent_metrics:
            self.agent_metrics[metrics.agent_id] = []
        self.agent_metrics[metrics.agent_id].append(metrics)
        
        # Update system metrics
        self.system_metrics.total_executions += 1
        if metrics.success:
            self.system_metrics.successful_executions += 1
        else:
            self.system_metrics.failed_executions += 1
        
        self.system_metrics.total_tokens_used += metrics.tokens_used
        self.system_metrics.total_cost += metrics.cost
        
        # Update average execution time
        total_time = (
            self.system_metrics.average_execution_time * 
            (self.system_metrics.total_executions - 1) + 
            metrics.duration_seconds
        )
        self.system_metrics.average_execution_time = (
            total_time / self.system_metrics.total_executions
        )
        
        # Update Prometheus metrics
        status = "success" if metrics.success else "failure"
        self.agent_execution_counter.labels(
            agent_id=metrics.agent_id,
            status=status
        ).inc()
        
        self.agent_execution_duration.labels(
            agent_id=metrics.agent_id
        ).observe(metrics.duration_seconds)
        
        self.agent_token_usage.labels(
            agent_id=metrics.agent_id,
            model=metrics.model,
            token_type="input"
        ).inc(metrics.input_tokens)
        
        self.agent_token_usage.labels(
            agent_id=metrics.agent_id,
            model=metrics.model,
            token_type="output"
        ).inc(metrics.output_tokens)
        
        self.agent_cost.labels(
            agent_id=metrics.agent_id,
            model=metrics.model
        ).inc(metrics.cost)
        
        # Update span with metrics
        if metrics.trace_id and metrics.span_id:
            span = self.active_spans.get(metrics.execution_id)
            if span:
                span.set_attributes({
                    "tokens.total": metrics.tokens_used,
                    "tokens.input": metrics.input_tokens,
                    "tokens.output": metrics.output_tokens,
                    "cost.dollars": metrics.cost,
                    "duration.seconds": metrics.duration_seconds
                })
    
    def update_token_usage(
        self,
        execution_id: str,
        input_tokens: int,
        output_tokens: int,
        model: Optional[str] = None
    ):
        """Update token usage for an ongoing execution"""
        for agent_metrics in self.agent_metrics.values():
            for metrics in agent_metrics:
                if metrics.execution_id == execution_id:
                    metrics.input_tokens = input_tokens
                    metrics.output_tokens = output_tokens
                    metrics.tokens_used = input_tokens + output_tokens
                    if model:
                        metrics.model = model
                    return
    
    def get_agent_metrics(
        self,
        agent_id: str,
        time_range: Optional[timedelta] = None
    ) -> List[AgentMetrics]:
        """Get metrics for a specific agent"""
        if agent_id not in self.agent_metrics:
            return []
        
        metrics_list = self.agent_metrics[agent_id]
        
        if time_range:
            cutoff = datetime.now() - time_range
            metrics_list = [
                m for m in metrics_list 
                if m.start_time > cutoff
            ]
        
        return metrics_list
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide metrics"""
        return {
            "total_executions": self.system_metrics.total_executions,
            "successful_executions": self.system_metrics.successful_executions,
            "failed_executions": self.system_metrics.failed_executions,
            "success_rate": (
                self.system_metrics.successful_executions / 
                max(self.system_metrics.total_executions, 1)
            ),
            "total_tokens_used": self.system_metrics.total_tokens_used,
            "total_cost": self.system_metrics.total_cost,
            "average_execution_time": self.system_metrics.average_execution_time,
            "active_nexus": self.system_metrics.active_nexus,
            "timestamp": self.system_metrics.timestamp.isoformat()
        }
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format"""
        return generate_latest(self.registry).decode('utf-8')
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format"""
        data = {
            "system_metrics": self.get_system_metrics(),
            "agent_metrics": {
                agent_id: [m.to_dict() for m in metrics_list]
                for agent_id, metrics_list in self.agent_metrics.items()
            },
            "timestamp": datetime.now().isoformat()
        }
        
        if format == "json":
            return json.dumps(data, indent=2)
        elif format == "prometheus":
            return self.get_prometheus_metrics()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def instrument_agent(self, agent_func: Callable) -> Callable:
        """Decorator to instrument agent functions with metrics collection"""
        @wraps(agent_func)
        async def wrapper(*args, **kwargs):
            # Extract agent_id and execution_id from arguments or generate
            agent_id = kwargs.get('agent_id', 'unknown')
            execution_id = kwargs.get('execution_id', f"exec_{int(time.time())}")
            model = kwargs.get('model', 'default')
            
            with self.trace_agent_execution(
                agent_id=agent_id,
                execution_id=execution_id,
                model=model,
                metadata={"args": str(args), "kwargs": str(kwargs)}
            ) as metrics:
                try:
                    result = await agent_func(*args, **kwargs)
                    
                    # Extract token usage from result if available
                    if isinstance(result, dict):
                        if 'usage' in result:
                            usage = result['usage']
                            self.update_token_usage(
                                execution_id=execution_id,
                                input_tokens=usage.get('prompt_tokens', 0),
                                output_tokens=usage.get('completion_tokens', 0),
                                model=model
                            )
                    
                    return result
                    
                except Exception as e:
                    metrics.error_message = str(e)
                    raise
        
        return wrapper
    
    def integrate_with_executor(self, executor: AgentExecutor):
        """Integrate metrics collector with existing executor"""
        original_execute = executor.execute
        
        async def instrumented_execute(context: ExecutionContext):
            with self.trace_agent_execution(
                agent_id=context.agent_id,
                execution_id=context.execution_id,
                model=getattr(context, 'model', 'default'),
                metadata={"context": str(context)}
            ) as metrics:
                try:
                    result = await original_execute(context)
                    
                    # Update metrics from execution context
                    if hasattr(context, 'token_usage'):
                        self.update_token_usage(
                            execution_id=context.execution_id,
                            input_tokens=context.token_usage.get('input', 0),
                            output_tokens=context.token_usage.get('output', 0)
                        )
                    
                    return result
                    
                except Exception as e:
                    metrics.error_message = str(e)
                    raise
        
        executor.execute = instrumented_execute
        return executor
    
    def stop(self):
        """Stop background collection"""
        self._running = False


# Global metrics collector instance
metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance"""
    return metrics_collector


# Convenience decorators
def trace_agent(agent_id: str, model: str = "default"):
    """Decorator to trace agent execution"""
    def decorator(func):
        return metrics_collector.instrument_agent(func)
    return decorator


def record_tokens(execution_id: str):
    """Decorator to record token usage"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            # Token recording logic would go here
            return result
        return wrapper
    return decorator


# Example usage
if __name__ == "__main__":
    # Initialize metrics collector
    collector = get_metrics_collector()
    
    # Example: Instrument an agent function
    @trace_agent(agent_id="example_agent", model="gpt-4")
    async def example_agent(prompt: str):
        # Simulate agent execution
        await asyncio.sleep(0.5)
        return {
            "response": "Example response",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        }
    
    # Run example
    async def main():
        result = await example_agent("Test prompt")
        print(f"Result: {result}")
        
        # Export metrics
        print("\nSystem Metrics:")
        print(collector.export_metrics("json"))
        
        # Stop collector
        collector.stop()
    
    asyncio.run(main())