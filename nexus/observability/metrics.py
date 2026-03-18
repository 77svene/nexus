"""Production Observability Suite for nexus.

Comprehensive monitoring with distributed tracing, performance metrics, cost tracking for LLM calls, and real-time debugging dashboard.
"""

import time
import asyncio
import logging
import json
import os
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
from contextlib import contextmanager, asynccontextmanager
from functools import wraps
import threading
import uuid

# OpenTelemetry imports
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.propagators.composite import CompositeHTTPPropagator

# Prometheus imports
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, Summary, Info

# FastAPI for dashboard
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# Import existing modules for integration
from nexus.agent.service import AgentService
from nexus.agent.views import AgentResponse, AgentStep
from nexus.actor.page import PageActor

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics we track."""
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    SUMMARY = "summary"


class OperationType(Enum):
    """Types of operations we monitor."""
    BROWSER_ACTION = "browser_action"
    LLM_CALL = "llm_call"
    AGENT_STEP = "agent_step"
    PAGE_NAVIGATION = "page_navigation"
    ELEMENT_INTERACTION = "element_interaction"
    PARSING = "parsing"
    VALIDATION = "validation"


@dataclass
class TraceContext:
    """Context for distributed tracing."""
    trace_id: str
    span_id: str
    trace_flags: int = 1
    trace_state: Optional[Dict[str, str]] = None
    
    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers for propagation."""
        headers = {}
        propagator = CompositeHTTPPropagator([
            TraceContextTextMapPropagator(),
            W3CBaggagePropagator()
        ])
        carrier = {}
        ctx = trace.set_span_in_context(trace.NonRecordingSpan(trace.SpanContext(
            trace_id=int(self.trace_id, 16),
            span_id=int(self.span_id, 16),
            is_remote=True,
            trace_flags=trace.TraceFlags(self.trace_flags)
        )))
        propagator.inject(carrier, context=ctx)
        return carrier


@dataclass
class LLMMetrics:
    """Metrics for LLM calls."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    model: str = ""
    provider: str = ""
    request_id: str = ""
    cached: bool = False
    error: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations."""
    operation_type: OperationType
    operation_name: str
    duration_ms: float
    success: bool
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    trace_context: Optional[TraceContext] = None
    error_message: Optional[str] = None


@dataclass
class CostEstimate:
    """Cost estimation for operations."""
    operation_id: str
    operation_type: OperationType
    llm_calls: List[LLMMetrics] = field(default_factory=list)
    total_cost_usd: float = 0.0
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    
    def add_llm_call(self, llm_metrics: LLMMetrics):
        """Add an LLM call to this cost estimate."""
        self.llm_calls.append(llm_metrics)
        self.total_cost_usd += llm_metrics.cost_usd
    
    def finalize(self):
        """Finalize the cost estimate."""
        self.end_time = datetime.utcnow()
        return self


class TokenCounter:
    """Counts tokens for different LLM providers."""
    
    # Approximate cost per token for different models (USD)
    MODEL_COSTS = {
        "gpt-4": {"prompt": 0.03 / 1000, "completion": 0.06 / 1000},
        "gpt-4-turbo": {"prompt": 0.01 / 1000, "completion": 0.03 / 1000},
        "gpt-3.5-turbo": {"prompt": 0.0015 / 1000, "completion": 0.002 / 1000},
        "claude-3-opus": {"prompt": 0.015 / 1000, "completion": 0.075 / 1000},
        "claude-3-sonnet": {"prompt": 0.003 / 1000, "completion": 0.015 / 1000},
        "claude-3-haiku": {"prompt": 0.00025 / 1000, "completion": 0.00125 / 1000},
        "default": {"prompt": 0.002 / 1000, "completion": 0.002 / 1000},
    }
    
    @classmethod
    def count_tokens(cls, text: str, model: str = "default") -> int:
        """Count tokens in text. Simplified implementation."""
        # In production, use tiktoken or model-specific tokenizer
        return len(text.split()) * 1.3  # Approximate
    
    @classmethod
    def calculate_cost(cls, 
                      prompt_tokens: int, 
                      completion_tokens: int, 
                      model: str = "default") -> float:
        """Calculate cost based on token counts."""
        costs = cls.MODEL_COSTS.get(model, cls.MODEL_COSTS["default"])
        prompt_cost = prompt_tokens * costs["prompt"]
        completion_cost = completion_tokens * costs["completion"]
        return prompt_cost + completion_cost
    
    @classmethod
    def extract_from_response(cls, response: Any, model: str = "default") -> LLMMetrics:
        """Extract token metrics from LLM response."""
        # This would be customized per LLM provider
        if hasattr(response, 'usage'):
            usage = response.usage
            prompt_tokens = getattr(usage, 'prompt_tokens', 0)
            completion_tokens = getattr(usage, 'completion_tokens', 0)
            total_tokens = getattr(usage, 'total_tokens', prompt_tokens + completion_tokens)
            cost = cls.calculate_cost(prompt_tokens, completion_tokens, model)
            
            return LLMMetrics(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=cost,
                model=model,
                provider=model.split("-")[0] if "-" in model else "unknown"
            )
        return LLMMetrics(model=model)


class PrometheusMetrics:
    """Prometheus metrics collection."""
    
    def __init__(self, prefix: str = "nexus"):
        self.prefix = prefix
        
        # Counters
        self.operations_total = Counter(
            f"{prefix}_operations_total",
            "Total number of operations",
            ["operation_type", "operation_name", "status"]
        )
        
        self.llm_calls_total = Counter(
            f"{prefix}_llm_calls_total",
            "Total number of LLM calls",
            ["model", "provider", "status"]
        )
        
        self.tokens_total = Counter(
            f"{prefix}_tokens_total",
            "Total tokens used",
            ["model", "provider", "token_type"]
        )
        
        # Histograms
        self.operation_duration_seconds = Histogram(
            f"{prefix}_operation_duration_seconds",
            "Duration of operations in seconds",
            ["operation_type", "operation_name"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
        )
        
        self.llm_latency_seconds = Histogram(
            f"{prefix}_llm_latency_seconds",
            "LLM call latency in seconds",
            ["model", "provider"],
            buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        )
        
        # Gauges
        self.active_operations = Gauge(
            f"{prefix}_active_operations",
            "Number of currently active operations",
            ["operation_type"]
        )
        
        self.cost_usd = Gauge(
            f"{prefix}_cost_usd",
            "Estimated cost in USD",
            ["operation_type", "model"]
        )
        
        # Summaries
        self.operation_latency = Summary(
            f"{prefix}_operation_latency_seconds",
            "Operation latency in seconds",
            ["operation_type"]
        )
        
        # Info metric
        self.nexus_info = Info(
            f"{prefix}_info",
            "Information about nexus instance"
        )
        
    def record_operation(self, metrics: PerformanceMetrics):
        """Record operation metrics."""
        labels = {
            "operation_type": metrics.operation_type.value,
            "operation_name": metrics.operation_name,
            "status": "success" if metrics.success else "failure"
        }
        
        self.operations_total.labels(**labels).inc()
        self.operation_duration_seconds.labels(
            operation_type=metrics.operation_type.value,
            operation_name=metrics.operation_name
        ).observe(metrics.duration_ms / 1000)
        
        self.operation_latency.labels(
            operation_type=metrics.operation_type.value
        ).observe(metrics.duration_ms / 1000)
    
    def record_llm_call(self, llm_metrics: LLMMetrics):
        """Record LLM call metrics."""
        labels = {
            "model": llm_metrics.model,
            "provider": llm_metrics.provider,
            "status": "success" if not llm_metrics.error else "failure"
        }
        
        self.llm_calls_total.labels(**labels).inc()
        self.llm_latency_seconds.labels(
            model=llm_metrics.model,
            provider=llm_metrics.provider
        ).observe(llm_metrics.latency_ms / 1000)
        
        # Token counts
        self.tokens_total.labels(
            model=llm_metrics.model,
            provider=llm_metrics.provider,
            token_type="prompt"
        ).inc(llm_metrics.prompt_tokens)
        
        self.tokens_total.labels(
            model=llm_metrics.model,
            provider=llm_metrics.provider,
            token_type="completion"
        ).inc(llm_metrics.completion_tokens)
        
        # Cost
        self.cost_usd.labels(
            operation_type=OperationType.LLM_CALL.value,
            model=llm_metrics.model
        ).set(llm_metrics.cost_usd)
    
    def start_operation(self, operation_type: OperationType):
        """Mark start of operation."""
        self.active_operations.labels(operation_type=operation_type.value).inc()
    
    def end_operation(self, operation_type: OperationType):
        """Mark end of operation."""
        self.active_operations.labels(operation_type=operation_type.value).dec()


class DistributedTracer:
    """Distributed tracing with OpenTelemetry."""
    
    def __init__(self, 
                 service_name: str = "nexus",
                 otlp_endpoint: Optional[str] = None,
                 console_export: bool = False):
        
        resource = Resource.create({
            "service.name": service_name,
            "service.version": "1.0.0",
            "deployment.environment": os.getenv("ENVIRONMENT", "production")
        })
        
        # Set up tracer provider
        tracer_provider = TracerProvider(resource=resource)
        
        # Add span processors
        if otlp_endpoint:
            otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        
        if console_export:
            console_exporter = ConsoleSpanExporter()
            tracer_provider.add_span_processor(BatchSpanProcessor(console_exporter))
        
        trace.set_tracer_provider(tracer_provider)
        self.tracer = trace.get_tracer(__name__)
        
        # Propagator for context propagation
        self.propagator = CompositeHTTPPropagator([
            TraceContextTextMapPropagator(),
            W3CBaggagePropagator()
        ])
    
    @contextmanager
    def start_span(self, 
                   name: str, 
                   attributes: Optional[Dict[str, Any]] = None,
                   kind: trace.SpanKind = trace.SpanKind.INTERNAL):
        """Start a new span."""
        with self.tracer.start_as_current_span(name, kind=kind) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            
            # Add trace context to span
            span_context = span.get_span_context()
            trace_context = TraceContext(
                trace_id=format(span_context.trace_id, '032x'),
                span_id=format(span_context.span_id, '016x'),
                trace_flags=span_context.trace_flags
            )
            
            span.set_attribute("trace_id", trace_context.trace_id)
            span.set_attribute("span_id", trace_context.span_id)
            
            yield span, trace_context
    
    @asynccontextmanager
    async def start_async_span(self, 
                              name: str, 
                              attributes: Optional[Dict[str, Any]] = None,
                              kind: trace.SpanKind = trace.SpanKind.INTERNAL):
        """Start a new async span."""
        with self.tracer.start_as_current_span(name, kind=kind) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            
            span_context = span.get_span_context()
            trace_context = TraceContext(
                trace_id=format(span_context.trace_id, '032x'),
                span_id=format(span_context.span_id, '016x'),
                trace_flags=span_context.trace_flags
            )
            
            span.set_attribute("trace_id", trace_context.trace_id)
            span.set_attribute("span_id", trace_context.span_id)
            
            yield span, trace_context
    
    def inject_context(self, carrier: Dict[str, str]):
        """Inject trace context into carrier."""
        self.propagator.inject(carrier)
    
    def extract_context(self, carrier: Dict[str, str]):
        """Extract trace context from carrier."""
        return self.propagator.extract(carrier)


class MetricsCollector:
    """Collects and aggregates metrics."""
    
    def __init__(self, 
                 prometheus_metrics: PrometheusMetrics,
                 tracer: DistributedTracer,
                 retention_hours: int = 24):
        
        self.prometheus = prometheus_metrics
        self.tracer = tracer
        self.retention_hours = retention_hours
        
        # In-memory storage for real-time dashboard
        self.recent_operations: List[PerformanceMetrics] = []
        self.recent_llm_calls: List[LLMMetrics] = []
        self.active_cost_estimates: Dict[str, CostEstimate] = {}
        self.completed_cost_estimates: List[CostEstimate] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_old_data, daemon=True)
        self._cleanup_thread.start()
    
    def record_operation(self, 
                        operation_type: OperationType,
                        operation_name: str,
                        duration_ms: float,
                        success: bool,
                        metadata: Optional[Dict[str, Any]] = None,
                        trace_context: Optional[TraceContext] = None,
                        error_message: Optional[str] = None):
        """Record an operation."""
        metrics = PerformanceMetrics(
            operation_type=operation_type,
            operation_name=operation_name,
            duration_ms=duration_ms,
            success=success,
            metadata=metadata or {},
            trace_context=trace_context,
            error_message=error_message
        )
        
        # Record to Prometheus
        self.prometheus.record_operation(metrics)
        
        # Store in memory
        with self._lock:
            self.recent_operations.append(metrics)
            # Keep only recent data
            cutoff = datetime.utcnow() - timedelta(hours=self.retention_hours)
            self.recent_operations = [
                op for op in self.recent_operations 
                if op.timestamp > cutoff
            ]
    
    def record_llm_call(self, llm_metrics: LLMMetrics):
        """Record an LLM call."""
        # Record to Prometheus
        self.prometheus.record_llm_call(llm_metrics)
        
        # Store in memory
        with self._lock:
            self.recent_llm_calls.append(llm_metrics)
            # Keep only recent data
            cutoff = datetime.utcnow() - timedelta(hours=self.retention_hours)
            self.recent_llm_calls = [
                call for call in self.recent_llm_calls
                if hasattr(call, 'timestamp') and call.timestamp > cutoff
            ]
            
            # Update active cost estimates
            for estimate in self.active_cost_estimates.values():
                estimate.add_llm_call(llm_metrics)
    
    def start_cost_estimate(self, 
                           operation_type: OperationType,
                           operation_id: Optional[str] = None) -> str:
        """Start tracking cost for an operation."""
        operation_id = operation_id or str(uuid.uuid4())
        
        estimate = CostEstimate(
            operation_id=operation_id,
            operation_type=operation_type
        )
        
        with self._lock:
            self.active_cost_estimates[operation_id] = estimate
        
        return operation_id
    
    def end_cost_estimate(self, operation_id: str) -> Optional[CostEstimate]:
        """Finalize cost tracking for an operation."""
        with self._lock:
            estimate = self.active_cost_estimates.pop(operation_id, None)
            if estimate:
                estimate.finalize()
                self.completed_cost_estimates.append(estimate)
                # Keep only recent estimates
                cutoff = datetime.utcnow() - timedelta(hours=self.retention_hours)
                self.completed_cost_estimates = [
                    est for est in self.completed_cost_estimates
                    if est.end_time and est.end_time > cutoff
                ]
                return estimate
        return None
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        with self._lock:
            now = datetime.utcnow()
            last_hour = now - timedelta(hours=1)
            
            recent_ops = [
                op for op in self.recent_operations
                if op.timestamp > last_hour
            ]
            
            recent_llm = [
                call for call in self.recent_llm_calls
                if hasattr(call, 'timestamp') and call.timestamp > last_hour
            ]
            
            total_cost = sum(
                est.total_cost_usd 
                for est in self.completed_cost_estimates
                if est.end_time and est.end_time > last_hour
            )
            
            return {
                "timestamp": now.isoformat(),
                "operations": {
                    "total": len(recent_ops),
                    "success": len([op for op in recent_ops if op.success]),
                    "failure": len([op for op in recent_ops if not op.success]),
                    "avg_duration_ms": sum(op.duration_ms for op in recent_ops) / len(recent_ops) if recent_ops else 0
                },
                "llm_calls": {
                    "total": len(recent_llm),
                    "total_tokens": sum(call.total_tokens for call in recent_llm),
                    "total_cost_usd": sum(call.cost_usd for call in recent_llm),
                    "avg_latency_ms": sum(call.latency_ms for call in recent_llm) / len(recent_llm) if recent_llm else 0
                },
                "cost_estimates": {
                    "active": len(self.active_cost_estimates),
                    "completed_last_hour": len([
                        est for est in self.completed_cost_estimates
                        if est.end_time and est.end_time > last_hour
                    ]),
                    "total_cost_last_hour_usd": total_cost
                }
            }
    
    def _cleanup_old_data(self):
        """Cleanup old data periodically."""
        while True:
            time.sleep(3600)  # Run every hour
            cutoff = datetime.utcnow() - timedelta(hours=self.retention_hours)
            
            with self._lock:
                self.recent_operations = [
                    op for op in self.recent_operations
                    if op.timestamp > cutoff
                ]
                
                self.recent_llm_calls = [
                    call for call in self.recent_llm_calls
                    if hasattr(call, 'timestamp') and call.timestamp > cutoff
                ]
                
                self.completed_cost_estimates = [
                    est for est in self.completed_cost_estimates
                    if est.end_time and est.end_time > cutoff
                ]


class ObservabilityDashboard:
    """Real-time monitoring dashboard."""
    
    def __init__(self, 
                 metrics_collector: MetricsCollector,
                 host: str = "0.0.0.0",
                 port: int = 9090):
        
        self.metrics_collector = metrics_collector
        self.host = host
        self.port = port
        self.app = FastAPI(title="Browser-Use Observability Dashboard")
        self.active_websockets: List[WebSocket] = []
        
        self._setup_routes()
        self._start_metrics_broadcast()
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home(request: Request):
            """Serve dashboard HTML."""
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Browser-Use Observability</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .container { max-width: 1200px; margin: 0 auto; }
                    .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
                    .metric-card { background: #f5f5f5; padding: 20px; border-radius: 8px; }
                    .metric-value { font-size: 24px; font-weight: bold; margin: 10px 0; }
                    .metric-label { color: #666; font-size: 14px; }
                    .success { color: #4CAF50; }
                    .failure { color: #F44336; }
                    .warning { color: #FF9800; }
                    #log { background: #1e1e1e; color: #fff; padding: 15px; border-radius: 8px; 
                           height: 300px; overflow-y: auto; font-family: monospace; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Browser-Use Observability Dashboard</h1>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-label">Operations (Last Hour)</div>
                            <div class="metric-value" id="ops-total">0</div>
                            <div>
                                <span class="success" id="ops-success">0</span> success /
                                <span class="failure" id="ops-failure">0</span> failure
                            </div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">LLM Calls (Last Hour)</div>
                            <div class="metric-value" id="llm-total">0</div>
                            <div>
                                <span id="llm-tokens">0</span> tokens /
                                $<span id="llm-cost">0.00</span>
                            </div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Active Operations</div>
                            <div class="metric-value" id="active-ops">0</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Avg Response Time</div>
                            <div class="metric-value" id="avg-response">0 ms</div>
                        </div>
                    </div>
                    <h2>Real-time Log</h2>
                    <div id="log"></div>
                </div>
                <script>
                    const ws = new WebSocket(`ws://${window.location.host}/ws/metrics`);
                    const logDiv = document.getElementById('log');
                    
                    ws.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        
                        // Update metrics
                        if (data.type === 'metrics') {
                            document.getElementById('ops-total').textContent = data.metrics.operations.total;
                            document.getElementById('ops-success').textContent = data.metrics.operations.success;
                            document.getElementById('ops-failure').textContent = data.metrics.operations.failure;
                            document.getElementById('llm-total').textContent = data.metrics.llm_calls.total;
                            document.getElementById('llm-tokens').textContent = data.metrics.llm_calls.total_tokens;
                            document.getElementById('llm-cost').textContent = data.metrics.llm_calls.total_cost_usd.toFixed(4);
                            document.getElementById('active-ops').textContent = data.metrics.cost_estimates.active;
                            document.getElementById('avg-response').textContent = 
                                `${data.metrics.operations.avg_duration_ms.toFixed(2)} ms`;
                        }
                        
                        // Add to log
                        if (data.type === 'log') {
                            const logEntry = document.createElement('div');
                            logEntry.textContent = `[${new Date().toLocaleTimeString()}] ${data.message}`;
                            logDiv.appendChild(logEntry);
                            logDiv.scrollTop = logDiv.scrollHeight;
                        }
                    };
                    
                    ws.onclose = function() {
                        console.log('WebSocket connection closed');
                    };
                </script>
            </body>
            </html>
            """
        
        @self.app.get("/metrics")
        async def prometheus_metrics():
            """Expose Prometheus metrics."""
            return Response(
                content=prometheus_client.generate_latest(),
                media_type="text/plain"
            )
        
        @self.app.get("/api/metrics/summary")
        async def metrics_summary():
            """Get metrics summary as JSON."""
            return self.metrics_collector.get_metrics_summary()
        
        @self.app.get("/api/operations/recent")
        async def recent_operations(limit: int = 100):
            """Get recent operations."""
            with self.metrics_collector._lock:
                operations = self.metrics_collector.recent_operations[-limit:]
                return [asdict(op) for op in operations]
        
        @self.app.get("/api/llm/recent")
        async def recent_llm_calls(limit: int = 100):
            """Get recent LLM calls."""
            with self.metrics_collector._lock:
                calls = self.metrics_collector.recent_llm_calls[-limit:]
                return [asdict(call) for call in calls]
        
        @self.app.websocket("/ws/metrics")
        async def websocket_metrics(websocket: WebSocket):
            """WebSocket endpoint for real-time metrics."""
            await websocket.accept()
            self.active_websockets.append(websocket)
            
            try:
                while True:
                    # Keep connection alive
                    await websocket.receive_text()
            except WebSocketDisconnect:
                self.active_websockets.remove(websocket)
    
    def _start_metrics_broadcast(self):
        """Start broadcasting metrics to WebSocket clients."""
        async def broadcast_metrics():
            while True:
                if self.active_websockets:
                    metrics = self.metrics_collector.get_metrics_summary()
                    message = {
                        "type": "metrics",
                        "metrics": metrics,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
                    disconnected = []
                    for ws in self.active_websockets:
                        try:
                            await ws.send_json(message)
                        except:
                            disconnected.append(ws)
                    
                    for ws in disconnected:
                        self.active_websockets.remove(ws)
                
                await asyncio.sleep(1)  # Update every second
        
        asyncio.create_task(broadcast_metrics())
    
    def log(self, message: str, level: str = "INFO"):
        """Send log message to dashboard."""
        log_message = {
            "type": "log",
            "level": level,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        for ws in self.active_websockets:
            try:
                asyncio.create_task(ws.send_json(log_message))
            except:
                pass
    
    def start(self):
        """Start the dashboard server."""
        import threading
        thread = threading.Thread(
            target=uvicorn.run,
            args=(self.app,),
            kwargs={"host": self.host, "port": self.port},
            daemon=True
        )
        thread.start()
        logger.info(f"Observability dashboard started on http://{self.host}:{self.port}")


class ObservabilitySuite:
    """Main observability suite integrating all components."""
    
    def __init__(self, 
                 service_name: str = "nexus",
                 otlp_endpoint: Optional[str] = None,
                 enable_prometheus: bool = True,
                 enable_dashboard: bool = True,
                 dashboard_port: int = 9090):
        
        # Initialize tracer
        self.tracer = DistributedTracer(
            service_name=service_name,
            otlp_endpoint=otlp_endpoint,
            console_export=os.getenv("OTEL_CONSOLE_EXPORT", "false").lower() == "true"
        )
        
        # Initialize Prometheus metrics
        self.prometheus_metrics = PrometheusMetrics(prefix=service_name.replace("-", "_"))
        
        # Initialize metrics collector
        self.metrics_collector = MetricsCollector(
            prometheus_metrics=self.prometheus_metrics,
            tracer=self.tracer
        )
        
        # Initialize dashboard if enabled
        self.dashboard = None
        if enable_dashboard:
            self.dashboard = ObservabilityDashboard(
                metrics_collector=self.metrics_collector,
                port=dashboard_port
            )
            self.dashboard.start()
        
        # Set up logging integration
        self._setup_logging_integration()
        
        logger.info(f"Observability suite initialized for {service_name}")
    
    def _setup_logging_integration(self):
        """Integrate with Python logging."""
        class MetricsHandler(logging.Handler):
            def __init__(self, observability):
                super().__init__()
                self.observability = observability
            
            def emit(self, record):
                if record.levelno >= logging.ERROR:
                    self.observability.log_error(record.getMessage())
                elif record.levelno >= logging.WARNING:
                    self.observability.log_warning(record.getMessage())
        
        handler = MetricsHandler(self)
        logging.getLogger().addHandler(handler)
    
    @contextmanager
    def trace_operation(self, 
                       operation_type: OperationType,
                       operation_name: str,
                       attributes: Optional[Dict[str, Any]] = None):
        """Context manager for tracing operations."""
        start_time = time.time()
        success = True
        error_message = None
        trace_context = None
        
        # Start Prometheus metrics
        self.prometheus_metrics.start_operation(operation_type)
        
        try:
            # Start trace span
            with self.tracer.start_span(
                f"{operation_type.value}.{operation_name}",
                attributes=attributes,
                kind=trace.SpanKind.INTERNAL
            ) as (span, ctx):
                trace_context = ctx
                
                # Add operation attributes
                span.set_attribute("operation.type", operation_type.value)
                span.set_attribute("operation.name", operation_name)
                
                yield span, ctx
                
        except Exception as e:
            success = False
            error_message = str(e)
            raise
        finally:
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Record metrics
            self.metrics_collector.record_operation(
                operation_type=operation_type,
                operation_name=operation_name,
                duration_ms=duration_ms,
                success=success,
                metadata=attributes,
                trace_context=trace_context,
                error_message=error_message
            )
            
            # End Prometheus metrics
            self.prometheus_metrics.end_operation(operation_type)
            
            # Log to dashboard
            if self.dashboard:
                status = "SUCCESS" if success else "FAILURE"
                self.dashboard.log(
                    f"{operation_type.value}.{operation_name}: {status} ({duration_ms:.2f}ms)",
                    "INFO" if success else "ERROR"
                )
    
    @asynccontextmanager
    async def trace_async_operation(self,
                                  operation_type: OperationType,
                                  operation_name: str,
                                  attributes: Optional[Dict[str, Any]] = None):
        """Async context manager for tracing operations."""
        start_time = time.time()
        success = True
        error_message = None
        trace_context = None
        
        # Start Prometheus metrics
        self.prometheus_metrics.start_operation(operation_type)
        
        try:
            # Start trace span
            async with self.tracer.start_async_span(
                f"{operation_type.value}.{operation_name}",
                attributes=attributes,
                kind=trace.SpanKind.INTERNAL
            ) as (span, ctx):
                trace_context = ctx
                
                # Add operation attributes
                span.set_attribute("operation.type", operation_type.value)
                span.set_attribute("operation.name", operation_name)
                
                yield span, ctx
                
        except Exception as e:
            success = False
            error_message = str(e)
            raise
        finally:
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Record metrics
            self.metrics_collector.record_operation(
                operation_type=operation_type,
                operation_name=operation_name,
                duration_ms=duration_ms,
                success=success,
                metadata=attributes,
                trace_context=trace_context,
                error_message=error_message
            )
            
            # End Prometheus metrics
            self.prometheus_metrics.end_operation(operation_type)
            
            # Log to dashboard
            if self.dashboard:
                status = "SUCCESS" if success else "FAILURE"
                self.dashboard.log(
                    f"{operation_type.value}.{operation_name}: {status} ({duration_ms:.2f}ms)",
                    "INFO" if success else "ERROR"
                )
    
    def trace_llm_call(self, model: str = "default", provider: str = "unknown"):
        """Decorator for tracing LLM calls."""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                error = None
                llm_metrics = None
                
                try:
                    # Start cost tracking
                    cost_id = self.metrics_collector.start_cost_estimate(
                        OperationType.LLM_CALL,
                        f"{model}_{provider}_{int(start_time)}"
                    )
                    
                    # Execute LLM call
                    result = await func(*args, **kwargs)
                    
                    # Extract metrics from response
                    llm_metrics = TokenCounter.extract_from_response(result, model)
                    llm_metrics.latency_ms = (time.time() - start_time) * 1000
                    llm_metrics.provider = provider
                    
                    # Record metrics
                    self.metrics_collector.record_llm_call(llm_metrics)
                    
                    # End cost tracking
                    self.metrics_collector.end_cost_estimate(cost_id)
                    
                    return result
                    
                except Exception as e:
                    error = str(e)
                    if llm_metrics:
                        llm_metrics.error = error
                        self.metrics_collector.record_llm_call(llm_metrics)
                    raise
                    
                finally:
                    # Log to dashboard
                    if self.dashboard:
                        if error:
                            self.dashboard.log(
                                f"LLM call failed: {model} - {error}",
                                "ERROR"
                            )
                        elif llm_metrics:
                            self.dashboard.log(
                                f"LLM call: {model} - {llm_metrics.total_tokens} tokens, ${llm_metrics.cost_usd:.4f}",
                                "INFO"
                            )
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                error = None
                llm_metrics = None
                
                try:
                    # Start cost tracking
                    cost_id = self.metrics_collector.start_cost_estimate(
                        OperationType.LLM_CALL,
                        f"{model}_{provider}_{int(start_time)}"
                    )
                    
                    # Execute LLM call
                    result = func(*args, **kwargs)
                    
                    # Extract metrics from response
                    llm_metrics = TokenCounter.extract_from_response(result, model)
                    llm_metrics.latency_ms = (time.time() - start_time) * 1000
                    llm_metrics.provider = provider
                    
                    # Record metrics
                    self.metrics_collector.record_llm_call(llm_metrics)
                    
                    # End cost tracking
                    self.metrics_collector.end_cost_estimate(cost_id)
                    
                    return result
                    
                except Exception as e:
                    error = str(e)
                    if llm_metrics:
                        llm_metrics.error = error
                        self.metrics_collector.record_llm_call(llm_metrics)
                    raise
                    
                finally:
                    # Log to dashboard
                    if self.dashboard:
                        if error:
                            self.dashboard.log(
                                f"LLM call failed: {model} - {error}",
                                "ERROR"
                            )
                        elif llm_metrics:
                            self.dashboard.log(
                                f"LLM call: {model} - {llm_metrics.total_tokens} tokens, ${llm_metrics.cost_usd:.4f}",
                                "INFO"
                            )
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper
        
        return decorator
    
    def instrument_agent_service(self, agent_service: AgentService):
        """Instrument an AgentService instance with observability."""
        original_step = agent_service.step
        
        @wraps(original_step)
        async def instrumented_step(*args, **kwargs):
            async with self.trace_async_operation(
                OperationType.AGENT_STEP,
                "agent_step",
                {"agent_id": agent_service.agent_id}
            ):
                return await original_step(*args, **kwargs)
        
        agent_service.step = instrumented_step
        
        # Instrument LLM calls if available
        if hasattr(agent_service, 'llm'):
            agent_service.llm = self.trace_llm_call(
                model=getattr(agent_service.llm, 'model_name', 'unknown'),
                provider="openai"  # Adjust based on actual provider
            )(agent_service.llm)
        
        return agent_service
    
    def instrument_page_actor(self, page_actor: PageActor):
        """Instrument a PageActor instance with observability."""
        original_methods = {
            'navigate': page_actor.navigate,
            'click': page_actor.click,
            'type': page_actor.type,
            'screenshot': page_actor.screenshot
        }
        
        for method_name, method in original_methods.items():
            if method:
                @wraps(method)
                async def instrumented_method(*args, **kwargs):
                    async with self.trace_async_operation(
                        OperationType.BROWSER_ACTION,
                        method_name,
                        {"page_id": id(page_actor.page) if hasattr(page_actor, 'page') else None}
                    ):
                        return await method(*args, **kwargs)
                
                setattr(page_actor, method_name, instrumented_method)
        
        return page_actor
    
    def log_error(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Log an error with observability."""
        if self.dashboard:
            self.dashboard.log(message, "ERROR")
        
        # Record as failed operation
        self.metrics_collector.record_operation(
            operation_type=OperationType.VALIDATION,
            operation_name="error_log",
            duration_ms=0,
            success=False,
            metadata=metadata or {},
            error_message=message
        )
    
    def log_warning(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Log a warning with observability."""
        if self.dashboard:
            self.dashboard.log(message, "WARNING")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics summary."""
        return self.metrics_collector.get_metrics_summary()
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format."""
        return prometheus_client.generate_latest()


# Global observability instance
_observability_instance: Optional[ObservabilitySuite] = None


def initialize_observability(
    service_name: str = "nexus",
    otlp_endpoint: Optional[str] = None,
    enable_prometheus: bool = True,
    enable_dashboard: bool = True,
    dashboard_port: int = 9090
) -> ObservabilitySuite:
    """Initialize global observability suite."""
    global _observability_instance
    
    if _observability_instance is None:
        _observability_instance = ObservabilitySuite(
            service_name=service_name,
            otlp_endpoint=otlp_endpoint,
            enable_prometheus=enable_prometheus,
            enable_dashboard=enable_dashboard,
            dashboard_port=dashboard_port
        )
    
    return _observability_instance


def get_observability() -> Optional[ObservabilitySuite]:
    """Get the global observability instance."""
    return _observability_instance


# Convenience decorators
def trace_operation(operation_type: OperationType, operation_name: str):
    """Decorator for tracing operations."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            observability = get_observability()
            if observability:
                async with observability.trace_async_operation(
                    operation_type, operation_name
                ):
                    return await func(*args, **kwargs)
            else:
                return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            observability = get_observability()
            if observability:
                with observability.trace_operation(operation_type, operation_name):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# Integration with existing modules
def setup_observability_for_agent(agent_service: AgentService) -> AgentService:
    """Setup observability for an agent service."""
    observability = get_observability()
    if observability:
        return observability.instrument_agent_service(agent_service)
    return agent_service


def setup_observability_for_page(page_actor: PageActor) -> PageActor:
    """Setup observability for a page actor."""
    observability = get_observability()
    if observability:
        return observability.instrument_page_actor(page_actor)
    return page_actor


# Auto-initialization if environment variables are set
if os.getenv("BROWSER_USE_OBSERVABILITY", "false").lower() == "true":
    initialize_observability(
        service_name=os.getenv("OTEL_SERVICE_NAME", "nexus"),
        otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
        enable_prometheus=os.getenv("ENABLE_PROMETHEUS", "true").lower() == "true",
        enable_dashboard=os.getenv("ENABLE_DASHBOARD", "true").lower() == "true",
        dashboard_port=int(os.getenv("DASHBOARD_PORT", "9090"))
    )