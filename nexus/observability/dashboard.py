"""
nexus/observability/dashboard.py

Production Observability Suite for nexus.
Comprehensive monitoring with distributed tracing, performance metrics, 
LLM cost tracking, and real-time debugging dashboard.
"""

import time
import asyncio
import json
import logging
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from functools import wraps
import uuid
import os

# OpenTelemetry imports
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

# Prometheus client for direct metric exposure
from prometheus_client import start_http_server, Counter, Histogram, Gauge, Summary

# FastAPI for dashboard
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

# Import existing nexus modules
from nexus.agent.service import AgentService
from nexus.actor.page import PageActor
from nexus.agent.message_manager.service import MessageManager

logger = logging.getLogger(__name__)


@dataclass
class TraceSpan:
    """Represents a trace span for monitoring."""
    span_id: str
    trace_id: str
    parent_span_id: Optional[str] = None
    name: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration: Optional[float] = None
    status: str = "started"  # started, completed, error
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class LLMCallMetrics:
    """Metrics for LLM API calls."""
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class OperationMetrics:
    """Metrics for browser operations."""
    operation_type: str  # click, type, navigate, etc.
    element_selector: Optional[str] = None
    page_url: Optional[str] = None
    duration_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class TokenCounter:
    """Token counting and cost estimation for LLM calls."""
    
    # Pricing per 1000 tokens (as of 2024)
    PRICING = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    }
    
    @classmethod
    def estimate_cost(cls, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost in USD for LLM call."""
        model_pricing = cls.PRICING.get(model, {"input": 0.01, "output": 0.03})
        input_cost = (prompt_tokens / 1000) * model_pricing["input"]
        output_cost = (completion_tokens / 1000) * model_pricing["output"]
        return input_cost + output_cost
    
    @classmethod
    def count_tokens_approximate(cls, text: str) -> int:
        """Approximate token count (4 chars per token average)."""
        return len(text) // 4


class DistributedTracer:
    """Distributed tracing with OpenTelemetry integration."""
    
    def __init__(self, service_name: str = "nexus"):
        self.service_name = service_name
        self.tracer_provider = None
        self.tracer = None
        self.propagator = TraceContextTextMapPropagator()
        self.active_spans: Dict[str, TraceSpan] = {}
        self.completed_spans: deque = deque(maxlen=1000)
        self._setup_tracing()
    
    def _setup_tracing(self):
        """Initialize OpenTelemetry tracing."""
        resource = Resource.create({"service.name": self.service_name})
        self.tracer_provider = TracerProvider(resource=resource)
        
        # Add console exporter for development
        self.tracer_provider.add_span_processor(
            BatchSpanProcessor(ConsoleSpanExporter())
        )
        
        trace.set_tracer_provider(self.tracer_provider)
        self.tracer = trace.get_tracer(__name__)
    
    def start_span(self, name: str, parent_span_id: Optional[str] = None, 
                   attributes: Optional[Dict[str, Any]] = None) -> TraceSpan:
        """Start a new trace span."""
        span_id = str(uuid.uuid4())
        trace_id = str(uuid.uuid4())
        
        if parent_span_id and parent_span_id in self.active_spans:
            parent_span = self.active_spans[parent_span_id]
            trace_id = parent_span.trace_id
        
        span = TraceSpan(
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            name=name,
            attributes=attributes or {}
        )
        
        self.active_spans[span_id] = span
        
        # Start OpenTelemetry span
        otel_span = self.tracer.start_span(name)
        otel_span.set_attribute("span.id", span_id)
        otel_span.set_attribute("trace.id", trace_id)
        
        if attributes:
            for key, value in attributes.items():
                otel_span.set_attribute(key, str(value))
        
        span.attributes["otel_span"] = otel_span
        
        return span
    
    def end_span(self, span_id: str, status: str = "completed", 
                 error: Optional[str] = None):
        """End a trace span."""
        if span_id not in self.active_spans:
            return
        
        span = self.active_spans[span_id]
        span.end_time = time.time()
        span.duration = span.end_time - span.start_time
        span.status = status
        span.error = error
        
        # End OpenTelemetry span
        if "otel_span" in span.attributes:
            otel_span = span.attributes["otel_span"]
            if error:
                otel_span.set_status(trace.Status(trace.StatusCode.ERROR, error))
            otel_span.end()
        
        # Move to completed spans
        self.completed_spans.append(span)
        del self.active_spans[span_id]
        
        return span
    
    def add_span_event(self, span_id: str, event_name: str, 
                       attributes: Optional[Dict[str, Any]] = None):
        """Add an event to a span."""
        if span_id not in self.active_spans:
            return
        
        span = self.active_spans[span_id]
        event = {
            "name": event_name,
            "timestamp": time.time(),
            "attributes": attributes or {}
        }
        span.events.append(event)
        
        # Add to OpenTelemetry span
        if "otel_span" in span.attributes:
            otel_span = span.attributes["otel_span"]
            otel_span.add_event(event_name, attributes=attributes or {})


class MetricsCollector:
    """Collects and exposes Prometheus metrics."""
    
    def __init__(self):
        # Operation metrics
        self.operations_total = Counter(
            'browser_operations_total',
            'Total browser operations',
            ['operation_type', 'status']
        )
        
        self.operation_duration = Histogram(
            'browser_operation_duration_seconds',
            'Browser operation duration in seconds',
            ['operation_type'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        )
        
        # LLM metrics
        self.llm_calls_total = Counter(
            'llm_calls_total',
            'Total LLM API calls',
            ['model', 'status']
        )
        
        self.llm_tokens_total = Counter(
            'llm_tokens_total',
            'Total tokens used in LLM calls',
            ['model', 'token_type']  # prompt, completion
        )
        
        self.llm_cost_usd = Counter(
            'llm_cost_usd_total',
            'Total cost in USD for LLM calls',
            ['model']
        )
        
        self.llm_latency = Histogram(
            'llm_latency_seconds',
            'LLM API call latency in seconds',
            ['model'],
            buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
        )
        
        # System metrics
        self.active_operations = Gauge(
            'active_operations',
            'Number of currently active operations'
        )
        
        self.active_llm_calls = Gauge(
            'active_llm_calls',
            'Number of currently active LLM calls'
        )
        
        # Agent metrics
        self.agent_steps_total = Counter(
            'agent_steps_total',
            'Total agent steps executed',
            ['agent_id', 'status']
        )
        
        self.agent_step_duration = Histogram(
            'agent_step_duration_seconds',
            'Agent step duration in seconds',
            ['agent_id'],
            buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        )
    
    def record_operation(self, operation_type: str, duration: float, success: bool):
        """Record a browser operation."""
        status = "success" if success else "failure"
        self.operations_total.labels(operation_type=operation_type, status=status).inc()
        self.operation_duration.labels(operation_type=operation_type).observe(duration)
    
    def record_llm_call(self, metrics: LLMCallMetrics):
        """Record LLM call metrics."""
        status = "success" if metrics.success else "failure"
        self.llm_calls_total.labels(model=metrics.model, status=status).inc()
        self.llm_tokens_total.labels(model=metrics.model, token_type="prompt").inc(metrics.prompt_tokens)
        self.llm_tokens_total.labels(model=metrics.model, token_type="completion").inc(metrics.completion_tokens)
        self.llm_cost_usd.labels(model=metrics.model).inc(metrics.cost_usd)
        self.llm_latency.labels(model=metrics.model).observe(metrics.latency_ms / 1000)
    
    def record_agent_step(self, agent_id: str, duration: float, success: bool):
        """Record agent step metrics."""
        status = "success" if success else "failure"
        self.agent_steps_total.labels(agent_id=agent_id, status=status).inc()
        self.agent_step_duration.labels(agent_id=agent_id).observe(duration)


class RealTimeDashboard:
    """Real-time web dashboard for monitoring."""
    
    def __init__(self, tracer: DistributedTracer, metrics: MetricsCollector):
        self.tracer = tracer
        self.metrics = metrics
        self.app = FastAPI(title="Browser-Use Observability Dashboard")
        self.connected_clients: List[WebSocket] = []
        self._setup_routes()
        self._start_background_tasks()
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/")
        async def dashboard():
            return HTMLResponse(self._get_dashboard_html())
        
        @self.app.get("/api/metrics")
        async def get_metrics():
            return {
                "active_spans": len(self.tracer.active_spans),
                "completed_spans": len(self.tracer.completed_spans),
                "timestamp": time.time()
            }
        
        @self.app.get("/api/traces")
        async def get_traces():
            traces = []
            for span in list(self.tracer.completed_spans)[-100:]:
                traces.append(asdict(span))
            return {"traces": traces}
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.connected_clients.append(websocket)
            try:
                while True:
                    data = await websocket.receive_text()
                    # Handle client messages if needed
            except WebSocketDisconnect:
                self.connected_clients.remove(websocket)
    
    def _start_background_tasks(self):
        """Start background tasks for real-time updates."""
        async def broadcast_updates():
            while True:
                if self.connected_clients:
                    update = {
                        "type": "metrics_update",
                        "data": {
                            "active_operations": self.metrics.active_operations._value.get(),
                            "active_llm_calls": self.metrics.active_llm_calls._value.get(),
                            "timestamp": time.time()
                        }
                    }
                    for client in self.connected_clients:
                        try:
                            await client.send_json(update)
                        except:
                            self.connected_clients.remove(client)
                await asyncio.sleep(1)
        
        asyncio.create_task(broadcast_updates())
    
    def _get_dashboard_html(self) -> str:
        """Generate dashboard HTML."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Browser-Use Observability Dashboard</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .dashboard { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }
                .card { background: #f5f5f5; padding: 20px; border-radius: 8px; }
                .metric { font-size: 24px; font-weight: bold; }
                .label { font-size: 14px; color: #666; }
                .chart-container { height: 300px; }
            </style>
        </head>
        <body>
            <h1>Browser-Use Observability Dashboard</h1>
            <div class="dashboard">
                <div class="card">
                    <div class="label">Active Operations</div>
                    <div class="metric" id="active-ops">0</div>
                </div>
                <div class="card">
                    <div class="label">Active LLM Calls</div>
                    <div class="metric" id="active-llm">0</div>
                </div>
                <div class="card">
                    <div class="label">Total Operations (24h)</div>
                    <div class="metric" id="total-ops">0</div>
                </div>
                <div class="card">
                    <div class="label">Total LLM Cost (24h)</div>
                    <div class="metric" id="total-cost">$0.00</div>
                </div>
            </div>
            
            <div class="chart-container">
                <canvas id="operationsChart"></canvas>
            </div>
            
            <script>
                const ws = new WebSocket(`ws://${window.location.host}/ws`);
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    if (data.type === 'metrics_update') {
                        document.getElementById('active-ops').textContent = 
                            data.data.active_operations;
                        document.getElementById('active-llm').textContent = 
                            data.data.active_llm_calls;
                    }
                };
                
                // Initialize charts
                const ctx = document.getElementById('operationsChart').getContext('2d');
                const operationsChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Operations per minute',
                            data: [],
                            borderColor: 'rgb(75, 192, 192)',
                            tension: 0.1
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false
                    }
                });
                
                // Update chart data periodically
                setInterval(() => {
                    fetch('/api/metrics')
                        .then(response => response.json())
                        .then(data => {
                            const now = new Date(data.timestamp * 1000);
                            const timeLabel = now.toLocaleTimeString();
                            
                            operationsChart.data.labels.push(timeLabel);
                            operationsChart.data.datasets[0].data.push(data.active_spans);
                            
                            if (operationsChart.data.labels.length > 20) {
                                operationsChart.data.labels.shift();
                                operationsChart.data.datasets[0].data.shift();
                            }
                            
                            operationsChart.update();
                        });
                }, 5000);
            </script>
        </body>
        </html>
        """


class ObservabilityDashboard:
    """Main observability dashboard integrating all monitoring components."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.service_name = os.getenv("SERVICE_NAME", "nexus")
        self.tracer = DistributedTracer(self.service_name)
        self.metrics = MetricsCollector()
        self.dashboard = RealTimeDashboard(self.tracer, self.metrics)
        
        # Start Prometheus metrics server
        prometheus_port = int(os.getenv("PROMETHEUS_PORT", "8000"))
        start_http_server(prometheus_port)
        logger.info(f"Prometheus metrics server started on port {prometheus_port}")
        
        # Start dashboard server in background
        dashboard_port = int(os.getenv("DASHBOARD_PORT", "8001"))
        threading.Thread(
            target=self._run_dashboard,
            args=(dashboard_port,),
            daemon=True
        ).start()
        logger.info(f"Dashboard server started on port {dashboard_port}")
        
        self._initialized = True
    
    def _run_dashboard(self, port: int):
        """Run the dashboard server."""
        uvicorn.run(self.dashboard.app, host="0.0.0.0", port=port)
    
    def trace_operation(self, operation_name: str):
        """Decorator to trace browser operations."""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                span = self.tracer.start_span(operation_name)
                self.metrics.active_operations.inc()
                start_time = time.time()
                
                try:
                    result = await func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    self.tracer.end_span(span.span_id, "completed")
                    self.metrics.record_operation(operation_name, duration, True)
                    self.metrics.active_operations.dec()
                    
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    error_msg = str(e)
                    
                    self.tracer.end_span(span.span_id, "error", error_msg)
                    self.metrics.record_operation(operation_name, duration, False)
                    self.metrics.active_operations.dec()
                    
                    raise
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                span = self.tracer.start_span(operation_name)
                self.metrics.active_operations.inc()
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    self.tracer.end_span(span.span_id, "completed")
                    self.metrics.record_operation(operation_name, duration, True)
                    self.metrics.active_operations.dec()
                    
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    error_msg = str(e)
                    
                    self.tracer.end_span(span.span_id, "error", error_msg)
                    self.metrics.record_operation(operation_name, duration, False)
                    self.metrics.active_operations.dec()
                    
                    raise
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper
        return decorator
    
    def trace_llm_call(self, model: str):
        """Decorator to trace LLM API calls."""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                span = self.tracer.start_span(f"llm_call_{model}")
                self.metrics.active_llm_calls.inc()
                start_time = time.time()
                
                try:
                    result = await func(*args, **kwargs)
                    latency_ms = (time.time() - start_time) * 1000
                    
                    # Extract token counts from result if available
                    prompt_tokens = getattr(result, 'usage', {}).get('prompt_tokens', 0)
                    completion_tokens = getattr(result, 'usage', {}).get('completion_tokens', 0)
                    
                    cost = TokenCounter.estimate_cost(model, prompt_tokens, completion_tokens)
                    
                    llm_metrics = LLMCallMetrics(
                        model=model,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                        cost_usd=cost,
                        latency_ms=latency_ms,
                        success=True
                    )
                    
                    self.tracer.end_span(span.span_id, "completed")
                    self.metrics.record_llm_call(llm_metrics)
                    self.metrics.active_llm_calls.dec()
                    
                    # Add metrics to span attributes
                    span.attributes.update({
                        "llm.model": model,
                        "llm.prompt_tokens": prompt_tokens,
                        "llm.completion_tokens": completion_tokens,
                        "llm.cost_usd": cost,
                        "llm.latency_ms": latency_ms
                    })
                    
                    return result
                except Exception as e:
                    latency_ms = (time.time() - start_time) * 1000
                    error_msg = str(e)
                    
                    llm_metrics = LLMCallMetrics(
                        model=model,
                        latency_ms=latency_ms,
                        success=False,
                        error=error_msg
                    )
                    
                    self.tracer.end_span(span.span_id, "error", error_msg)
                    self.metrics.record_llm_call(llm_metrics)
                    self.metrics.active_llm_calls.dec()
                    
                    raise
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                span = self.tracer.start_span(f"llm_call_{model}")
                self.metrics.active_llm_calls.inc()
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    latency_ms = (time.time() - start_time) * 1000
                    
                    # Extract token counts from result if available
                    prompt_tokens = getattr(result, 'usage', {}).get('prompt_tokens', 0)
                    completion_tokens = getattr(result, 'usage', {}).get('completion_tokens', 0)
                    
                    cost = TokenCounter.estimate_cost(model, prompt_tokens, completion_tokens)
                    
                    llm_metrics = LLMCallMetrics(
                        model=model,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                        cost_usd=cost,
                        latency_ms=latency_ms,
                        success=True
                    )
                    
                    self.tracer.end_span(span.span_id, "completed")
                    self.metrics.record_llm_call(llm_metrics)
                    self.metrics.active_llm_calls.dec()
                    
                    return result
                except Exception as e:
                    latency_ms = (time.time() - start_time) * 1000
                    error_msg = str(e)
                    
                    llm_metrics = LLMCallMetrics(
                        model=model,
                        latency_ms=latency_ms,
                        success=False,
                        error=error_msg
                    )
                    
                    self.tracer.end_span(span.span_id, "error", error_msg)
                    self.metrics.record_llm_call(llm_metrics)
                    self.metrics.active_llm_calls.dec()
                    
                    raise
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper
        return decorator
    
    def trace_agent_step(self, agent_id: str):
        """Decorator to trace agent steps."""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                span = self.tracer.start_span(f"agent_step_{agent_id}")
                start_time = time.time()
                
                try:
                    result = await func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    self.tracer.end_span(span.span_id, "completed")
                    self.metrics.record_agent_step(agent_id, duration, True)
                    
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    error_msg = str(e)
                    
                    self.tracer.end_span(span.span_id, "error", error_msg)
                    self.metrics.record_agent_step(agent_id, duration, False)
                    
                    raise
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                span = self.tracer.start_span(f"agent_step_{agent_id}")
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    self.tracer.end_span(span.span_id, "completed")
                    self.metrics.record_agent_step(agent_id, duration, True)
                    
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    error_msg = str(e)
                    
                    self.tracer.end_span(span.span_id, "error", error_msg)
                    self.metrics.record_agent_step(agent_id, duration, False)
                    
                    raise
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper
        return decorator
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics."""
        return {
            "active_operations": self.metrics.active_operations._value.get(),
            "active_llm_calls": self.metrics.active_llm_calls._value.get(),
            "total_spans": len(self.tracer.completed_spans),
            "active_spans": len(self.tracer.active_spans),
            "timestamp": time.time()
        }
    
    def get_recent_traces(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trace spans."""
        traces = []
        for span in list(self.tracer.completed_spans)[-limit:]:
            traces.append(asdict(span))
        return traces


# Global instance for easy access
dashboard = ObservabilityDashboard()


# Convenience decorators for common operations
def trace_browser_operation(operation_name: str):
    """Convenience decorator for browser operations."""
    return dashboard.trace_operation(operation_name)


def trace_llm_call(model: str):
    """Convenience decorator for LLM calls."""
    return dashboard.trace_llm_call(model)


def trace_agent_step(agent_id: str):
    """Convenience decorator for agent steps."""
    return dashboard.trace_agent_step(agent_id)


# Integration with existing nexus modules
class InstrumentedAgentService(AgentService):
    """Agent service with built-in observability."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_id = str(uuid.uuid4())
    
    @trace_agent_step("agent_step")
    async def step(self, *args, **kwargs):
        return await super().step(*args, **kwargs)


class InstrumentedPageActor(PageActor):
    """Page actor with built-in observability."""
    
    @trace_browser_operation("click")
    async def click(self, selector: str, *args, **kwargs):
        return await super().click(selector, *args, **kwargs)
    
    @trace_browser_operation("type")
    async def type(self, selector: str, text: str, *args, **kwargs):
        return await super().type(selector, text, *args, **kwargs)
    
    @trace_browser_operation("navigate")
    async def navigate(self, url: str, *args, **kwargs):
        return await super().navigate(url, *args, **kwargs)


# Example usage and integration guide
"""
Integration Guide:

1. Basic Usage:
   from nexus.observability.dashboard import dashboard, trace_browser_operation

   @trace_browser_operation("custom_operation")
   async def my_custom_operation():
       # Your code here
       pass

2. LLM Call Tracking:
   from nexus.observability.dashboard import trace_llm_call

   @trace_llm_call("gpt-4")
   async def call_llm(prompt):
       # Your LLM API call
       pass

3. Agent Integration:
   from nexus.observability.dashboard import InstrumentedAgentService
   
   agent = InstrumentedAgentService(...)
   # All agent steps will be automatically traced

4. Page Actor Integration:
   from nexus.observability.dashboard import InstrumentedPageActor
   
   actor = InstrumentedPageActor(...)
   # All browser operations will be automatically traced

5. Custom Metrics:
   from nexus.observability.dashboard import dashboard
   
   # Record custom operation
   dashboard.metrics.record_operation("custom_op", 0.5, True)
   
   # Get metrics summary
   summary = dashboard.get_metrics_summary()

6. Environment Variables:
   SERVICE_NAME=nexus
   PROMETHEUS_PORT=8000
   DASHBOARD_PORT=8001

7. Accessing the Dashboard:
   - Web Dashboard: http://localhost:8001
   - Prometheus Metrics: http://localhost:8000/metrics
   - API Endpoints: http://localhost:8001/api/metrics
"""