"""nexus/observability/tracing.py

Production Observability Suite for Browser-Use
Comprehensive monitoring with distributed tracing, performance metrics, 
LLM cost tracking, and real-time debugging dashboard.
"""

import time
import asyncio
import functools
import logging
import threading
import json
import uuid
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
import inspect
from contextlib import contextmanager, asynccontextmanager
from collections import defaultdict
import traceback

# External dependencies - gracefully handle imports
try:
    from opentelemetry import trace, context
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    # Fallback implementations
    class MockTracer:
        def start_span(self, name, **kwargs):
            return MockSpan()
        
        def start_as_current_span(self, name, **kwargs):
            return MockSpan()
    
    class MockSpan:
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass
        
        def set_attribute(self, key, value):
            pass
        
        def set_status(self, status):
            pass
        
        def record_exception(self, exception):
            pass
        
        def end(self):
            pass
    
    trace = type('trace', (), {'get_tracer': lambda name: MockTracer()})()

try:
    from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest, REGISTRY
    from prometheus_client.multiprocess import MultiProcessCollector
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Fallback metric classes
    class Counter:
        def __init__(self, *args, **kwargs):
            pass
        def inc(self, *args, **kwargs):
            pass
        def labels(self, *args, **kwargs):
            return self
    
    class Histogram:
        def __init__(self, *args, **kwargs):
            pass
        def observe(self, *args, **kwargs):
            pass
        def time(self):
            return DummyTimer()
        def labels(self, *args, **kwargs):
            return self
    
    class Gauge:
        def __init__(self, *args, **kwargs):
            pass
        def set(self, *args, **kwargs):
            pass
        def inc(self, *args, **kwargs):
            pass
        def dec(self, *args, **kwargs):
            pass
        def labels(self, *args, **kwargs):
            return self
    
    class DummyTimer:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

class SpanType(Enum):
    """Types of spans for categorization"""
    AGENT = "agent"
    BROWSER = "browser"
    LLM = "llm"
    TOOL = "tool"
    API = "api"
    DATABASE = "database"
    CACHE = "cache"
    CUSTOM = "custom"

class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    SUMMARY = "summary"

@dataclass
class SpanContext:
    """Context for a traced operation"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "attributes": self.attributes,
            "start_time": self.start_time
        }

@dataclass
class LLMMetrics:
    """Metrics for LLM API calls"""
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class PerformanceMetrics:
    """Performance metrics for operations"""
    operation: str
    duration_ms: float
    success: bool
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

class TokenCounter:
    """Token counting and cost estimation for LLM calls"""
    
    # Pricing per 1000 tokens (as of 2024)
    PRICING = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        "default": {"input": 0.01, "output": 0.03}
    }
    
    def __init__(self):
        self.encoders = {}
        if TIKTOKEN_AVAILABLE:
            try:
                self.encoders["gpt-4"] = tiktoken.encoding_for_model("gpt-4")
                self.encoders["gpt-3.5-turbo"] = tiktoken.encoding_for_model("gpt-3.5-turbo")
            except Exception:
                pass
    
    def count_tokens(self, text: str, model: str = "gpt-4") -> int:
        """Count tokens in text for a given model"""
        if not TIKTOKEN_AVAILABLE:
            # Fallback: approximate token count (1 token ≈ 4 chars)
            return len(text) // 4
        
        try:
            if model in self.encoders:
                encoder = self.encoders[model]
            else:
                encoder = tiktoken.encoding_for_model(model)
                self.encoders[model] = encoder
            
            return len(encoder.encode(text))
        except Exception:
            # Fallback if model not found
            return len(text) // 4
    
    def estimate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Estimate cost in USD for token usage"""
        pricing = self.PRICING.get(model, self.PRICING["default"])
        
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        
        return input_cost + output_cost

class MetricsCollector:
    """Collects and manages Prometheus metrics"""
    
    def __init__(self, namespace: str = "nexus"):
        self.namespace = namespace
        self.metrics = {}
        self._lock = threading.Lock()
        
        if PROMETHEUS_AVAILABLE:
            self._initialize_metrics()
    
    def _initialize_metrics(self):
        """Initialize Prometheus metrics"""
        # Counters
        self.metrics["operations_total"] = Counter(
            f"{self.namespace}_operations_total",
            "Total number of operations",
            ["operation", "status", "service"]
        )
        
        self.metrics["llm_calls_total"] = Counter(
            f"{self.namespace}_llm_calls_total",
            "Total number of LLM API calls",
            ["model", "status"]
        )
        
        self.metrics["tokens_total"] = Counter(
            f"{self.namespace}_tokens_total",
            "Total tokens used",
            ["model", "type"]
        )
        
        self.metrics["cost_total"] = Counter(
            f"{self.namespace}_cost_total_usd",
            "Total cost in USD",
            ["model"]
        )
        
        # Histograms
        self.metrics["operation_duration_seconds"] = Histogram(
            f"{self.namespace}_operation_duration_seconds",
            "Operation duration in seconds",
            ["operation", "service"],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0)
        )
        
        self.metrics["llm_latency_seconds"] = Histogram(
            f"{self.namespace}_llm_latency_seconds",
            "LLM API call latency in seconds",
            ["model"],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)
        )
        
        # Gauges
        self.metrics["active_operations"] = Gauge(
            f"{self.namespace}_active_operations",
            "Number of active operations",
            ["operation"]
        )
        
        self.metrics["browser_pages"] = Gauge(
            f"{self.namespace}_browser_pages",
            "Number of open browser pages"
        )
        
        self.metrics["agent_tasks"] = Gauge(
            f"{self.namespace}_agent_tasks",
            "Number of agent tasks in progress",
            ["agent_type"]
        )
    
    def record_operation(self, operation: str, duration: float, success: bool, service: str = "default"):
        """Record an operation metric"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        status = "success" if success else "failure"
        
        with self._lock:
            self.metrics["operations_total"].labels(
                operation=operation,
                status=status,
                service=service
            ).inc()
            
            self.metrics["operation_duration_seconds"].labels(
                operation=operation,
                service=service
            ).observe(duration)
    
    def record_llm_call(self, metrics: LLMMetrics):
        """Record LLM call metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        status = "success" if metrics.success else "failure"
        
        with self._lock:
            self.metrics["llm_calls_total"].labels(
                model=metrics.model,
                status=status
            ).inc()
            
            self.metrics["tokens_total"].labels(
                model=metrics.model,
                type="input"
            ).inc(metrics.input_tokens)
            
            self.metrics["tokens_total"].labels(
                model=metrics.model,
                type="output"
            ).inc(metrics.output_tokens)
            
            self.metrics["cost_total"].labels(
                model=metrics.model
            ).inc(metrics.cost_usd)
            
            self.metrics["llm_latency_seconds"].labels(
                model=metrics.model
            ).observe(metrics.latency_ms / 1000)
    
    def update_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Update a gauge metric"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        with self._lock:
            if name in self.metrics:
                if labels:
                    self.metrics[name].labels(**labels).set(value)
                else:
                    self.metrics[name].set(value)
    
    def increment_gauge(self, name: str, labels: Optional[Dict[str, str]] = None, amount: float = 1):
        """Increment a gauge metric"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        with self._lock:
            if name in self.metrics:
                if labels:
                    self.metrics[name].labels(**labels).inc(amount)
                else:
                    self.metrics[name].inc(amount)
    
    def decrement_gauge(self, name: str, labels: Optional[Dict[str, str]] = None, amount: float = 1):
        """Decrement a gauge metric"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        with self._lock:
            if name in self.metrics:
                if labels:
                    self.metrics[name].labels(**labels).dec(amount)
                else:
                    self.metrics[name].dec(amount)
    
    def get_metrics(self) -> bytes:
        """Get Prometheus metrics in exposition format"""
        if not PROMETHEUS_AVAILABLE:
            return b""
        
        return generate_latest(REGISTRY)

class TraceManager:
    """Manages distributed tracing with OpenTelemetry"""
    
    def __init__(self, service_name: str = "nexus", 
                 exporter_type: str = "jaeger",
                 endpoint: Optional[str] = None):
        self.service_name = service_name
        self.exporter_type = exporter_type
        self.endpoint = endpoint
        self.tracer = None
        self.propagator = TraceContextTextMapPropagator() if OPENTELEMETRY_AVAILABLE else None
        
        if OPENTELEMETRY_AVAILABLE:
            self._initialize_tracer()
    
    def _initialize_tracer(self):
        """Initialize OpenTelemetry tracer"""
        resource = Resource.create({
            "service.name": self.service_name,
            "service.version": "1.0.0",
            "deployment.environment": "production"
        })
        
        tracer_provider = TracerProvider(resource=resource)
        
        if self.exporter_type == "jaeger" and self.endpoint:
            exporter = JaegerExporter(
                agent_host_name=self.endpoint.split(":")[0],
                agent_port=int(self.endpoint.split(":")[1]) if ":" in self.endpoint else 6831,
            )
        elif self.exporter_type == "otlp" and self.endpoint:
            exporter = OTLPSpanExporter(endpoint=self.endpoint)
        else:
            # Use console exporter for development
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
            exporter = ConsoleSpanExporter()
        
        # Use BatchSpanProcessor for production, SimpleSpanProcessor for debugging
        span_processor = BatchSpanProcessor(exporter)
        tracer_provider.add_span_processor(span_processor)
        
        trace.set_tracer_provider(tracer_provider)
        self.tracer = trace.get_tracer(self.service_name)
    
    def start_span(self, name: str, span_type: SpanType = SpanType.CUSTOM, 
                   attributes: Optional[Dict[str, Any]] = None,
                   parent_context: Optional[Any] = None) -> Any:
        """Start a new span"""
        if not OPENTELEMETRY_AVAILABLE or not self.tracer:
            return MockSpan()
        
        span_attributes = {
            "span.type": span_type.value,
            "service.name": self.service_name,
            **(attributes or {})
        }
        
        if parent_context:
            span = self.tracer.start_span(
                name,
                context=parent_context,
                attributes=span_attributes
            )
        else:
            span = self.tracer.start_span(
                name,
                attributes=span_attributes
            )
        
        return span
    
    @contextmanager
    def trace_operation(self, name: str, span_type: SpanType = SpanType.CUSTOM,
                        attributes: Optional[Dict[str, Any]] = None):
        """Context manager for tracing an operation"""
        span = self.start_span(name, span_type, attributes)
        
        try:
            yield span
            span.set_status(trace.StatusCode.OK)
        except Exception as e:
            span.set_status(trace.StatusCode.ERROR, str(e))
            span.record_exception(e)
            raise
        finally:
            span.end()
    
    @asynccontextmanager
    async def trace_async_operation(self, name: str, span_type: SpanType = SpanType.CUSTOM,
                                    attributes: Optional[Dict[str, Any]] = None):
        """Async context manager for tracing an operation"""
        span = self.start_span(name, span_type, attributes)
        
        try:
            yield span
            span.set_status(trace.StatusCode.OK)
        except Exception as e:
            span.set_status(trace.StatusCode.ERROR, str(e))
            span.record_exception(e)
            raise
        finally:
            span.end()
    
    def inject_context(self, carrier: Dict[str, str]):
        """Inject trace context into carrier for propagation"""
        if self.propagator:
            self.propagator.inject(carrier)
    
    def extract_context(self, carrier: Dict[str, str]):
        """Extract trace context from carrier"""
        if self.propagator:
            return self.propagator.extract(carrier)
        return None

class RealTimeDashboard:
    """Real-time debugging dashboard for monitoring"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 9090):
        self.host = host
        self.port = port
        self.metrics_history = defaultdict(list)
        self.active_traces = {}
        self.recent_errors = []
        self.performance_stats = defaultdict(list)
        self._lock = threading.Lock()
        self._server = None
        self._running = False
    
    def update_metrics(self, operation: str, metrics: PerformanceMetrics):
        """Update metrics for dashboard"""
        with self._lock:
            self.metrics_history[operation].append(metrics)
            
            # Keep only last 1000 metrics per operation
            if len(self.metrics_history[operation]) > 1000:
                self.metrics_history[operation] = self.metrics_history[operation][-1000:]
    
    def add_trace(self, trace_id: str, span_context: SpanContext):
        """Add active trace to dashboard"""
        with self._lock:
            self.active_traces[trace_id] = {
                "context": span_context.to_dict(),
                "start_time": time.time(),
                "status": "active"
            }
    
    def complete_trace(self, trace_id: str, success: bool = True, error: Optional[str] = None):
        """Mark trace as completed"""
        with self._lock:
            if trace_id in self.active_traces:
                self.active_traces[trace_id]["status"] = "completed" if success else "failed"
                self.active_traces[trace_id]["end_time"] = time.time()
                self.active_traces[trace_id]["success"] = success
                if error:
                    self.active_traces[trace_id]["error"] = error
    
    def add_error(self, error: Exception, context: Dict[str, Any]):
        """Add error to recent errors list"""
        with self._lock:
            error_entry = {
                "timestamp": datetime.now().isoformat(),
                "error_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
                "context": context
            }
            self.recent_errors.append(error_entry)
            
            # Keep only last 100 errors
            if len(self.recent_errors) > 100:
                self.recent_errors = self.recent_errors[-100:]
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data"""
        with self._lock:
            # Calculate performance stats
            perf_stats = {}
            for operation, metrics_list in self.metrics_history.items():
                if metrics_list:
                    durations = [m.duration_ms for m in metrics_list if m.success]
                    if durations:
                        perf_stats[operation] = {
                            "count": len(metrics_list),
                            "success_rate": sum(1 for m in metrics_list if m.success) / len(metrics_list) * 100,
                            "avg_duration_ms": sum(durations) / len(durations),
                            "p95_duration_ms": sorted(durations)[int(len(durations) * 0.95)] if durations else 0,
                            "p99_duration_ms": sorted(durations)[int(len(durations) * 0.99)] if durations else 0
                        }
            
            return {
                "timestamp": datetime.now().isoformat(),
                "active_traces": len(self.active_traces),
                "recent_errors": len(self.recent_errors),
                "performance_stats": perf_stats,
                "active_traces_details": dict(list(self.active_traces.items())[:50]),  # Last 50 active traces
                "recent_errors_details": self.recent_errors[-10:]  # Last 10 errors
            }
    
    def start(self):
        """Start the dashboard server"""
        if self._running:
            return
        
        self._running = True
        
        def run_server():
            try:
                from http.server import HTTPServer, BaseHTTPRequestHandler
                import socketserver
                
                dashboard = self
                
                class DashboardHandler(BaseHTTPRequestHandler):
                    def do_GET(self):
                        if self.path == "/":
                            self.send_response(200)
                            self.send_header("Content-type", "text/html")
                            self.end_headers()
                            self.wfile.write(self._get_dashboard_html().encode())
                        elif self.path == "/api/dashboard":
                            self.send_response(200)
                            self.send_header("Content-type", "application/json")
                            self.end_headers()
                            data = dashboard.get_dashboard_data()
                            self.wfile.write(json.dumps(data, default=str).encode())
                        elif self.path == "/metrics":
                            self.send_response(200)
                            self.send_header("Content-type", "text/plain")
                            self.end_headers()
                            # This would integrate with Prometheus metrics
                            self.wfile.write(b"# Metrics endpoint - integrate with Prometheus client")
                        else:
                            self.send_response(404)
                            self.end_headers()
                    
                    def _get_dashboard_html(self):
                        return """
                        <!DOCTYPE html>
                        <html>
                        <head>
                            <title>Browser-Use Observability Dashboard</title>
                            <meta http-equiv="refresh" content="5">
                            <style>
                                body { font-family: Arial, sans-serif; margin: 20px; }
                                .container { display: flex; flex-wrap: wrap; }
                                .card { border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin: 10px; min-width: 300px; }
                                .metric { margin: 10px 0; }
                                .error { color: #d32f2f; }
                                .success { color: #388e3c; }
                                table { width: 100%; border-collapse: collapse; }
                                th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                            </style>
                        </head>
                        <body>
                            <h1>Browser-Use Observability Dashboard</h1>
                            <div class="container">
                                <div class="card">
                                    <h2>Performance Metrics</h2>
                                    <div id="performance"></div>
                                </div>
                                <div class="card">
                                    <h2>Active Traces</h2>
                                    <div id="traces"></div>
                                </div>
                                <div class="card">
                                    <h2>Recent Errors</h2>
                                    <div id="errors"></div>
                                </div>
                            </div>
                            <script>
                                fetch('/api/dashboard')
                                    .then(response => response.json())
                                    .then(data => {
                                        // Update performance metrics
                                        let perfHtml = '<table><tr><th>Operation</th><th>Count</th><th>Success Rate</th><th>Avg Duration</th></tr>';
                                        for (const [op, stats] of Object.entries(data.performance_stats || {})) {
                                            perfHtml += `<tr>
                                                <td>${op}</td>
                                                <td>${stats.count}</td>
                                                <td>${stats.success_rate.toFixed(1)}%</td>
                                                <td>${stats.avg_duration_ms.toFixed(2)}ms</td>
                                            </tr>`;
                                        }
                                        perfHtml += '</table>';
                                        document.getElementById('performance').innerHTML = perfHtml;
                                        
                                        // Update active traces
                                        document.getElementById('traces').innerHTML = 
                                            `<p>Active traces: ${data.active_traces}</p>`;
                                        
                                        // Update recent errors
                                        let errorHtml = '<ul>';
                                        for (const error of data.recent_errors_details || []) {
                                            errorHtml += `<li class="error">${error.timestamp}: ${error.error_type} - ${error.message}</li>`;
                                        }
                                        errorHtml += '</ul>';
                                        document.getElementById('errors').innerHTML = errorHtml;
                                    });
                            </script>
                        </body>
                        </html>
                        """
                    
                    def log_message(self, format, *args):
                        # Suppress default logging
                        pass
                
                with HTTPServer((self.host, self.port), DashboardHandler) as httpd:
                    self._server = httpd
                    logger.info(f"Dashboard server started at http://{self.host}:{self.port}")
                    httpd.serve_forever()
                    
            except ImportError:
                logger.warning("HTTP server not available, dashboard disabled")
            except Exception as e:
                logger.error(f"Failed to start dashboard server: {e}")
        
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
    
    def stop(self):
        """Stop the dashboard server"""
        self._running = False
        if self._server:
            self._server.shutdown()

class ObservabilitySuite:
    """Main observability suite combining all components"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern for global observability"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, service_name: str = "nexus",
                 enable_tracing: bool = True,
                 enable_metrics: bool = True,
                 enable_dashboard: bool = True,
                 dashboard_host: str = "0.0.0.0",
                 dashboard_port: int = 9090,
                 trace_exporter: str = "jaeger",
                 trace_endpoint: Optional[str] = None):
        
        if self._initialized:
            return
        
        self.service_name = service_name
        self.enable_tracing = enable_tracing
        self.enable_metrics = enable_metrics
        self.enable_dashboard = enable_dashboard
        
        # Initialize components
        self.token_counter = TokenCounter()
        
        if enable_tracing:
            self.trace_manager = TraceManager(
                service_name=service_name,
                exporter_type=trace_exporter,
                endpoint=trace_endpoint
            )
        else:
            self.trace_manager = None
        
        if enable_metrics:
            self.metrics_collector = MetricsCollector(namespace=service_name.replace("-", "_"))
        else:
            self.metrics_collector = None
        
        if enable_dashboard:
            self.dashboard = RealTimeDashboard(host=dashboard_host, port=dashboard_port)
            self.dashboard.start()
        else:
            self.dashboard = None
        
        self._initialized = True
        logger.info(f"Observability suite initialized for service: {service_name}")
    
    def trace(self, name: Optional[str] = None, span_type: SpanType = SpanType.CUSTOM,
              attributes: Optional[Dict[str, Any]] = None):
        """Decorator for tracing functions"""
        def decorator(func):
            nonlocal name
            if name is None:
                name = f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                if not self.enable_tracing:
                    return func(*args, **kwargs)
                
                with self.trace_manager.trace_operation(name, span_type, attributes) as span:
                    start_time = time.time()
                    try:
                        result = func(*args, **kwargs)
                        duration = (time.time() - start_time) * 1000
                        
                        if self.enable_metrics:
                            self.metrics_collector.record_operation(
                                operation=name,
                                duration=duration / 1000,
                                success=True
                            )
                        
                        if self.dashboard:
                            self.dashboard.update_metrics(name, PerformanceMetrics(
                                operation=name,
                                duration_ms=duration,
                                success=True
                            ))
                        
                        return result
                    except Exception as e:
                        duration = (time.time() - start_time) * 1000
                        
                        if self.enable_metrics:
                            self.metrics_collector.record_operation(
                                operation=name,
                                duration=duration / 1000,
                                success=False
                            )
                        
                        if self.dashboard:
                            self.dashboard.update_metrics(name, PerformanceMetrics(
                                operation=name,
                                duration_ms=duration,
                                success=False
                            ))
                            self.dashboard.add_error(e, {"operation": name, "args": str(args)[:200]})
                        
                        raise
            
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not self.enable_tracing:
                    return await func(*args, **kwargs)
                
                async with self.trace_manager.trace_async_operation(name, span_type, attributes) as span:
                    start_time = time.time()
                    try:
                        result = await func(*args, **kwargs)
                        duration = (time.time() - start_time) * 1000
                        
                        if self.enable_metrics:
                            self.metrics_collector.record_operation(
                                operation=name,
                                duration=duration / 1000,
                                success=True
                            )
                        
                        if self.dashboard:
                            self.dashboard.update_metrics(name, PerformanceMetrics(
                                operation=name,
                                duration_ms=duration,
                                success=True
                            ))
                        
                        return result
                    except Exception as e:
                        duration = (time.time() - start_time) * 1000
                        
                        if self.enable_metrics:
                            self.metrics_collector.record_operation(
                                operation=name,
                                duration=duration / 1000,
                                success=False
                            )
                        
                        if self.dashboard:
                            self.dashboard.update_metrics(name, PerformanceMetrics(
                                operation=name,
                                duration_ms=duration,
                                success=False
                            ))
                            self.dashboard.add_error(e, {"operation": name, "args": str(args)[:200]})
                        
                        raise
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def trace_llm_call(self, model: str = "gpt-4"):
        """Decorator for tracing LLM calls with token counting"""
        def decorator(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                trace_id = str(uuid.uuid4())
                
                # Extract input for token counting
                input_text = ""
                if args and isinstance(args[0], str):
                    input_text = args[0]
                elif "prompt" in kwargs:
                    input_text = kwargs["prompt"]
                elif "messages" in kwargs:
                    # For chat models
                    messages = kwargs["messages"]
                    if isinstance(messages, list):
                        input_text = " ".join([m.get("content", "") for m in messages if isinstance(m, dict)])
                
                input_tokens = self.token_counter.count_tokens(input_text, model) if input_text else 0
                
                span_attributes = {
                    "llm.model": model,
                    "llm.input_tokens": input_tokens,
                    "trace.id": trace_id
                }
                
                if self.enable_tracing:
                    with self.trace_manager.trace_operation(
                        f"llm_call_{model}",
                        SpanType.LLM,
                        span_attributes
                    ) as span:
                        try:
                            result = await func(*args, **kwargs)
                            
                            # Extract output for token counting
                            output_text = ""
                            if isinstance(result, str):
                                output_text = result
                            elif hasattr(result, "content"):
                                output_text = result.content
                            elif isinstance(result, dict) and "content" in result:
                                output_text = result["content"]
                            
                            output_tokens = self.token_counter.count_tokens(output_text, model) if output_text else 0
                            total_tokens = input_tokens + output_tokens
                            cost = self.token_counter.estimate_cost(input_tokens, output_tokens, model)
                            latency_ms = (time.time() - start_time) * 1000
                            
                            # Update span with output metrics
                            span.set_attribute("llm.output_tokens", output_tokens)
                            span.set_attribute("llm.total_tokens", total_tokens)
                            span.set_attribute("llm.cost_usd", cost)
                            span.set_attribute("llm.latency_ms", latency_ms)
                            
                            # Record metrics
                            if self.enable_metrics:
                                self.metrics_collector.record_llm_call(LLMMetrics(
                                    model=model,
                                    input_tokens=input_tokens,
                                    output_tokens=output_tokens,
                                    total_tokens=total_tokens,
                                    cost_usd=cost,
                                    latency_ms=latency_ms,
                                    success=True
                                ))
                            
                            if self.dashboard:
                                self.dashboard.update_metrics(f"llm_{model}", PerformanceMetrics(
                                    operation=f"llm_{model}",
                                    duration_ms=latency_ms,
                                    success=True,
                                    metadata={
                                        "input_tokens": input_tokens,
                                        "output_tokens": output_tokens,
                                        "cost_usd": cost
                                    }
                                ))
                            
                            return result
                            
                        except Exception as e:
                            latency_ms = (time.time() - start_time) * 1000
                            
                            if self.enable_metrics:
                                self.metrics_collector.record_llm_call(LLMMetrics(
                                    model=model,
                                    input_tokens=input_tokens,
                                    latency_ms=latency_ms,
                                    success=False,
                                    error_message=str(e)
                                ))
                            
                            if self.dashboard:
                                self.dashboard.update_metrics(f"llm_{model}", PerformanceMetrics(
                                    operation=f"llm_{model}",
                                    duration_ms=latency_ms,
                                    success=False,
                                    metadata={"error": str(e)}
                                ))
                                self.dashboard.add_error(e, {"operation": f"llm_{model}", "model": model})
                            
                            raise
                else:
                    # No tracing, just execute with metrics
                    try:
                        result = await func(*args, **kwargs)
                        
                        output_text = ""
                        if isinstance(result, str):
                            output_text = result
                        elif hasattr(result, "content"):
                            output_text = result.content
                        
                        output_tokens = self.token_counter.count_tokens(output_text, model) if output_text else 0
                        total_tokens = input_tokens + output_tokens
                        cost = self.token_counter.estimate_cost(input_tokens, output_tokens, model)
                        latency_ms = (time.time() - start_time) * 1000
                        
                        if self.enable_metrics:
                            self.metrics_collector.record_llm_call(LLMMetrics(
                                model=model,
                                input_tokens=input_tokens,
                                output_tokens=output_tokens,
                                total_tokens=total_tokens,
                                cost_usd=cost,
                                latency_ms=latency_ms,
                                success=True
                            ))
                        
                        return result
                        
                    except Exception as e:
                        latency_ms = (time.time() - start_time) * 1000
                        
                        if self.enable_metrics:
                            self.metrics_collector.record_llm_call(LLMMetrics(
                                model=model,
                                input_tokens=input_tokens,
                                latency_ms=latency_ms,
                                success=False,
                                error_message=str(e)
                            ))
                        
                        raise
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # For synchronous functions, we'd need a different approach
                # For now, assume LLM calls are async
                return asyncio.get_event_loop().run_until_complete(async_wrapper(*args, **kwargs))
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    @contextmanager
    def span(self, name: str, span_type: SpanType = SpanType.CUSTOM,
             attributes: Optional[Dict[str, Any]] = None):
        """Context manager for creating spans"""
        if self.enable_tracing:
            with self.trace_manager.trace_operation(name, span_type, attributes) as span:
                trace_id = str(span.get_span_context().trace_id) if hasattr(span, 'get_span_context') else str(uuid.uuid4())
                span_id = str(span.get_span_context().span_id) if hasattr(span, 'get_span_context') else str(uuid.uuid4())
                
                span_context = SpanContext(
                    trace_id=trace_id,
                    span_id=span_id,
                    attributes=attributes or {}
                )
                
                if self.dashboard:
                    self.dashboard.add_trace(trace_id, span_context)
                
                start_time = time.time()
                success = True
                error_msg = None
                
                try:
                    yield span
                except Exception as e:
                    success = False
                    error_msg = str(e)
                    raise
                finally:
                    duration = (time.time() - start_time) * 1000
                    
                    if self.enable_metrics:
                        self.metrics_collector.record_operation(
                            operation=name,
                            duration=duration / 1000,
                            success=success
                        )
                    
                    if self.dashboard:
                        self.dashboard.update_metrics(name, PerformanceMetrics(
                            operation=name,
                            duration_ms=duration,
                            success=success
                        ))
                        self.dashboard.complete_trace(trace_id, success, error_msg)
        else:
            # No tracing, just execute
            start_time = time.time()
            success = True
            error_msg = None
            
            try:
                yield None
            except Exception as e:
                success = False
                error_msg = str(e)
                raise
            finally:
                duration = (time.time() - start_time) * 1000
                
                if self.enable_metrics:
                    self.metrics_collector.record_operation(
                        operation=name,
                        duration=duration / 1000,
                        success=success
                    )
    
    def record_metric(self, metric_type: MetricType, name: str, value: float,
                      labels: Optional[Dict[str, str]] = None):
        """Record a custom metric"""
        if not self.enable_metrics:
            return
        
        if metric_type == MetricType.COUNTER:
            # For counters, we need to use the appropriate Prometheus counter
            # This is a simplified implementation
            pass
        elif metric_type == MetricType.GAUGE:
            self.metrics_collector.update_gauge(name, value, labels)
        elif metric_type == MetricType.HISTOGRAM:
            # Histograms would need to be pre-defined
            pass
    
    def get_metrics(self) -> bytes:
        """Get all metrics in Prometheus format"""
        if self.enable_metrics:
            return self.metrics_collector.get_metrics()
        return b""
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data"""
        if self.dashboard:
            return self.dashboard.get_dashboard_data()
        return {}
    
    def shutdown(self):
        """Shutdown the observability suite"""
        if self.dashboard:
            self.dashboard.stop()
        logger.info("Observability suite shutdown")

# Global observability instance
_observability: Optional[ObservabilitySuite] = None

def get_observability(**kwargs) -> ObservabilitySuite:
    """Get or create the global observability instance"""
    global _observability
    if _observability is None:
        _observability = ObservabilitySuite(**kwargs)
    return _observability

def init_observability(service_name: str = "nexus", **kwargs) -> ObservabilitySuite:
    """Initialize observability with custom settings"""
    global _observability
    _observability = ObservabilitySuite(service_name=service_name, **kwargs)
    return _observability

# Convenience decorators
def trace(name: Optional[str] = None, span_type: SpanType = SpanType.CUSTOM,
          attributes: Optional[Dict[str, Any]] = None):
    """Convenience decorator for tracing"""
    obs = get_observability()
    return obs.trace(name, span_type, attributes)

def trace_llm(model: str = "gpt-4"):
    """Convenience decorator for LLM call tracing"""
    obs = get_observability()
    return obs.trace_llm_call(model)

# Integration with existing modules
def instrument_agent_service(agent_service_module):
    """Instrument the agent service module with observability"""
    obs = get_observability()
    
    # Trace agent service methods
    if hasattr(agent_service_module, 'AgentService'):
        agent_class = agent_service_module.AgentService
        
        # Trace main methods
        for method_name in ['run', 'step', 'execute_action']:
            if hasattr(agent_class, method_name):
                method = getattr(agent_class, method_name)
                traced_method = obs.trace(
                    f"agent.{method_name}",
                    SpanType.AGENT
                )(method)
                setattr(agent_class, method_name, traced_method)
        
        # Trace LLM calls
        if hasattr(agent_class, 'llm'):
            llm = agent_class.llm
            if hasattr(llm, 'predict'):
                llm.predict = obs.trace_llm_call()(llm.predict)

def instrument_browser_module(browser_module):
    """Instrument browser-related modules with observability"""
    obs = get_observability()
    
    # This would instrument browser operations
    # Implementation depends on the specific browser module structure
    pass

# Export main components
__all__ = [
    'ObservabilitySuite',
    'TraceManager',
    'MetricsCollector',
    'RealTimeDashboard',
    'TokenCounter',
    'SpanType',
    'MetricType',
    'get_observability',
    'init_observability',
    'trace',
    'trace_llm',
    'instrument_agent_service',
    'instrument_browser_module'
]