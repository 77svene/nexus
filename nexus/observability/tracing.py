"""
nexus/observability/tracing.py

Real-time Observability Dashboard with OpenTelemetry tracing, Prometheus metrics,
WebSocket live dashboard, and automatic anomaly detection.
"""

import asyncio
import json
import time
import threading
import statistics
from collections import defaultdict, deque
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse
import weakref

# OpenTelemetry imports
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    PeriodicExportingMetricReader,
    ConsoleMetricExporter,
)
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.trace import Status, StatusCode
from opentelemetry.metrics import Observation, CallbackOptions
import psutil

# Prometheus imports
from prometheus_client import start_http_server, Gauge, Counter, Histogram, Summary

# WebSocket imports
import websockets
from websockets.server import WebSocketServerProtocol

# Local imports
from nexus.agent.views import AgentState
from nexus.actor.page import Page


class MetricType(Enum):
    """Types of metrics we track"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricDefinition:
    """Definition of a metric to be tracked"""
    name: str
    description: str
    metric_type: MetricType
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms


@dataclass
class SpanContext:
    """Context for a traced operation"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    operation_name: str = ""
    start_time: float = field(default_factory=time.time)
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    status: StatusCode = StatusCode.UNSET


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    success_rate: float = 0.0
    latency_p50: float = 0.0
    latency_p90: float = 0.0
    latency_p99: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    active_spans: int = 0
    total_spans: int = 0


@dataclass
class Anomaly:
    """Detected anomaly"""
    timestamp: datetime
    metric_name: str
    current_value: float
    expected_range: Tuple[float, float]
    severity: str  # "warning", "critical"
    description: str
    trace_id: Optional[str] = None


class AnomalyDetector:
    """Automatic anomaly detection using statistical methods"""
    
    def __init__(self, window_size: int = 100, sensitivity: float = 2.0):
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.metric_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.anomalies: List[Anomaly] = []
        self._lock = threading.RLock()
    
    def update_metric(self, metric_name: str, value: float, trace_id: Optional[str] = None) -> Optional[Anomaly]:
        """Update metric and check for anomalies"""
        with self._lock:
            window = self.metric_windows[metric_name]
            window.append(value)
            
            if len(window) < 10:  # Need minimum data points
                return None
            
            mean = statistics.mean(window)
            stdev = statistics.stdev(window) if len(window) > 1 else 0
            
            # Check for anomalies using z-score
            if stdev > 0:
                z_score = abs(value - mean) / stdev
                if z_score > self.sensitivity:
                    expected_min = mean - (self.sensitivity * stdev)
                    expected_max = mean + (self.sensitivity * stdev)
                    
                    severity = "critical" if z_score > self.sensitivity * 1.5 else "warning"
                    
                    anomaly = Anomaly(
                        timestamp=datetime.now(),
                        metric_name=metric_name,
                        current_value=value,
                        expected_range=(expected_min, expected_max),
                        severity=severity,
                        description=f"Metric {metric_name} is {z_score:.1f} standard deviations from mean",
                        trace_id=trace_id
                    )
                    
                    self.anomalies.append(anomaly)
                    # Keep only last 1000 anomalies
                    if len(self.anomalies) > 1000:
                        self.anomalies = self.anomalies[-1000:]
                    
                    return anomaly
            
            return None


class PrometheusMetrics:
    """Prometheus metrics exporter"""
    
    def __init__(self, port: int = 9090):
        self.port = port
        self.metrics: Dict[str, Union[Counter, Gauge, Histogram, Summary]] = {}
        self._lock = threading.RLock()
        
        # Start Prometheus HTTP server
        start_http_server(port)
        
        # Initialize standard metrics
        self._init_standard_metrics()
    
    def _init_standard_metrics(self):
        """Initialize standard nexus metrics"""
        with self._lock:
            # Operation metrics
            self.metrics["browser_operations_total"] = Counter(
                "browser_operations_total",
                "Total number of browser operations",
                ["operation_type", "status"]
            )
            
            self.metrics["browser_operation_duration_seconds"] = Histogram(
                "browser_operation_duration_seconds",
                "Duration of browser operations in seconds",
                ["operation_type"],
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
            )
            
            self.metrics["browser_operation_latency"] = Summary(
                "browser_operation_latency",
                "Latency of browser operations",
                ["operation_type"]
            )
            
            # Agent metrics
            self.metrics["agent_steps_total"] = Counter(
                "agent_steps_total",
                "Total number of agent steps",
                ["agent_id", "status"]
            )
            
            self.metrics["agent_step_duration_seconds"] = Histogram(
                "agent_step_duration_seconds",
                "Duration of agent steps in seconds",
                ["agent_id"],
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
            )
            
            # Resource metrics
            self.metrics["cpu_usage_percent"] = Gauge(
                "cpu_usage_percent",
                "CPU usage percentage"
            )
            
            self.metrics["memory_usage_bytes"] = Gauge(
                "memory_usage_bytes",
                "Memory usage in bytes"
            )
            
            self.metrics["active_browser_pages"] = Gauge(
                "active_browser_pages",
                "Number of active browser pages"
            )
            
            # Success/failure metrics
            self.metrics["success_rate"] = Gauge(
                "success_rate",
                "Success rate of operations"
            )
            
            self.metrics["error_rate"] = Gauge(
                "error_rate",
                "Error rate of operations"
            )
            
            # WebSocket connections
            self.metrics["websocket_connections"] = Gauge(
                "websocket_connections",
                "Number of active WebSocket connections"
            )
    
    def record_operation(self, operation_type: str, duration: float, success: bool):
        """Record a browser operation"""
        status = "success" if success else "failure"
        
        with self._lock:
            self.metrics["browser_operations_total"].labels(
                operation_type=operation_type, 
                status=status
            ).inc()
            
            self.metrics["browser_operation_duration_seconds"].labels(
                operation_type=operation_type
            ).observe(duration)
            
            self.metrics["browser_operation_latency"].labels(
                operation_type=operation_type
            ).observe(duration)
    
    def record_agent_step(self, agent_id: str, duration: float, success: bool):
        """Record an agent step"""
        status = "success" if success else "failure"
        
        with self._lock:
            self.metrics["agent_steps_total"].labels(
                agent_id=agent_id,
                status=status
            ).inc()
            
            self.metrics["agent_step_duration_seconds"].labels(
                agent_id=agent_id
            ).observe(duration)
    
    def update_resource_metrics(self):
        """Update resource usage metrics"""
        process = psutil.Process()
        
        with self._lock:
            self.metrics["cpu_usage_percent"].set(process.cpu_percent())
            self.metrics["memory_usage_bytes"].set(process.memory_info().rss)
    
    def update_success_rate(self, rate: float):
        """Update success rate metric"""
        with self._lock:
            self.metrics["success_rate"].set(rate)
    
    def update_error_rate(self, rate: float):
        """Update error rate metric"""
        with self._lock:
            self.metrics["error_rate"].set(rate)
    
    def update_active_pages(self, count: int):
        """Update active pages metric"""
        with self._lock:
            self.metrics["active_browser_pages"].set(count)
    
    def update_websocket_connections(self, count: int):
        """Update WebSocket connections metric"""
        with self._lock:
            self.metrics["websocket_connections"].set(count)


class LiveDashboard:
    """WebSocket-based live dashboard"""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.connections: Set[WebSocketServerProtocol] = set()
        self._lock = threading.RLock()
        self._server = None
        self._running = False
        
        # Metrics history for dashboard
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.performance_metrics = PerformanceMetrics()
        
        # Start metrics collection thread
        self._metrics_thread = threading.Thread(target=self._collect_metrics_loop, daemon=True)
        self._metrics_thread.start()
    
    async def handler(self, websocket: WebSocketServerProtocol, path: str):
        """Handle WebSocket connections"""
        with self._lock:
            self.connections.add(websocket)
        
        try:
            # Send initial metrics
            await self._send_metrics(websocket)
            
            # Keep connection alive and send periodic updates
            async for message in websocket:
                # Handle client messages if needed
                data = json.loads(message)
                if data.get("type") == "subscribe":
                    await self._handle_subscription(websocket, data)
        
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            with self._lock:
                self.connections.discard(websocket)
    
    async def _send_metrics(self, websocket: WebSocketServerProtocol):
        """Send current metrics to a client"""
        metrics_data = {
            "type": "metrics_update",
            "timestamp": datetime.now().isoformat(),
            "performance": asdict(self.performance_metrics),
            "history": {
                metric_name: list(values) 
                for metric_name, values in self.metrics_history.items()
            },
            "anomalies": [
                asdict(anomaly) 
                for anomaly in getattr(self, 'anomalies', [])[-10:]  # Last 10 anomalies
            ]
        }
        
        await websocket.send(json.dumps(metrics_data))
    
    async def _handle_subscription(self, websocket: WebSocketServerProtocol, data: Dict):
        """Handle subscription requests"""
        # Could implement filtering based on subscription
        pass
    
    def _collect_metrics_loop(self):
        """Background thread to collect and broadcast metrics"""
        while True:
            try:
                # Update performance metrics
                self._update_performance_metrics()
                
                # Broadcast to all connected clients
                asyncio.run(self._broadcast_metrics())
                
                time.sleep(1)  # Update every second
            
            except Exception as e:
                print(f"Error in metrics collection: {e}")
                time.sleep(5)
    
    def _update_performance_metrics(self):
        """Update performance metrics from various sources"""
        # This would be connected to actual metrics collection
        # For now, we'll simulate with some basic updates
        process = psutil.Process()
        
        self.performance_metrics.cpu_usage = process.cpu_percent()
        self.performance_metrics.memory_usage = process.memory_info().rss
        
        # Update history
        self.metrics_history["cpu_usage"].append(self.performance_metrics.cpu_usage)
        self.metrics_history["memory_usage"].append(self.performance_metrics.memory_usage)
    
    async def _broadcast_metrics(self):
        """Broadcast metrics to all connected clients"""
        if not self.connections:
            return
        
        metrics_data = {
            "type": "metrics_update",
            "timestamp": datetime.now().isoformat(),
            "performance": asdict(self.performance_metrics),
            "history": {
                metric_name: list(values)[-10:]  # Last 10 values
                for metric_name, values in self.metrics_history.items()
            }
        }
        
        message = json.dumps(metrics_data)
        
        # Send to all connections
        disconnected = set()
        for connection in self.connections:
            try:
                await connection.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(connection)
        
        # Clean up disconnected clients
        with self._lock:
            self.connections -= disconnected
    
    async def start(self):
        """Start the WebSocket server"""
        self._running = True
        self._server = await websockets.serve(self.handler, self.host, self.port)
        print(f"Live dashboard started at ws://{self.host}:{self.port}")
    
    async def stop(self):
        """Stop the WebSocket server"""
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()


class BrowserUseTracer:
    """Main tracing and observability class for nexus"""
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        
        # Initialize OpenTelemetry
        self._init_opentelemetry()
        
        # Initialize Prometheus metrics
        self.prometheus = PrometheusMetrics(port=9090)
        
        # Initialize anomaly detector
        self.anomaly_detector = AnomalyDetector(window_size=100, sensitivity=2.5)
        
        # Initialize live dashboard
        self.dashboard = LiveDashboard(host="localhost", port=8765)
        
        # Active spans
        self.active_spans: Dict[str, SpanContext] = {}
        self.completed_spans: deque = deque(maxlen=1000)
        
        # Metrics tracking
        self.operation_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.operation_latencies: Dict[str, List[float]] = defaultdict(list)
        
        # Start background threads
        self._start_background_tasks()
        
        # Store weak references to pages for monitoring
        self._page_refs: weakref.WeakSet = weakref.WeakSet()
    
    def _init_opentelemetry(self):
        """Initialize OpenTelemetry tracing and metrics"""
        # Set up tracing
        trace.set_tracer_provider(TracerProvider())
        self.tracer = trace.get_tracer(__name__)
        
        # Add span processors
        trace.get_tracer_provider().add_span_processor(
            BatchSpanProcessor(ConsoleSpanExporter())
        )
        
        # Set up metrics
        metrics.set_meter_provider(MeterProvider())
        self.meter = metrics.get_meter(__name__)
        
        # Create metrics
        self.operation_counter = self.meter.create_counter(
            name="browser_operations",
            description="Number of browser operations",
            unit="1"
        )
        
        self.operation_histogram = self.meter.create_histogram(
            name="browser_operation_duration",
            description="Duration of browser operations",
            unit="ms"
        )
    
    def _start_background_tasks(self):
        """Start background monitoring tasks"""
        # Start metrics collection thread
        self._metrics_thread = threading.Thread(target=self._collect_system_metrics, daemon=True)
        self._metrics_thread.start()
        
        # Start dashboard in background thread
        self._dashboard_thread = threading.Thread(target=self._run_dashboard, daemon=True)
        self._dashboard_thread.start()
    
    def _collect_system_metrics(self):
        """Collect system metrics periodically"""
        while True:
            try:
                # Update Prometheus metrics
                self.prometheus.update_resource_metrics()
                
                # Check for anomalies
                cpu_anomaly = self.anomaly_detector.update_metric(
                    "cpu_usage", 
                    psutil.cpu_percent()
                )
                if cpu_anomaly:
                    print(f"CPU anomaly detected: {cpu_anomaly.description}")
                
                memory_anomaly = self.anomaly_detector.update_metric(
                    "memory_usage",
                    psutil.Process().memory_info().rss
                )
                if memory_anomaly:
                    print(f"Memory anomaly detected: {memory_anomaly.description}")
                
                # Update success/error rates
                self._update_rates()
                
                time.sleep(5)  # Collect every 5 seconds
            
            except Exception as e:
                print(f"Error collecting system metrics: {e}")
                time.sleep(10)
    
    def _update_rates(self):
        """Update success and error rates"""
        total_ops = sum(sum(counts.values()) for counts in self.operation_counts.values())
        if total_ops > 0:
            success_ops = sum(
                counts.get("success", 0) 
                for counts in self.operation_counts.values()
            )
            success_rate = success_ops / total_ops
            error_rate = 1 - success_rate
            
            self.prometheus.update_success_rate(success_rate)
            self.prometheus.update_error_rate(error_rate)
            
            # Check for rate anomalies
            rate_anomaly = self.anomaly_detector.update_metric("success_rate", success_rate)
            if rate_anomaly:
                print(f"Success rate anomaly: {rate_anomaly.description}")
    
    def _run_dashboard(self):
        """Run the dashboard in a separate thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self.dashboard.start())
            loop.run_forever()
        except Exception as e:
            print(f"Dashboard error: {e}")
        finally:
            loop.close()
    
    @contextmanager
    def trace_operation(self, operation_name: str, attributes: Optional[Dict] = None):
        """Context manager for tracing operations"""
        span_id = f"span_{int(time.time() * 1000)}"
        trace_id = f"trace_{int(time.time() * 1000)}"
        
        context = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            operation_name=operation_name,
            attributes=attributes or {}
        )
        
        self.active_spans[span_id] = context
        start_time = time.time()
        
        # Start OpenTelemetry span
        with self.tracer.start_as_current_span(operation_name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            
            try:
                yield context
                
                # Record success
                duration = time.time() - start_time
                self._record_operation(operation_name, duration, True)
                context.status = StatusCode.OK
                
            except Exception as e:
                # Record failure
                duration = time.time() - start_time
                self._record_operation(operation_name, duration, False)
                context.status = StatusCode.ERROR
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
            
            finally:
                # Clean up
                if span_id in self.active_spans:
                    self.completed_spans.append(self.active_spans.pop(span_id))
    
    def _record_operation(self, operation_name: str, duration: float, success: bool):
        """Record operation metrics"""
        status = "success" if success else "failure"
        
        # Update local tracking
        self.operation_counts[operation_name][status] += 1
        self.operation_latencies[operation_name].append(duration)
        
        # Keep only last 1000 latencies per operation
        if len(self.operation_latencies[operation_name]) > 1000:
            self.operation_latencies[operation_name] = self.operation_latencies[operation_name][-1000:]
        
        # Update Prometheus metrics
        self.prometheus.record_operation(operation_name, duration, success)
        
        # Update OpenTelemetry metrics
        self.operation_counter.add(1, {"operation": operation_name, "status": status})
        self.operation_histogram.record(duration * 1000, {"operation": operation_name})  # Convert to ms
        
        # Check for latency anomalies
        latency_anomaly = self.anomaly_detector.update_metric(
            f"{operation_name}_latency",
            duration
        )
        if latency_anomaly:
            print(f"Latency anomaly for {operation_name}: {latency_anomaly.description}")
    
    def register_page(self, page: Page):
        """Register a page for monitoring"""
        self._page_refs.add(page)
        self.prometheus.update_active_pages(len(self._page_refs))
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        metrics = PerformanceMetrics()
        
        # Calculate percentiles
        all_latencies = []
        for op_latencies in self.operation_latencies.values():
            all_latencies.extend(op_latencies)
        
        if all_latencies:
            all_latencies.sort()
            metrics.latency_p50 = all_latencies[len(all_latencies) // 2]
            metrics.latency_p90 = all_latencies[int(len(all_latencies) * 0.9)]
            metrics.latency_p99 = all_latencies[int(len(all_latencies) * 0.99)]
        
        # Calculate rates
        total_ops = sum(sum(counts.values()) for counts in self.operation_counts.values())
        if total_ops > 0:
            success_ops = sum(
                counts.get("success", 0) 
                for counts in self.operation_counts.values()
            )
            metrics.success_rate = success_ops / total_ops
            metrics.error_rate = 1 - metrics.success_rate
        
        metrics.active_spans = len(self.active_spans)
        metrics.total_spans = len(self.completed_spans)
        
        # Resource usage
        process = psutil.Process()
        metrics.cpu_usage = process.cpu_percent()
        metrics.memory_usage = process.memory_info().rss
        
        return metrics
    
    def get_anomalies(self, limit: int = 50) -> List[Anomaly]:
        """Get recent anomalies"""
        return self.anomaly_detector.anomalies[-limit:]
    
    def export_metrics_json(self) -> str:
        """Export all metrics as JSON"""
        data = {
            "performance": asdict(self.get_performance_metrics()),
            "operation_counts": dict(self.operation_counts),
            "operation_latencies": {
                op: {
                    "count": len(latencies),
                    "mean": statistics.mean(latencies) if latencies else 0,
                    "p50": statistics.median(latencies) if latencies else 0,
                    "p90": latencies[int(len(latencies) * 0.9)] if latencies else 0,
                    "p99": latencies[int(len(latencies) * 0.99)] if latencies else 0,
                }
                for op, latencies in self.operation_latencies.items()
            },
            "active_spans": len(self.active_spans),
            "total_spans": len(self.completed_spans),
            "anomalies": [asdict(a) for a in self.get_anomalies()],
            "timestamp": datetime.now().isoformat()
        }
        
        return json.dumps(data, indent=2)


# Global tracer instance
_tracer_instance = None


def get_tracer() -> BrowserUseTracer:
    """Get or create the global tracer instance"""
    global _tracer_instance
    if _tracer_instance is None:
        _tracer_instance = BrowserUseTracer()
    return _tracer_instance


def trace_browser_operation(operation_name: str):
    """Decorator for tracing browser operations"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            
            # Extract attributes from arguments
            attributes = {}
            if args and hasattr(args[0], '__class__'):
                attributes['class'] = args[0].__class__.__name__
            
            with tracer.trace_operation(operation_name, attributes):
                return func(*args, **kwargs)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracer()
            
            # Extract attributes from arguments
            attributes = {}
            if args and hasattr(args[0], '__class__'):
                attributes['class'] = args[0].__class__.__name__
            
            with tracer.trace_operation(operation_name, attributes):
                return await func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


def instrument_page(page: Page):
    """Instrument a Page instance with tracing"""
    tracer = get_tracer()
    tracer.register_page(page)
    
    # Wrap key methods
    original_methods = {}
    
    methods_to_trace = [
        'goto', 'click', 'type', 'screenshot', 'evaluate',
        'wait_for_selector', 'wait_for_navigation'
    ]
    
    for method_name in methods_to_trace:
        if hasattr(page, method_name):
            original_method = getattr(page, method_name)
            original_methods[method_name] = original_method
            
            traced_method = trace_browser_operation(f"page_{method_name}")(original_method)
            setattr(page, method_name, traced_method)
    
    return original_methods


def restore_page(page: Page, original_methods: Dict[str, Callable]):
    """Restore original methods to a Page instance"""
    for method_name, original_method in original_methods.items():
        setattr(page, method_name, original_method)


# Export public API
__all__ = [
    'BrowserUseTracer',
    'get_tracer',
    'trace_browser_operation',
    'instrument_page',
    'restore_page',
    'PerformanceMetrics',
    'Anomaly',
    'LiveDashboard',
    'PrometheusMetrics'
]