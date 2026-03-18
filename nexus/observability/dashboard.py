"""
Real-time Observability Dashboard for nexus
Provides comprehensive monitoring with live metrics, debugging tools, and performance analytics.
"""

import asyncio
import json
import time
import uuid
import threading
import statistics
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable, Set
from enum import Enum
from collections import deque, defaultdict
import logging
import weakref
import psutil
import os

# OpenTelemetry imports
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Status, StatusCode

# Prometheus metrics
from prometheus_client import start_http_server, Counter, Histogram, Gauge, Summary

# WebSocket for live dashboard
import websockets
from websockets.server import WebSocketServerProtocol

# FastAPI for REST API (optional but recommended)
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics tracked"""
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    SUMMARY = "summary"


class AnomalyLevel(Enum):
    """Severity levels for anomalies"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class MetricDefinition:
    """Definition of a metric to track"""
    name: str
    metric_type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms
    percentiles: Optional[List[float]] = None  # For summaries


@dataclass
class Anomaly:
    """Detected anomaly"""
    id: str
    timestamp: float
    metric_name: str
    level: AnomalyLevel
    message: str
    current_value: float
    expected_range: tuple
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DashboardConfig:
    """Configuration for the observability dashboard"""
    websocket_port: int = 8765
    prometheus_port: int = 9090
    api_port: int = 8000
    enable_tracing: bool = True
    enable_metrics: bool = True
    enable_anomaly_detection: bool = True
    anomaly_detection_window: int = 300  # seconds
    anomaly_sensitivity: float = 2.0  # standard deviations
    metrics_export_interval: int = 15  # seconds
    trace_sample_rate: float = 1.0
    max_trace_spans: int = 10000
    dashboard_update_interval: float = 1.0  # seconds


class MetricsCollector:
    """Collects and manages metrics"""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.metrics: Dict[str, Any] = {}
        self.metric_definitions: Dict[str, MetricDefinition] = {}
        self.time_series: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._setup_prometheus_metrics()
        
    def _setup_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        self.prom_counters = {}
        self.prom_histograms = {}
        self.prom_gauges = {}
        self.prom_summaries = {}
        
        # Default metrics for nexus
        self.define_metric(MetricDefinition(
            name="browser_action_total",
            metric_type=MetricType.COUNTER,
            description="Total browser actions performed",
            labels=["action_type", "status"]
        ))
        
        self.define_metric(MetricDefinition(
            name="browser_action_duration_seconds",
            metric_type=MetricType.HISTOGRAM,
            description="Duration of browser actions",
            labels=["action_type"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        ))
        
        self.define_metric(MetricDefinition(
            name="browser_page_load_duration_seconds",
            metric_type=MetricType.HISTOGRAM,
            description="Page load duration",
            labels=["domain"],
            buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
        ))
        
        self.define_metric(MetricDefinition(
            name="browser_element_interaction_duration_seconds",
            metric_type=MetricType.HISTOGRAM,
            description="Element interaction duration",
            labels=["element_type", "action"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
        ))
        
        self.define_metric(MetricDefinition(
            name="browser_agent_task_duration_seconds",
            metric_type=MetricType.HISTOGRAM,
            description="Agent task completion duration",
            labels=["task_type", "status"],
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]
        ))
        
        self.define_metric(MetricDefinition(
            name="browser_success_rate",
            metric_type=MetricType.GAUGE,
            description="Success rate of browser operations",
            labels=["operation_type"]
        ))
        
        self.define_metric(MetricDefinition(
            name="browser_error_total",
            metric_type=MetricType.COUNTER,
            description="Total errors encountered",
            labels=["error_type", "component"]
        ))
        
        self.define_metric(MetricDefinition(
            name="browser_resource_usage",
            metric_type=MetricType.GAUGE,
            description="Resource usage metrics",
            labels=["resource_type", "process"]
        ))
    
    def define_metric(self, definition: MetricDefinition):
        """Define a new metric"""
        self.metric_definitions[definition.name] = definition
        
        if definition.metric_type == MetricType.COUNTER:
            self.prom_counters[definition.name] = Counter(
                definition.name,
                definition.description,
                definition.labels
            )
        elif definition.metric_type == MetricType.HISTOGRAM:
            self.prom_histograms[definition.name] = Histogram(
                definition.name,
                definition.description,
                definition.labels,
                buckets=definition.buckets or Histogram.DEFAULT_BUCKETS
            )
        elif definition.metric_type == MetricType.GAUGE:
            self.prom_gauges[definition.name] = Gauge(
                definition.name,
                definition.description,
                definition.labels
            )
        elif definition.metric_type == MetricType.SUMMARY:
            self.prom_summaries[definition.name] = Summary(
                definition.name,
                definition.description,
                definition.labels
            )
    
    def record_metric(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Record a metric value"""
        labels = labels or {}
        label_tuple = tuple(sorted(labels.items()))
        
        if name not in self.metrics:
            self.metrics[name] = {}
        
        if label_tuple not in self.metrics[name]:
            self.metrics[name][label_tuple] = {
                'values': deque(maxlen=1000),
                'timestamps': deque(maxlen=1000),
                'count': 0,
                'sum': 0.0
            }
        
        metric_data = self.metrics[name][label_tuple]
        metric_data['values'].append(value)
        metric_data['timestamps'].append(time.time())
        metric_data['count'] += 1
        metric_data['sum'] += value
        
        # Update time series for anomaly detection
        self.time_series[name].append({
            'timestamp': time.time(),
            'value': value,
            'labels': labels
        })
        
        # Update Prometheus metrics
        if name in self.prom_counters:
            self.prom_counters[name].labels(**labels).inc(value)
        elif name in self.prom_histograms:
            self.prom_histograms[name].labels(**labels).observe(value)
        elif name in self.prom_gauges:
            self.prom_gauges[name].labels(**labels).set(value)
        elif name in self.prom_summaries:
            self.prom_summaries[name].labels(**labels).observe(value)
    
    def get_metric_stats(self, name: str, labels: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Get statistics for a metric"""
        if name not in self.metrics:
            return {}
        
        if labels:
            label_tuple = tuple(sorted(labels.items()))
            if label_tuple not in self.metrics[name]:
                return {}
            
            data = self.metrics[name][label_tuple]
            values = list(data['values'])
            
            if not values:
                return {}
            
            return {
                'count': data['count'],
                'sum': data['sum'],
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'min': min(values),
                'max': max(values),
                'stddev': statistics.stdev(values) if len(values) > 1 else 0,
                'p95': self._percentile(values, 95),
                'p99': self._percentile(values, 99),
                'rate': self._calculate_rate(data['timestamps'])
            }
        
        # Aggregate across all labels
        all_values = []
        total_count = 0
        total_sum = 0.0
        
        for label_data in self.metrics[name].values():
            all_values.extend(list(label_data['values']))
            total_count += label_data['count']
            total_sum += label_data['sum']
        
        if not all_values:
            return {}
        
        return {
            'count': total_count,
            'sum': total_sum,
            'mean': statistics.mean(all_values),
            'median': statistics.median(all_values),
            'min': min(all_values),
            'max': max(all_values),
            'stddev': statistics.stdev(all_values) if len(all_values) > 1 else 0,
            'p95': self._percentile(all_values, 95),
            'p99': self._percentile(all_values, 99)
        }
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        lower = int(index)
        upper = lower + 1
        if upper >= len(sorted_data):
            return sorted_data[lower]
        weight = index - lower
        return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight
    
    def _calculate_rate(self, timestamps: deque) -> float:
        """Calculate rate per second"""
        if len(timestamps) < 2:
            return 0.0
        
        time_window = 60  # seconds
        now = time.time()
        recent = [ts for ts in timestamps if now - ts <= time_window]
        
        if len(recent) < 2:
            return 0.0
        
        return len(recent) / time_window
    
    def get_all_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        summary = {}
        for name in self.metrics:
            summary[name] = self.get_metric_stats(name)
        return summary


class TracingManager:
    """Manages distributed tracing with OpenTelemetry"""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.tracer = None
        self.active_spans = weakref.WeakSet()
        self.span_history = deque(maxlen=config.max_trace_spans)
        
        if config.enable_tracing:
            self._setup_tracing()
    
    def _setup_tracing(self):
        """Initialize OpenTelemetry tracing"""
        resource = Resource.create({
            "service.name": "nexus",
            "service.version": "1.0.0"
        })
        
        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)
        
        # Add console exporter for debugging
        tracer_provider.add_span_processor(
            BatchSpanProcessor(ConsoleSpanExporter())
        )
        
        self.tracer = trace.get_tracer(__name__)
    
    def start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> Any:
        """Start a new span"""
        if not self.config.enable_tracing or not self.tracer:
            return None
        
        span = self.tracer.start_span(name)
        
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        
        self.active_spans.add(span)
        self.span_history.append({
            'span_id': format(span.get_span_context().span_id, '016x'),
            'trace_id': format(span.get_span_context().trace_id, '032x'),
            'name': name,
            'start_time': time.time(),
            'attributes': attributes or {}
        })
        
        return span
    
    def end_span(self, span: Any, status: Status = Status(StatusCode.OK), error: Optional[Exception] = None):
        """End a span"""
        if not span:
            return
        
        if error:
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.record_exception(error)
        else:
            span.set_status(status)
        
        span.end()
        
        # Update span history
        for span_data in self.span_history:
            if span_data.get('span_id') == format(span.get_span_context().span_id, '016x'):
                span_data['end_time'] = time.time()
                span_data['duration'] = span_data['end_time'] - span_data['start_time']
                span_data['status'] = status.status_code.name
                break
    
    def get_active_spans(self) -> List[Dict[str, Any]]:
        """Get information about active spans"""
        active = []
        for span in self.active_spans:
            if span.is_recording():
                active.append({
                    'span_id': format(span.get_span_context().span_id, '016x'),
                    'trace_id': format(span.get_span_context().trace_id, '032x'),
                    'name': span.name,
                    'start_time': span.start_time,
                    'attributes': dict(span.attributes) if span.attributes else {}
                })
        return active
    
    def get_span_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent span history"""
        return list(self.span_history)[-limit:]


class AnomalyDetector:
    """Detects anomalies in metrics using statistical methods"""
    
    def __init__(self, config: DashboardConfig, metrics_collector: MetricsCollector):
        self.config = config
        self.metrics_collector = metrics_collector
        self.anomalies: List[Anomaly] = []
        self.anomaly_callbacks: List[Callable[[Anomaly], None]] = []
        self._baseline_stats: Dict[str, Dict[str, float]] = {}
        
    def register_callback(self, callback: Callable[[Anomaly], None]):
        """Register a callback for anomaly notifications"""
        self.anomaly_callbacks.append(callback)
    
    def detect_anomalies(self):
        """Run anomaly detection on all metrics"""
        if not self.config.enable_anomaly_detection:
            return
        
        current_time = time.time()
        window_start = current_time - self.config.anomaly_detection_window
        
        for metric_name, time_series in self.metrics_collector.time_series.items():
            # Filter to recent data
            recent_data = [
                point for point in time_series
                if point['timestamp'] >= window_start
            ]
            
            if len(recent_data) < 10:  # Need enough data points
                continue
            
            values = [point['value'] for point in recent_data]
            
            # Calculate baseline statistics
            mean = statistics.mean(values)
            stddev = statistics.stdev(values) if len(values) > 1 else 0
            
            # Check latest value
            latest = recent_data[-1]
            latest_value = latest['value']
            
            # Simple anomaly detection: check if value is outside N standard deviations
            if stddev > 0:
                z_score = abs(latest_value - mean) / stddev
                
                if z_score > self.config.anomaly_sensitivity:
                    anomaly = Anomaly(
                        id=str(uuid.uuid4()),
                        timestamp=current_time,
                        metric_name=metric_name,
                        level=AnomalyLevel.WARNING if z_score < 3 else AnomalyLevel.CRITICAL,
                        message=f"Anomaly detected in {metric_name}: value {latest_value:.2f} is {z_score:.1f} standard deviations from mean {mean:.2f}",
                        current_value=latest_value,
                        expected_range=(mean - self.config.anomaly_sensitivity * stddev,
                                      mean + self.config.anomaly_sensitivity * stddev),
                        metadata={
                            'z_score': z_score,
                            'mean': mean,
                            'stddev': stddev,
                            'labels': latest.get('labels', {})
                        }
                    )
                    
                    self.anomalies.append(anomaly)
                    
                    # Notify callbacks
                    for callback in self.anomaly_callbacks:
                        try:
                            callback(anomaly)
                        except Exception as e:
                            logger.error(f"Error in anomaly callback: {e}")
        
        # Keep only recent anomalies
        self.anomalies = [
            a for a in self.anomalies
            if current_time - a.timestamp < self.config.anomaly_detection_window
        ]
    
    def get_anomalies(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent anomalies"""
        return [asdict(a) for a in self.anomalies[-limit:]]


class ResourceMonitor:
    """Monitors system resource usage"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.process = psutil.Process(os.getpid())
        self._monitoring = False
        self._monitor_thread = None
    
    def start_monitoring(self, interval: float = 5.0):
        """Start resource monitoring"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
    
    def _monitor_loop(self, interval: float):
        """Monitor loop"""
        while self._monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.metrics_collector.record_metric(
                    "browser_resource_usage",
                    cpu_percent,
                    {"resource_type": "cpu", "process": "nexus"}
                )
                
                # Memory usage
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                self.metrics_collector.record_metric(
                    "browser_resource_usage",
                    memory_mb,
                    {"resource_type": "memory_mb", "process": "nexus"}
                )
                
                # Thread count
                thread_count = self.process.num_threads()
                self.metrics_collector.record_metric(
                    "browser_resource_usage",
                    thread_count,
                    {"resource_type": "threads", "process": "nexus"}
                )
                
                # File descriptors
                try:
                    fd_count = self.process.num_fds()
                    self.metrics_collector.record_metric(
                        "browser_resource_usage",
                        fd_count,
                        {"resource_type": "file_descriptors", "process": "nexus"}
                    )
                except AttributeError:
                    # Windows doesn't have num_fds
                    pass
                
            except Exception as e:
                logger.error(f"Error monitoring resources: {e}")
            
            time.sleep(interval)


class DashboardServer:
    """WebSocket server for real-time dashboard updates"""
    
    def __init__(self, config: DashboardConfig, metrics_collector: MetricsCollector,
                 tracing_manager: TracingManager, anomaly_detector: AnomalyDetector):
        self.config = config
        self.metrics_collector = metrics_collector
        self.tracing_manager = tracing_manager
        self.anomaly_detector = anomaly_detector
        self.connected_clients: Set[WebSocket] = set()
        self._running = False
        self._update_task = None
        
        # FastAPI app for REST API and WebSocket
        self.app = FastAPI(title="Browser-Use Observability Dashboard")
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/")
        async def get_dashboard():
            """Serve dashboard HTML"""
            return HTMLResponse(self._get_dashboard_html())
        
        @self.app.get("/api/metrics")
        async def get_metrics():
            """Get all metrics"""
            return self.metrics_collector.get_all_metrics_summary()
        
        @self.app.get("/api/metrics/{metric_name}")
        async def get_metric(metric_name: str):
            """Get specific metric"""
            stats = self.metrics_collector.get_metric_stats(metric_name)
            if not stats:
                raise HTTPException(status_code=404, detail="Metric not found")
            return stats
        
        @self.app.get("/api/traces")
        async def get_traces():
            """Get trace information"""
            return {
                "active_spans": self.tracing_manager.get_active_spans(),
                "recent_spans": self.tracing_manager.get_span_history(50)
            }
        
        @self.app.get("/api/anomalies")
        async def get_anomalies():
            """Get detected anomalies"""
            return self.anomaly_detector.get_anomalies()
        
        @self.app.get("/api/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "metrics_count": len(self.metrics_collector.metrics),
                "active_spans": len(self.tracing_manager.active_spans),
                "anomalies": len(self.anomaly_detector.anomalies)
            }
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            await websocket.accept()
            self.connected_clients.add(websocket)
            
            try:
                while True:
                    # Keep connection alive
                    data = await websocket.receive_text()
                    # Handle any incoming messages if needed
            except WebSocketDisconnect:
                self.connected_clients.remove(websocket)
    
    def _get_dashboard_html(self) -> str:
        """Generate dashboard HTML"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Browser-Use Observability Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; }
                .card { background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .metric-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }
                .metric-card { background: #f8f9fa; border-radius: 6px; padding: 15px; }
                .metric-title { font-weight: bold; margin-bottom: 10px; color: #333; }
                .metric-value { font-size: 24px; font-weight: bold; color: #007bff; }
                .metric-stats { font-size: 12px; color: #666; margin-top: 5px; }
                .status-indicator { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; }
                .status-healthy { background: #28a745; }
                .status-warning { background: #ffc107; }
                .status-critical { background: #dc3545; }
                .anomaly-list { max-height: 300px; overflow-y: auto; }
                .anomaly-item { padding: 10px; border-left: 4px solid; margin-bottom: 5px; }
                .anomaly-warning { border-color: #ffc107; background: #fff3cd; }
                .anomaly-critical { border-color: #dc3545; background: #f8d7da; }
                .trace-table { width: 100%; border-collapse: collapse; }
                .trace-table th, .trace-table td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                .trace-table tr:hover { background: #f5f5f5; }
                .refresh-btn { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
                .refresh-btn:hover { background: #0056b3; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Browser-Use Observability Dashboard</h1>
                <p>Real-time monitoring and performance analytics</p>
                
                <div class="card">
                    <h2>System Status</h2>
                    <div id="system-status">Loading...</div>
                </div>
                
                <div class="card">
                    <h2>Key Metrics</h2>
                    <div class="metric-grid" id="metrics-grid">
                        Loading metrics...
                    </div>
                </div>
                
                <div class="card">
                    <h2>Active Traces</h2>
                    <div id="active-traces">
                        Loading traces...
                    </div>
                </div>
                
                <div class="card">
                    <h2>Detected Anomalies</h2>
                    <div class="anomaly-list" id="anomalies-list">
                        Loading anomalies...
                    </div>
                </div>
                
                <button class="refresh-btn" onclick="refreshDashboard()">Refresh Dashboard</button>
            </div>
            
            <script>
                let ws = null;
                let reconnectAttempts = 0;
                const maxReconnectAttempts = 5;
                
                function connectWebSocket() {
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    const wsUrl = `${protocol}//${window.location.host}/ws`;
                    
                    ws = new WebSocket(wsUrl);
                    
                    ws.onopen = function() {
                        console.log('WebSocket connected');
                        reconnectAttempts = 0;
                        updateConnectionStatus('connected');
                    };
                    
                    ws.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        handleUpdate(data);
                    };
                    
                    ws.onclose = function() {
                        console.log('WebSocket disconnected');
                        updateConnectionStatus('disconnected');
                        
                        // Attempt to reconnect
                        if (reconnectAttempts < maxReconnectAttempts) {
                            reconnectAttempts++;
                            setTimeout(connectWebSocket, 2000 * reconnectAttempts);
                        }
                    };
                    
                    ws.onerror = function(error) {
                        console.error('WebSocket error:', error);
                    };
                }
                
                function updateConnectionStatus(status) {
                    const statusElement = document.getElementById('system-status');
                    if (statusElement) {
                        const statusClass = status === 'connected' ? 'status-healthy' : 'status-warning';
                        statusElement.innerHTML = `
                            <span class="status-indicator ${statusClass}"></span>
                            WebSocket: ${status}
                            <br>Last update: ${new Date().toLocaleTimeString()}
                        `;
                    }
                }
                
                function handleUpdate(data) {
                    if (data.type === 'metrics_update') {
                        updateMetrics(data.metrics);
                    } else if (data.type === 'anomaly') {
                        addAnomaly(data.anomaly);
                    } else if (data.type === 'trace_update') {
                        updateTraces(data.traces);
                    }
                }
                
                function updateMetrics(metrics) {
                    const grid = document.getElementById('metrics-grid');
                    if (!grid) return;
                    
                    let html = '';
                    for (const [name, stats] of Object.entries(metrics)) {
                        if (stats && stats.count > 0) {
                            html += `
                                <div class="metric-card">
                                    <div class="metric-title">${name}</div>
                                    <div class="metric-value">${stats.mean ? stats.mean.toFixed(2) : 'N/A'}</div>
                                    <div class="metric-stats">
                                        Count: ${stats.count} | 
                                        P95: ${stats.p95 ? stats.p95.toFixed(2) : 'N/A'} | 
                                        Rate: ${stats.rate ? stats.rate.toFixed(2) + '/s' : 'N/A'}
                                    </div>
                                </div>
                            `;
                        }
                    }
                    grid.innerHTML = html || '<p>No metrics available</p>';
                }
                
                function addAnomaly(anomaly) {
                    const list = document.getElementById('anomalies-list');
                    if (!list) return;
                    
                    const anomalyClass = anomaly.level === 'critical' ? 'anomaly-critical' : 'anomaly-warning';
                    const html = `
                        <div class="anomaly-item ${anomalyClass}">
                            <strong>${anomaly.metric_name}</strong><br>
                            ${anomaly.message}<br>
                            <small>${new Date(anomaly.timestamp * 1000).toLocaleString()}</small>
                        </div>
                    `;
                    
                    list.innerHTML = html + list.innerHTML;
                    
                    // Keep only last 20 anomalies
                    const items = list.querySelectorAll('.anomaly-item');
                    if (items.length > 20) {
                        items[items.length - 1].remove();
                    }
                }
                
                function updateTraces(traces) {
                    const container = document.getElementById('active-traces');
                    if (!container) return;
                    
                    let html = '<table class="trace-table"><thead><tr><th>Span ID</th><th>Name</th><th>Duration</th><th>Status</th></tr></thead><tbody>';
                    
                    for (const span of traces.active_spans) {
                        const duration = span.end_time ? 
                            ((span.end_time - span.start_time) * 1000).toFixed(2) + 'ms' : 
                            'Active';
                        html += `
                            <tr>
                                <td>${span.span_id.substring(0, 8)}...</td>
                                <td>${span.name}</td>
                                <td>${duration}</td>
                                <td>Active</td>
                            </tr>
                        `;
                    }
                    
                    html += '</tbody></table>';
                    container.innerHTML = html;
                }
                
                async function refreshDashboard() {
                    try {
                        // Fetch metrics
                        const metricsResponse = await fetch('/api/metrics');
                        const metrics = await metricsResponse.json();
                        updateMetrics(metrics);
                        
                        // Fetch traces
                        const tracesResponse = await fetch('/api/traces');
                        const traces = await tracesResponse.json();
                        updateTraces(traces);
                        
                        // Fetch anomalies
                        const anomaliesResponse = await fetch('/api/anomalies');
                        const anomalies = await anomaliesResponse.json();
                        
                        const list = document.getElementById('anomalies-list');
                        if (list) {
                            let html = '';
                            for (const anomaly of anomalies.reverse()) {
                                const anomalyClass = anomaly.level === 'critical' ? 'anomaly-critical' : 'anomaly-warning';
                                html += `
                                    <div class="anomaly-item ${anomalyClass}">
                                        <strong>${anomaly.metric_name}</strong><br>
                                        ${anomaly.message}<br>
                                        <small>${new Date(anomaly.timestamp * 1000).toLocaleString()}</small>
                                    </div>
                                `;
                            }
                            list.innerHTML = html || '<p>No anomalies detected</p>';
                        }
                        
                    } catch (error) {
                        console.error('Error refreshing dashboard:', error);
                    }
                }
                
                // Initialize
                document.addEventListener('DOMContentLoaded', function() {
                    connectWebSocket();
                    refreshDashboard();
                    
                    // Auto-refresh every 30 seconds
                    setInterval(refreshDashboard, 30000);
                });
            </script>
        </body>
        </html>
        """
    
    async def _broadcast_updates(self):
        """Broadcast updates to all connected clients"""
        while self._running:
            try:
                if self.connected_clients:
                    # Get current metrics
                    metrics = self.metrics_collector.get_all_metrics_summary()
                    
                    # Get active traces
                    traces = {
                        "active_spans": self.tracing_manager.get_active_spans(),
                        "recent_spans": self.tracing_manager.get_span_history(10)
                    }
                    
                    # Create update message
                    update = {
                        "type": "metrics_update",
                        "metrics": metrics,
                        "traces": traces,
                        "timestamp": time.time()
                    }
                    
                    # Broadcast to all clients
                    disconnected = set()
                    for client in self.connected_clients:
                        try:
                            await client.send_json(update)
                        except Exception:
                            disconnected.add(client)
                    
                    # Remove disconnected clients
                    self.connected_clients -= disconnected
                
                await asyncio.sleep(self.config.dashboard_update_interval)
                
            except Exception as e:
                logger.error(f"Error broadcasting updates: {e}")
                await asyncio.sleep(5)
    
    def start(self):
        """Start the dashboard server"""
        self._running = True
        
        # Start Prometheus metrics server
        if self.config.enable_metrics:
            start_http_server(self.config.prometheus_port)
            logger.info(f"Prometheus metrics server started on port {self.config.prometheus_port}")
        
        # Start WebSocket update task
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._update_task = loop.create_task(self._broadcast_updates())
        
        # Start FastAPI server in a separate thread
        import threading
        api_thread = threading.Thread(
            target=lambda: uvicorn.run(
                self.app,
                host="0.0.0.0",
                port=self.config.api_port,
                log_level="info"
            ),
            daemon=True
        )
        api_thread.start()
        
        logger.info(f"Dashboard API server started on port {self.config.api_port}")
        logger.info(f"Dashboard available at http://localhost:{self.config.api_port}")
    
    def stop(self):
        """Stop the dashboard server"""
        self._running = False
        if self._update_task:
            self._update_task.cancel()


class ObservabilityDashboard:
    """Main observability dashboard class"""
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        self.config = config or DashboardConfig()
        self.metrics_collector = MetricsCollector(self.config)
        self.tracing_manager = TracingManager(self.config)
        self.anomaly_detector = AnomalyDetector(self.config, self.metrics_collector)
        self.resource_monitor = ResourceMonitor(self.metrics_collector)
        self.dashboard_server = DashboardServer(
            self.config,
            self.metrics_collector,
            self.tracing_manager,
            self.anomaly_detector
        )
        
        # Register anomaly callback to log anomalies
        self.anomaly_detector.register_callback(self._log_anomaly)
        
        self._initialized = False
    
    def _log_anomaly(self, anomaly: Anomaly):
        """Log detected anomaly"""
        if anomaly.level == AnomalyLevel.CRITICAL:
            logger.critical(f"CRITICAL ANOMALY: {anomaly.message}")
        elif anomaly.level == AnomalyLevel.WARNING:
            logger.warning(f"ANOMALY: {anomaly.message}")
        else:
            logger.info(f"ANOMALY: {anomaly.message}")
    
    def initialize(self):
        """Initialize the observability dashboard"""
        if self._initialized:
            return
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring(interval=10.0)
        
        # Start dashboard server
        self.dashboard_server.start()
        
        # Start anomaly detection loop
        self._start_anomaly_detection()
        
        self._initialized = True
        logger.info("Observability dashboard initialized")
    
    def _start_anomaly_detection(self):
        """Start anomaly detection in background"""
        def detection_loop():
            while True:
                try:
                    self.anomaly_detector.detect_anomalies()
                except Exception as e:
                    logger.error(f"Error in anomaly detection: {e}")
                time.sleep(30)  # Check every 30 seconds
        
        thread = threading.Thread(target=detection_loop, daemon=True)
        thread.start()
    
    def record_action(self, action_type: str, duration: float, success: bool, 
                     labels: Optional[Dict[str, str]] = None):
        """Record a browser action"""
        labels = labels or {}
        labels.update({
            "action_type": action_type,
            "status": "success" if success else "failure"
        })
        
        # Record counter
        self.metrics_collector.record_metric(
            "browser_action_total",
            1.0,
            labels
        )
        
        # Record duration
        self.metrics_collector.record_metric(
            "browser_action_duration_seconds",
            duration,
            {"action_type": action_type}
        )
        
        # Update success rate
        self._update_success_rate(action_type, success)
    
    def record_page_load(self, url: str, duration: float, success: bool):
        """Record page load metrics"""
        from urllib.parse import urlparse
        domain = urlparse(url).netloc or "unknown"
        
        self.metrics_collector.record_metric(
            "browser_page_load_duration_seconds",
            duration,
            {"domain": domain}
        )
        
        if not success:
            self.metrics_collector.record_metric(
                "browser_error_total",
                1.0,
                {"error_type": "page_load_failure", "component": "browser"}
            )
    
    def record_element_interaction(self, element_type: str, action: str, 
                                 duration: float, success: bool):
        """Record element interaction metrics"""
        self.metrics_collector.record_metric(
            "browser_element_interaction_duration_seconds",
            duration,
            {"element_type": element_type, "action": action}
        )
        
        if not success:
            self.metrics_collector.record_metric(
                "browser_error_total",
                1.0,
                {"error_type": "element_interaction_failure", "component": "actor"}
            )
    
    def record_agent_task(self, task_type: str, duration: float, success: bool):
        """Record agent task metrics"""
        self.metrics_collector.record_metric(
            "browser_agent_task_duration_seconds",
            duration,
            {"task_type": task_type, "status": "success" if success else "failure"}
        )
    
    def _update_success_rate(self, operation_type: str, success: bool):
        """Update success rate metric"""
        # This is a simplified implementation
        # In production, you'd want a more sophisticated calculation
        labels = {"operation_type": operation_type}
        
        # Get current stats
        stats = self.metrics_collector.get_metric_stats("browser_action_total", labels)
        if stats and stats['count'] > 0:
            # Calculate success rate (simplified)
            # In reality, you'd track successes and failures separately
            success_rate = 1.0 if success else 0.0  # Simplified
            self.metrics_collector.record_metric(
                "browser_success_rate",
                success_rate,
                labels
            )
    
    def start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> Any:
        """Start a tracing span"""
        return self.tracing_manager.start_span(name, attributes)
    
    def end_span(self, span: Any, status: Status = Status(StatusCode.OK), 
                error: Optional[Exception] = None):
        """End a tracing span"""
        self.tracing_manager.end_span(span, status, error)
    
    def get_dashboard_url(self) -> str:
        """Get the URL for the dashboard"""
        return f"http://localhost:{self.config.api_port}"
    
    def shutdown(self):
        """Shutdown the observability dashboard"""
        self.resource_monitor.stop_monitoring()
        self.dashboard_server.stop()
        self._initialized = False


# Global instance for easy access
_dashboard_instance: Optional[ObservabilityDashboard] = None


def get_dashboard() -> ObservabilityDashboard:
    """Get the global dashboard instance"""
    global _dashboard_instance
    if _dashboard_instance is None:
        _dashboard_instance = ObservabilityDashboard()
    return _dashboard_instance


def initialize_dashboard(config: Optional[DashboardConfig] = None) -> ObservabilityDashboard:
    """Initialize the global dashboard"""
    global _dashboard_instance
    _dashboard_instance = ObservabilityDashboard(config)
    _dashboard_instance.initialize()
    return _dashboard_instance


# Decorators for easy instrumentation
def trace_span(name: Optional[str] = None):
    """Decorator to trace function execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            dashboard = get_dashboard()
            span_name = name or f"{func.__module__}.{func.__name__}"
            span = dashboard.start_span(span_name)
            
            try:
                result = func(*args, **kwargs)
                dashboard.end_span(span)
                return result
            except Exception as e:
                dashboard.end_span(span, error=e)
                raise
        
        return wrapper
    return decorator


def record_metric(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator to record function execution as a metric"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            dashboard = get_dashboard()
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record success
                dashboard.metrics_collector.record_metric(
                    metric_name,
                    duration,
                    labels or {}
                )
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                
                # Record failure
                error_labels = (labels or {}).copy()
                error_labels["status"] = "error"
                dashboard.metrics_collector.record_metric(
                    metric_name,
                    duration,
                    error_labels
                )
                
                raise
        
        return wrapper
    return decorator


# Context manager for tracing
class traced_operation:
    """Context manager for tracing operations"""
    
    def __init__(self, operation_name: str, attributes: Optional[Dict[str, Any]] = None):
        self.operation_name = operation_name
        self.attributes = attributes or {}
        self.span = None
        self.dashboard = get_dashboard()
    
    def __enter__(self):
        self.span = self.dashboard.start_span(self.operation_name, self.attributes)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.dashboard.end_span(self.span, error=exc_val)
        else:
            self.dashboard.end_span(self.span)
        return False


# Example usage in existing modules:
# from nexus.observability.dashboard import get_dashboard, trace_span, record_metric
#
# @trace_span("browser_action")
# @record_metric("browser_action_duration_seconds", {"action_type": "click"})
# def perform_click(element):
#     # ... existing code ...
#     pass