# nexus/observability/metrics_collector.py

import asyncio
import time
import json
import logging
import statistics
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import weakref
import psutil
import os

# OpenTelemetry imports
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.metrics import set_meter_provider
from opentelemetry.sdk.resources import Resource

# Prometheus
from prometheus_client import start_http_server, Counter, Histogram, Gauge, Summary

# WebSocket for live dashboard
import websockets
from websockets.server import serve

logger = logging.getLogger(__name__)


class MetricType(Enum):
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    SUMMARY = "summary"


@dataclass
class MetricDefinition:
    name: str
    description: str
    metric_type: MetricType
    labels: List[str] = None
    buckets: List[float] = None  # For histograms


@dataclass
class PerformanceMetrics:
    timestamp: float
    success_rate: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    error_rate: float
    resource_usage: Dict[str, float]
    active_sessions: int
    actions_per_second: float


@dataclass
class Anomaly:
    timestamp: float
    metric_name: str
    value: float
    threshold: float
    severity: str  # "warning", "critical"
    description: str


class MetricsCollector:
    """
    Central observability system for nexus with real-time metrics,
    distributed tracing, and anomaly detection.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, 
                 service_name: str = "nexus",
                 enable_tracing: bool = True,
                 enable_metrics: bool = True,
                 enable_dashboard: bool = True,
                 prometheus_port: int = 8000,
                 websocket_port: int = 8001,
                 otlp_endpoint: Optional[str] = None,
                 sampling_rate: float = 1.0):
        
        if self._initialized:
            return
            
        self.service_name = service_name
        self.enable_tracing = enable_tracing
        self.enable_metrics = enable_metrics
        self.enable_dashboard = enable_dashboard
        self.prometheus_port = prometheus_port
        self.websocket_port = websocket_port
        self.otlp_endpoint = otlp_endpoint
        self.sampling_rate = sampling_rate
        
        # Metric storage
        self._metrics: Dict[str, Any] = {}
        self._metric_definitions: Dict[str, MetricDefinition] = {}
        self._latency_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._success_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Anomaly detection
        self._anomalies: List[Anomaly] = []
        self._anomaly_thresholds: Dict[str, Dict[str, float]] = {
            "latency_p95": {"warning": 5.0, "critical": 10.0},
            "success_rate": {"warning": 0.95, "critical": 0.90},
            "error_rate": {"warning": 0.05, "critical": 0.10}
        }
        
        # WebSocket connections
        self._websocket_clients: weakref.WeakSet = weakref.WeakSet()
        self._broadcast_queue = asyncio.Queue()
        
        # Resource monitoring
        self._process = psutil.Process(os.getpid())
        
        # Initialize components
        self._setup_tracing()
        self._setup_metrics()
        self._register_default_metrics()
        
        # Start background tasks
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="metrics_collector")
        self._running = True
        
        if self.enable_dashboard:
            self._start_dashboard()
        
        self._initialized = True
        logger.info(f"MetricsCollector initialized for {service_name}")
    
    def _setup_tracing(self):
        """Initialize OpenTelemetry tracing"""
        if not self.enable_tracing:
            return
            
        try:
            resource = Resource.create({"service.name": self.service_name})
            tracer_provider = TracerProvider(resource=resource)
            
            if self.otlp_endpoint:
                otlp_exporter = OTLPSpanExporter(endpoint=self.otlp_endpoint)
                span_processor = BatchSpanProcessor(otlp_exporter)
                tracer_provider.add_span_processor(span_processor)
            
            trace.set_tracer_provider(tracer_provider)
            self.tracer = trace.get_tracer(__name__)
            
            logger.info("OpenTelemetry tracing initialized")
        except Exception as e:
            logger.error(f"Failed to initialize tracing: {e}")
            self.enable_tracing = False
    
    def _setup_metrics(self):
        """Initialize Prometheus metrics"""
        if not self.enable_metrics:
            return
            
        try:
            # Start Prometheus HTTP server
            start_http_server(self.prometheus_port)
            logger.info(f"Prometheus metrics server started on port {self.prometheus_port}")
            
            # Initialize metric reader
            resource = Resource.create({"service.name": self.service_name})
            reader = PrometheusMetricReader(prefix="nexus_")
            meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
            set_meter_provider(meter_provider)
            self.meter = metrics.get_meter(__name__)
            
        except Exception as e:
            logger.error(f"Failed to initialize metrics: {e}")
            self.enable_metrics = False
    
    def _register_default_metrics(self):
        """Register default metrics for browser automation"""
        default_metrics = [
            MetricDefinition(
                name="browser_action_duration_seconds",
                description="Duration of browser actions in seconds",
                metric_type=MetricType.HISTOGRAM,
                labels=["action_type", "page_url", "success"],
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
            ),
            MetricDefinition(
                name="browser_action_total",
                description="Total number of browser actions",
                metric_type=MetricType.COUNTER,
                labels=["action_type", "success"]
            ),
            MetricDefinition(
                name="agent_step_duration_seconds",
                description="Duration of agent steps in seconds",
                metric_type=MetricType.HISTOGRAM,
                labels=["step_type", "success"],
                buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
            ),
            MetricDefinition(
                name="agent_step_total",
                description="Total number of agent steps",
                metric_type=MetricType.COUNTER,
                labels=["step_type", "success"]
            ),
            MetricDefinition(
                name="resource_usage_percent",
                description="Resource usage percentage",
                metric_type=MetricType.GAUGE,
                labels=["resource_type"]
            ),
            MetricDefinition(
                name="active_sessions",
                description="Number of active browser sessions",
                metric_type=MetricType.GAUGE
            ),
            MetricDefinition(
                name="success_rate",
                description="Success rate of operations",
                metric_type=MetricType.GAUGE,
                labels=["operation_type"]
            ),
            MetricDefinition(
                name="latency_percentile",
                description="Latency percentiles",
                metric_type=MetricType.GAUGE,
                labels=["operation_type", "percentile"]
            )
        ]
        
        for metric_def in default_metrics:
            self.register_metric(metric_def)
    
    def register_metric(self, metric_def: MetricDefinition):
        """Register a new metric"""
        if not self.enable_metrics:
            return
            
        self._metric_definitions[metric_def.name] = metric_def
        
        try:
            if metric_def.metric_type == MetricType.COUNTER:
                self._metrics[metric_def.name] = Counter(
                    metric_def.name,
                    metric_def.description,
                    metric_def.labels or []
                )
            elif metric_def.metric_type == MetricType.HISTOGRAM:
                self._metrics[metric_def.name] = Histogram(
                    metric_def.name,
                    metric_def.description,
                    metric_def.labels or [],
                    buckets=metric_def.buckets
                )
            elif metric_def.metric_type == MetricType.GAUGE:
                self._metrics[metric_def.name] = Gauge(
                    metric_def.name,
                    metric_def.description,
                    metric_def.labels or []
                )
            elif metric_def.metric_type == MetricType.SUMMARY:
                self._metrics[metric_def.name] = Summary(
                    metric_def.name,
                    metric_def.description,
                    metric_def.labels or []
                )
                
            logger.debug(f"Registered metric: {metric_def.name}")
        except Exception as e:
            logger.error(f"Failed to register metric {metric_def.name}: {e}")
    
    def record_metric(self, metric_name: str, value: float, labels: Dict[str, str] = None):
        """Record a metric value"""
        if not self.enable_metrics or metric_name not in self._metrics:
            return
            
        try:
            metric = self._metrics[metric_name]
            label_values = tuple(labels.values()) if labels else ()
            
            if metric_name in self._metric_definitions:
                metric_type = self._metric_definitions[metric_name].metric_type
                
                if metric_type == MetricType.COUNTER:
                    metric.labels(*label_values).inc(value)
                elif metric_type == MetricType.HISTOGRAM:
                    metric.labels(*label_values).observe(value)
                elif metric_type == MetricType.GAUGE:
                    metric.labels(*label_values).set(value)
                elif metric_type == MetricType.SUMMARY:
                    metric.labels(*label_values).observe(value)
            
            # Store for anomaly detection and dashboard
            self._store_metric_for_analysis(metric_name, value, labels)
            
        except Exception as e:
            logger.error(f"Failed to record metric {metric_name}: {e}")
    
    def _store_metric_for_analysis(self, metric_name: str, value: float, labels: Dict[str, str] = None):
        """Store metric values for analysis and anomaly detection"""
        label_key = json.dumps(labels, sort_keys=True) if labels else "default"
        key = f"{metric_name}:{label_key}"
        
        if "duration" in metric_name or "latency" in metric_name:
            self._latency_history[key].append(value)
        elif "success" in metric_name:
            self._success_history[key].append(value)
    
    def trace_operation(self, operation_name: str, attributes: Dict[str, Any] = None):
        """Decorator/context manager for tracing operations"""
        def decorator(func: Callable):
            async def async_wrapper(*args, **kwargs):
                if not self.enable_tracing:
                    return await func(*args, **kwargs)
                
                with self.tracer.start_as_current_span(operation_name) as span:
                    if attributes:
                        for key, value in attributes.items():
                            span.set_attribute(key, value)
                    
                    start_time = time.time()
                    try:
                        result = await func(*args, **kwargs)
                        span.set_attribute("success", True)
                        self.record_metric("browser_action_total", 1, {
                            "action_type": operation_name,
                            "success": "true"
                        })
                        return result
                    except Exception as e:
                        span.set_attribute("success", False)
                        span.set_attribute("error", str(e))
                        span.record_exception(e)
                        self.record_metric("browser_action_total", 1, {
                            "action_type": operation_name,
                            "success": "false"
                        })
                        raise
                    finally:
                        duration = time.time() - start_time
                        span.set_attribute("duration_seconds", duration)
                        self.record_metric("browser_action_duration_seconds", duration, {
                            "action_type": operation_name,
                            "success": "true" if 'result' in locals() else "false"
                        })
            
            def sync_wrapper(*args, **kwargs):
                if not self.enable_tracing:
                    return func(*args, **kwargs)
                
                with self.tracer.start_as_current_span(operation_name) as span:
                    if attributes:
                        for key, value in attributes.items():
                            span.set_attribute(key, value)
                    
                    start_time = time.time()
                    try:
                        result = func(*args, **kwargs)
                        span.set_attribute("success", True)
                        self.record_metric("browser_action_total", 1, {
                            "action_type": operation_name,
                            "success": "true"
                        })
                        return result
                    except Exception as e:
                        span.set_attribute("success", False)
                        span.set_attribute("error", str(e))
                        span.record_exception(e)
                        self.record_metric("browser_action_total", 1, {
                            "action_type": operation_name,
                            "success": "false"
                        })
                        raise
                    finally:
                        duration = time.time() - start_time
                        span.set_attribute("duration_seconds", duration)
                        self.record_metric("browser_action_duration_seconds", duration, {
                            "action_type": operation_name,
                            "success": "true" if 'result' in locals() else "false"
                        })
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def trace_span(self, span_name: str, attributes: Dict[str, Any] = None):
        """Context manager for tracing a span"""
        class SpanContext:
            def __init__(self, collector, name, attrs):
                self.collector = collector
                self.name = name
                self.attrs = attrs
                self.span = None
                self.start_time = None
            
            def __enter__(self):
                if not self.collector.enable_tracing:
                    return self
                
                self.span = self.collector.tracer.start_span(self.name)
                self.start_time = time.time()
                
                if self.attrs:
                    for key, value in self.attrs.items():
                        self.span.set_attribute(key, value)
                
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if not self.collector.enable_tracing or not self.span:
                    return
                
                duration = time.time() - self.start_time
                self.span.set_attribute("duration_seconds", duration)
                
                if exc_type:
                    self.span.set_attribute("success", False)
                    self.span.set_attribute("error", str(exc_val))
                    self.span.record_exception(exc_val)
                else:
                    self.span.set_attribute("success", True)
                
                self.span.end()
        
        return SpanContext(self, span_name, attributes)
    
    def update_resource_metrics(self):
        """Update resource usage metrics"""
        try:
            cpu_percent = self._process.cpu_percent()
            memory_info = self._process.memory_info()
            memory_percent = self._process.memory_percent()
            
            self.record_metric("resource_usage_percent", cpu_percent, {"resource_type": "cpu"})
            self.record_metric("resource_usage_percent", memory_percent, {"resource_type": "memory"})
            
            # Update for dashboard
            self._current_resources = {
                "cpu_percent": cpu_percent,
                "memory_mb": memory_info.rss / 1024 / 1024,
                "memory_percent": memory_percent,
                "threads": self._process.num_threads(),
                "open_files": len(self._process.open_files())
            }
            
        except Exception as e:
            logger.error(f"Failed to update resource metrics: {e}")
    
    def get_performance_summary(self, window_seconds: int = 60) -> PerformanceMetrics:
        """Get performance summary for the specified time window"""
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        # Calculate success rate
        success_rates = []
        for key, history in self._success_history.items():
            if history:
                recent = [v for i, v in enumerate(history) if i >= len(history) - 100]
                if recent:
                    success_rates.append(statistics.mean(recent))
        
        avg_success_rate = statistics.mean(success_rates) if success_rates else 1.0
        
        # Calculate latency percentiles
        latencies = []
        for key, history in self._latency_history.items():
            if history:
                recent = [v for i, v in enumerate(history) if i >= len(history) - 100]
                latencies.extend(recent)
        
        latencies_sorted = sorted(latencies)
        latency_p50 = statistics.median(latencies_sorted) if latencies_sorted else 0
        latency_p95 = self._percentile(latencies_sorted, 95) if latencies_sorted else 0
        latency_p99 = self._percentile(latencies_sorted, 99) if latencies_sorted else 0
        
        # Calculate error rate
        error_rate = 1.0 - avg_success_rate
        
        # Calculate actions per second (approximate)
        total_actions = sum(len(h) for h in self._success_history.values())
        actions_per_second = total_actions / window_seconds if window_seconds > 0 else 0
        
        return PerformanceMetrics(
            timestamp=current_time,
            success_rate=avg_success_rate,
            latency_p50=latency_p50,
            latency_p95=latency_p95,
            latency_p99=latency_p99,
            error_rate=error_rate,
            resource_usage=getattr(self, '_current_resources', {}),
            active_sessions=0,  # Will be updated by session manager
            actions_per_second=actions_per_second
        )
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of a list"""
        if not data:
            return 0.0
        k = (len(data) - 1) * (percentile / 100)
        f = int(k)
        c = k - f
        if f + 1 < len(data):
            return data[f] + c * (data[f + 1] - data[f])
        return data[f]
    
    def detect_anomalies(self, metrics: PerformanceMetrics) -> List[Anomaly]:
        """Detect anomalies in performance metrics"""
        anomalies = []
        
        # Check latency anomalies
        if metrics.latency_p95 > self._anomaly_thresholds["latency_p95"]["critical"]:
            anomalies.append(Anomaly(
                timestamp=metrics.timestamp,
                metric_name="latency_p95",
                value=metrics.latency_p95,
                threshold=self._anomaly_thresholds["latency_p95"]["critical"],
                severity="critical",
                description=f"Critical latency detected: {metrics.latency_p95:.2f}s"
            ))
        elif metrics.latency_p95 > self._anomaly_thresholds["latency_p95"]["warning"]:
            anomalies.append(Anomaly(
                timestamp=metrics.timestamp,
                metric_name="latency_p95",
                value=metrics.latency_p95,
                threshold=self._anomaly_thresholds["latency_p95"]["warning"],
                severity="warning",
                description=f"High latency detected: {metrics.latency_p95:.2f}s"
            ))
        
        # Check success rate anomalies
        if metrics.success_rate < self._anomaly_thresholds["success_rate"]["critical"]:
            anomalies.append(Anomaly(
                timestamp=metrics.timestamp,
                metric_name="success_rate",
                value=metrics.success_rate,
                threshold=self._anomaly_thresholds["success_rate"]["critical"],
                severity="critical",
                description=f"Critical success rate: {metrics.success_rate:.2%}"
            ))
        elif metrics.success_rate < self._anomaly_thresholds["success_rate"]["warning"]:
            anomalies.append(Anomaly(
                timestamp=metrics.timestamp,
                metric_name="success_rate",
                value=metrics.success_rate,
                threshold=self._anomaly_thresholds["success_rate"]["warning"],
                severity="warning",
                description=f"Low success rate: {metrics.success_rate:.2%}"
            ))
        
        # Check error rate anomalies
        if metrics.error_rate > self._anomaly_thresholds["error_rate"]["critical"]:
            anomalies.append(Anomaly(
                timestamp=metrics.timestamp,
                metric_name="error_rate",
                value=metrics.error_rate,
                threshold=self._anomaly_thresholds["error_rate"]["critical"],
                severity="critical",
                description=f"Critical error rate: {metrics.error_rate:.2%}"
            ))
        elif metrics.error_rate > self._anomaly_thresholds["error_rate"]["warning"]:
            anomalies.append(Anomaly(
                timestamp=metrics.timestamp,
                metric_name="error_rate",
                value=metrics.error_rate,
                threshold=self._anomaly_thresholds["error_rate"]["warning"],
                severity="warning",
                description=f"High error rate: {metrics.error_rate:.2%}"
            ))
        
        # Store anomalies
        self._anomalies.extend(anomalies)
        # Keep only last 100 anomalies
        if len(self._anomalies) > 100:
            self._anomalies = self._anomalies[-100:]
        
        return anomalies
    
    def _start_dashboard(self):
        """Start WebSocket dashboard server"""
        def run_websocket_server():
            asyncio.set_event_loop(asyncio.new_event_loop())
            loop = asyncio.get_event_loop()
            
            async def handler(websocket, path):
                self._websocket_clients.add(websocket)
                logger.info(f"Dashboard client connected. Total clients: {len(self._websocket_clients)}")
                
                try:
                    # Send initial data
                    await self._send_initial_data(websocket)
                    
                    # Keep connection alive and handle messages
                    async for message in websocket:
                        await self._handle_dashboard_message(websocket, message)
                        
                except websockets.exceptions.ConnectionClosed:
                    pass
                finally:
                    self._websocket_clients.discard(websocket)
                    logger.info(f"Dashboard client disconnected. Total clients: {len(self._websocket_clients)}")
            
            async def broadcast_metrics():
                while self._running:
                    try:
                        # Get performance metrics
                        metrics = self.get_performance_summary(window_seconds=30)
                        anomalies = self.detect_anomalies(metrics)
                        
                        # Prepare data for broadcast
                        data = {
                            "type": "metrics_update",
                            "timestamp": time.time(),
                            "metrics": asdict(metrics),
                            "anomalies": [asdict(a) for a in anomalies],
                            "resource_usage": getattr(self, '_current_resources', {})
                        }
                        
                        # Broadcast to all connected clients
                        if self._websocket_clients:
                            message = json.dumps(data)
                            disconnected = set()
                            
                            for client in self._websocket_clients:
                                try:
                                    await client.send(message)
                                except:
                                    disconnected.add(client)
                            
                            # Clean up disconnected clients
                            for client in disconnected:
                                self._websocket_clients.discard(client)
                        
                        # Update resource metrics
                        self.update_resource_metrics()
                        
                        # Wait before next update
                        await asyncio.sleep(2)  # Update every 2 seconds
                        
                    except Exception as e:
                        logger.error(f"Error in broadcast loop: {e}")
                        await asyncio.sleep(5)
            
            # Start servers
            start_server = serve(handler, "localhost", self.websocket_port)
            loop.run_until_complete(start_server)
            logger.info(f"WebSocket dashboard server started on port {self.websocket_port}")
            
            # Start broadcast task
            loop.create_task(broadcast_metrics())
            loop.run_forever()
        
        # Run in separate thread
        thread = threading.Thread(target=run_websocket_server, daemon=True)
        thread.start()
    
    async def _send_initial_data(self, websocket):
        """Send initial data to newly connected dashboard client"""
        try:
            # Send historical data (last 5 minutes)
            historical_data = {
                "type": "historical_data",
                "metrics_history": self._get_historical_metrics(300),  # 5 minutes
                "anomalies_history": [asdict(a) for a in self._anomalies[-50:]],  # Last 50 anomalies
                "metric_definitions": {
                    name: asdict(defn) 
                    for name, defn in self._metric_definitions.items()
                }
            }
            
            await websocket.send(json.dumps(historical_data))
            
        except Exception as e:
            logger.error(f"Failed to send initial data: {e}")
    
    def _get_historical_metrics(self, window_seconds: int) -> Dict[str, List]:
        """Get historical metrics for dashboard"""
        historical = {
            "timestamps": [],
            "success_rates": [],
            "latency_p95": [],
            "error_rates": [],
            "resource_usage": {
                "cpu": [],
                "memory": []
            }
        }
        
        # Generate sample historical data (in production, this would come from a time-series database)
        current_time = time.time()
        for i in range(window_seconds, 0, -10):  # Every 10 seconds
            timestamp = current_time - i
            historical["timestamps"].append(timestamp)
            
            # Simulate metrics (replace with actual historical data retrieval)
            historical["success_rates"].append(0.95 + (i % 10) * 0.005)
            historical["latency_p95"].append(1.0 + (i % 20) * 0.1)
            historical["error_rates"].append(0.05 - (i % 10) * 0.005)
            historical["resource_usage"]["cpu"].append(30 + (i % 15) * 2)
            historical["resource_usage"]["memory"].append(40 + (i % 10) * 3)
        
        return historical
    
    async def _handle_dashboard_message(self, websocket, message: str):
        """Handle incoming messages from dashboard client"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "get_details":
                # Send detailed metrics for a specific time range
                start_time = data.get("start_time", time.time() - 3600)
                end_time = data.get("end_time", time.time())
                
                details = {
                    "type": "details_response",
                    "time_range": {"start": start_time, "end": end_time},
                    "detailed_metrics": self._get_detailed_metrics(start_time, end_time)
                }
                
                await websocket.send(json.dumps(details))
            
            elif message_type == "update_thresholds":
                # Update anomaly detection thresholds
                thresholds = data.get("thresholds", {})
                for metric, values in thresholds.items():
                    if metric in self._anomaly_thresholds:
                        self._anomaly_thresholds[metric].update(values)
                
                await websocket.send(json.dumps({
                    "type": "thresholds_updated",
                    "success": True
                }))
                
        except Exception as e:
            logger.error(f"Failed to handle dashboard message: {e}")
    
    def _get_detailed_metrics(self, start_time: float, end_time: float) -> Dict[str, Any]:
        """Get detailed metrics for a specific time range"""
        # In production, this would query a time-series database
        # For now, return simulated data
        return {
            "actions_by_type": {
                "click": {"count": 150, "success_rate": 0.96, "avg_latency": 1.2},
                "type": {"count": 200, "success_rate": 0.98, "avg_latency": 0.8},
                "navigate": {"count": 50, "success_rate": 0.92, "avg_latency": 2.5},
                "extract": {"count": 300, "success_rate": 0.94, "avg_latency": 1.5}
            },
            "errors_by_type": {
                "timeout": 15,
                "element_not_found": 8,
                "network_error": 3,
                "javascript_error": 2
            },
            "performance_trends": {
                "hourly_success_rate": [0.95, 0.96, 0.94, 0.97, 0.93, 0.95],
                "hourly_latency_p95": [1.2, 1.1, 1.3, 1.0, 1.4, 1.2]
            }
        }
    
    def get_dashboard_url(self) -> str:
        """Get URL for the live dashboard"""
        return f"ws://localhost:{self.websocket_port}"
    
    def get_prometheus_url(self) -> str:
        """Get URL for Prometheus metrics"""
        return f"http://localhost:{self.prometheus_port}/metrics"
    
    def shutdown(self):
        """Shutdown the metrics collector"""
        self._running = False
        self._executor.shutdown(wait=False)
        logger.info("MetricsCollector shutdown complete")


# Global instance for easy access
metrics_collector = MetricsCollector()


def trace_browser_action(action_type: str):
    """Decorator for tracing browser actions"""
    return metrics_collector.trace_operation(
        f"browser_action_{action_type}",
        {"action_type": action_type}
    )


def trace_agent_step(step_type: str):
    """Decorator for tracing agent steps"""
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            with metrics_collector.trace_span(f"agent_step_{step_type}", {"step_type": step_type}):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    success = True
                    return result
                except Exception as e:
                    success = False
                    raise
                finally:
                    duration = time.time() - start_time
                    metrics_collector.record_metric("agent_step_duration_seconds", duration, {
                        "step_type": step_type,
                        "success": str(success).lower()
                    })
                    metrics_collector.record_metric("agent_step_total", 1, {
                        "step_type": step_type,
                        "success": str(success).lower()
                    })
        return wrapper
    return decorator


def record_metric(metric_name: str, value: float, labels: Dict[str, str] = None):
    """Helper function to record a metric"""
    metrics_collector.record_metric(metric_name, value, labels)


def get_performance_summary(window_seconds: int = 60) -> PerformanceMetrics:
    """Get performance summary"""
    return metrics_collector.get_performance_summary(window_seconds)


def get_dashboard_url() -> str:
    """Get dashboard URL"""
    return metrics_collector.get_dashboard_url()


# Integration with existing modules
def integrate_with_agent_service():
    """Integrate metrics collection with AgentService"""
    try:
        from nexus.agent.service import AgentService
        
        # Wrap key methods with tracing
        original_execute_step = AgentService.execute_step
        
        async def traced_execute_step(self, *args, **kwargs):
            with metrics_collector.trace_span("agent_execute_step", {"agent_id": str(id(self))}):
                return await original_execute_step(self, *args, **kwargs)
        
        AgentService.execute_step = traced_execute_step
        
        logger.info("Integrated metrics with AgentService")
    except ImportError:
        logger.warning("AgentService not available for integration")


def integrate_with_actor_page():
    """Integrate metrics collection with Page actor"""
    try:
        from nexus.actor.page import Page
        
        # Wrap navigation and interaction methods
        methods_to_trace = ['goto', 'click', 'fill', 'screenshot', 'evaluate']
        
        for method_name in methods_to_trace:
            if hasattr(Page, method_name):
                original_method = getattr(Page, method_name)
                
                if asyncio.iscoroutinefunction(original_method):
                    async def traced_method(self, *args, **kwargs):
                        with metrics_collector.trace_span(
                            f"page_{method_name}",
                            {"page_url": self.url if hasattr(self, 'url') else "unknown"}
                        ):
                            return await original_method(self, *args, **kwargs)
                    
                    setattr(Page, method_name, traced_method)
        
        logger.info("Integrated metrics with Page actor")
    except ImportError:
        logger.warning("Page actor not available for integration")


# Auto-integrate when module is imported
try:
    integrate_with_agent_service()
    integrate_with_actor_page()
except Exception as e:
    logger.error(f"Failed to auto-integrate metrics: {e}")