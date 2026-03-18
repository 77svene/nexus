"""
nexus/debugging/replay_engine.py

Visual Debugging & Replay System — Record every action with screenshots, DOM snapshots, and timing data.
Enables time-travel debugging, failure analysis, and one-click reproduction of bugs.
"""

import asyncio
import base64
import gzip
import hashlib
import io
import json
import logging
import os
import tempfile
import time
import uuid
import zipfile
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from PIL import Image
from playwright.async_api import ConsoleMessage, Page, Request, Response

from nexus.actor.page import PageActor
from nexus.agent.message_manager.service import MessageManager
from nexus.agent.message_manager.views import Message

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    ACTION_START = "action_start"
    ACTION_END = "action_end"
    NAVIGATION = "navigation"
    NETWORK_REQUEST = "network_request"
    NETWORK_RESPONSE = "network_response"
    CONSOLE_LOG = "console_log"
    DOM_MUTATION = "dom_mutation"
    AGENT_REASONING = "agent_reasoning"
    AGENT_DECISION = "agent_decision"
    ERROR = "error"
    SCREENSHOT = "screenshot"


@dataclass
class NetworkRequest:
    url: str
    method: str
    headers: Dict[str, str]
    post_data: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class NetworkResponse:
    url: str
    status: int
    headers: Dict[str, str]
    request_id: str
    timestamp: float = field(default_factory=time.time)
    response_time_ms: float = 0.0


@dataclass
class ConsoleLog:
    text: str
    type: str  # 'log', 'error', 'warning', 'info', 'debug'
    location: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class DOMSnapshot:
    html: str
    url: str
    timestamp: float = field(default_factory=time.time)
    compressed_html: Optional[bytes] = None
    
    def compress(self) -> bytes:
        """Compress HTML using gzip for storage efficiency."""
        if self.compressed_html:
            return self.compressed_html
        self.compressed_html = gzip.compress(self.html.encode('utf-8'))
        return self.compressed_html
    
    @classmethod
    def decompress(cls, compressed: bytes, url: str, timestamp: float) -> 'DOMSnapshot':
        """Create DOMSnapshot from compressed data."""
        html = gzip.decompress(compressed).decode('utf-8')
        return cls(html=html, url=url, timestamp=timestamp, compressed_html=compressed)


@dataclass
class Screenshot:
    image_data: bytes  # PNG format
    timestamp: float = field(default_factory=time.time)
    viewport_width: int = 0
    viewport_height: int = 0
    full_page: bool = False
    
    def to_base64(self) -> str:
        return base64.b64encode(self.image_data).decode('utf-8')


@dataclass
class AgentStep:
    step_id: str
    reasoning: str
    decision: str
    action: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)
    tokens_used: int = 0
    model: str = ""


@dataclass
class DebugEvent:
    event_id: str
    event_type: EventType
    timestamp: float
    data: Dict[str, Any]
    sequence_number: int
    action_id: Optional[str] = None
    parent_event_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionTrace:
    action_id: str
    action_type: str
    parameters: Dict[str, Any]
    start_time: float
    end_time: Optional[float] = None
    success: bool = True
    error: Optional[str] = None
    before_state: Optional[Dict[str, Any]] = None
    after_state: Optional[Dict[str, Any]] = None
    screenshots: List[Screenshot] = field(default_factory=list)
    dom_snapshots: List[DOMSnapshot] = field(default_factory=list)
    network_requests: List[NetworkRequest] = field(default_factory=list)
    network_responses: List[NetworkResponse] = field(default_factory=list)
    console_logs: List[ConsoleLog] = field(default_factory=list)
    agent_steps: List[AgentStep] = field(default_factory=list)
    events: List[DebugEvent] = field(default_factory=list)


class RingBuffer:
    """Fixed-size ring buffer for efficient memory usage."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)
        self.total_items = 0
    
    def append(self, item: Any) -> None:
        self.buffer.append(item)
        self.total_items += 1
    
    def get_all(self) -> List[Any]:
        return list(self.buffer)
    
    def get_last(self, n: int) -> List[Any]:
        if n >= len(self.buffer):
            return list(self.buffer)
        return list(self.buffer)[-n:]
    
    def clear(self) -> None:
        self.buffer.clear()
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def __iter__(self):
        return iter(self.buffer)


class ReplayEngine:
    """
    Visual debugging and replay system for browser automation.
    
    Features:
    - Ring buffer for efficient memory usage
    - Before/after screenshots with DOM snapshots
    - Network request/response tracking
    - Console log capture
    - Agent reasoning step recording
    - Time-travel debugging
    - One-click debug bundle generation
    """
    
    def __init__(
        self,
        page: Page,
        buffer_size: int = 1000,
        screenshot_quality: int = 80,
        capture_network: bool = True,
        capture_console: bool = True,
        capture_dom: bool = True,
        capture_screenshots: bool = True,
        storage_path: Optional[Path] = None,
        compress_dom: bool = True,
        max_screenshot_width: int = 1280,
        max_screenshot_height: int = 720,
    ):
        self.page = page
        self.buffer_size = buffer_size
        self.screenshot_quality = screenshot_quality
        self.capture_network = capture_network
        self.capture_console = capture_console
        self.capture_dom = capture_dom
        self.capture_screenshots = capture_screenshots
        self.compress_dom = compress_dom
        self.max_screenshot_width = max_screenshot_width
        self.max_screenshot_height = max_screenshot_height
        
        # Storage
        self.storage_path = storage_path or Path(tempfile.mkdtemp(prefix="nexus_debug_"))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Ring buffers for different data types
        self.action_traces: RingBuffer = RingBuffer(buffer_size)
        self.network_requests: RingBuffer = RingBuffer(buffer_size * 10)
        self.network_responses: RingBuffer = RingBuffer(buffer_size * 10)
        self.console_logs: RingBuffer = RingBuffer(buffer_size * 5)
        self.dom_snapshots: RingBuffer = RingBuffer(buffer_size)
        self.screenshots: RingBuffer = RingBuffer(buffer_size * 2)
        self.agent_steps: RingBuffer = RingBuffer(buffer_size * 3)
        self.events: RingBuffer = RingBuffer(buffer_size * 5)
        
        # Current action tracking
        self.current_action_id: Optional[str] = None
        self.current_action_start_time: Optional[float] = None
        self.sequence_counter: int = 0
        self.action_counter: int = 0
        
        # Event listeners
        self._listeners_initialized = False
        self._request_ids: Dict[str, str] = {}  # Maps request to action_id
        
        # Message manager integration
        self.message_manager: Optional[MessageManager] = None
        
        logger.info(f"ReplayEngine initialized with buffer_size={buffer_size}, storage_path={self.storage_path}")
    
    async def initialize(self) -> None:
        """Initialize event listeners on the page."""
        if self._listeners_initialized:
            return
        
        # Network listeners
        if self.capture_network:
            self.page.on("request", self._on_request)
            self.page.on("response", self._on_response)
            self.page.on("requestfailed", self._on_request_failed)
        
        # Console listener
        if self.capture_console:
            self.page.on("console", self._on_console)
        
        # DOM mutation observer (simplified)
        if self.capture_dom:
            await self._setup_dom_observer()
        
        self._listeners_initialized = True
        logger.debug("ReplayEngine event listeners initialized")
    
    async def _setup_dom_observer(self) -> None:
        """Set up a DOM mutation observer."""
        try:
            # Inject a mutation observer script
            await self.page.evaluate("""
                () => {
                    if (!window.__browserUseDebugObserver) {
                        const observer = new MutationObserver((mutations) => {
                            window.__browserUseDebugMutations = window.__browserUseDebugMutations || [];
                            mutations.forEach(mutation => {
                                window.__browserUseDebugMutations.push({
                                    type: mutation.type,
                                    target: mutation.target.tagName,
                                    addedNodes: mutation.addedNodes.length,
                                    removedNodes: mutation.removedNodes.length,
                                    timestamp: Date.now()
                                });
                            });
                            // Keep only last 100 mutations
                            if (window.__browserUseDebugMutations.length > 100) {
                                window.__browserUseDebugMutations = window.__browserUseDebugMutations.slice(-100);
                            }
                        });
                        observer.observe(document.body, {
                            childList: true,
                            subtree: true,
                            attributes: true,
                            characterData: true
                        });
                        window.__browserUseDebugObserver = observer;
                    }
                }
            """)
        except Exception as e:
            logger.warning(f"Failed to set up DOM observer: {e}")
    
    def _generate_id(self, prefix: str = "") -> str:
        """Generate a unique ID."""
        return f"{prefix}{uuid.uuid4().hex[:12]}"
    
    async def _capture_screenshot(self, full_page: bool = False) -> Optional[Screenshot]:
        """Capture a screenshot of the current page."""
        if not self.capture_screenshots:
            return None
        
        try:
            # Get viewport size
            viewport = self.page.viewport_size
            if not viewport:
                viewport = {"width": 1280, "height": 720}
            
            # Take screenshot
            screenshot_bytes = await self.page.screenshot(
                type="png",
                quality=self.screenshot_quality,
                full_page=full_page,
                animations="disabled"
            )
            
            # Resize if needed
            if viewport["width"] > self.max_screenshot_width or viewport["height"] > self.max_screenshot_height:
                img = Image.open(io.BytesIO(screenshot_bytes))
                img.thumbnail((self.max_screenshot_width, self.max_screenshot_height), Image.Resampling.LANCZOS)
                buffer = io.BytesIO()
                img.save(buffer, format="PNG", optimize=True)
                screenshot_bytes = buffer.getvalue()
            
            return Screenshot(
                image_data=screenshot_bytes,
                viewport_width=viewport["width"],
                viewport_height=viewport["height"],
                full_page=full_page
            )
        except Exception as e:
            logger.error(f"Failed to capture screenshot: {e}")
            return None
    
    async def _capture_dom_snapshot(self) -> Optional[DOMSnapshot]:
        """Capture a DOM snapshot of the current page."""
        if not self.capture_dom:
            return None
        
        try:
            html = await self.page.content()
            url = self.page.url
            snapshot = DOMSnapshot(html=html, url=url)
            
            if self.compress_dom:
                snapshot.compress()
            
            return snapshot
        except Exception as e:
            logger.error(f"Failed to capture DOM snapshot: {e}")
            return None
    
    def _on_request(self, request: Request) -> None:
        """Handle network request event."""
        try:
            action_id = self.current_action_id or self._generate_id("req_")
            self._request_ids[request.url] = action_id
            
            network_request = NetworkRequest(
                url=request.url,
                method=request.method,
                headers=dict(request.headers),
                post_data=request.post_data
            )
            
            self.network_requests.append(network_request)
            
            event = DebugEvent(
                event_id=self._generate_id("evt_"),
                event_type=EventType.NETWORK_REQUEST,
                timestamp=time.time(),
                data=asdict(network_request),
                sequence_number=self.sequence_counter,
                action_id=action_id
            )
            self.events.append(event)
            self.sequence_counter += 1
            
        except Exception as e:
            logger.error(f"Error handling request: {e}")
    
    def _on_response(self, response: Response) -> None:
        """Handle network response event."""
        try:
            request = response.request
            action_id = self._request_ids.get(request.url, self.current_action_id or self._generate_id("resp_"))
            
            # Calculate response time
            response_time = 0.0
            if hasattr(request, 'timing'):
                timing = request.timing
                if timing and 'responseEnd' in timing and 'requestStart' in timing:
                    response_time = (timing['responseEnd'] - timing['requestStart']) * 1000
            
            network_response = NetworkResponse(
                url=response.url,
                status=response.status,
                headers=dict(response.headers),
                request_id=request.url,
                response_time_ms=response_time
            )
            
            self.network_responses.append(network_response)
            
            event = DebugEvent(
                event_id=self._generate_id("evt_"),
                event_type=EventType.NETWORK_RESPONSE,
                timestamp=time.time(),
                data=asdict(network_response),
                sequence_number=self.sequence_counter,
                action_id=action_id
            )
            self.events.append(event)
            self.sequence_counter += 1
            
        except Exception as e:
            logger.error(f"Error handling response: {e}")
    
    def _on_request_failed(self, request: Request) -> None:
        """Handle failed network request."""
        try:
            action_id = self._request_ids.get(request.url, self.current_action_id)
            
            event = DebugEvent(
                event_id=self._generate_id("evt_"),
                event_type=EventType.ERROR,
                timestamp=time.time(),
                data={
                    "type": "network_request_failed",
                    "url": request.url,
                    "method": request.method,
                    "error": str(request.failure) if request.failure else "Unknown error"
                },
                sequence_number=self.sequence_counter,
                action_id=action_id
            )
            self.events.append(event)
            self.sequence_counter += 1
            
        except Exception as e:
            logger.error(f"Error handling request failure: {e}")
    
    def _on_console(self, message: ConsoleMessage) -> None:
        """Handle console message event."""
        try:
            console_log = ConsoleLog(
                text=message.text,
                type=message.type,
                location=f"{message.location.get('url', '')}:{message.location.get('lineNumber', '')}"
            )
            
            self.console_logs.append(console_log)
            
            event = DebugEvent(
                event_id=self._generate_id("evt_"),
                event_type=EventType.CONSOLE_LOG,
                timestamp=time.time(),
                data=asdict(console_log),
                sequence_number=self.sequence_counter,
                action_id=self.current_action_id
            )
            self.events.append(event)
            self.sequence_counter += 1
            
        except Exception as e:
            logger.error(f"Error handling console message: {e}")
    
    async def start_action(
        self,
        action_type: str,
        parameters: Dict[str, Any],
        agent_step: Optional[AgentStep] = None
    ) -> str:
        """
        Start recording an action.
        
        Returns:
            action_id: Unique identifier for this action
        """
        action_id = self._generate_id("act_")
        self.current_action_id = action_id
        self.current_action_start_time = time.time()
        self.action_counter += 1
        
        # Capture before state
        before_screenshot = await self._capture_screenshot()
        before_dom = await self._capture_dom_snapshot()
        
        # Create action trace
        action_trace = ActionTrace(
            action_id=action_id,
            action_type=action_type,
            parameters=parameters,
            start_time=self.current_action_start_time,
            before_state={
                "url": self.page.url,
                "title": await self.page.title() if hasattr(self.page, 'title') else "",
                "timestamp": self.current_action_start_time
            }
        )
        
        if before_screenshot:
            action_trace.screenshots.append(before_screenshot)
        
        if before_dom:
            action_trace.dom_snapshots.append(before_dom)
        
        if agent_step:
            action_trace.agent_steps.append(agent_step)
            self.agent_steps.append(agent_step)
        
        # Create start event
        event = DebugEvent(
            event_id=self._generate_id("evt_"),
            event_type=EventType.ACTION_START,
            timestamp=self.current_action_start_time,
            data={
                "action_type": action_type,
                "parameters": parameters,
                "before_url": self.page.url
            },
            sequence_number=self.sequence_counter,
            action_id=action_id
        )
        action_trace.events.append(event)
        self.events.append(event)
        self.sequence_counter += 1
        
        # Store in buffer
        self.action_traces.append(action_trace)
        
        logger.debug(f"Started recording action {action_id}: {action_type}")
        return action_id
    
    async def end_action(
        self,
        action_id: str,
        success: bool = True,
        error: Optional[str] = None,
        result: Optional[Any] = None
    ) -> None:
        """
        End recording an action.
        """
        if not self.current_action_id or self.current_action_id != action_id:
            logger.warning(f"Action {action_id} not found or not current")
            return
        
        end_time = time.time()
        
        # Find the action trace
        action_trace = None
        for trace in self.action_traces.get_all():
            if trace.action_id == action_id:
                action_trace = trace
                break
        
        if not action_trace:
            logger.error(f"Action trace not found for {action_id}")
            return
        
        # Capture after state
        after_screenshot = await self._capture_screenshot()
        after_dom = await self._capture_dom_snapshot()
        
        # Update action trace
        action_trace.end_time = end_time
        action_trace.success = success
        action_trace.error = error
        action_trace.after_state = {
            "url": self.page.url,
            "title": await self.page.title() if hasattr(self.page, 'title') else "",
            "timestamp": end_time,
            "result": result
        }
        
        if after_screenshot:
            action_trace.screenshots.append(after_screenshot)
        
        if after_dom:
            action_trace.dom_snapshots.append(after_dom)
        
        # Add network and console logs that occurred during this action
        # (This is simplified - in production you'd filter by timestamp)
        action_trace.network_requests.extend(self.network_requests.get_last(100))
        action_trace.network_responses.extend(self.network_responses.get_last(100))
        action_trace.console_logs.extend(self.console_logs.get_last(50))
        
        # Create end event
        event = DebugEvent(
            event_id=self._generate_id("evt_"),
            event_type=EventType.ACTION_END,
            timestamp=end_time,
            data={
                "success": success,
                "error": error,
                "duration_ms": (end_time - action_trace.start_time) * 1000,
                "after_url": self.page.url,
                "result_summary": str(result)[:500] if result else None
            },
            sequence_number=self.sequence_counter,
            action_id=action_id
        )
        action_trace.events.append(event)
        self.events.append(event)
        self.sequence_counter += 1
        
        # Reset current action
        self.current_action_id = None
        self.current_action_start_time = None
        
        logger.debug(f"Ended recording action {action_id}: success={success}")
    
    async def record_agent_step(
        self,
        reasoning: str,
        decision: str,
        action: Optional[Dict[str, Any]] = None,
        tokens_used: int = 0,
        model: str = ""
    ) -> str:
        """Record an agent reasoning step."""
        step_id = self._generate_id("step_")
        
        agent_step = AgentStep(
            step_id=step_id,
            reasoning=reasoning,
            decision=decision,
            action=action,
            tokens_used=tokens_used,
            model=model
        )
        
        self.agent_steps.append(agent_step)
        
        # Add to current action if exists
        if self.current_action_id:
            for trace in self.action_traces.get_all():
                if trace.action_id == self.current_action_id:
                    trace.agent_steps.append(agent_step)
                    break
        
        # Create event
        event = DebugEvent(
            event_id=self._generate_id("evt_"),
            event_type=EventType.AGENT_REASONING,
            timestamp=time.time(),
            data=asdict(agent_step),
            sequence_number=self.sequence_counter,
            action_id=self.current_action_id
        )
        self.events.append(event)
        self.sequence_counter += 1
        
        return step_id
    
    async def record_navigation(self, url: str, navigation_type: str = "goto") -> None:
        """Record a navigation event."""
        event = DebugEvent(
            event_id=self._generate_id("evt_"),
            event_type=EventType.NAVIGATION,
            timestamp=time.time(),
            data={
                "url": url,
                "navigation_type": navigation_type,
                "previous_url": self.page.url if hasattr(self.page, 'url') else ""
            },
            sequence_number=self.sequence_counter,
            action_id=self.current_action_id
        )
        self.events.append(event)
        self.sequence_counter += 1
    
    async def get_action_trace(self, action_id: str) -> Optional[ActionTrace]:
        """Get trace for a specific action."""
        for trace in self.action_traces.get_all():
            if trace.action_id == action_id:
                return trace
        return None
    
    async def get_recent_actions(self, count: int = 10) -> List[ActionTrace]:
        """Get the most recent action traces."""
        return self.action_traces.get_last(count)
    
    async def get_timeline(self, start_time: Optional[float] = None, end_time: Optional[float] = None) -> List[DebugEvent]:
        """Get events within a time range."""
        events = self.events.get_all()
        
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        
        return sorted(events, key=lambda e: e.timestamp)
    
    async def create_debug_bundle(
        self,
        output_path: Optional[Path] = None,
        include_screenshots: bool = True,
        include_dom: bool = True,
        include_network: bool = True,
        include_console: bool = True,
        include_agent_steps: bool = True,
        compress: bool = True
    ) -> Path:
        """
        Create a shareable debug bundle with all recorded data.
        
        Returns:
            Path to the created bundle file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.storage_path / f"debug_bundle_{timestamp}.zip"
        
        logger.info(f"Creating debug bundle at {output_path}")
        
        # Create metadata
        metadata = {
            "created_at": datetime.now().isoformat(),
            "page_url": self.page.url,
            "total_actions": len(self.action_traces),
            "total_events": len(self.events),
            "buffer_size": self.buffer_size,
            "settings": {
                "capture_network": self.capture_network,
                "capture_console": self.capture_console,
                "capture_dom": self.capture_dom,
                "capture_screenshots": self.capture_screenshots,
                "compress_dom": self.compress_dom
            }
        }
        
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Write metadata
            zipf.writestr("metadata.json", json.dumps(metadata, indent=2))
            
            # Write action traces
            actions_data = []
            for trace in self.action_traces.get_all():
                trace_dict = asdict(trace)
                # Convert bytes to base64 for JSON serialization
                for screenshot in trace_dict.get('screenshots', []):
                    if 'image_data' in screenshot and isinstance(screenshot['image_data'], bytes):
                        screenshot['image_data'] = base64.b64encode(screenshot['image_data']).decode('utf-8')
                actions_data.append(trace_dict)
            
            zipf.writestr("actions.json", json.dumps(actions_data, indent=2))
            
            # Write events
            events_data = [asdict(event) for event in self.events.get_all()]
            zipf.writestr("events.json", json.dumps(events_data, indent=2))
            
            # Write network data
            if include_network:
                network_data = {
                    "requests": [asdict(req) for req in self.network_requests.get_all()],
                    "responses": [asdict(resp) for resp in self.network_responses.get_all()]
                }
                zipf.writestr("network.json", json.dumps(network_data, indent=2))
            
            # Write console logs
            if include_console:
                console_data = [asdict(log) for log in self.console_logs.get_all()]
                zipf.writestr("console.json", json.dumps(console_data, indent=2))
            
            # Write agent steps
            if include_agent_steps:
                agent_data = [asdict(step) for step in self.agent_steps.get_all()]
                zipf.writestr("agent_steps.json", json.dumps(agent_data, indent=2))
            
            # Write screenshots as separate files
            if include_screenshots:
                screenshot_index = 0
                for trace in self.action_traces.get_all():
                    for screenshot in trace.screenshots:
                        filename = f"screenshots/action_{trace.action_id}_{screenshot_index}.png"
                        zipf.writestr(filename, screenshot.image_data)
                        screenshot_index += 1
            
            # Write DOM snapshots
            if include_dom:
                dom_index = 0
                for trace in self.action_traces.get_all():
                    for dom in trace.dom_snapshots:
                        if dom.compressed_html:
                            filename = f"dom/action_{trace.action_id}_{dom_index}.html.gz"
                            zipf.writestr(filename, dom.compressed_html)
                        else:
                            filename = f"dom/action_{trace.action_id}_{dom_index}.html"
                            zipf.writestr(filename, dom.html)
                        dom_index += 1
            
            # Write timeline HTML viewer
            viewer_html = self._generate_viewer_html()
            zipf.writestr("viewer.html", viewer_html)
        
        logger.info(f"Debug bundle created: {output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")
        return output_path
    
    def _generate_viewer_html(self) -> str:
        """Generate an HTML viewer for the debug bundle."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Browser Use Debug Viewer</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .timeline { border: 1px solid #ddd; padding: 10px; margin: 20px 0; }
        .event { padding: 5px; margin: 2px 0; border-left: 3px solid #007bff; }
        .event.action_start { border-left-color: #28a745; }
        .event.action_end { border-left-color: #dc3545; }
        .event.error { border-left-color: #ffc107; background: #fff3cd; }
        .screenshot { max-width: 300px; border: 1px solid #ddd; margin: 5px; }
        .action-details { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .tabs { display: flex; margin: 20px 0; }
        .tab { padding: 10px 20px; cursor: pointer; border: 1px solid #ddd; background: #f8f9fa; }
        .tab.active { background: #007bff; color: white; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Browser Use Debug Viewer</h1>
        <div id="metadata"></div>
        
        <div class="tabs">
            <div class="tab active" onclick="showTab('timeline')">Timeline</div>
            <div class="tab" onclick="showTab('actions')">Actions</div>
            <div class="tab" onclick="showTab('network')">Network</div>
            <div class="tab" onclick="showTab('console')">Console</div>
            <div class="tab" onclick="showTab('agent')">Agent Steps</div>
        </div>
        
        <div id="timeline" class="tab-content active">
            <h2>Event Timeline</h2>
            <div id="timeline-events"></div>
        </div>
        
        <div id="actions" class="tab-content">
            <h2>Action Traces</h2>
            <div id="action-list"></div>
        </div>
        
        <div id="network" class="tab-content">
            <h2>Network Activity</h2>
            <div id="network-list"></div>
        </div>
        
        <div id="console" class="tab-content">
            <h2>Console Logs</h2>
            <div id="console-list"></div>
        </div>
        
        <div id="agent" class="tab-content">
            <h2>Agent Reasoning</h2>
            <div id="agent-list"></div>
        </div>
    </div>
    
    <script>
        // Load data from JSON files
        let metadata = {};
        let events = [];
        let actions = [];
        let network = { requests: [], responses: [] };
        let consoleLogs = [];
        let agentSteps = [];
        
        async function loadData() {
            try {
                metadata = await (await fetch('metadata.json')).json();
                events = await (await fetch('events.json')).json();
                actions = await (await fetch('actions.json')).json();
                network = await (await fetch('network.json')).json();
                consoleLogs = await (await fetch('console.json')).json();
                agentSteps = await (await fetch('agent_steps.json')).json();
                
                renderMetadata();
                renderTimeline();
                renderActions();
                renderNetwork();
                renderConsole();
                renderAgent();
            } catch (e) {
                console.error('Error loading data:', e);
            }
        }
        
        function renderMetadata() {
            document.getElementById('metadata').innerHTML = `
                <p><strong>URL:</strong> ${metadata.page_url}</p>
                <p><strong>Created:</strong> ${metadata.created_at}</p>
                <p><strong>Total Actions:</strong> ${metadata.total_actions}</p>
                <p><strong>Total Events:</strong> ${metadata.total_events}</p>
            `;
        }
        
        function renderTimeline() {
            const container = document.getElementById('timeline-events');
            container.innerHTML = events.map(event => `
                <div class="event ${event.event_type}">
                    <strong>${new Date(event.timestamp * 1000).toLocaleTimeString()}</strong>
                    [${event.event_type}]
                    ${JSON.stringify(event.data).substring(0, 100)}...
                </div>
            `).join('');
        }
        
        function renderActions() {
            const container = document.getElementById('action-list');
            container.innerHTML = actions.map(action => `
                <div class="action-details">
                    <h3>${action.action_type}</h3>
                    <p><strong>ID:</strong> ${action.action_id}</p>
                    <p><strong>Success:</strong> ${action.success}</p>
                    <p><strong>Duration:</strong> ${action.end_time ? ((action.end_time - action.start_time) * 1000).toFixed(2) + 'ms' : 'N/A'}</p>
                    <p><strong>Parameters:</strong> ${JSON.stringify(action.parameters)}</p>
                    ${action.error ? `<p><strong>Error:</strong> ${action.error}</p>` : ''}
                </div>
            `).join('');
        }
        
        function renderNetwork() {
            const container = document.getElementById('network-list');
            container.innerHTML = `
                <h3>Requests (${network.requests.length})</h3>
                ${network.requests.map(req => `
                    <div>${req.method} ${req.url}</div>
                `).join('')}
                <h3>Responses (${network.responses.length})</h3>
                ${network.responses.map(resp => `
                    <div>${resp.status} ${resp.url} (${resp.response_time_ms.toFixed(2)}ms)</div>
                `).join('')}
            `;
        }
        
        function renderConsole() {
            const container = document.getElementById('console-list');
            container.innerHTML = consoleLogs.map(log => `
                <div class="event console_${log.type}">
                    <strong>[${log.type}]</strong> ${log.text}
                    ${log.location ? `<br><small>${log.location}</small>` : ''}
                </div>
            `).join('');
        }
        
        function renderAgent() {
            const container = document.getElementById('agent-list');
            container.innerHTML = agentSteps.map(step => `
                <div class="action-details">
                    <h3>Step ${step.step_id}</h3>
                    <p><strong>Reasoning:</strong> ${step.reasoning}</p>
                    <p><strong>Decision:</strong> ${step.decision}</p>
                    ${step.action ? `<p><strong>Action:</strong> ${JSON.stringify(step.action)}</p>` : ''}
                    <p><strong>Model:</strong> ${step.model} (${step.tokens_used} tokens)</p>
                </div>
            `).join('');
        }
        
        function showTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected tab
            document.getElementById(tabName).classList.add('active');
            document.querySelector(`.tab[onclick="showTab('${tabName}')"]`).classList.add('active');
        }
        
        // Load data on page load
        loadData();
    </script>
</body>
</html>
        """
    
    async def replay_action(self, action_id: str, page: Optional[Page] = None) -> bool:
        """
        Replay a specific action on a page.
        
        Args:
            action_id: ID of the action to replay
            page: Page to replay on (uses current page if None)
        
        Returns:
            True if replay was successful
        """
        target_page = page or self.page
        action_trace = await self.get_action_trace(action_id)
        
        if not action_trace:
            logger.error(f"Action {action_id} not found")
            return False
        
        logger.info(f"Replaying action {action_id}: {action_trace.action_type}")
        
        try:
            # This is a simplified replay - in production you'd need to
            # handle different action types and restore state
            if action_trace.action_type == "goto":
                url = action_trace.parameters.get("url")
                if url:
                    await target_page.goto(url)
                    return True
            
            # Add more action types as needed
            logger.warning(f"Replay not implemented for action type: {action_trace.action_type}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to replay action {action_id}: {e}")
            return False
    
    async def get_failure_analysis(self) -> Dict[str, Any]:
        """Analyze failures in the recorded actions."""
        failed_actions = []
        for trace in self.action_traces.get_all():
            if not trace.success:
                failed_actions.append({
                    "action_id": trace.action_id,
                    "action_type": trace.action_type,
                    "error": trace.error,
                    "timestamp": trace.start_time,
                    "duration": trace.end_time - trace.start_time if trace.end_time else None
                })
        
        # Analyze patterns
        error_types = {}
        for action in failed_actions:
            error = action.get("error", "Unknown")
            error_types[error] = error_types.get(error, 0) + 1
        
        return {
            "total_failed": len(failed_actions),
            "failed_actions": failed_actions,
            "error_patterns": error_types,
            "failure_rate": len(failed_actions) / max(len(self.action_traces), 1)
        }
    
    async def clear(self) -> None:
        """Clear all recorded data."""
        self.action_traces.clear()
        self.network_requests.clear()
        self.network_responses.clear()
        self.console_logs.clear()
        self.dom_snapshots.clear()
        self.screenshots.clear()
        self.agent_steps.clear()
        self.events.clear()
        self.current_action_id = None
        self.current_action_start_time = None
        self.sequence_counter = 0
        self.action_counter = 0
        self._request_ids.clear()
        
        logger.info("ReplayEngine cleared")
    
    async def close(self) -> None:
        """Clean up resources."""
        if self._listeners_initialized:
            # Remove event listeners
            if self.capture_network:
                self.page.remove_listener("request", self._on_request)
                self.page.remove_listener("response", self._on_response)
                self.page.remove_listener("requestfailed", self._on_request_failed)
            
            if self.capture_console:
                self.page.remove_listener("console", self._on_console)
        
        # Clean up DOM observer
        if self.capture_dom:
            try:
                await self.page.evaluate("""
                    () => {
                        if (window.__browserUseDebugObserver) {
                            window.__browserUseDebugObserver.disconnect();
                            delete window.__browserUseDebugObserver;
                            delete window.__browserUseDebugMutations;
                        }
                    }
                """)
            except:
                pass
        
        logger.info("ReplayEngine closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


class PageActorWithReplay(PageActor):
    """PageActor extended with replay engine integration."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_engine: Optional[ReplayEngine] = None
        self._replay_enabled = False
    
    async def enable_replay(
        self,
        buffer_size: int = 1000,
        capture_network: bool = True,
        capture_console: bool = True,
        capture_dom: bool = True,
        capture_screenshots: bool = True,
        storage_path: Optional[Path] = None
    ) -> ReplayEngine:
        """Enable replay engine for this page actor."""
        if not self.page:
            raise RuntimeError("Page not initialized")
        
        self.replay_engine = ReplayEngine(
            page=self.page,
            buffer_size=buffer_size,
            capture_network=capture_network,
            capture_console=capture_console,
            capture_dom=capture_dom,
            capture_screenshots=capture_screenshots,
            storage_path=storage_path
        )
        
        await self.replay_engine.initialize()
        self._replay_enabled = True
        
        logger.info("Replay engine enabled for PageActor")
        return self.replay_engine
    
    async def disable_replay(self) -> None:
        """Disable replay engine."""
        if self.replay_engine:
            await self.replay_engine.close()
            self.replay_engine = None
            self._replay_enabled = False
            logger.info("Replay engine disabled")
    
    async def click(self, selector: str, **kwargs) -> None:
        """Override click to record replay data."""
        if not self._replay_enabled or not self.replay_engine:
            return await super().click(selector, **kwargs)
        
        action_id = await self.replay_engine.start_action(
            action_type="click",
            parameters={"selector": selector, **kwargs}
        )
        
        try:
            result = await super().click(selector, **kwargs)
            await self.replay_engine.end_action(action_id, success=True)
            return result
        except Exception as e:
            await self.replay_engine.end_action(action_id, success=False, error=str(e))
            raise
    
    async def type(self, selector: str, text: str, **kwargs) -> None:
        """Override type to record replay data."""
        if not self._replay_enabled or not self.replay_engine:
            return await super().type(selector, text, **kwargs)
        
        action_id = await self.replay_engine.start_action(
            action_type="type",
            parameters={"selector": selector, "text": text, **kwargs}
        )
        
        try:
            result = await super().type(selector, text, **kwargs)
            await self.replay_engine.end_action(action_id, success=True)
            return result
        except Exception as e:
            await self.replay_engine.end_action(action_id, success=False, error=str(e))
            raise
    
    async def goto(self, url: str, **kwargs) -> None:
        """Override goto to record replay data."""
        if not self._replay_enabled or not self.replay_engine:
            return await super().goto(url, **kwargs)
        
        action_id = await self.replay_engine.start_action(
            action_type="goto",
            parameters={"url": url, **kwargs}
        )
        
        try:
            result = await super().goto(url, **kwargs)
            await self.replay_engine.record_navigation(url, "goto")
            await self.replay_engine.end_action(action_id, success=True)
            return result
        except Exception as e:
            await self.replay_engine.end_action(action_id, success=False, error=str(e))
            raise


# Integration with MessageManager for agent reasoning capture
class MessageManagerWithReplay(MessageManager):
    """MessageManager extended with replay engine integration."""
    
    def __init__(self, *args, replay_engine: Optional[ReplayEngine] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_engine = replay_engine
    
    async def add_message(self, message: Message, **kwargs) -> None:
        """Override add_message to record agent reasoning."""
        await super().add_message(message, **kwargs)
        
        if self.replay_engine and hasattr(message, 'content'):
            # Extract reasoning from message content
            # This is simplified - in production you'd parse the message more carefully
            content = message.content
            if isinstance(content, str) and len(content) > 50:
                # Assume longer messages contain reasoning
                await self.replay_engine.record_agent_step(
                    reasoning=content[:500],  # Truncate for storage
                    decision="Generated response",
                    model=getattr(message, 'model', 'unknown'),
                    tokens_used=getattr(message, 'tokens', 0)
                )


# Utility functions
def create_debug_bundle_from_directory(directory: Path, output_path: Path) -> Path:
    """Create a debug bundle from an existing directory of debug data."""
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(directory)
                zipf.write(file_path, arcname)
    
    return output_path


async def analyze_debug_bundle(bundle_path: Path) -> Dict[str, Any]:
    """Analyze a debug bundle and return insights."""
    with zipfile.ZipFile(bundle_path, 'r') as zipf:
        # Read metadata
        metadata = json.loads(zipf.read('metadata.json'))
        
        # Read actions
        actions = json.loads(zipf.read('actions.json'))
        
        # Calculate statistics
        total_actions = len(actions)
        failed_actions = sum(1 for a in actions if not a.get('success', True))
        
        # Calculate average action duration
        durations = []
        for action in actions:
            if action.get('end_time') and action.get('start_time'):
                durations.append(action['end_time'] - action['start_time'])
        
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        # Get unique action types
        action_types = set(a.get('action_type', 'unknown') for a in actions)
        
        return {
            "metadata": metadata,
            "statistics": {
                "total_actions": total_actions,
                "failed_actions": failed_actions,
                "success_rate": (total_actions - failed_actions) / max(total_actions, 1),
                "average_duration_ms": avg_duration * 1000,
                "unique_action_types": list(action_types)
            },
            "actions": actions
        }