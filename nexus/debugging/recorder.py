"""
Visual Debugging & Replay System for nexus.
Records every action with screenshots, DOM snapshots, network requests,
console logs, and agent reasoning steps. Enables time-travel debugging,
failure analysis, and one-click bug reproduction.
"""

import asyncio
import base64
import json
import time
import uuid
import zipfile
from collections import deque
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Deque
from io import BytesIO

from PIL import Image
from playwright.async_api import Page, Response, Request, ConsoleMessage

from nexus.agent.message_manager.service import MessageManager
from nexus.agent.views import AgentStepInfo
from nexus.actor.page import PageActor


class EventType(Enum):
    """Types of recorded events."""
    ACTION_START = "action_start"
    ACTION_END = "action_end"
    NAVIGATION = "navigation"
    NETWORK_REQUEST = "network_request"
    NETWORK_RESPONSE = "network_response"
    CONSOLE_LOG = "console_log"
    DOM_CHANGE = "dom_change"
    SCREENSHOT = "screenshot"
    AGENT_REASONING = "agent_reasoning"
    ERROR = "error"
    CUSTOM = "custom"


@dataclass
class NetworkRequest:
    """Network request data."""
    url: str
    method: str
    headers: Dict[str, str]
    post_data: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class NetworkResponse:
    """Network response data."""
    url: str
    status: int
    headers: Dict[str, str]
    request_id: str
    timestamp: float = field(default_factory=time.time)
    body_size: Optional[int] = None


@dataclass
class ConsoleLog:
    """Console log entry."""
    message: str
    level: str
    timestamp: float = field(default_factory=time.time)
    location: Optional[str] = None


@dataclass
class DOMSnapshot:
    """DOM snapshot data."""
    html: str
    timestamp: float = field(default_factory=time.time)
    selector: Optional[str] = None
    diff: Optional[str] = None


@dataclass
class Screenshot:
    """Screenshot data."""
    image_data: bytes
    timestamp: float = field(default_factory=time.time)
    full_page: bool = False
    element_selector: Optional[str] = None


@dataclass
class AgentReasoning:
    """Agent reasoning step."""
    thought: str
    action: Optional[str] = None
    action_params: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class DebugEvent:
    """Unified debug event container."""
    event_type: EventType
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Optional typed data
    network_request: Optional[NetworkRequest] = None
    network_response: Optional[NetworkResponse] = None
    console_log: Optional[ConsoleLog] = None
    dom_snapshot: Optional[DOMSnapshot] = None
    screenshot: Optional[Screenshot] = None
    agent_reasoning: Optional[AgentReasoning] = None


@dataclass
class DebugBundle:
    """Shareable debug bundle."""
    bundle_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    events: List[DebugEvent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    summary: Optional[str] = None


class RingBuffer:
    """Ring buffer for storing debug events with fixed capacity."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer: Deque[DebugEvent] = deque(maxlen=max_size)
        self.total_events = 0
        self._lock = asyncio.Lock()
    
    async def add(self, event: DebugEvent) -> None:
        """Add event to ring buffer."""
        async with self._lock:
            self.buffer.append(event)
            self.total_events += 1
    
    async def get_events(self, 
                        event_types: Optional[Set[EventType]] = None,
                        start_time: Optional[float] = None,
                        end_time: Optional[float] = None,
                        limit: Optional[int] = None) -> List[DebugEvent]:
        """Get filtered events from buffer."""
        async with self._lock:
            events = list(self.buffer)
        
        # Apply filters
        if event_types:
            events = [e for e in events if e.event_type in event_types]
        
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        
        # Apply limit
        if limit:
            events = events[-limit:]
        
        return events
    
    async def clear(self) -> None:
        """Clear the ring buffer."""
        async with self._lock:
            self.buffer.clear()
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def __iter__(self):
        return iter(self.buffer)


class DebugRecorder:
    """
    Main debug recorder that captures all browser and agent activity.
    Integrates with nexus components to provide comprehensive debugging.
    """
    
    def __init__(self, 
                 page: Page,
                 page_actor: Optional[PageActor] = None,
                 message_manager: Optional[MessageManager] = None,
                 buffer_size: int = 1000,
                 capture_screenshots: bool = True,
                 capture_dom: bool = True,
                 capture_network: bool = True,
                 capture_console: bool = True,
                 capture_agent_reasoning: bool = True,
                 screenshot_quality: int = 80,
                 dom_snapshot_selector: Optional[str] = None):
        
        self.page = page
        self.page_actor = page_actor
        self.message_manager = message_manager
        self.ring_buffer = RingBuffer(max_size=buffer_size)
        
        # Capture settings
        self.capture_screenshots = capture_screenshots
        self.capture_dom = capture_dom
        self.capture_network = capture_network
        self.capture_console = capture_console
        self.capture_agent_reasoning = capture_agent_reasoning
        self.screenshot_quality = screenshot_quality
        self.dom_snapshot_selector = dom_snapshot_selector
        
        # State tracking
        self.current_action_id: Optional[str] = None
        self.is_recording = False
        self._listeners_setup = False
        self._network_requests: Dict[str, NetworkRequest] = {}
        
        # Event handlers
        self._event_handlers: Dict[EventType, List[callable]] = {}
        
        # Setup default handlers
        self._setup_default_handlers()
    
    def _setup_default_handlers(self) -> None:
        """Setup default event handlers."""
        self.register_handler(EventType.ERROR, self._handle_error_event)
        self.register_handler(EventType.NAVIGATION, self._handle_navigation_event)
    
    def register_handler(self, event_type: EventType, handler: callable) -> None:
        """Register an event handler."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
    
    async def _emit_event(self, event: DebugEvent) -> None:
        """Emit event to handlers and add to buffer."""
        # Add to ring buffer
        await self.ring_buffer.add(event)
        
        # Call registered handlers
        handlers = self._event_handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                print(f"Error in event handler: {e}")
    
    async def start_recording(self) -> None:
        """Start recording debug events."""
        if self.is_recording:
            return
        
        self.is_recording = True
        
        if not self._listeners_setup:
            await self._setup_listeners()
            self._listeners_setup = True
        
        # Record start event
        await self._emit_event(DebugEvent(
            event_type=EventType.CUSTOM,
            data={"action": "recording_started"},
            metadata={"recorder": "DebugRecorder"}
        ))
    
    async def stop_recording(self) -> None:
        """Stop recording debug events."""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        # Record stop event
        await self._emit_event(DebugEvent(
            event_type=EventType.CUSTOM,
            data={"action": "recording_stopped"},
            metadata={"recorder": "DebugRecorder"}
        ))
    
    async def _setup_listeners(self) -> None:
        """Setup page event listeners."""
        if self.capture_network:
            self.page.on("request", self._on_request)
            self.page.on("response", self._on_response)
            self.page.on("requestfailed", self._on_request_failed)
        
        if self.capture_console:
            self.page.on("console", self._on_console)
        
        # Navigation events
        self.page.on("framenavigated", self._on_navigation)
    
    async def _on_request(self, request: Request) -> None:
        """Handle network request event."""
        if not self.is_recording or not self.capture_network:
            return
        
        network_request = NetworkRequest(
            url=request.url,
            method=request.method,
            headers=dict(request.headers),
            post_data=request.post_data
        )
        
        self._network_requests[request._id] = network_request
        
        await self._emit_event(DebugEvent(
            event_type=EventType.NETWORK_REQUEST,
            network_request=network_request,
            data={
                "resource_type": request.resource_type,
                "is_navigation_request": request.is_navigation_request()
            }
        ))
    
    async def _on_response(self, response: Response) -> None:
        """Handle network response event."""
        if not self.is_recording or not self.capture_network:
            return
        
        request_id = response.request._id
        network_request = self._network_requests.get(request_id)
        
        if network_request:
            network_response = NetworkResponse(
                url=response.url,
                status=response.status,
                headers=dict(response.headers),
                request_id=network_request.request_id,
                body_size=len(await response.body()) if response.ok else None
            )
            
            await self._emit_event(DebugEvent(
                event_type=EventType.NETWORK_RESPONSE,
                network_response=network_response,
                data={
                    "ok": response.ok,
                    "from_cache": response.from_service_worker,
                    "request_method": network_request.method
                }
            ))
    
    async def _on_request_failed(self, request: Request) -> None:
        """Handle failed network request."""
        if not self.is_recording or not self.capture_network:
            return
        
        await self._emit_event(DebugEvent(
            event_type=EventType.ERROR,
            data={
                "type": "network_request_failed",
                "url": request.url,
                "method": request.method,
                "failure": request.failure
            }
        ))
    
    async def _on_console(self, console_message: ConsoleMessage) -> None:
        """Handle console log event."""
        if not self.is_recording or not self.capture_console:
            return
        
        console_log = ConsoleLog(
            message=console_message.text,
            level=console_message.type,
            location=f"{console_message.location.get('url', '')}:{console_message.location.get('lineNumber', '')}"
        )
        
        await self._emit_event(DebugEvent(
            event_type=EventType.CONSOLE_LOG,
            console_log=console_log,
            data={"args_count": len(console_message.args)}
        ))
    
    async def _on_navigation(self, frame) -> None:
        """Handle navigation event."""
        if not self.is_recording:
            return
        
        await self._emit_event(DebugEvent(
            event_type=EventType.NAVIGATION,
            data={
                "url": frame.url,
                "name": frame.name,
                "parent_frame": frame.parent_frame.name if frame.parent_frame else None
            }
        ))
    
    async def _handle_error_event(self, event: DebugEvent) -> None:
        """Handle error events."""
        print(f"Debug Error: {event.data}")
    
    async def _handle_navigation_event(self, event: DebugEvent) -> None:
        """Handle navigation events - capture state after navigation."""
        if self.capture_screenshots:
            await self.capture_screenshot()
        
        if self.capture_dom:
            await self.capture_dom_snapshot()
    
    async def capture_screenshot(self, 
                                full_page: bool = False,
                                element_selector: Optional[str] = None) -> Optional[Screenshot]:
        """Capture a screenshot of the current page."""
        if not self.capture_screenshots:
            return None
        
        try:
            if element_selector:
                element = await self.page.query_selector(element_selector)
                if element:
                    screenshot_bytes = await element.screenshot(
                        type="jpeg",
                        quality=self.screenshot_quality
                    )
                else:
                    # Fallback to full page if element not found
                    screenshot_bytes = await self.page.screenshot(
                        type="jpeg",
                        quality=self.screenshot_quality,
                        full_page=full_page
                    )
            else:
                screenshot_bytes = await self.page.screenshot(
                    type="jpeg",
                    quality=self.screenshot_quality,
                    full_page=full_page
                )
            
            screenshot = Screenshot(
                image_data=screenshot_bytes,
                full_page=full_page,
                element_selector=element_selector
            )
            
            await self._emit_event(DebugEvent(
                event_type=EventType.SCREENSHOT,
                screenshot=screenshot,
                data={"size_bytes": len(screenshot_bytes)}
            ))
            
            return screenshot
            
        except Exception as e:
            await self._emit_event(DebugEvent(
                event_type=EventType.ERROR,
                data={"type": "screenshot_failed", "error": str(e)}
            ))
            return None
    
    async def capture_dom_snapshot(self, selector: Optional[str] = None) -> Optional[DOMSnapshot]:
        """Capture a DOM snapshot."""
        if not self.capture_dom:
            return None
        
        try:
            selector = selector or self.dom_snapshot_selector
            
            if selector:
                element = await self.page.query_selector(selector)
                if element:
                    html = await element.inner_html()
                else:
                    html = await self.page.content()
            else:
                html = await self.page.content()
            
            dom_snapshot = DOMSnapshot(
                html=html,
                selector=selector
            )
            
            await self._emit_event(DebugEvent(
                event_type=EventType.DOM_CHANGE,
                dom_snapshot=dom_snapshot,
                data={"html_length": len(html)}
            ))
            
            return dom_snapshot
            
        except Exception as e:
            await self._emit_event(DebugEvent(
                event_type=EventType.ERROR,
                data={"type": "dom_snapshot_failed", "error": str(e)}
            ))
            return None
    
    async def record_agent_reasoning(self, 
                                   thought: str,
                                   action: Optional[str] = None,
                                   action_params: Optional[Dict[str, Any]] = None) -> None:
        """Record agent reasoning step."""
        if not self.capture_agent_reasoning:
            return
        
        agent_reasoning = AgentReasoning(
            thought=thought,
            action=action,
            action_params=action_params
        )
        
        await self._emit_event(DebugEvent(
            event_type=EventType.AGENT_REASONING,
            agent_reasoning=agent_reasoning,
            data={"has_action": action is not None}
        ))
    
    async def start_action(self, 
                          action_name: str,
                          action_params: Optional[Dict[str, Any]] = None) -> str:
        """Start recording an action with before state."""
        action_id = str(uuid.uuid4())
        self.current_action_id = action_id
        
        # Capture before state
        before_screenshot = None
        before_dom = None
        
        if self.capture_screenshots:
            before_screenshot = await self.capture_screenshot()
        
        if self.capture_dom:
            before_dom = await self.capture_dom_snapshot()
        
        await self._emit_event(DebugEvent(
            event_type=EventType.ACTION_START,
            event_id=action_id,
            data={
                "action_name": action_name,
                "action_params": action_params or {},
                "before_screenshot_captured": before_screenshot is not None,
                "before_dom_captured": before_dom is not None
            },
            metadata={"action_id": action_id}
        ))
        
        return action_id
    
    async def end_action(self, 
                        action_id: str,
                        success: bool = True,
                        error: Optional[str] = None) -> None:
        """End recording an action with after state."""
        if self.current_action_id != action_id:
            await self._emit_event(DebugEvent(
                event_type=EventType.ERROR,
                data={
                    "type": "action_id_mismatch",
                    "expected": self.current_action_id,
                    "received": action_id
                }
            ))
            return
        
        # Capture after state
        after_screenshot = None
        after_dom = None
        
        if self.capture_screenshots:
            after_screenshot = await self.capture_screenshot()
        
        if self.capture_dom:
            after_dom = await self.capture_dom_snapshot()
        
        await self._emit_event(DebugEvent(
            event_type=EventType.ACTION_END,
            event_id=action_id,
            data={
                "success": success,
                "error": error,
                "after_screenshot_captured": after_screenshot is not None,
                "after_dom_captured": after_dom is not None
            },
            metadata={"action_id": action_id}
        ))
        
        self.current_action_id = None
    
    async def record_custom_event(self, 
                                 event_name: str,
                                 data: Optional[Dict[str, Any]] = None) -> None:
        """Record a custom event."""
        await self._emit_event(DebugEvent(
            event_type=EventType.CUSTOM,
            data={"event_name": event_name, **(data or {})}
        ))
    
    async def get_debug_bundle(self, 
                              include_screenshots: bool = True,
                              include_dom: bool = True,
                              max_events: Optional[int] = None) -> DebugBundle:
        """Create a debug bundle from recorded events."""
        events = await self.ring_buffer.get_events(limit=max_events)
        
        # Filter out large data if requested
        if not include_screenshots:
            for event in events:
                if event.screenshot:
                    event.screenshot.image_data = b""
        
        if not include_dom:
            for event in events:
                if event.dom_snapshot:
                    event.dom_snapshot.html = ""
        
        bundle = DebugBundle(
            events=events,
            metadata={
                "total_events_recorded": self.ring_buffer.total_events,
                "buffer_size": len(self.ring_buffer),
                "recording_settings": {
                    "capture_screenshots": self.capture_screenshots,
                    "capture_dom": self.capture_dom,
                    "capture_network": self.capture_network,
                    "capture_console": self.capture_console,
                    "capture_agent_reasoning": self.capture_agent_reasoning
                }
            }
        )
        
        return bundle
    
    async def export_debug_bundle(self, 
                                 filepath: Union[str, Path],
                                 include_screenshots: bool = True,
                                 include_dom: bool = True,
                                 compress: bool = True) -> Path:
        """Export debug bundle to a file."""
        bundle = await self.get_debug_bundle(
            include_screenshots=include_screenshots,
            include_dom=include_dom
        )
        
        filepath = Path(filepath)
        
        if compress:
            # Export as zip with JSON and images
            with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add bundle metadata
                bundle_dict = asdict(bundle)
                
                # Convert screenshots to base64 for JSON serialization
                for event in bundle_dict['events']:
                    if event.get('screenshot') and event['screenshot'].get('image_data'):
                        event['screenshot']['image_data'] = base64.b64encode(
                            event['screenshot']['image_data']
                        ).decode('utf-8')
                
                # Add main JSON file
                zipf.writestr(
                    'debug_bundle.json',
                    json.dumps(bundle_dict, indent=2, default=str)
                )
                
                # Add summary
                summary = self._generate_bundle_summary(bundle)
                zipf.writestr('summary.txt', summary)
                
                # Add README with instructions
                readme = self._generate_readme(bundle)
                zipf.writestr('README.md', readme)
        else:
            # Export as plain JSON
            bundle_dict = asdict(bundle)
            
            # Convert screenshots to base64
            for event in bundle_dict['events']:
                if event.get('screenshot') and event['screenshot'].get('image_data'):
                    event['screenshot']['image_data'] = base64.b64encode(
                        event['screenshot']['image_data']
                    ).decode('utf-8')
            
            with open(filepath, 'w') as f:
                json.dump(bundle_dict, f, indent=2, default=str)
        
        return filepath
    
    def _generate_bundle_summary(self, bundle: DebugBundle) -> str:
        """Generate a human-readable summary of the debug bundle."""
        lines = [
            "DEBUG BUNDLE SUMMARY",
            "=" * 50,
            f"Created: {bundle.created_at}",
            f"Bundle ID: {bundle.bundle_id}",
            f"Total Events: {len(bundle.events)}",
            ""
        ]
        
        # Count event types
        event_counts = {}
        for event in bundle.events:
            event_type = event.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        lines.append("Event Counts:")
        for event_type, count in sorted(event_counts.items()):
            lines.append(f"  - {event_type}: {count}")
        
        lines.append("")
        
        # Timeline summary
        if bundle.events:
            start_time = bundle.events[0].timestamp
            end_time = bundle.events[-1].timestamp
            duration = end_time - start_time
            
            lines.append(f"Timeline: {duration:.2f} seconds")
            lines.append(f"From: {datetime.fromtimestamp(start_time)}")
            lines.append(f"To: {datetime.fromtimestamp(end_time)}")
        
        return "\n".join(lines)
    
    def _generate_readme(self, bundle: DebugBundle) -> str:
        """Generate README with instructions for using the debug bundle."""
        return f"""# Debug Bundle: {bundle.bundle_id}

## Overview
This debug bundle contains recorded browser and agent activity for debugging purposes.

## Contents
- `debug_bundle.json`: Complete event data in JSON format
- `summary.txt`: Human-readable summary of the recording
- This README file

## Event Types Recorded
{chr(10).join(f'- {et.value}' for et in EventType)}

## How to Use

### Loading the Bundle
```python
import json
from pathlib import Path

with open('debug_bundle.json', 'r') as f:
    bundle_data = json.load(f)
```

### Analyzing Events
Events are stored in chronological order in the `events` array. Each event has:
- `event_type`: Type of event
- `timestamp`: Unix timestamp
- `data`: Event-specific data
- `metadata`: Additional metadata

### Screenshots
Screenshots are stored as base64-encoded JPEG images in the `screenshot.image_data` field.

### Replaying Actions
Use the `ACTION_START` and `ACTION_END` events to identify and replay specific actions.

## Integration with nexus
This bundle was generated by the nexus DebugRecorder.

## Support
For issues or questions, please refer to the nexus documentation.
"""
    
    async def clear(self) -> None:
        """Clear all recorded events."""
        await self.ring_buffer.clear()
        self._network_requests.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get recorder statistics."""
        return {
            "is_recording": self.is_recording,
            "buffer_size": len(self.ring_buffer),
            "total_events": self.ring_buffer.total_events,
            "current_action_id": self.current_action_id,
            "listeners_setup": self._listeners_setup,
            "settings": {
                "capture_screenshots": self.capture_screenshots,
                "capture_dom": self.capture_dom,
                "capture_network": self.capture_network,
                "capture_console": self.capture_console,
                "capture_agent_reasoning": self.capture_agent_reasoning,
                "screenshot_quality": self.screenshot_quality,
                "dom_snapshot_selector": self.dom_snapshot_selector
            }
        }


class DebugRecorderIntegration:
    """
    Integration helper for connecting DebugRecorder with nexus components.
    Provides convenient methods for common debugging scenarios.
    """
    
    @staticmethod
    async def create_for_agent(page: Page,
                              page_actor: PageActor,
                              message_manager: MessageManager,
                              **kwargs) -> DebugRecorder:
        """Create a DebugRecorder configured for agent debugging."""
        recorder = DebugRecorder(
            page=page,
            page_actor=page_actor,
            message_manager=message_manager,
            capture_agent_reasoning=True,
            **kwargs
        )
        
        # Hook into message manager if available
        if hasattr(message_manager, 'add_message'):
            original_add_message = message_manager.add_message
            
            async def hooked_add_message(*args, **kwargs):
                result = await original_add_message(*args, **kwargs)
                
                # Record agent reasoning from messages
                if args and hasattr(args[0], 'content'):
                    content = args[0].content
                    if isinstance(content, str) and len(content) > 0:
                        await recorder.record_agent_reasoning(
                            thought=content[:500] + "..." if len(content) > 500 else content
                        )
                
                return result
            
            message_manager.add_message = hooked_add_message
        
        return recorder
    
    @staticmethod
    async def record_action_sequence(recorder: DebugRecorder,
                                   action_func: callable,
                                   action_name: str,
                                   action_params: Optional[Dict[str, Any]] = None) -> Any:
        """Record a complete action sequence with before/after states."""
        action_id = await recorder.start_action(action_name, action_params)
        
        try:
            result = await action_func()
            await recorder.end_action(action_id, success=True)
            return result
        except Exception as e:
            await recorder.end_action(action_id, success=False, error=str(e))
            raise
    
    @staticmethod
    async def create_failure_bundle(recorder: DebugRecorder,
                                  error: Exception,
                                  context: Optional[Dict[str, Any]] = None) -> Path:
        """Create a debug bundle specifically for failure analysis."""
        # Record the error
        await recorder.record_custom_event(
            "failure_occurred",
            {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context or {}
            }
        )
        
        # Capture final state
        await recorder.capture_screenshot(full_page=True)
        await recorder.capture_dom_snapshot()
        
        # Export bundle
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bundle_path = Path(f"debug_failure_{timestamp}.zip")
        
        return await recorder.export_debug_bundle(
            bundle_path,
            include_screenshots=True,
            include_dom=True,
            compress=True
        )


# Convenience functions for quick debugging
async def quick_record(page: Page, 
                      duration_seconds: float = 10.0,
                      buffer_size: int = 500) -> DebugRecorder:
    """Quickly start recording for a specified duration."""
    recorder = DebugRecorder(page, buffer_size=buffer_size)
    await recorder.start_recording()
    
    # Record for specified duration
    await asyncio.sleep(duration_seconds)
    
    await recorder.stop_recording()
    return recorder


async def record_and_export(page: Page,
                           filepath: Union[str, Path],
                           duration_seconds: float = 10.0) -> Path:
    """Record for a duration and export to file."""
    recorder = await quick_record(page, duration_seconds)
    return await recorder.export_debug_bundle(filepath)


# Export main classes
__all__ = [
    'DebugRecorder',
    'DebugRecorderIntegration',
    'RingBuffer',
    'DebugBundle',
    'DebugEvent',
    'EventType',
    'quick_record',
    'record_and_export'
]