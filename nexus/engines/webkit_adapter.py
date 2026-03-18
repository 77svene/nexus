"""
WebKit browser engine adapter for multi-browser support.
Implements WebKit Inspector Protocol translation layer.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class BrowserType(Enum):
    WEBKIT = "webkit"
    FIREFOX = "firefox"
    CHROMIUM = "chromium"
    MOBILE_WEBKIT = "mobile_webkit"


@dataclass
class WebKitCapabilities:
    """WebKit-specific capabilities and configuration."""
    headless: bool = True
    viewport_width: int = 1280
    viewport_height: int = 720
    user_agent: Optional[str] = None
    locale: Optional[str] = None
    timezone: Optional[str] = None
    mobile_emulation: bool = False
    device_name: Optional[str] = None
    touch_enabled: bool = False
    proxy: Optional[Dict[str, str]] = None
    ignore_https_errors: bool = False
    java_script_enabled: bool = True
    bypass_csp: bool = False
    extra_http_headers: Optional[Dict[str, str]] = None


@dataclass
class WebKitSession:
    """Represents a WebKit browser session."""
    session_id: str
    target_id: str
    capabilities: WebKitCapabilities
    websocket_url: Optional[str] = None
    pages: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


class WebKitProtocolError(Exception):
    """WebKit protocol-specific errors."""
    pass


class WebKitAdapter:
    """
    Adapter for WebKit browser engine.
    Translates WebKit Inspector Protocol to unified browser API.
    """
    
    # WebKit Inspector Protocol commands
    WIP_COMMANDS = {
        "Page.enable": "page_enable",
        "Page.navigate": "page_navigate",
        "Page.captureScreenshot": "page_screenshot",
        "Runtime.evaluate": "runtime_evaluate",
        "DOM.getDocument": "dom_get_document",
        "DOM.querySelector": "dom_query_selector",
        "DOM.querySelectorAll": "dom_query_selector_all",
        "DOM.setAttributeValue": "dom_set_attribute",
        "DOM.setNodeValue": "dom_set_node_value",
        "Input.dispatchMouseEvent": "input_mouse_event",
        "Input.dispatchKeyEvent": "input_key_event",
        "Input.insertText": "input_insert_text",
        "Network.enable": "network_enable",
        "Network.setExtraHTTPHeaders": "network_set_headers",
        "Console.enable": "console_enable",
        "Console.disable": "console_disable",
    }
    
    # Mobile device presets for WebKit
    MOBILE_DEVICES = {
        "iPhone 12": {
            "width": 390,
            "height": 844,
            "device_scale_factor": 3,
            "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
            "touch_enabled": True,
            "mobile": True,
        },
        "iPhone SE": {
            "width": 375,
            "height": 667,
            "device_scale_factor": 2,
            "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
            "touch_enabled": True,
            "mobile": True,
        },
        "iPad Pro": {
            "width": 1024,
            "height": 1366,
            "device_scale_factor": 2,
            "user_agent": "Mozilla/5.0 (iPad; CPU OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
            "touch_enabled": True,
            "mobile": True,
        },
        "Pixel 5": {
            "width": 393,
            "height": 851,
            "device_scale_factor": 2.75,
            "user_agent": "Mozilla/5.0 (Linux; Android 11; Pixel 5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.91 Mobile Safari/537.36",
            "touch_enabled": True,
            "mobile": True,
        }
    }
    
    def __init__(self, capabilities: Optional[WebKitCapabilities] = None):
        self.capabilities = capabilities or WebKitCapabilities()
        self.sessions: Dict[str, WebKitSession] = {}
        self._command_id = 0
        self._pending_commands: Dict[int, asyncio.Future] = {}
        self._event_handlers: Dict[str, List[callable]] = {}
        self._connected = False
        self._websocket = None
        
    async def connect(self, websocket_url: str = "ws://127.0.0.1:9222") -> bool:
        """Connect to WebKit browser instance."""
        try:
            # In a real implementation, this would establish WebSocket connection
            # to WebKit's inspector server
            self._websocket_url = websocket_url
            self._connected = True
            logger.info(f"Connected to WebKit at {websocket_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to WebKit: {e}")
            raise WebKitProtocolError(f"Connection failed: {e}")
    
    async def disconnect(self):
        """Disconnect from WebKit browser."""
        self._connected = False
        self._websocket = None
        logger.info("Disconnected from WebKit")
    
    async def create_session(self, capabilities: Optional[WebKitCapabilities] = None) -> WebKitSession:
        """Create a new browser session."""
        if not self._connected:
            raise WebKitProtocolError("Not connected to WebKit")
        
        caps = capabilities or self.capabilities
        session_id = f"webkit_session_{len(self.sessions) + 1}"
        
        # Apply mobile emulation if specified
        if caps.mobile_emulation and caps.device_name:
            await self._apply_mobile_emulation(caps.device_name, caps)
        
        session = WebKitSession(
            session_id=session_id,
            target_id=f"target_{session_id}",
            capabilities=caps
        )
        
        self.sessions[session_id] = session
        
        # Enable necessary domains
        await self._send_command("Page.enable", session_id=session_id)
        await self._send_command("Runtime.enable", session_id=session_id)
        await self._send_command("Console.enable", session_id=session_id)
        
        if caps.viewport_width and caps.viewport_height:
            await self.set_viewport(
                session_id,
                caps.viewport_width,
                caps.viewport_height
            )
        
        logger.info(f"Created WebKit session: {session_id}")
        return session
    
    async def close_session(self, session_id: str):
        """Close a browser session."""
        if session_id in self.sessions:
            # Close all pages in the session
            session = self.sessions[session_id]
            for page_id in session.pages:
                await self.close_page(session_id, page_id)
            
            del self.sessions[session_id]
            logger.info(f"Closed WebKit session: {session_id}")
    
    async def new_page(self, session_id: str, url: Optional[str] = None) -> str:
        """Create a new page/tab."""
        if session_id not in self.sessions:
            raise WebKitProtocolError(f"Session {session_id} not found")
        
        # WebKit uses target-based architecture
        result = await self._send_command(
            "Target.createTarget",
            params={"url": url or "about:blank"},
            session_id=session_id
        )
        
        target_id = result.get("targetId")
        page_id = f"page_{target_id}"
        
        session = self.sessions[session_id]
        session.pages.append(page_id)
        
        # Attach to target
        await self._send_command(
            "Target.attachToTarget",
            params={"targetId": target_id, "flatten": True},
            session_id=session_id
        )
        
        if url:
            await self.navigate(session_id, page_id, url)
        
        logger.info(f"Created new page {page_id} in session {session_id}")
        return page_id
    
    async def close_page(self, session_id: str, page_id: str):
        """Close a page/tab."""
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        if page_id in session.pages:
            target_id = page_id.replace("page_", "")
            await self._send_command(
                "Target.closeTarget",
                params={"targetId": target_id},
                session_id=session_id
            )
            session.pages.remove(page_id)
    
    async def navigate(self, session_id: str, page_id: str, url: str) -> Dict[str, Any]:
        """Navigate to a URL."""
        return await self._send_command(
            "Page.navigate",
            params={"url": url},
            session_id=session_id,
            page_id=page_id
        )
    
    async def set_viewport(self, session_id: str, width: int, height: int, device_scale_factor: float = 1.0):
        """Set viewport dimensions."""
        await self._send_command(
            "Emulation.setDeviceMetricsOverride",
            params={
                "width": width,
                "height": height,
                "deviceScaleFactor": device_scale_factor,
                "mobile": self.capabilities.mobile_emulation
            },
            session_id=session_id
        )
    
    async def evaluate(self, session_id: str, page_id: str, expression: str, 
                      await_promise: bool = False) -> Any:
        """Evaluate JavaScript expression."""
        result = await self._send_command(
            "Runtime.evaluate",
            params={
                "expression": expression,
                "awaitPromise": await_promise,
                "returnByValue": True
            },
            session_id=session_id,
            page_id=page_id
        )
        
        if "exceptionDetails" in result:
            raise WebKitProtocolError(f"JavaScript error: {result['exceptionDetails']}")
        
        return result.get("result", {}).get("value")
    
    async def query_selector(self, session_id: str, page_id: str, selector: str) -> Optional[str]:
        """Find element by CSS selector."""
        # First get document root
        doc_result = await self._send_command(
            "DOM.getDocument",
            params={"depth": 0},
            session_id=session_id,
            page_id=page_id
        )
        
        root_node_id = doc_result["root"]["nodeId"]
        
        # Query selector
        result = await self._send_command(
            "DOM.querySelector",
            params={
                "nodeId": root_node_id,
                "selector": selector
            },
            session_id=session_id,
            page_id=page_id
        )
        
        return str(result.get("nodeId")) if result.get("nodeId") else None
    
    async def query_selector_all(self, session_id: str, page_id: str, selector: str) -> List[str]:
        """Find all elements by CSS selector."""
        doc_result = await self._send_command(
            "DOM.getDocument",
            params={"depth": 0},
            session_id=session_id,
            page_id=page_id
        )
        
        root_node_id = doc_result["root"]["nodeId"]
        
        result = await self._send_command(
            "DOM.querySelectorAll",
            params={
                "nodeId": root_node_id,
                "selector": selector
            },
            session_id=session_id,
            page_id=page_id
        )
        
        return [str(node_id) for node_id in result.get("nodeIds", [])]
    
    async def click(self, session_id: str, page_id: str, x: int, y: int, 
                   button: str = "left", click_count: int = 1):
        """Simulate mouse click."""
        # Mouse down
        await self._send_command(
            "Input.dispatchMouseEvent",
            params={
                "type": "mousePressed",
                "x": x,
                "y": y,
                "button": button,
                "clickCount": click_count
            },
            session_id=session_id,
            page_id=page_id
        )
        
        # Mouse up
        await self._send_command(
            "Input.dispatchMouseEvent",
            params={
                "type": "mouseReleased",
                "x": x,
                "y": y,
                "button": button,
                "clickCount": click_count
            },
            session_id=session_id,
            page_id=page_id
        )
    
    async def type_text(self, session_id: str, page_id: str, text: str, 
                       delay: int = 0):
        """Type text with optional delay between keystrokes."""
        for char in text:
            # Key down
            await self._send_command(
                "Input.dispatchKeyEvent",
                params={
                    "type": "keyDown",
                    "text": char,
                    "key": char,
                    "code": f"Key{char.upper()}" if char.isalpha() else "",
                    "windowsVirtualKeyCode": ord(char.upper()) if char.isalpha() else 0
                },
                session_id=session_id,
                page_id=page_id
            )
            
            # Key up
            await self._send_command(
                "Input.dispatchKeyEvent",
                params={
                    "type": "keyUp",
                    "key": char,
                    "code": f"Key{char.upper()}" if char.isalpha() else "",
                    "windowsVirtualKeyCode": ord(char.upper()) if char.isalpha() else 0
                },
                session_id=session_id,
                page_id=page_id
            )
            
            if delay > 0:
                await asyncio.sleep(delay / 1000)
    
    async def screenshot(self, session_id: str, page_id: str, 
                        format: str = "png", quality: int = 80) -> bytes:
        """Capture screenshot of the page."""
        result = await self._send_command(
            "Page.captureScreenshot",
            params={
                "format": format,
                "quality": quality if format == "jpeg" else None
            },
            session_id=session_id,
            page_id=page_id
        )
        
        import base64
        return base64.b64decode(result["data"])
    
    async def get_cookies(self, session_id: str, page_id: str) -> List[Dict[str, Any]]:
        """Get all cookies for the page."""
        result = await self._send_command(
            "Network.getCookies",
            params={},
            session_id=session_id,
            page_id=page_id
        )
        return result.get("cookies", [])
    
    async def set_cookie(self, session_id: str, page_id: str, 
                        cookie: Dict[str, Any]):
        """Set a cookie."""
        await self._send_command(
            "Network.setCookie",
            params=cookie,
            session_id=session_id,
            page_id=page_id
        )
    
    async def clear_cookies(self, session_id: str, page_id: str):
        """Clear all cookies."""
        await self._send_command(
            "Network.clearBrowserCookies",
            params={},
            session_id=session_id,
            page_id=page_id
        )
    
    async def set_extra_headers(self, session_id: str, page_id: str, 
                               headers: Dict[str, str]):
        """Set extra HTTP headers."""
        await self._send_command(
            "Network.setExtraHTTPHeaders",
            params={"headers": headers},
            session_id=session_id,
            page_id=page_id
        )
    
    async def enable_console(self, session_id: str, page_id: str):
        """Enable console events."""
        await self._send_command(
            "Console.enable",
            params={},
            session_id=session_id,
            page_id=page_id
        )
    
    async def disable_console(self, session_id: str, page_id: str):
        """Disable console events."""
        await self._send_command(
            "Console.disable",
            params={},
            session_id=session_id,
            page_id=page_id
        )
    
    async def get_console_messages(self, session_id: str, page_id: str) -> List[Dict[str, Any]]:
        """Get console messages."""
        result = await self._send_command(
            "Console.getMessages",
            params={},
            session_id=session_id,
            page_id=page_id
        )
        return result.get("messages", [])
    
    async def _apply_mobile_emulation(self, device_name: str, 
                                     capabilities: WebKitCapabilities):
        """Apply mobile device emulation settings."""
        device = self.MOBILE_DEVICES.get(device_name)
        if not device:
            raise ValueError(f"Unknown device: {device_name}")
        
        capabilities.viewport_width = device["width"]
        capabilities.viewport_height = device["height"]
        capabilities.touch_enabled = device["touch_enabled"]
        capabilities.mobile_emulation = device["mobile"]
        
        if not capabilities.user_agent:
            capabilities.user_agent = device["user_agent"]
        
        logger.info(f"Applied mobile emulation for {device_name}")
    
    async def _send_command(self, method: str, params: Dict[str, Any] = None,
                           session_id: str = None, page_id: str = None) -> Dict[str, Any]:
        """Send command to WebKit browser."""
        if not self._connected:
            raise WebKitProtocolError("Not connected to WebKit")
        
        self._command_id += 1
        command = {
            "id": self._command_id,
            "method": method,
            "params": params or {}
        }
        
        # Add session context if provided
        if session_id:
            command["sessionId"] = session_id
        
        # Add target context if provided
        if page_id:
            target_id = page_id.replace("page_", "")
            command["targetId"] = target_id
        
        # In a real implementation, this would send via WebSocket
        # and wait for response with matching id
        logger.debug(f"Sending WebKit command: {method}")
        
        # Simulate protocol translation
        translated_method = self.WIP_COMMANDS.get(method, method)
        
        # Mock response for demonstration
        response = {
            "id": self._command_id,
            "result": self._mock_command_response(translated_method, params)
        }
        
        return response.get("result", {})
    
    def _mock_command_response(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mock command responses for development/testing."""
        mock_responses = {
            "page_enable": {},
            "page_navigate": {"frameId": "frame_1", "loaderId": "loader_1"},
            "page_screenshot": {"data": "base64_encoded_image_data"},
            "runtime_evaluate": {
                "result": {
                    "type": "string",
                    "value": "Evaluation result"
                }
            },
            "dom_get_document": {
                "root": {
                    "nodeId": 1,
                    "nodeName": "document",
                    "nodeType": 9
                }
            },
            "dom_query_selector": {"nodeId": 2},
            "dom_query_selector_all": {"nodeIds": [2, 3, 4]},
            "input_mouse_event": {},
            "input_key_event": {},
            "input_insert_text": {},
            "network_enable": {},
            "network_set_headers": {},
            "console_enable": {},
            "console_disable": {},
        }
        
        return mock_responses.get(method, {})
    
    def on_event(self, event_name: str, handler: callable):
        """Register event handler."""
        if event_name not in self._event_handlers:
            self._event_handlers[event_name] = []
        self._event_handlers[event_name].append(handler)
    
    def _emit_event(self, event_name: str, data: Any = None):
        """Emit event to registered handlers."""
        if event_name in self._event_handlers:
            for handler in self._event_handlers[event_name]:
                try:
                    if data:
                        handler(data)
                    else:
                        handler()
                except Exception as e:
                    logger.error(f"Error in event handler for {event_name}: {e}")
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to WebKit."""
        return self._connected
    
    @property
    def browser_type(self) -> BrowserType:
        """Get browser type."""
        if self.capabilities.mobile_emulation:
            return BrowserType.MOBILE_WEBKIT
        return BrowserType.WEBKIT


class WebKitAdapterFactory:
    """Factory for creating WebKit adapter instances."""
    
    @staticmethod
    def create_adapter(capabilities: Optional[WebKitCapabilities] = None) -> WebKitAdapter:
        """Create a new WebKit adapter instance."""
        return WebKitAdapter(capabilities)
    
    @staticmethod
    def create_mobile_adapter(device_name: str, **kwargs) -> WebKitAdapter:
        """Create adapter configured for mobile device emulation."""
        capabilities = WebKitCapabilities(
            mobile_emulation=True,
            device_name=device_name,
            touch_enabled=True,
            **kwargs
        )
        return WebKitAdapter(capabilities)
    
    @staticmethod
    def create_headless_adapter(**kwargs) -> WebKitAdapter:
        """Create headless WebKit adapter."""
        capabilities = WebKitCapabilities(headless=True, **kwargs)
        return WebKitAdapter(capabilities)


# Integration with existing nexus modules
class WebKitPageAdapter:
    """
    Adapter to integrate WebKit with existing Page actor.
    Translates WebKit operations to Page actor interface.
    """
    
    def __init__(self, webkit_adapter: WebKitAdapter, session_id: str, page_id: str):
        self.webkit = webkit_adapter
        self.session_id = session_id
        self.page_id = page_id
    
    async def goto(self, url: str) -> Dict[str, Any]:
        """Navigate to URL."""
        return await self.webkit.navigate(self.session_id, self.page_id, url)
    
    async def evaluate(self, expression: str) -> Any:
        """Evaluate JavaScript."""
        return await self.webkit.evaluate(self.session_id, self.page_id, expression)
    
    async def query_selector(self, selector: str) -> Optional[str]:
        """Find element by selector."""
        return await self.webkit.query_selector(self.session_id, self.page_id, selector)
    
    async def click(self, selector: str) -> None:
        """Click element."""
        element_id = await self.query_selector(selector)
        if element_id:
            # In real implementation, would get element position first
            # For now, click at center of viewport
            await self.webkit.click(self.session_id, self.page_id, 640, 360)
    
    async def type(self, selector: str, text: str) -> None:
        """Type text into element."""
        element_id = await self.query_selector(selector)
        if element_id:
            await self.webkit.click(self.session_id, self.page_id, 640, 360)
            await self.webkit.type_text(self.session_id, self.page_id, text)
    
    async def screenshot(self, path: Optional[str] = None) -> bytes:
        """Take screenshot."""
        data = await self.webkit.screenshot(self.session_id, self.page_id)
        if path:
            with open(path, "wb") as f:
                f.write(data)
        return data


# Export for use in other modules
__all__ = [
    "WebKitAdapter",
    "WebKitAdapterFactory", 
    "WebKitCapabilities",
    "WebKitSession",
    "WebKitPageAdapter",
    "BrowserType",
    "WebKitProtocolError"
]