"""Multi-Browser Engine Support with Mobile Adaptation.

This module provides unified browser automation across Chrome, Firefox, WebKit,
and mobile browsers through an adapter pattern with protocol translation layers.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from urllib.parse import urlparse
import websockets
import aiohttp

logger = logging.getLogger(__name__)


class BrowserType(Enum):
    """Supported browser types."""
    CHROME = "chrome"
    FIREFOX = "firefox"
    WEBKIT = "webkit"
    CHROME_MOBILE = "chrome_mobile"
    FIREFOX_MOBILE = "firefox_mobile"
    SAFARI_MOBILE = "safari_mobile"


class DeviceType(Enum):
    """Mobile device types for emulation."""
    IPHONE_12 = "iphone_12"
    IPHONE_12_PRO = "iphone_12_pro"
    IPHONE_SE = "iphone_se"
    IPAD_PRO = "ipad_pro"
    GALAXY_S21 = "galaxy_s21"
    GALAXY_NOTE_20 = "galaxy_note_20"
    PIXEL_5 = "pixel_5"
    CUSTOM = "custom"


@dataclass
class MobileDeviceConfig:
    """Configuration for mobile device emulation."""
    device_type: DeviceType
    width: int
    height: int
    pixel_ratio: float = 2.0
    user_agent: str = ""
    touch_enabled: bool = True
    mobile: bool = True
    screen_orientation: str = "portrait"
    device_scale_factor: float = 2.0
    
    @classmethod
    def get_preset(cls, device_type: DeviceType) -> "MobileDeviceConfig":
        """Get preset configuration for common devices."""
        presets = {
            DeviceType.IPHONE_12: cls(
                device_type=DeviceType.IPHONE_12,
                width=390,
                height=844,
                pixel_ratio=3.0,
                user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
                device_scale_factor=3.0
            ),
            DeviceType.IPHONE_12_PRO: cls(
                device_type=DeviceType.IPHONE_12_PRO,
                width=390,
                height=844,
                pixel_ratio=3.0,
                user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
                device_scale_factor=3.0
            ),
            DeviceType.GALAXY_S21: cls(
                device_type=DeviceType.GALAXY_S21,
                width=360,
                height=800,
                pixel_ratio=3.0,
                user_agent="Mozilla/5.0 (Linux; Android 11; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36",
                device_scale_factor=3.0
            ),
            DeviceType.PIXEL_5: cls(
                device_type=DeviceType.PIXEL_5,
                width=393,
                height=851,
                pixel_ratio=2.75,
                user_agent="Mozilla/5.0 (Linux; Android 11; Pixel 5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36",
                device_scale_factor=2.75
            ),
        }
        return presets.get(device_type, cls(device_type=DeviceType.CUSTOM, width=375, height=667))


@dataclass
class BrowserConfig:
    """Unified browser configuration."""
    browser_type: BrowserType
    headless: bool = True
    viewport_width: int = 1280
    viewport_height: int = 720
    user_agent: Optional[str] = None
    mobile_config: Optional[MobileDeviceConfig] = None
    proxy: Optional[Dict[str, str]] = None
    args: List[str] = field(default_factory=list)
    ignore_https_errors: bool = False
    java_script_enabled: bool = True
    locale: str = "en-US"
    timezone: str = "America/New_York"
    geolocation: Optional[Dict[str, float]] = None
    permissions: List[str] = field(default_factory=list)
    extra_http_headers: Dict[str, str] = field(default_factory=dict)


class BrowserEngineProtocol(ABC):
    """Abstract base class for browser engine protocol implementations."""
    
    @abstractmethod
    async def connect(self, config: BrowserConfig) -> None:
        """Connect to browser instance."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from browser instance."""
        pass
    
    @abstractmethod
    async def new_page(self) -> "BrowserPage":
        """Create a new browser page."""
        pass
    
    @abstractmethod
    async def close_page(self, page_id: str) -> None:
        """Close a browser page."""
        pass
    
    @abstractmethod
    async def get_pages(self) -> List["BrowserPage"]:
        """Get all open pages."""
        pass
    
    @abstractmethod
    async def set_mobile_emulation(self, config: MobileDeviceConfig) -> None:
        """Set mobile device emulation."""
        pass
    
    @abstractmethod
    async def execute_cdp(self, method: str, params: Dict[str, Any] = None) -> Any:
        """Execute Chrome DevTools Protocol command."""
        pass


class ChromeProtocol(BrowserEngineProtocol):
    """Chrome DevTools Protocol implementation."""
    
    def __init__(self):
        self.websocket = None
        self.session_id = None
        self.target_id = None
        self.pages = {}
        self.message_id = 0
        self.callbacks = {}
    
    async def connect(self, config: BrowserConfig) -> None:
        """Connect to Chrome via CDP."""
        # In production, this would launch Chrome with remote debugging
        # or connect to existing Chrome instance
        chrome_url = "ws://localhost:9222/devtools/browser"
        
        try:
            self.websocket = await websockets.connect(chrome_url)
            logger.info("Connected to Chrome DevTools Protocol")
            
            # Get browser version
            response = await self._send_command("Browser.getVersion")
            logger.info(f"Chrome version: {response.get('product', 'Unknown')}")
            
            # Create new target (page)
            target_response = await self._send_command("Target.createTarget", {
                "url": "about:blank",
                "width": config.viewport_width,
                "height": config.viewport_height
            })
            self.target_id = target_response.get("targetId")
            
            # Attach to target
            session_response = await self._send_command("Target.attachToTarget", {
                "targetId": self.target_id,
                "flatten": True
            })
            self.session_id = session_response.get("sessionId")
            
        except Exception as e:
            logger.error(f"Failed to connect to Chrome: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from Chrome."""
        if self.websocket:
            await self.websocket.close()
    
    async def new_page(self) -> "BrowserPage":
        """Create a new Chrome page."""
        target_response = await self._send_command("Target.createTarget", {
            "url": "about:blank"
        })
        target_id = target_response.get("targetId")
        
        # Attach to target
        session_response = await self._send_command("Target.attachToTarget", {
            "targetId": target_id,
            "flatten": True
        })
        session_id = session_response.get("sessionId")
        
        page = ChromePage(self, target_id, session_id)
        self.pages[target_id] = page
        return page
    
    async def close_page(self, page_id: str) -> None:
        """Close a Chrome page."""
        if page_id in self.pages:
            await self._send_command("Target.closeTarget", {"targetId": page_id})
            del self.pages[page_id]
    
    async def get_pages(self) -> List["BrowserPage"]:
        """Get all Chrome pages."""
        return list(self.pages.values())
    
    async def set_mobile_emulation(self, config: MobileDeviceConfig) -> None:
        """Set mobile emulation via CDP."""
        await self._send_command("Emulation.setDeviceMetricsOverride", {
            "width": config.width,
            "height": config.height,
            "deviceScaleFactor": config.device_scale_factor,
            "mobile": config.mobile,
            "screenOrientation": {
                "type": config.screen_orientation,
                "angle": 0
            }
        })
        
        if config.user_agent:
            await self._send_command("Emulation.setUserAgentOverride", {
                "userAgent": config.user_agent,
                "platform": "iPhone" if "iPhone" in config.user_agent else "Android"
            })
        
        if config.touch_enabled:
            await self._send_command("Emulation.setTouchEmulationEnabled", {
                "enabled": True,
                "maxTouchPoints": 5
            })
    
    async def execute_cdp(self, method: str, params: Dict[str, Any] = None) -> Any:
        """Execute CDP command."""
        return await self._send_command(method, params)
    
    async def _send_command(self, method: str, params: Dict[str, Any] = None) -> Any:
        """Send CDP command and wait for response."""
        self.message_id += 1
        message = {
            "id": self.message_id,
            "method": method,
            "params": params or {}
        }
        
        if self.session_id:
            message["sessionId"] = self.session_id
        
        await self.websocket.send(json.dumps(message))
        
        # Wait for response
        while True:
            response = await self.websocket.recv()
            data = json.loads(response)
            
            if data.get("id") == self.message_id:
                if "error" in data:
                    raise Exception(f"CDP error: {data['error']}")
                return data.get("result", {})
            
            # Handle events
            if "method" in data:
                await self._handle_event(data)
    
    async def _handle_event(self, event: Dict[str, Any]) -> None:
        """Handle CDP events."""
        method = event.get("method")
        params = event.get("params", {})
        
        if method == "Target.targetCreated":
            logger.info(f"Target created: {params.get('targetInfo', {}).get('targetId')}")
        elif method == "Page.loadEventFired":
            logger.debug("Page load event fired")


class FirefoxProtocol(BrowserEngineProtocol):
    """Firefox DevTools Protocol implementation."""
    
    def __init__(self):
        self.websocket = None
        self.browser_id = None
        self.pages = {}
        self.message_id = 0
    
    async def connect(self, config: BrowserConfig) -> None:
        """Connect to Firefox via Marionette/Remote Protocol."""
        # Firefox uses Marionette protocol for automation
        firefox_url = "ws://localhost:2828/session"
        
        try:
            self.websocket = await websockets.connect(firefox_url)
            logger.info("Connected to Firefox DevTools Protocol")
            
            # Create new session
            session_response = await self._send_command("WebDriver:NewSession", {
                "capabilities": {
                    "alwaysMatch": {
                        "browserName": "firefox",
                        "moz:firefoxOptions": {
                            "args": ["-headless"] if config.headless else []
                        }
                    }
                }
            })
            self.browser_id = session_response.get("sessionId")
            
        except Exception as e:
            logger.error(f"Failed to connect to Firefox: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from Firefox."""
        if self.websocket:
            await self._send_command("WebDriver:DeleteSession", {})
            await self.websocket.close()
    
    async def new_page(self) -> "BrowserPage":
        """Create a new Firefox page."""
        # Firefox uses tabs/windows
        response = await self._send_command("WebDriver:NewWindow", {
            "type": "tab"
        })
        handle = response.get("handle")
        
        page = FirefoxPage(self, handle)
        self.pages[handle] = page
        return page
    
    async def close_page(self, page_id: str) -> None:
        """Close a Firefox page."""
        if page_id in self.pages:
            await self._send_command("WebDriver:CloseWindow", {})
            del self.pages[page_id]
    
    async def get_pages(self) -> List["BrowserPage"]:
        """Get all Firefox pages."""
        return list(self.pages.values())
    
    async def set_mobile_emulation(self, config: MobileDeviceConfig) -> None:
        """Set mobile emulation for Firefox."""
        # Firefox mobile emulation via user agent and viewport
        if config.user_agent:
            await self._send_command("Marionette:SetContext", {
                "context": "chrome"
            })
            
            await self._send_command("Marionette:ExecuteScript", {
                "script": """
                    Services.prefs.setCharPref(
                        'general.useragent.override',
                        arguments[0]
                    );
                """,
                "args": [config.user_agent]
            })
            
            await self._send_command("Marionette:SetContext", {
                "context": "content"
            })
        
        # Set viewport
        await self._send_command("WebDriver:SetWindowRect", {
            "width": config.width,
            "height": config.height
        })
    
    async def execute_cdp(self, method: str, params: Dict[str, Any] = None) -> Any:
        """Execute Firefox command (CDP translation layer)."""
        # Translate CDP commands to Firefox equivalents
        firefox_method = self._translate_cdp_method(method)
        firefox_params = self._translate_cdp_params(method, params)
        
        return await self._send_command(firefox_method, firefox_params)
    
    def _translate_cdp_method(self, cdp_method: str) -> str:
        """Translate CDP method to Firefox equivalent."""
        translations = {
            "Page.navigate": "WebDriver:NavigateTo",
            "Runtime.evaluate": "WebDriver:ExecuteScript",
            "DOM.getDocument": "Marionette:GetPageSource",
            "Input.dispatchMouseEvent": "WebDriver:PerformActions",
            "Input.dispatchKeyEvent": "WebDriver:PerformActions",
            "Network.enable": "Marionette:SetContext",
            "Emulation.setDeviceMetricsOverride": "WebDriver:SetWindowRect",
        }
        return translations.get(cdp_method, cdp_method)
    
    def _translate_cdp_params(self, cdp_method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Translate CDP parameters to Firefox equivalents."""
        if cdp_method == "Page.navigate" and params:
            return {"url": params.get("url")}
        elif cdp_method == "Runtime.evaluate" and params:
            return {
                "script": params.get("expression"),
                "args": []
            }
        return params or {}
    
    async def _send_command(self, method: str, params: Dict[str, Any] = None) -> Any:
        """Send Firefox command."""
        self.message_id += 1
        message = {
            "id": self.message_id,
            "name": method,
            "parameters": params or {}
        }
        
        await self.websocket.send(json.dumps(message))
        
        response = await self.websocket.recv()
        data = json.loads(response)
        
        if data.get("error"):
            raise Exception(f"Firefox error: {data['error']}")
        
        return data.get("value", data.get("data"))


class WebKitProtocol(BrowserEngineProtocol):
    """WebKit Inspector Protocol implementation."""
    
    def __init__(self):
        self.websocket = None
        self.page_id = None
        self.pages = {}
        self.message_id = 0
    
    async def connect(self, config: BrowserConfig) -> None:
        """Connect to WebKit via Inspector Protocol."""
        # WebKit uses its own inspector protocol
        webkit_url = "ws://localhost:9222/devtools/page"
        
        try:
            self.websocket = await websockets.connect(webkit_url)
            logger.info("Connected to WebKit Inspector Protocol")
            
            # Get available pages
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:9222/json") as resp:
                    targets = await resp.json()
                    if targets:
                        self.page_id = targets[0].get("id")
            
        except Exception as e:
            logger.error(f"Failed to connect to WebKit: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from WebKit."""
        if self.websocket:
            await self.websocket.close()
    
    async def new_page(self) -> "BrowserPage":
        """Create a new WebKit page."""
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:9222/json/new") as resp:
                target = await resp.json()
                page_id = target.get("id")
                
                page = WebKitPage(self, page_id)
                self.pages[page_id] = page
                return page
    
    async def close_page(self, page_id: str) -> None:
        """Close a WebKit page."""
        if page_id in self.pages:
            async with aiohttp.ClientSession() as session:
                await session.get(f"http://localhost:9222/json/close/{page_id}")
            del self.pages[page_id]
    
    async def get_pages(self) -> List["BrowserPage"]:
        """Get all WebKit pages."""
        return list(self.pages.values())
    
    async def set_mobile_emulation(self, config: MobileDeviceConfig) -> None:
        """Set mobile emulation for WebKit."""
        # WebKit mobile emulation
        await self._send_command("Page.setDeviceMetricsOverride", {
            "width": config.width,
            "height": config.height,
            "deviceScaleFactor": config.device_scale_factor,
            "mobile": config.mobile,
            "screenOrientation": {
                "type": config.screen_orientation,
                "angle": 0
            }
        })
        
        if config.user_agent:
            await self._send_command("Network.setUserAgentOverride", {
                "userAgent": config.user_agent
            })
    
    async def execute_cdp(self, method: str, params: Dict[str, Any] = None) -> Any:
        """Execute WebKit command (CDP translation layer)."""
        # WebKit uses similar protocol to CDP with some differences
        return await self._send_command(method, params)
    
    async def _send_command(self, method: str, params: Dict[str, Any] = None) -> Any:
        """Send WebKit command."""
        self.message_id += 1
        message = {
            "id": self.message_id,
            "method": method,
            "params": params or {}
        }
        
        await self.websocket.send(json.dumps(message))
        
        response = await self.websocket.recv()
        data = json.loads(response)
        
        if "error" in data:
            raise Exception(f"WebKit error: {data['error']}")
        
        return data.get("result", {})


class BrowserPage(ABC):
    """Abstract base class for browser page operations."""
    
    def __init__(self, engine: BrowserEngineProtocol, page_id: str):
        self.engine = engine
        self.page_id = page_id
        self.url = "about:blank"
        self.title = ""
    
    @abstractmethod
    async def navigate(self, url: str) -> None:
        """Navigate to URL."""
        pass
    
    @abstractmethod
    async def get_content(self) -> str:
        """Get page HTML content."""
        pass
    
    @abstractmethod
    async def evaluate(self, expression: str) -> Any:
        """Evaluate JavaScript expression."""
        pass
    
    @abstractmethod
    async def click(self, selector: str) -> None:
        """Click element by selector."""
        pass
    
    @abstractmethod
    async def type(self, selector: str, text: str) -> None:
        """Type text into element."""
        pass
    
    @abstractmethod
    async def screenshot(self, path: str = None) -> bytes:
        """Take screenshot."""
        pass
    
    @abstractmethod
    async def wait_for_selector(self, selector: str, timeout: int = 30000) -> None:
        """Wait for element to appear."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the page."""
        pass


class ChromePage(BrowserPage):
    """Chrome page implementation."""
    
    async def navigate(self, url: str) -> None:
        """Navigate using CDP."""
        await self.engine.execute_cdp("Page.navigate", {"url": url})
        self.url = url
    
    async def get_content(self) -> str:
        """Get page content via CDP."""
        result = await self.engine.execute_cdp("Runtime.evaluate", {
            "expression": "document.documentElement.outerHTML"
        })
        return result.get("result", {}).get("value", "")
    
    async def evaluate(self, expression: str) -> Any:
        """Evaluate JavaScript via CDP."""
        result = await self.engine.execute_cdp("Runtime.evaluate", {
            "expression": expression,
            "returnByValue": True
        })
        return result.get("result", {}).get("value")
    
    async def click(self, selector: str) -> None:
        """Click element via CDP."""
        # Get element position
        result = await self.engine.execute_cdp("Runtime.evaluate", {
            "expression": f"""
                const element = document.querySelector('{selector}');
                if (!element) throw new Error('Element not found: {selector}');
                const rect = element.getBoundingClientRect();
                JSON.stringify({{
                    x: rect.x + rect.width / 2,
                    y: rect.y + rect.height / 2
                }});
            """
        })
        
        position = json.loads(result.get("result", {}).get("value", "{}"))
        
        # Dispatch mouse events
        await self.engine.execute_cdp("Input.dispatchMouseEvent", {
            "type": "mousePressed",
            "x": position["x"],
            "y": position["y"],
            "button": "left",
            "clickCount": 1
        })
        
        await self.engine.execute_cdp("Input.dispatchMouseEvent", {
            "type": "mouseReleased",
            "x": position["x"],
            "y": position["y"],
            "button": "left",
            "clickCount": 1
        })
    
    async def type(self, selector: str, text: str) -> None:
        """Type text via CDP."""
        # Focus element
        await self.engine.execute_cdp("Runtime.evaluate", {
            "expression": f"document.querySelector('{selector}').focus()"
        })
        
        # Type each character
        for char in text:
            await self.engine.execute_cdp("Input.dispatchKeyEvent", {
                "type": "keyDown",
                "text": char,
                "key": char
            })
            
            await self.engine.execute_cdp("Input.dispatchKeyEvent", {
                "type": "keyUp",
                "key": char
            })
    
    async def screenshot(self, path: str = None) -> bytes:
        """Take screenshot via CDP."""
        result = await self.engine.execute_cdp("Page.captureScreenshot", {
            "format": "png"
        })
        
        screenshot_data = result.get("data", "")
        if path:
            import base64
            with open(path, "wb") as f:
                f.write(base64.b64decode(screenshot_data))
        
        return base64.b64decode(screenshot_data) if screenshot_data else b""
    
    async def wait_for_selector(self, selector: str, timeout: int = 30000) -> None:
        """Wait for selector via CDP."""
        script = f"""
            new Promise((resolve, reject) => {{
                const timeout = setTimeout(() => reject(new Error('Timeout')), {timeout});
                const check = () => {{
                    const element = document.querySelector('{selector}');
                    if (element) {{
                        clearTimeout(timeout);
                        resolve();
                    }} else {{
                        setTimeout(check, 100);
                    }}
                }};
                check();
            }});
        """
        
        await self.engine.execute_cdp("Runtime.evaluate", {
            "expression": script,
            "awaitPromise": True
        })
    
    async def close(self) -> None:
        """Close Chrome page."""
        await self.engine.close_page(self.page_id)


class FirefoxPage(BrowserPage):
    """Firefox page implementation."""
    
    async def navigate(self, url: str) -> None:
        """Navigate using Firefox protocol."""
        await self.engine.execute_cdp("Page.navigate", {"url": url})
        self.url = url
    
    async def get_content(self) -> str:
        """Get page content."""
        result = await self.engine.execute_cdp("DOM.getDocument", {})
        return result.get("root", {}).get("outerHTML", "")
    
    async def evaluate(self, expression: str) -> Any:
        """Evaluate JavaScript."""
        result = await self.engine.execute_cdp("Runtime.evaluate", {
            "expression": expression
        })
        return result
    
    async def click(self, selector: str) -> None:
        """Click element."""
        await self.engine.execute_cdp("Runtime.evaluate", {
            "expression": f"document.querySelector('{selector}').click()"
        })
    
    async def type(self, selector: str, text: str) -> None:
        """Type text."""
        await self.engine.execute_cdp("Runtime.evaluate", {
            "expression": f"""
                const element = document.querySelector('{selector}');
                element.value = '{text}';
                element.dispatchEvent(new Event('input', {{ bubbles: true }}));
            """
        })
    
    async def screenshot(self, path: str = None) -> bytes:
        """Take screenshot."""
        # Firefox screenshot implementation
        result = await self.engine.execute_cdp("Page.captureScreenshot", {})
        return b""  # Placeholder
    
    async def wait_for_selector(self, selector: str, timeout: int = 30000) -> None:
        """Wait for selector."""
        await self.engine.execute_cdp("Runtime.evaluate", {
            "expression": f"""
                await new Promise((resolve) => {{
                    const check = () => {{
                        if (document.querySelector('{selector}')) resolve();
                        else setTimeout(check, 100);
                    }};
                    check();
                }});
            """
        })
    
    async def close(self) -> None:
        """Close Firefox page."""
        await self.engine.close_page(self.page_id)


class WebKitPage(BrowserPage):
    """WebKit page implementation."""
    
    async def navigate(self, url: str) -> None:
        """Navigate using WebKit protocol."""
        await self.engine.execute_cdp("Page.navigate", {"url": url})
        self.url = url
    
    async def get_content(self) -> str:
        """Get page content."""
        result = await self.engine.execute_cdp("DOM.getDocument", {})
        return result.get("root", {}).get("outerHTML", "")
    
    async def evaluate(self, expression: str) -> Any:
        """Evaluate JavaScript."""
        result = await self.engine.execute_cdp("Runtime.evaluate", {
            "expression": expression
        })
        return result.get("result", {}).get("value")
    
    async def click(self, selector: str) -> None:
        """Click element."""
        await self.engine.execute_cdp("Runtime.evaluate", {
            "expression": f"document.querySelector('{selector}').click()"
        })
    
    async def type(self, selector: str, text: str) -> None:
        """Type text."""
        await self.engine.execute_cdp("Runtime.evaluate", {
            "expression": f"""
                const element = document.querySelector('{selector}');
                element.value = '{text}';
                element.dispatchEvent(new Event('input', {{ bubbles: true }}));
            """
        })
    
    async def screenshot(self, path: str = None) -> bytes:
        """Take screenshot."""
        result = await self.engine.execute_cdp("Page.captureScreenshot", {})
        return b""  # Placeholder
    
    async def wait_for_selector(self, selector: str, timeout: int = 30000) -> None:
        """Wait for selector."""
        await self.engine.execute_cdp("Runtime.evaluate", {
            "expression": f"""
                await new Promise((resolve) => {{
                    const check = () => {{
                        if (document.querySelector('{selector}')) resolve();
                        else setTimeout(check, 100);
                    }};
                    check();
                }});
            """
        })
    
    async def close(self) -> None:
        """Close WebKit page."""
        await self.engine.close_page(self.page_id)


class MobileAdapter:
    """Mobile browser adaptation layer."""
    
    def __init__(self, engine: BrowserEngineProtocol):
        self.engine = engine
        self.original_config = None
        self.mobile_enabled = False
    
    async def enable_mobile_emulation(self, device_config: MobileDeviceConfig) -> None:
        """Enable mobile device emulation."""
        self.original_config = {
            "viewport": (1280, 720),  # Default desktop viewport
            "user_agent": None
        }
        
        await self.engine.set_mobile_emulation(device_config)
        self.mobile_enabled = True
        
        logger.info(f"Mobile emulation enabled for {device_config.device_type.value}")
    
    async def disable_mobile_emulation(self) -> None:
        """Disable mobile emulation and restore desktop settings."""
        if self.original_config:
            # Restore original settings
            await self.engine.set_mobile_emulation(MobileDeviceConfig(
                device_type=DeviceType.CUSTOM,
                width=self.original_config["viewport"][0],
                height=self.original_config["viewport"][1],
                user_agent=self.original_config["user_agent"] or "",
                mobile=False,
                touch_enabled=False
            ))
            self.mobile_enabled = False
    
    async def rotate_orientation(self, orientation: str) -> None:
        """Rotate device orientation (portrait/landscape)."""
        if not self.mobile_enabled:
            raise RuntimeError("Mobile emulation not enabled")
        
        # Get current mobile config and update orientation
        # This would require storing the current config
        pass
    
    async def simulate_touch_gesture(self, gesture_type: str, **kwargs) -> None:
        """Simulate touch gestures (swipe, pinch, etc.)."""
        gestures = {
            "swipe": self._simulate_swipe,
            "pinch": self._simulate_pinch,
            "tap": self._simulate_tap,
            "long_press": self._simulate_long_press,
        }
        
        if gesture_type in gestures:
            await gestures[gesture_type](**kwargs)
    
    async def _simulate_swipe(self, start_x: int, start_y: int, 
                             end_x: int, end_y: int, duration: int = 300) -> None:
        """Simulate swipe gesture."""
        # Implementation would use Input.dispatchTouchEvent
        pass
    
    async def _simulate_pinch(self, center_x: int, center_y: int, 
                             scale: float, duration: int = 300) -> None:
        """Simulate pinch gesture."""
        pass
    
    async def _simulate_tap(self, x: int, y: int) -> None:
        """Simulate tap gesture."""
        pass
    
    async def _simulate_long_press(self, x: int, y: int, duration: int = 1000) -> None:
        """Simulate long press gesture."""
        pass


class BrowserEngineFactory:
    """Factory for creating browser engine instances."""
    
    @staticmethod
    def create_engine(browser_type: BrowserType) -> BrowserEngineProtocol:
        """Create browser engine instance based on type."""
        engines = {
            BrowserType.CHROME: ChromeProtocol,
            BrowserType.FIREFOX: FirefoxProtocol,
            BrowserType.WEBKIT: WebKitProtocol,
            BrowserType.CHROME_MOBILE: ChromeProtocol,  # Chrome with mobile emulation
            BrowserType.FIREFOX_MOBILE: FirefoxProtocol,  # Firefox with mobile emulation
            BrowserType.SAFARI_MOBILE: WebKitProtocol,  # WebKit with mobile emulation
        }
        
        engine_class = engines.get(browser_type)
        if not engine_class:
            raise ValueError(f"Unsupported browser type: {browser_type}")
        
        return engine_class()
    
    @staticmethod
    def get_mobile_config(browser_type: BrowserType, device_type: DeviceType) -> MobileDeviceConfig:
        """Get mobile configuration for browser and device type."""
        config = MobileDeviceConfig.get_preset(device_type)
        
        # Adjust user agent based on browser
        if browser_type == BrowserType.CHROME_MOBILE:
            config.user_agent = config.user_agent.replace("Safari", "Chrome")
        elif browser_type == BrowserType.FIREFOX_MOBILE:
            config.user_agent = config.user_agent.replace("Chrome", "Firefox")
        
        return config


class UnifiedBrowserSession:
    """Unified browser session manager with multi-engine support."""
    
    def __init__(self, config: BrowserConfig):
        self.config = config
        self.engine = None
        self.mobile_adapter = None
        self.pages = []
        self._is_connected = False
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    async def connect(self) -> None:
        """Connect to browser."""
        if self._is_connected:
            return
        
        # Create engine
        self.engine = BrowserEngineFactory.create_engine(self.config.browser_type)
        
        # Connect to browser
        await self.engine.connect(self.config)
        
        # Setup mobile adapter if needed
        if self.config.mobile_config:
            self.mobile_adapter = MobileAdapter(self.engine)
            await self.mobile_adapter.enable_mobile_emulation(self.config.mobile_config)
        
        self._is_connected = True
        logger.info(f"Connected to {self.config.browser_type.value}")
    
    async def disconnect(self) -> None:
        """Disconnect from browser."""
        if not self._is_connected:
            return
        
        # Close all pages
        for page in self.pages:
            try:
                await page.close()
            except Exception as e:
                logger.warning(f"Error closing page: {e}")
        
        # Disconnect engine
        if self.engine:
            await self.engine.disconnect()
        
        self._is_connected = False
        logger.info("Disconnected from browser")
    
    async def new_page(self) -> BrowserPage:
        """Create new browser page."""
        if not self._is_connected:
            await self.connect()
        
        page = await self.engine.new_page()
        self.pages.append(page)
        return page
    
    async def execute_javascript(self, page: BrowserPage, script: str) -> Any:
        """Execute JavaScript in page context."""
        return await page.evaluate(script)
    
    async def take_screenshot(self, page: BrowserPage, path: str = None) -> bytes:
        """Take screenshot of page."""
        return await page.screenshot(path)
    
    async def navigate(self, page: BrowserPage, url: str) -> None:
        """Navigate page to URL."""
        await page.navigate(url)
    
    async def get_page_content(self, page: BrowserPage) -> str:
        """Get page HTML content."""
        return await page.get_content()
    
    async def click_element(self, page: BrowserPage, selector: str) -> None:
        """Click element on page."""
        await page.click(selector)
    
    async def type_text(self, page: BrowserPage, selector: str, text: str) -> None:
        """Type text into element."""
        await page.type(selector, text)
    
    async def wait_for_element(self, page: BrowserPage, selector: str, timeout: int = 30000) -> None:
        """Wait for element to appear."""
        await page.wait_for_selector(selector, timeout)
    
    async def switch_to_mobile(self, device_type: DeviceType) -> None:
        """Switch to mobile emulation."""
        if not self.mobile_adapter:
            self.mobile_adapter = MobileAdapter(self.engine)
        
        mobile_config = BrowserEngineFactory.get_mobile_config(
            self.config.browser_type, device_type
        )
        await self.mobile_adapter.enable_mobile_emulation(mobile_config)
    
    async def switch_to_desktop(self) -> None:
        """Switch back to desktop mode."""
        if self.mobile_adapter:
            await self.mobile_adapter.disable_mobile_emulation()
    
    async def simulate_mobile_gesture(self, gesture_type: str, **kwargs) -> None:
        """Simulate mobile gesture."""
        if self.mobile_adapter:
            await self.mobile_adapter.simulate_touch_gesture(gesture_type, **kwargs)


# Integration with existing nexus codebase
def integrate_with_existing_codebase():
    """
    Integration points with existing nexus modules:
    
    1. nexus/actor/page.py:
       - Replace direct Playwright usage with UnifiedBrowserSession
       - Add browser_type parameter to Page class
    
    2. nexus/actor/element.py:
       - Update element interactions to use unified API
       - Add mobile-specific element handling
    
    3. nexus/agent/service.py:
       - Add browser engine configuration to agent setup
       - Support multiple browser types in agent workflows
    
    4. nexus/actor/playground/:
       - Update playground to support browser switching
       - Add mobile emulation controls to playground UI
    """
    pass


# Example usage
async def example_usage():
    """Example of using the unified browser API."""
    
    # Desktop Chrome
    chrome_config = BrowserConfig(
        browser_type=BrowserType.CHROME,
        headless=True,
        viewport_width=1920,
        viewport_height=1080
    )
    
    async with UnifiedBrowserSession(chrome_config) as session:
        page = await session.new_page()
        await session.navigate(page, "https://example.com")
        content = await session.get_page_content(page)
        print(f"Page title: {await page.evaluate('document.title')}")
    
    # Mobile Safari
    mobile_config = BrowserConfig(
        browser_type=BrowserType.SAFARI_MOBILE,
        headless=True,
        mobile_config=MobileDeviceConfig.get_preset(DeviceType.IPHONE_12)
    )
    
    async with UnifiedBrowserSession(mobile_config) as session:
        page = await session.new_page()
        await session.navigate(page, "https://example.com")
        
        # Simulate mobile interactions
        await session.simulate_mobile_gesture("tap", x=100, y=200)
        
        # Rotate to landscape
        await session.mobile_adapter.rotate_orientation("landscape")


if __name__ == "__main__":
    asyncio.run(example_usage())