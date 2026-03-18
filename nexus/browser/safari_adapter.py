"""
Cross-Browser Support Layer for nexus.

This module provides Safari-specific adapter implementation with unified API,
WebDriver BiDi protocol support, browser capability detection, and compatibility layers.
"""

import asyncio
import json
import logging
import platform
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import aiohttp
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.safari.options import Options as SafariOptions
from selenium.webdriver.safari.service import Service as SafariService
from selenium.common.exceptions import (
    WebDriverException,
    NoSuchWindowException,
    NoSuchElementException,
    JavascriptException,
    TimeoutException,
)

from nexus.actor.element import Element
from nexus.actor.page import Page

logger = logging.getLogger(__name__)


class BrowserType(Enum):
    """Supported browser types."""
    CHROME = "chrome"
    FIREFOX = "firefox"
    SAFARI = "safari"
    EDGE = "edge"


class BrowserCapability(Enum):
    """Browser capability flags."""
    CDP = "cdp"  # Chrome DevTools Protocol
    WEBDRIVER_BIDI = "webdriver_bidi"  # WebDriver BiDi protocol
    JAVASCRIPT = "javascript"
    HEADLESS = "headless"
    EXTENSIONS = "extensions"
    TOUCH = "touch"
    PROXY = "proxy"


@dataclass
class BrowserInfo:
    """Information about a browser instance."""
    browser_type: BrowserType
    version: str
    capabilities: Set[BrowserCapability]
    platform: str
    driver_path: Optional[str] = None
    executable_path: Optional[str] = None


class BrowserAdapter(ABC):
    """Abstract base class for browser-specific adapters."""
    
    def __init__(self, browser_type: BrowserType):
        self.browser_type = browser_type
        self.driver: Optional[WebDriver] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._bidi_connection = None
        self._capabilities: Set[BrowserCapability] = set()
        self._page_load_timeout = 30
        self._script_timeout = 30
        self._implicit_wait = 10
        
    @abstractmethod
    async def start(self, **kwargs) -> None:
        """Start the browser session."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the browser session."""
        pass
    
    @abstractmethod
    async def navigate(self, url: str) -> None:
        """Navigate to a URL."""
        pass
    
    @abstractmethod
    async def get_current_url(self) -> str:
        """Get the current URL."""
        pass
    
    @abstractmethod
    async def get_title(self) -> str:
        """Get the page title."""
        pass
    
    @abstractmethod
    async def get_page_source(self) -> str:
        """Get the page source HTML."""
        pass
    
    @abstractmethod
    async def find_element(self, selector: str, by: By = By.CSS_SELECTOR) -> Optional[Element]:
        """Find an element on the page."""
        pass
    
    @abstractmethod
    async def find_elements(self, selector: str, by: By = By.CSS_SELECTOR) -> List[Element]:
        """Find multiple elements on the page."""
        pass
    
    @abstractmethod
    async def execute_script(self, script: str, *args) -> Any:
        """Execute JavaScript in the browser."""
        pass
    
    @abstractmethod
    async def execute_async_script(self, script: str, *args) -> Any:
        """Execute asynchronous JavaScript in the browser."""
        pass
    
    @abstractmethod
    async def take_screenshot(self) -> bytes:
        """Take a screenshot of the current page."""
        pass
    
    @abstractmethod
    async def get_cookies(self) -> List[Dict[str, Any]]:
        """Get all cookies."""
        pass
    
    @abstractmethod
    async def set_cookie(self, cookie: Dict[str, Any]) -> None:
        """Set a cookie."""
        pass
    
    @abstractmethod
    async def delete_cookies(self) -> None:
        """Delete all cookies."""
        pass
    
    @abstractmethod
    async def get_window_size(self) -> Tuple[int, int]:
        """Get the browser window size."""
        pass
    
    @abstractmethod
    async def set_window_size(self, width: int, height: int) -> None:
        """Set the browser window size."""
        pass
    
    @abstractmethod
    async def get_capabilities(self) -> Set[BrowserCapability]:
        """Get the capabilities of this browser."""
        pass
    
    @abstractmethod
    async def is_javascript_enabled(self) -> bool:
        """Check if JavaScript is enabled."""
        pass
    
    @abstractmethod
    async def wait_for_element(
        self, 
        selector: str, 
        by: By = By.CSS_SELECTOR,
        timeout: float = 10
    ) -> Element:
        """Wait for an element to appear."""
        pass
    
    async def create_page(self) -> Page:
        """Create a Page object for the current browser session."""
        return Page(self)


class BiDiProtocol:
    """WebDriver BiDi protocol implementation for cross-browser automation."""
    
    def __init__(self, websocket_url: str):
        self.websocket_url = websocket_url
        self._ws = None
        self._session_id = None
        self._event_handlers = {}
        self._pending_commands = {}
        self._command_id = 0
    
    async def connect(self) -> None:
        """Connect to the BiDi websocket."""
        try:
            import websockets
            self._ws = await websockets.connect(self.websocket_url)
            logger.info(f"Connected to BiDi websocket at {self.websocket_url}")
        except Exception as e:
            logger.error(f"Failed to connect to BiDi websocket: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from the BiDi websocket."""
        if self._ws:
            await self._ws.close()
            self._ws = None
    
    async def send_command(self, method: str, params: Dict[str, Any] = None) -> Any:
        """Send a command via BiDi protocol."""
        if not self._ws:
            raise RuntimeError("BiDi websocket not connected")
        
        self._command_id += 1
        command = {
            "id": self._command_id,
            "method": method,
            "params": params or {}
        }
        
        if self._session_id:
            command["sessionId"] = self._session_id
        
        await self._ws.send(json.dumps(command))
        
        # Wait for response
        response = await self._ws.recv()
        response_data = json.loads(response)
        
        if "error" in response_data:
            raise RuntimeError(f"BiDi command failed: {response_data['error']}")
        
        return response_data.get("result")
    
    async def subscribe(self, event: str, handler) -> None:
        """Subscribe to a BiDi event."""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        
        self._event_handlers[event].append(handler)
        
        # Subscribe to the event via BiDi
        await self.send_command("session.subscribe", {
            "events": [event]
        })
    
    async def _handle_event(self, event_data: Dict[str, Any]) -> None:
        """Handle incoming BiDi events."""
        event_type = event_data.get("method")
        if event_type in self._event_handlers:
            for handler in self._event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event_data.get("params", {}))
                    else:
                        handler(event_data.get("params", {}))
                except Exception as e:
                    logger.error(f"Error in BiDi event handler: {e}")


class SafariAdapter(BrowserAdapter):
    """Safari-specific browser adapter implementation."""
    
    SAFARI_CAPABILITIES = {
        BrowserCapability.JAVASCRIPT,
        BrowserCapability.TOUCH,
        BrowserCapability.WEBDRIVER_BIDI,
    }
    
    def __init__(self):
        super().__init__(BrowserType.SAFARI)
        self._service = None
        self._safari_options = SafariOptions()
        self._driver_path = self._find_safaridriver()
        self._capabilities = self.SAFARI_CAPABILITIES.copy()
        
    def _find_safaridriver(self) -> Optional[str]:
        """Find the SafariDriver executable path."""
        try:
            # Try to find safaridriver in PATH
            result = subprocess.run(
                ["which", "safaridriver"],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                return result.stdout.strip()
            
            # Default locations on macOS
            if platform.system() == "Darwin":
                default_paths = [
                    "/usr/bin/safaridriver",
                    "/Applications/Safari.app/Contents/MacOS/safaridriver"
                ]
                for path in default_paths:
                    if Path(path).exists():
                        return path
            
            logger.warning("SafariDriver not found in standard locations")
            return None
            
        except Exception as e:
            logger.error(f"Error finding SafariDriver: {e}")
            return None
    
    async def start(self, **kwargs) -> None:
        """Start Safari browser session."""
        try:
            # Configure Safari options
            if "headless" in kwargs and kwargs["headless"]:
                logger.warning("Safari does not support headless mode")
            
            if "private_browsing" in kwargs and kwargs["private_browsing"]:
                self._safari_options.add_argument("--private")
            
            # Set up Safari service
            if self._driver_path:
                self._service = SafariService(executable_path=self._driver_path)
            else:
                self._service = SafariService()
            
            # Initialize the driver
            self.driver = webdriver.Safari(
                options=self._safari_options,
                service=self._service
            )
            
            # Configure timeouts
            self.driver.set_page_load_timeout(self._page_load_timeout)
            self.driver.set_script_timeout(self._script_timeout)
            self.driver.implicitly_wait(self._implicit_wait)
            
            # Try to establish BiDi connection
            await self._setup_bidi_connection()
            
            logger.info("Safari browser session started successfully")
            
        except WebDriverException as e:
            logger.error(f"Failed to start Safari: {e}")
            raise RuntimeError(f"Could not start Safari browser: {e}")
    
    async def _setup_bidi_connection(self) -> None:
        """Set up WebDriver BiDi connection for Safari."""
        try:
            # Safari supports BiDi protocol
            capabilities = self.driver.capabilities
            bidi_url = capabilities.get("webSocketUrl")
            
            if bidi_url:
                self._bidi_connection = BiDiProtocol(bidi_url)
                await self._bidi_connection.connect()
                logger.info("BiDi connection established for Safari")
                
                # Subscribe to useful events
                await self._bidi_connection.subscribe("log.entryAdded", self._handle_log_event)
                await self._bidi_connection.subscribe("network.responseCompleted", self._handle_network_event)
            else:
                logger.warning("BiDi not available for this Safari session")
                
        except Exception as e:
            logger.warning(f"Could not establish BiDi connection: {e}")
            self._bidi_connection = None
    
    async def _handle_log_event(self, params: Dict[str, Any]) -> None:
        """Handle log events from BiDi."""
        logger.debug(f"Browser log: {params}")
    
    async def _handle_network_event(self, params: Dict[str, Any]) -> None:
        """Handle network events from BiDi."""
        logger.debug(f"Network event: {params}")
    
    async def stop(self) -> None:
        """Stop Safari browser session."""
        try:
            if self._bidi_connection:
                await self._bidi_connection.disconnect()
                self._bidi_connection = None
            
            if self.driver:
                self.driver.quit()
                self.driver = None
            
            if self._service:
                self._service.stop()
                self._service = None
            
            logger.info("Safari browser session stopped")
            
        except Exception as e:
            logger.error(f"Error stopping Safari: {e}")
    
    async def navigate(self, url: str) -> None:
        """Navigate to a URL in Safari."""
        if not self.driver:
            raise RuntimeError("Browser not started")
        
        try:
            self.driver.get(url)
            logger.debug(f"Navigated to {url}")
        except WebDriverException as e:
            logger.error(f"Navigation failed: {e}")
            raise
    
    async def get_current_url(self) -> str:
        """Get the current URL from Safari."""
        if not self.driver:
            raise RuntimeError("Browser not started")
        return self.driver.current_url
    
    async def get_title(self) -> str:
        """Get the page title from Safari."""
        if not self.driver:
            raise RuntimeError("Browser not started")
        return self.driver.title
    
    async def get_page_source(self) -> str:
        """Get the page source HTML from Safari."""
        if not self.driver:
            raise RuntimeError("Browser not started")
        return self.driver.page_source
    
    async def find_element(self, selector: str, by: By = By.CSS_SELECTOR) -> Optional[Element]:
        """Find an element on the page in Safari."""
        if not self.driver:
            raise RuntimeError("Browser not started")
        
        try:
            web_element = self.driver.find_element(by, selector)
            return Element(web_element, selector, by)
        except NoSuchElementException:
            return None
        except WebDriverException as e:
            logger.error(f"Error finding element {selector}: {e}")
            raise
    
    async def find_elements(self, selector: str, by: By = By.CSS_SELECTOR) -> List[Element]:
        """Find multiple elements on the page in Safari."""
        if not self.driver:
            raise RuntimeError("Browser not started")
        
        try:
            web_elements = self.driver.find_elements(by, selector)
            return [Element(we, selector, by) for we in web_elements]
        except WebDriverException as e:
            logger.error(f"Error finding elements {selector}: {e}")
            raise
    
    async def execute_script(self, script: str, *args) -> Any:
        """Execute JavaScript in Safari."""
        if not self.driver:
            raise RuntimeError("Browser not started")
        
        try:
            return self.driver.execute_script(script, *args)
        except JavascriptException as e:
            logger.error(f"JavaScript execution failed: {e}")
            raise
        except WebDriverException as e:
            logger.error(f"Script execution error: {e}")
            raise
    
    async def execute_async_script(self, script: str, *args) -> Any:
        """Execute asynchronous JavaScript in Safari."""
        if not self.driver:
            raise RuntimeError("Browser not started")
        
        try:
            return self.driver.execute_async_script(script, *args)
        except JavascriptException as e:
            logger.error(f"Async JavaScript execution failed: {e}")
            raise
        except TimeoutException:
            logger.error("Async script execution timed out")
            raise
        except WebDriverException as e:
            logger.error(f"Async script execution error: {e}")
            raise
    
    async def take_screenshot(self) -> bytes:
        """Take a screenshot in Safari."""
        if not self.driver:
            raise RuntimeError("Browser not started")
        
        try:
            return self.driver.get_screenshot_as_png()
        except WebDriverException as e:
            logger.error(f"Screenshot failed: {e}")
            raise
    
    async def get_cookies(self) -> List[Dict[str, Any]]:
        """Get all cookies from Safari."""
        if not self.driver:
            raise RuntimeError("Browser not started")
        
        try:
            return self.driver.get_cookies()
        except WebDriverException as e:
            logger.error(f"Failed to get cookies: {e}")
            raise
    
    async def set_cookie(self, cookie: Dict[str, Any]) -> None:
        """Set a cookie in Safari."""
        if not self.driver:
            raise RuntimeError("Browser not started")
        
        try:
            self.driver.add_cookie(cookie)
        except WebDriverException as e:
            logger.error(f"Failed to set cookie: {e}")
            raise
    
    async def delete_cookies(self) -> None:
        """Delete all cookies in Safari."""
        if not self.driver:
            raise RuntimeError("Browser not started")
        
        try:
            self.driver.delete_all_cookies()
        except WebDriverException as e:
            logger.error(f"Failed to delete cookies: {e}")
            raise
    
    async def get_window_size(self) -> Tuple[int, int]:
        """Get the Safari window size."""
        if not self.driver:
            raise RuntimeError("Browser not started")
        
        try:
            size = self.driver.get_window_size()
            return (size["width"], size["height"])
        except WebDriverException as e:
            logger.error(f"Failed to get window size: {e}")
            raise
    
    async def set_window_size(self, width: int, height: int) -> None:
        """Set the Safari window size."""
        if not self.driver:
            raise RuntimeError("Browser not started")
        
        try:
            self.driver.set_window_size(width, height)
        except WebDriverException as e:
            logger.error(f"Failed to set window size: {e}")
            raise
    
    async def get_capabilities(self) -> Set[BrowserCapability]:
        """Get Safari capabilities."""
        return self._capabilities.copy()
    
    async def is_javascript_enabled(self) -> bool:
        """Check if JavaScript is enabled in Safari."""
        try:
            result = await self.execute_script("return true;")
            return result is True
        except:
            return False
    
    async def wait_for_element(
        self, 
        selector: str, 
        by: By = By.CSS_SELECTOR,
        timeout: float = 10
    ) -> Element:
        """Wait for an element to appear in Safari."""
        if not self.driver:
            raise RuntimeError("Browser not started")
        
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        
        try:
            wait = WebDriverWait(self.driver, timeout)
            web_element = wait.until(
                EC.presence_of_element_located((by, selector))
            )
            return Element(web_element, selector, by)
        except TimeoutException:
            logger.error(f"Timeout waiting for element {selector}")
            raise
        except WebDriverException as e:
            logger.error(f"Error waiting for element {selector}: {e}")
            raise
    
    async def get_browser_info(self) -> BrowserInfo:
        """Get information about the Safari browser."""
        if not self.driver:
            raise RuntimeError("Browser not started")
        
        capabilities = self.driver.capabilities
        version = capabilities.get("browserVersion", "Unknown")
        
        return BrowserInfo(
            browser_type=BrowserType.SAFARI,
            version=version,
            capabilities=self._capabilities,
            platform=platform.system(),
            driver_path=self._driver_path,
            executable_path=capabilities.get("browserPath")
        )
    
    async def enable_javascript(self) -> None:
        """Enable JavaScript in Safari (note: Safari has JS enabled by default)."""
        # Safari doesn't have a direct way to enable/disable JS via WebDriver
        # JavaScript is typically always enabled in Safari
        logger.info("JavaScript is enabled by default in Safari")
    
    async def disable_javascript(self) -> None:
        """Disable JavaScript in Safari (limited support)."""
        logger.warning("Safari WebDriver does not support disabling JavaScript")
    
    async def set_user_agent(self, user_agent: str) -> None:
        """Set custom user agent in Safari."""
        if not self.driver:
            raise RuntimeError("Browser not started")
        
        try:
            # Safari doesn't support changing UA via WebDriver directly
            # We can try using BiDi if available
            if self._bidi_connection:
                await self._bidi_connection.send_command(
                    "emulation.setUserAgentOverride",
                    {"userAgent": user_agent}
                )
            else:
                logger.warning("Cannot set user agent: BiDi not available")
        except Exception as e:
            logger.error(f"Failed to set user agent: {e}")
    
    async def clear_cache(self) -> None:
        """Clear browser cache in Safari."""
        if not self.driver:
            raise RuntimeError("Browser not started")
        
        try:
            # Try using BiDi to clear cache
            if self._bidi_connection:
                await self._bidi_connection.send_command("storage.clearCookies")
                await self._bidi_connection.send_command("storage.clearCache")
            else:
                # Fallback: clear via JavaScript
                await self.execute_script("""
                    if (window.caches) {
                        caches.keys().then(names => {
                            names.forEach(name => caches.delete(name));
                        });
                    }
                """)
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    async def get_network_logs(self) -> List[Dict[str, Any]]:
        """Get network logs (requires BiDi support)."""
        if not self._bidi_connection:
            logger.warning("Network logs require BiDi support")
            return []
        
        try:
            # This would require implementing network event collection
            # For now, return empty list
            return []
        except Exception as e:
            logger.error(f"Failed to get network logs: {e}")
            return []
    
    async def handle_safari_specific_issue(self, issue_type: str) -> bool:
        """Handle Safari-specific compatibility issues."""
        safari_issues = {
            "popup_blocking": self._handle_popup_blocking,
            "cross_origin": self._handle_cross_origin,
            "mixed_content": self._handle_mixed_content,
            "autoplay": self._handle_autoplay,
        }
        
        handler = safari_issues.get(issue_type)
        if handler:
            return await handler()
        return False
    
    async def _handle_popup_blocking(self) -> bool:
        """Handle Safari's popup blocking."""
        try:
            # Try to detect if popups are blocked
            result = await self.execute_script("""
                try {
                    var popup = window.open('', '_blank');
                    if (popup && popup.closed) {
                        return 'blocked';
                    }
                    if (popup) {
                        popup.close();
                        return 'allowed';
                    }
                } catch (e) {
                    return 'blocked';
                }
                return 'unknown';
            """)
            return result == 'blocked'
        except:
            return False
    
    async def _handle_cross_origin(self) -> bool:
        """Handle Safari's cross-origin restrictions."""
        # Safari has strict CORS policies
        logger.info("Safari enforces strict CORS policies")
        return True
    
    async def _handle_mixed_content(self) -> bool:
        """Handle Safari's mixed content blocking."""
        try:
            result = await self.execute_script("""
                return {
                    hasInsecureContent: document.querySelectorAll(
                        'img[src^="http:"], script[src^="http:"], link[href^="http:"]'
                    ).length > 0
                };
            """)
            return result.get('hasInsecureContent', False)
        except:
            return False
    
    async def _handle_autoplay(self) -> bool:
        """Handle Safari's autoplay restrictions."""
        try:
            # Check if autoplay is allowed
            result = await self.execute_script("""
                var video = document.createElement('video');
                video.src = 'data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDE=';
                var promise = video.play();
                if (promise !== undefined) {
                    promise.catch(function(error) {
                        return 'blocked';
                    }).then(function() {
                        return 'allowed';
                    });
                }
                return 'unknown';
            """)
            return result == 'blocked'
        except:
            return False


class BrowserFactory:
    """Factory for creating browser adapters based on browser type."""
    
    _adapters = {
        BrowserType.SAFARI: SafariAdapter,
        # Other adapters would be registered here
    }
    
    @classmethod
    def register_adapter(cls, browser_type: BrowserType, adapter_class):
        """Register a new browser adapter."""
        cls._adapters[browser_type] = adapter_class
    
    @classmethod
    def create_adapter(cls, browser_type: BrowserType) -> BrowserAdapter:
        """Create a browser adapter for the specified browser type."""
        adapter_class = cls._adapters.get(browser_type)
        if not adapter_class:
            raise ValueError(f"No adapter registered for browser type: {browser_type}")
        
        return adapter_class()
    
    @classmethod
    def detect_browser(cls, browser_hint: Optional[str] = None) -> BrowserType:
        """Detect which browser to use based on system and hints."""
        if browser_hint:
            try:
                return BrowserType(browser_hint.lower())
            except ValueError:
                pass
        
        # Default detection logic
        system = platform.system()
        
        if system == "Darwin":  # macOS
            # Check if Safari is available
            if cls._is_safari_available():
                return BrowserType.SAFARI
            # Fallback to Chrome
            return BrowserType.CHROME
        
        elif system == "Windows":
            # Prefer Edge on Windows
            return BrowserType.EDGE
        
        else:  # Linux and others
            # Default to Firefox or Chrome
            return BrowserType.FIREFOX
    
    @staticmethod
    def _is_safari_available() -> bool:
        """Check if Safari is available on the system."""
        if platform.system() != "Darwin":
            return False
        
        try:
            # Check if SafariDriver is available
            result = subprocess.run(
                ["which", "safaridriver"],
                capture_output=True,
                text=True,
                check=False
            )
            return result.returncode == 0
        except:
            return False


class CrossBrowserManager:
    """Manager for cross-browser automation with fallback support."""
    
    def __init__(self, preferred_browsers: Optional[List[BrowserType]] = None):
        self.preferred_browsers = preferred_browsers or [
            BrowserType.CHROME,
            BrowserType.SAFARI,
            BrowserType.FIREFOX,
            BrowserType.EDGE
        ]
        self.current_adapter: Optional[BrowserAdapter] = None
        self.browser_info: Optional[BrowserInfo] = None
    
    async def initialize(self, **kwargs) -> BrowserAdapter:
        """Initialize a browser with fallback support."""
        for browser_type in self.preferred_browsers:
            try:
                adapter = BrowserFactory.create_adapter(browser_type)
                await adapter.start(**kwargs)
                
                # Test if the browser is working
                if await self._test_browser(adapter):
                    self.current_adapter = adapter
                    self.browser_info = await adapter.get_browser_info()
                    logger.info(f"Successfully initialized {browser_type.value}")
                    return adapter
                
                await adapter.stop()
                
            except Exception as e:
                logger.warning(f"Failed to initialize {browser_type.value}: {e}")
                continue
        
        raise RuntimeError("Could not initialize any supported browser")
    
    async def _test_browser(self, adapter: BrowserAdapter) -> bool:
        """Test if a browser adapter is working correctly."""
        try:
            # Test basic navigation
            await adapter.navigate("about:blank")
            
            # Test JavaScript execution
            result = await adapter.execute_script("return 1 + 1;")
            if result != 2:
                return False
            
            # Test element finding
            await adapter.execute_script("""
                document.body.innerHTML = '<div id="test-element">Test</div>';
            """)
            
            element = await adapter.find_element("#test-element")
            if not element:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Browser test failed: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Clean up the current browser session."""
        if self.current_adapter:
            await self.current_adapter.stop()
            self.current_adapter = None
            self.browser_info = None
    
    async def get_optimal_capabilities(self) -> Set[BrowserCapability]:
        """Get the optimal capabilities for the current browser."""
        if not self.current_adapter:
            return set()
        
        capabilities = await self.current_adapter.get_capabilities()
        
        # Add Safari-specific optimizations
        if self.browser_info and self.browser_info.browser_type == BrowserType.SAFARI:
            capabilities.add(BrowserCapability.TOUCH)
            # Safari has excellent BiDi support
            if BrowserCapability.WEBDRIVER_BIDI in capabilities:
                logger.info("Using BiDi protocol for Safari")
        
        return capabilities
    
    async def execute_with_fallback(self, operation, *args, **kwargs):
        """Execute an operation with fallback to alternative methods."""
        if not self.current_adapter:
            raise RuntimeError("No browser initialized")
        
        try:
            return await operation(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Operation failed: {e}")
            
            # Try fallback strategies based on browser type
            if self.browser_info:
                if self.browser_info.browser_type == BrowserType.SAFARI:
                    return await self._safari_fallback(operation, *args, **kwargs)
            
            raise
    
    async def _safari_fallback(self, operation, *args, **kwargs):
        """Safari-specific fallback strategies."""
        # Implement Safari-specific workarounds
        logger.info("Applying Safari fallback strategy")
        
        # Example: Safari sometimes needs extra time for page loads
        await asyncio.sleep(1)
        
        try:
            return await operation(*args, **kwargs)
        except Exception as e:
            logger.error(f"Safari fallback also failed: {e}")
            raise


# Compatibility layer for existing code
class SafariCompatibilityLayer:
    """Provides compatibility layer for Safari-specific issues."""
    
    @staticmethod
    def patch_webdriver():
        """Apply patches to WebDriver for Safari compatibility."""
        # Patch any Safari-specific WebDriver issues
        pass
    
    @staticmethod
    async def handle_safari_quirks(adapter: SafariAdapter):
        """Handle known Safari quirks and issues."""
        quirks = [
            SafariCompatibilityLayer._fix_iframe_focus,
            SafariCompatibilityLayer._fix_input_events,
            SafariCompatibilityLayer._fix_css_animations,
        ]
        
        for quirk in quirks:
            try:
                await quirk(adapter)
            except Exception as e:
                logger.warning(f"Failed to apply Safari quirk fix: {e}")
    
    @staticmethod
    async def _fix_iframe_focus(adapter: SafariAdapter):
        """Fix iframe focus issues in Safari."""
        await adapter.execute_script("""
            // Safari sometimes loses focus in iframes
            window.addEventListener('blur', function() {
                setTimeout(function() {
                    window.focus();
                }, 100);
            });
        """)
    
    @staticmethod
    async def _fix_input_events(adapter: SafariAdapter):
        """Fix input event issues in Safari."""
        await adapter.execute_script("""
            // Safari may not trigger input events properly
            document.addEventListener('input', function(e) {
                if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
                    var event = new Event('change', { bubbles: true });
                    e.target.dispatchEvent(event);
                }
            });
        """)
    
    @staticmethod
    async def _fix_css_animations(adapter: SafariAdapter):
        """Fix CSS animation issues in Safari."""
        await adapter.execute_script("""
            // Safari may have issues with CSS animations
            if (typeof CSS !== 'undefined' && CSS.supports) {
                if (!CSS.supports('animation', 'test 1s')) {
                    document.documentElement.classList.add('no-css-animations');
                }
            }
        """)


# Export public API
__all__ = [
    'BrowserType',
    'BrowserCapability',
    'BrowserInfo',
    'BrowserAdapter',
    'SafariAdapter',
    'BiDiProtocol',
    'BrowserFactory',
    'CrossBrowserManager',
    'SafariCompatibilityLayer',
]