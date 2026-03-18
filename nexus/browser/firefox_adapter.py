"""
Cross-Browser Support Layer for SOVEREIGN's nexus module.
Implements Firefox adapter with WebDriver BiDi protocol support.
"""

import asyncio
import json
import logging
import platform
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from urllib.parse import urlparse

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.firefox.options import Options as FirefoxOptions
    from selenium.webdriver.firefox.service import Service as FirefoxService
    from selenium.webdriver.remote.webdriver import WebDriver
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.common.exceptions import (
        TimeoutException,
        NoSuchElementException,
        WebDriverException,
        JavascriptException
    )
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

from nexus.actor.element import Element
from nexus.actor.page import Page
from nexus.actor.mouse import Mouse
from nexus.actor.utils import retry_async, timeout_async


logger = logging.getLogger(__name__)


class BrowserType(Enum):
    """Supported browser types."""
    CHROME = "chrome"
    FIREFOX = "firefox"
    SAFARI = "safari"
    EDGE = "edge"


class BrowserCapability(Enum):
    """Browser capabilities that can be detected."""
    CDP = "cdp"  # Chrome DevTools Protocol
    BIDI = "bidi"  # WebDriver BiDi
    WEBDRIVER = "webdriver"  # Classic WebDriver
    HEADLESS = "headless"
    EXTENSIONS = "extensions"
    PROXY = "proxy"
    DOWNLOAD = "download"
    PERMISSIONS = "permissions"
    INTERCEPT = "network_intercept"
    PERFORMANCE = "performance_metrics"


@dataclass
class BrowserConfig:
    """Configuration for browser instances."""
    browser_type: BrowserType = BrowserType.FIREFOX
    headless: bool = True
    proxy: Optional[str] = None
    user_agent: Optional[str] = None
    window_size: Tuple[int, int] = (1920, 1080)
    disable_images: bool = False
    disable_javascript: bool = False
    disable_extensions: bool = False
    disable_notifications: bool = True
    disable_popup_blocking: bool = False
    args: List[str] = field(default_factory=list)
    prefs: Dict[str, Any] = field(default_factory=dict)
    geckodriver_path: Optional[str] = None
    firefox_binary: Optional[str] = None
    log_level: str = "info"
    timeout: int = 30
    page_load_strategy: str = "normal"  # normal, eager, none


class BrowserAdapter(ABC):
    """Abstract base class for browser adapters."""
    
    @abstractmethod
    async def start(self) -> None:
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
        """Get current page URL."""
        pass
    
    @abstractmethod
    async def get_title(self) -> str:
        """Get current page title."""
        pass
    
    @abstractmethod
    async def find_element(self, selector: str, by: str = "css") -> Optional[Element]:
        """Find a single element."""
        pass
    
    @abstractmethod
    async def find_elements(self, selector: str, by: str = "css") -> List[Element]:
        """Find multiple elements."""
        pass
    
    @abstractmethod
    async def execute_script(self, script: str, *args) -> Any:
        """Execute JavaScript in the browser."""
        pass
    
    @abstractmethod
    async def screenshot(self, element: Optional[Element] = None) -> bytes:
        """Take a screenshot."""
        pass
    
    @abstractmethod
    async def get_page_source(self) -> str:
        """Get the page source HTML."""
        pass
    
    @abstractmethod
    async def go_back(self) -> None:
        """Navigate back in history."""
        pass
    
    @abstractmethod
    async def go_forward(self) -> None:
        """Navigate forward in history."""
        pass
    
    @abstractmethod
    async def refresh(self) -> None:
        """Refresh the current page."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Set[BrowserCapability]:
        """Get browser capabilities."""
        pass


class BiDiProtocol:
    """Implementation of WebDriver BiDi protocol for Firefox."""
    
    def __init__(self, websocket_url: str):
        self.websocket_url = websocket_url
        self.connection = None
        self.command_id = 0
        self.pending_commands = {}
        self.event_handlers = {}
        self._running = False
    
    async def connect(self) -> None:
        """Establish BiDi WebSocket connection."""
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets package required for BiDi protocol")
        
        try:
            self.connection = await websockets.connect(self.websocket_url)
            self._running = True
            asyncio.create_task(self._listen_for_messages())
            logger.info(f"BiDi connection established to {self.websocket_url}")
        except Exception as e:
            logger.error(f"Failed to connect to BiDi: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Close BiDi connection."""
        self._running = False
        if self.connection:
            await self.connection.close()
            logger.info("BiDi connection closed")
    
    async def _listen_for_messages(self) -> None:
        """Listen for incoming BiDi messages."""
        try:
            async for message in self.connection:
                data = json.loads(message)
                await self._handle_message(data)
        except websockets.exceptions.ConnectionClosed:
            logger.warning("BiDi connection closed unexpectedly")
        except Exception as e:
            logger.error(f"Error in BiDi listener: {e}")
    
    async def _handle_message(self, data: Dict) -> None:
        """Handle incoming BiDi message."""
        if "id" in data:
            # Response to a command
            command_id = data["id"]
            if command_id in self.pending_commands:
                future = self.pending_commands.pop(command_id)
                if "error" in data:
                    future.set_exception(Exception(data["error"]))
                else:
                    future.set_result(data.get("result"))
        elif "method" in data:
            # Event notification
            method = data["method"]
            if method in self.event_handlers:
                for handler in self.event_handlers[method]:
                    try:
                        await handler(data.get("params", {}))
                    except Exception as e:
                        logger.error(f"Error in BiDi event handler for {method}: {e}")
    
    async def send_command(self, method: str, params: Dict = None) -> Any:
        """Send a BiDi command and wait for response."""
        if not self.connection:
            raise RuntimeError("BiDi connection not established")
        
        self.command_id += 1
        command = {
            "id": self.command_id,
            "method": method,
            "params": params or {}
        }
        
        future = asyncio.get_event_loop().create_future()
        self.pending_commands[self.command_id] = future
        
        try:
            await self.connection.send(json.dumps(command))
            return await asyncio.wait_for(future, timeout=30)
        except asyncio.TimeoutError:
            self.pending_commands.pop(self.command_id, None)
            raise TimeoutError(f"BiDi command {method} timed out")
        except Exception as e:
            self.pending_commands.pop(self.command_id, None)
            raise
    
    def add_event_handler(self, event: str, handler: Callable) -> None:
        """Add handler for BiDi events."""
        if event not in self.event_handlers:
            self.event_handlers[event] = []
        self.event_handlers[event].append(handler)


class FirefoxAdapter(BrowserAdapter):
    """Firefox browser adapter with BiDi protocol support."""
    
    def __init__(self, config: BrowserConfig = None):
        if not SELENIUM_AVAILABLE:
            raise ImportError("Selenium package required for browser automation")
        
        self.config = config or BrowserConfig()
        self.driver: Optional[WebDriver] = None
        self.bidi: Optional[BiDiProtocol] = None
        self._capabilities = self._detect_capabilities()
        self._temp_dir = tempfile.mkdtemp(prefix="sovereign_firefox_")
        self._process = None
    
    def _detect_capabilities(self) -> Set[BrowserCapability]:
        """Detect Firefox capabilities."""
        capabilities = {
            BrowserCapability.WEBDRIVER,
            BrowserCapability.HEADLESS,
            BrowserCapability.PROXY,
            BrowserCapability.DOWNLOAD,
            BrowserCapability.PERMISSIONS
        }
        
        # Firefox 106+ supports BiDi
        try:
            result = subprocess.run(
                ["firefox", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                version_str = result.stdout.strip()
                # Extract version number
                if "Firefox" in version_str:
                    version = int(version_str.split()[-1].split('.')[0])
                    if version >= 106:
                        capabilities.add(BrowserCapability.BIDI)
                        logger.info(f"Firefox {version} supports BiDi protocol")
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass
        
        return capabilities
    
    def _create_firefox_options(self) -> FirefoxOptions:
        """Create Firefox options from configuration."""
        options = FirefoxOptions()
        
        if self.config.headless:
            options.add_argument("--headless")
        
        if self.config.proxy:
            options.set_preference("network.proxy.type", 1)
            proxy_parts = self.config.proxy.split("://")
            if len(proxy_parts) == 2:
                scheme, address = proxy_parts
                host, port = address.split(":")
                options.set_preference(f"network.proxy.{scheme}", host)
                options.set_preference(f"network.proxy.{scheme}_port", int(port))
        
        if self.config.user_agent:
            options.set_preference("general.useragent.override", self.config.user_agent)
        
        if self.config.disable_images:
            options.set_preference("permissions.default.image", 2)
        
        if self.config.disable_notifications:
            options.set_preference("dom.webnotifications.enabled", False)
            options.set_preference("dom.push.enabled", False)
        
        if self.config.disable_popup_blocking:
            options.set_preference("dom.disable_open_during_load", False)
        
        # Set download preferences
        options.set_preference("browser.download.folderList", 2)
        options.set_preference("browser.download.dir", self._temp_dir)
        options.set_preference("browser.download.useDownloadDir", True)
        options.set_preference("browser.helperApps.neverAsk.saveToDisk", 
                             "application/octet-stream,application/pdf,text/csv")
        
        # Performance and logging preferences
        options.set_preference("devtools.console.stdout.content", True)
        options.set_preference("network.http.speculative-parallel-limit", 0)
        
        # Set window size
        options.add_argument(f"--width={self.config.window_size[0]}")
        options.add_argument(f"--height={self.config.window_size[1]}")
        
        # Add custom arguments
        for arg in self.config.args:
            options.add_argument(arg)
        
        # Set custom preferences
        for key, value in self.config.prefs.items():
            options.set_preference(key, value)
        
        # Set Firefox binary if specified
        if self.config.firefox_binary:
            options.binary_location = self.config.firefox_binary
        
        # Enable BiDi if supported and available
        if BrowserCapability.BIDI in self._capabilities:
            options.set_preference("remote.active-protocols", 1)  # Enable BiDi
            logger.info("Enabled WebDriver BiDi protocol for Firefox")
        
        return options
    
    async def _start_geckodriver(self) -> str:
        """Start geckodriver process and return WebSocket URL for BiDi."""
        geckodriver_path = self.config.geckodriver_path or self._find_geckodriver()
        
        if not geckodriver_path:
            raise FileNotFoundError("geckodriver not found. Please install geckodriver.")
        
        # Start geckodriver with BiDi support
        cmd = [
            geckodriver_path,
            "--port", "0",  # Use random available port
            "--websocket-port", "0",  # Random port for BiDi
        ]
        
        if self.config.headless:
            cmd.append("--headless")
        
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for geckodriver to start and get ports
            for line in iter(self._process.stderr.readline, ''):
                if "Listening on" in line:
                    # Extract ports from output
                    # Example: "Listening on 127.0.0.1:4444"
                    parts = line.split()
                    if len(parts) >= 3:
                        address = parts[2]
                        host, port = address.split(':')
                        # For BiDi, we need the WebSocket URL
                        # Firefox BiDi typically uses ws://host:port/session
                        # We'll construct it from the geckodriver output
                        return f"ws://{host}:{port}/session"
                elif "WebDriver BiDi" in line:
                    logger.info("Geckodriver started with BiDi support")
            
            # If we didn't get BiDi URL, fallback to standard WebDriver
            logger.warning("Could not detect BiDi WebSocket URL, falling back to WebDriver")
            return None
            
        except Exception as e:
            logger.error(f"Failed to start geckodriver: {e}")
            raise
    
    def _find_geckodriver(self) -> Optional[str]:
        """Find geckodriver executable in system PATH."""
        import shutil
        return shutil.which("geckodriver")
    
    async def start(self) -> None:
        """Start Firefox browser session."""
        logger.info("Starting Firefox browser...")
        
        try:
            options = self._create_firefox_options()
            
            # Try to start with BiDi support first
            if BrowserCapability.BIDI in self._capabilities:
                try:
                    websocket_url = await self._start_geckodriver()
                    if websocket_url:
                        # Create driver with BiDi capability
                        capabilities = options.to_capabilities()
                        capabilities["webSocketUrl"] = True
                        
                        service = FirefoxService(
                            executable_path=self.config.geckodriver_path,
                            log_path=self.config.log_level.lower() == "debug" and "/tmp/geckodriver.log"
                        )
                        
                        self.driver = webdriver.Firefox(
                            service=service,
                            options=options,
                            capabilities=capabilities
                        )
                        
                        # Initialize BiDi protocol
                        if hasattr(self.driver, 'capabilities') and 'webSocketUrl' in self.driver.capabilities:
                            bidi_url = self.driver.capabilities['webSocketUrl']
                            self.bidi = BiDiProtocol(bidi_url)
                            await self.bidi.connect()
                            logger.info("Firefox started with BiDi protocol")
                        else:
                            logger.warning("BiDi not available in driver capabilities")
                    else:
                        raise RuntimeError("Failed to get BiDi WebSocket URL")
                except Exception as e:
                    logger.warning(f"Failed to start with BiDi: {e}, falling back to WebDriver")
                    self.bidi = None
            
            # Fallback to standard WebDriver if BiDi failed or not supported
            if not self.driver:
                service = FirefoxService(
                    executable_path=self.config.geckodriver_path,
                    log_path=self.config.log_level.lower() == "debug" and "/tmp/geckodriver.log"
                )
                
                self.driver = webdriver.Firefox(
                    service=service,
                    options=options
                )
                logger.info("Firefox started with WebDriver protocol")
            
            # Set timeouts
            self.driver.implicitly_wait(self.config.timeout)
            self.driver.set_page_load_timeout(self.config.timeout)
            self.driver.set_script_timeout(self.config.timeout)
            
            # Set window size
            self.driver.set_window_size(*self.config.window_size)
            
            logger.info(f"Firefox browser started successfully (headless: {self.config.headless})")
            
        except Exception as e:
            logger.error(f"Failed to start Firefox: {e}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop Firefox browser session."""
        logger.info("Stopping Firefox browser...")
        
        try:
            if self.bidi:
                await self.bidi.disconnect()
                self.bidi = None
            
            if self.driver:
                self.driver.quit()
                self.driver = None
            
            if self._process:
                self._process.terminate()
                self._process.wait(timeout=5)
                self._process = None
            
            # Clean up temp directory
            import shutil
            if Path(self._temp_dir).exists():
                shutil.rmtree(self._temp_dir, ignore_errors=True)
            
            logger.info("Firefox browser stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping Firefox: {e}")
    
    async def navigate(self, url: str) -> None:
        """Navigate to a URL."""
        if not self.driver:
            raise RuntimeError("Browser not started")
        
        try:
            # Use BiDi for navigation if available
            if self.bidi:
                await self.bidi.send_command("browsingContext.navigate", {
                    "url": url,
                    "wait": "complete"
                })
            else:
                # Fallback to WebDriver
                self.driver.get(url)
            
            logger.debug(f"Navigated to: {url}")
            
        except Exception as e:
            logger.error(f"Navigation failed to {url}: {e}")
            raise
    
    async def get_current_url(self) -> str:
        """Get current page URL."""
        if not self.driver:
            raise RuntimeError("Browser not started")
        
        try:
            if self.bidi:
                result = await self.bidi.send_command("browsingContext.getTree", {})
                if result and "contexts" in result and result["contexts"]:
                    return result["contexts"][0].get("url", "")
            
            return self.driver.current_url
            
        except Exception as e:
            logger.error(f"Failed to get current URL: {e}")
            return ""
    
    async def get_title(self) -> str:
        """Get current page title."""
        if not self.driver:
            raise RuntimeError("Browser not started")
        
        try:
            if self.bidi:
                result = await self.bidi.send_command("script.evaluate", {
                    "expression": "document.title",
                    "target": {"context": await self._get_context_id()}
                })
                if result and "result" in result:
                    return result["result"].get("value", "")
            
            return self.driver.title
            
        except Exception as e:
            logger.error(f"Failed to get page title: {e}")
            return ""
    
    async def _get_context_id(self) -> str:
        """Get the current browsing context ID for BiDi."""
        if not self.bidi:
            raise RuntimeError("BiDi not available")
        
        result = await self.bidi.send_command("browsingContext.getTree", {})
        if result and "contexts" in result and result["contexts"]:
            return result["contexts"][0].get("context", "")
        
        raise RuntimeError("No browsing context available")
    
    async def find_element(self, selector: str, by: str = "css") -> Optional[Element]:
        """Find a single element."""
        if not self.driver:
            raise RuntimeError("Browser not started")
        
        try:
            # Convert by string to Selenium By
            by_mapping = {
                "css": By.CSS_SELECTOR,
                "xpath": By.XPATH,
                "id": By.ID,
                "name": By.NAME,
                "class": By.CLASS_NAME,
                "tag": By.TAG_NAME,
                "link": By.LINK_TEXT,
                "partial_link": By.PARTIAL_LINK_TEXT
            }
            
            selenium_by = by_mapping.get(by.lower(), By.CSS_SELECTOR)
            
            # Use BiDi for element finding if available
            if self.bidi:
                # BiDi uses CSS selectors by default, convert if needed
                if by.lower() == "css":
                    expression = f"document.querySelector('{selector}')"
                elif by.lower() == "xpath":
                    expression = f"document.evaluate('{selector}', document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue"
                else:
                    # Fallback to WebDriver for non-CSS selectors with BiDi
                    element = self.driver.find_element(selenium_by, selector)
                    return self._create_element_from_selenium(element)
                
                result = await self.bidi.send_command("script.evaluate", {
                    "expression": expression,
                    "target": {"context": await self._get_context_id()}
                })
                
                if result and "result" in result and result["result"].get("value"):
                    # We found an element, but BiDi doesn't return element references directly
                    # We need to use WebDriver to get the actual element
                    # This is a limitation of current BiDi spec
                    pass
            
            # Fallback to WebDriver
            element = WebDriverWait(self.driver, self.config.timeout).until(
                EC.presence_of_element_located((selenium_by, selector))
            )
            
            return self._create_element_from_selenium(element)
            
        except TimeoutException:
            logger.debug(f"Element not found: {selector} (by: {by})")
            return None
        except Exception as e:
            logger.error(f"Error finding element {selector}: {e}")
            return None
    
    async def find_elements(self, selector: str, by: str = "css") -> List[Element]:
        """Find multiple elements."""
        if not self.driver:
            raise RuntimeError("Browser not started")
        
        try:
            by_mapping = {
                "css": By.CSS_SELECTOR,
                "xpath": By.XPATH,
                "id": By.ID,
                "name": By.NAME,
                "class": By.CLASS_NAME,
                "tag": By.TAG_NAME,
                "link": By.LINK_TEXT,
                "partial_link": By.PARTIAL_LINK_TEXT
            }
            
            selenium_by = by_mapping.get(by.lower(), By.CSS_SELECTOR)
            
            # Wait for at least one element to be present
            WebDriverWait(self.driver, self.config.timeout).until(
                EC.presence_of_element_located((selenium_by, selector))
            )
            
            # Find all matching elements
            elements = self.driver.find_elements(selenium_by, selector)
            return [self._create_element_from_selenium(el) for el in elements]
            
        except TimeoutException:
            logger.debug(f"No elements found: {selector} (by: {by})")
            return []
        except Exception as e:
            logger.error(f"Error finding elements {selector}: {e}")
            return []
    
    def _create_element_from_selenium(self, selenium_element) -> Element:
        """Create Element instance from Selenium WebElement."""
        return Element(
            selector="",  # We don't have the original selector
            element=selenium_element,
            browser_adapter=self
        )
    
    async def execute_script(self, script: str, *args) -> Any:
        """Execute JavaScript in the browser."""
        if not self.driver:
            raise RuntimeError("Browser not started")
        
        try:
            # Use BiDi if available
            if self.bidi:
                # Convert args to BiDi format
                bidi_args = []
                for arg in args:
                    if isinstance(arg, Element):
                        # For elements, we need to get their reference
                        # This is complex in BiDi, fallback to WebDriver
                        return self.driver.execute_script(script, *[el.element if isinstance(el, Element) else el for el in args])
                    else:
                        bidi_args.append({"value": arg})
                
                result = await self.bidi.send_command("script.evaluate", {
                    "expression": script,
                    "target": {"context": await self._get_context_id()},
                    "arguments": bidi_args
                })
                
                if result and "result" in result:
                    return result["result"].get("value")
            
            # Fallback to WebDriver
            processed_args = []
            for arg in args:
                if isinstance(arg, Element):
                    processed_args.append(arg.element)
                else:
                    processed_args.append(arg)
            
            return self.driver.execute_script(script, *processed_args)
            
        except JavascriptException as e:
            logger.error(f"JavaScript execution error: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to execute script: {e}")
            raise
    
    async def screenshot(self, element: Optional[Element] = None) -> bytes:
        """Take a screenshot."""
        if not self.driver:
            raise RuntimeError("Browser not started")
        
        try:
            if element and hasattr(element, 'element'):
                # Screenshot of specific element
                return element.element.screenshot_as_png
            else:
                # Full page screenshot
                return self.driver.get_screenshot_as_png()
                
        except Exception as e:
            logger.error(f"Failed to take screenshot: {e}")
            return b""
    
    async def get_page_source(self) -> str:
        """Get the page source HTML."""
        if not self.driver:
            raise RuntimeError("Browser not started")
        
        try:
            if self.bidi:
                result = await self.bidi.send_command("script.evaluate", {
                    "expression": "document.documentElement.outerHTML",
                    "target": {"context": await self._get_context_id()}
                })
                if result and "result" in result:
                    return result["result"].get("value", "")
            
            return self.driver.page_source
            
        except Exception as e:
            logger.error(f"Failed to get page source: {e}")
            return ""
    
    async def go_back(self) -> None:
        """Navigate back in history."""
        if not self.driver:
            raise RuntimeError("Browser not started")
        
        try:
            if self.bidi:
                await self.bidi.send_command("browsingContext.traverseHistory", {
                    "delta": -1
                })
            else:
                self.driver.back()
                
        except Exception as e:
            logger.error(f"Failed to go back: {e}")
            raise
    
    async def go_forward(self) -> None:
        """Navigate forward in history."""
        if not self.driver:
            raise RuntimeError("Browser not started")
        
        try:
            if self.bidi:
                await self.bidi.send_command("browsingContext.traverseHistory", {
                    "delta": 1
                })
            else:
                self.driver.forward()
                
        except Exception as e:
            logger.error(f"Failed to go forward: {e}")
            raise
    
    async def refresh(self) -> None:
        """Refresh the current page."""
        if not self.driver:
            raise RuntimeError("Browser not started")
        
        try:
            if self.bidi:
                await self.bidi.send_command("browsingContext.navigate", {
                    "url": await self.get_current_url(),
                    "wait": "complete"
                })
            else:
                self.driver.refresh()
                
        except Exception as e:
            logger.error(f"Failed to refresh: {e}")
            raise
    
    def get_capabilities(self) -> Set[BrowserCapability]:
        """Get browser capabilities."""
        return self._capabilities.copy()
    
    async def wait_for_element(self, selector: str, by: str = "css", 
                              timeout: Optional[int] = None) -> Optional[Element]:
        """Wait for an element to appear."""
        if not self.driver:
            raise RuntimeError("Browser not started")
        
        timeout = timeout or self.config.timeout
        
        try:
            by_mapping = {
                "css": By.CSS_SELECTOR,
                "xpath": By.XPATH,
                "id": By.ID,
                "name": By.NAME,
                "class": By.CLASS_NAME,
                "tag": By.TAG_NAME,
                "link": By.LINK_TEXT,
                "partial_link": By.PARTIAL_LINK_TEXT
            }
            
            selenium_by = by_mapping.get(by.lower(), By.CSS_SELECTOR)
            
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((selenium_by, selector))
            )
            
            return self._create_element_from_selenium(element)
            
        except TimeoutException:
            logger.debug(f"Timeout waiting for element: {selector}")
            return None
        except Exception as e:
            logger.error(f"Error waiting for element {selector}: {e}")
            return None
    
    async def wait_for_navigation(self, timeout: Optional[int] = None) -> bool:
        """Wait for page navigation to complete."""
        if not self.driver:
            raise RuntimeError("Browser not started")
        
        timeout = timeout or self.config.timeout
        
        try:
            # Wait for document.readyState to be complete
            def check_ready_state(driver):
                return driver.execute_script("return document.readyState") == "complete"
            
            WebDriverWait(self.driver, timeout).until(check_ready_state)
            return True
            
        except TimeoutException:
            logger.debug("Timeout waiting for navigation")
            return False
        except Exception as e:
            logger.error(f"Error waiting for navigation: {e}")
            return False
    
    async def set_page_load_strategy(self, strategy: str) -> None:
        """Set page load strategy (normal, eager, none)."""
        if not self.driver:
            raise RuntimeError("Browser not started")
        
        valid_strategies = ["normal", "eager", "none"]
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy. Must be one of: {valid_strategies}")
        
        try:
            self.driver.execute_script(f"window.performance.setResourceTimingBufferSize(0);")
            # Note: Selenium doesn't directly support changing page load strategy after driver creation
            # This would need to be set during driver initialization
            logger.warning("Page load strategy can only be set during browser initialization")
            
        except Exception as e:
            logger.error(f"Failed to set page load strategy: {e}")
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get browser performance metrics."""
        if not self.driver:
            raise RuntimeError("Browser not started")
        
        try:
            metrics = {}
            
            # Get navigation timing
            nav_timing = await self.execute_script("""
                const timing = window.performance.timing;
                return {
                    navigationStart: timing.navigationStart,
                    loadEventEnd: timing.loadEventEnd,
                    domComplete: timing.domComplete,
                    responseStart: timing.responseStart,
                    requestStart: timing.requestStart
                };
            """)
            metrics["navigation_timing"] = nav_timing
            
            # Get resource timing
            resource_timing = await self.execute_script("""
                return window.performance.getEntriesByType('resource').map(entry => ({
                    name: entry.name,
                    duration: entry.duration,
                    initiatorType: entry.initiatorType
                }));
            """)
            metrics["resource_timing"] = resource_timing
            
            # Get memory info if available
            try:
                memory_info = await self.execute_script("""
                    if (window.performance.memory) {
                        return {
                            usedJSHeapSize: window.performance.memory.usedJSHeapSize,
                            totalJSHeapSize: window.performance.memory.totalJSHeapSize
                        };
                    }
                    return null;
                """)
                if memory_info:
                    metrics["memory"] = memory_info
            except:
                pass
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {}


class BrowserFactory:
    """Factory for creating browser adapter instances."""
    
    @staticmethod
    def create_adapter(config: BrowserConfig) -> BrowserAdapter:
        """Create a browser adapter based on configuration."""
        if config.browser_type == BrowserType.FIREFOX:
            return FirefoxAdapter(config)
        elif config.browser_type == BrowserType.CHROME:
            # Import Chrome adapter if available
            try:
                from nexus.browser.chrome_adapter import ChromeAdapter
                return ChromeAdapter(config)
            except ImportError:
                raise NotImplementedError("Chrome adapter not implemented")
        elif config.browser_type == BrowserType.SAFARI:
            raise NotImplementedError("Safari adapter not implemented")
        elif config.browser_type == BrowserType.EDGE:
            raise NotImplementedError("Edge adapter not implemented")
        else:
            raise ValueError(f"Unsupported browser type: {config.browser_type}")
    
    @staticmethod
    def get_available_browsers() -> List[BrowserType]:
        """Get list of available browsers on the system."""
        available = []
        
        # Check Firefox
        try:
            subprocess.run(["firefox", "--version"], 
                          capture_output=True, timeout=2)
            available.append(BrowserType.FIREFOX)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Check Chrome
        try:
            if platform.system() == "Windows":
                subprocess.run(["chrome", "--version"], 
                              capture_output=True, timeout=2)
            else:
                subprocess.run(["google-chrome", "--version"], 
                              capture_output=True, timeout=2)
            available.append(BrowserType.CHROME)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Check Edge
        try:
            if platform.system() == "Windows":
                subprocess.run(["msedge", "--version"], 
                              capture_output=True, timeout=2)
            else:
                subprocess.run(["microsoft-edge", "--version"], 
                              capture_output=True, timeout=2)
            available.append(BrowserType.EDGE)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        return available


# Cross-browser testing utilities
class CrossBrowserTester:
    """Utility for cross-browser testing."""
    
    def __init__(self, test_url: str = "https://example.com"):
        self.test_url = test_url
        self.results = {}
    
    async def test_browser(self, browser_type: BrowserType, 
                          config_overrides: Dict = None) -> Dict[str, Any]:
        """Test a specific browser."""
        config = BrowserConfig(browser_type=browser_type)
        
        if config_overrides:
            for key, value in config_overrides.items():
                setattr(config, key, value)
        
        adapter = BrowserFactory.create_adapter(config)
        test_results = {
            "browser": browser_type.value,
            "success": False,
            "errors": [],
            "metrics": {}
        }
        
        try:
            # Start browser
            await adapter.start()
            test_results["started"] = True
            
            # Navigate to test URL
            await adapter.navigate(self.test_url)
            test_results["navigated"] = True
            
            # Get page info
            title = await adapter.get_title()
            url = await adapter.get_current_url()
            test_results["title"] = title
            test_results["url"] = url
            
            # Test element finding
            elements = await adapter.find_elements("a")
            test_results["elements_found"] = len(elements)
            
            # Test JavaScript execution
            js_result = await adapter.execute_script("return document.title;")
            test_results["js_executed"] = js_result == title
            
            # Get capabilities
            capabilities = adapter.get_capabilities()
            test_results["capabilities"] = [cap.value for cap in capabilities]
            
            # Get performance metrics if available
            if hasattr(adapter, 'get_performance_metrics'):
                metrics = await adapter.get_performance_metrics()
                test_results["metrics"] = metrics
            
            test_results["success"] = True
            
        except Exception as e:
            test_results["errors"].append(str(e))
            logger.error(f"Browser test failed for {browser_type}: {e}")
        
        finally:
            try:
                await adapter.stop()
            except:
                pass
        
        return test_results
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run tests on all available browsers."""
        available = BrowserFactory.get_available_browsers()
        
        for browser in available:
            logger.info(f"Testing {browser.value}...")
            self.results[browser.value] = await self.test_browser(browser)
        
        return self.results
    
    def generate_report(self) -> str:
        """Generate test report."""
        report = ["Cross-Browser Test Report", "=" * 40, ""]
        
        for browser, result in self.results.items():
            report.append(f"Browser: {browser}")
            report.append(f"  Success: {result['success']}")
            report.append(f"  Started: {result.get('started', False)}")
            report.append(f"  Navigated: {result.get('navigated', False)}")
            report.append(f"  Elements Found: {result.get('elements_found', 0)}")
            report.append(f"  JS Executed: {result.get('js_executed', False)}")
            
            if result.get('errors'):
                report.append(f"  Errors: {', '.join(result['errors'])}")
            
            if result.get('capabilities'):
                report.append(f"  Capabilities: {', '.join(result['capabilities'])}")
            
            report.append("")
        
        return "\n".join(report)


# Integration with existing Page and Element classes
class FirefoxPage(Page):
    """Page implementation for Firefox browser."""
    
    def __init__(self, adapter: FirefoxAdapter):
        self.adapter = adapter
        self._mouse = None
    
    @property
    def mouse(self) -> Mouse:
        """Get mouse controller."""
        if not self._mouse:
            from nexus.actor.mouse import Mouse
            self._mouse = Mouse(self)
        return self._mouse
    
    async def goto(self, url: str) -> None:
        """Navigate to URL."""
        await self.adapter.navigate(url)
    
    async def query_selector(self, selector: str) -> Optional[Element]:
        """Find single element."""
        return await self.adapter.find_element(selector)
    
    async def query_selector_all(self, selector: str) -> List[Element]:
        """Find multiple elements."""
        return await self.adapter.find_elements(selector)
    
    async def evaluate(self, script: str, *args) -> Any:
        """Execute JavaScript."""
        return await self.adapter.execute_script(script, *args)
    
    async def screenshot(self, **kwargs) -> bytes:
        """Take screenshot."""
        return await self.adapter.screenshot()
    
    @property
    def url(self) -> str:
        """Get current URL."""
        # This would need to be async in real implementation
        # Simplified for example
        return ""
    
    @property
    def title(self) -> str:
        """Get page title."""
        # This would need to be async in real implementation
        return ""


# Example usage and testing
async def example_usage():
    """Example of using Firefox adapter."""
    # Create Firefox configuration
    config = BrowserConfig(
        browser_type=BrowserType.FIREFOX,
        headless=True,
        window_size=(1280, 720),
        disable_notifications=True
    )
    
    # Create adapter
    adapter = FirefoxAdapter(config)
    
    try:
        # Start browser
        await adapter.start()
        print(f"Browser capabilities: {adapter.get_capabilities()}")
        
        # Navigate to a page
        await adapter.navigate("https://example.com")
        print(f"Page title: {await adapter.get_title()}")
        print(f"Current URL: {await adapter.get_current_url()}")
        
        # Find elements
        heading = await adapter.find_element("h1")
        if heading:
            print(f"Heading text: {await heading.text()}")
        
        # Execute JavaScript
        js_result = await adapter.execute_script("return document.domain;")
        print(f"JavaScript result: {js_result}")
        
        # Take screenshot
        screenshot = await adapter.screenshot()
        print(f"Screenshot size: {len(screenshot)} bytes")
        
        # Get performance metrics
        metrics = await adapter.get_performance_metrics()
        print(f"Performance metrics: {json.dumps(metrics, indent=2)}")
        
    finally:
        # Stop browser
        await adapter.stop()


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
    
    # Or run cross-browser tests
    # tester = CrossBrowserTester()
    # results = asyncio.run(tester.run_all_tests())
    # print(tester.generate_report())