"""Cross-browser compatibility layer for nexus.

This module provides a unified API for browser automation across Chrome, Firefox,
Safari, and Edge with browser-specific optimizations and fallbacks.
"""

import asyncio
import logging
import platform
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from playwright.async_api import Page, Browser, BrowserContext
    from selenium.webdriver.remote.webdriver import WebDriver

logger = logging.getLogger(__name__)


class BrowserType(Enum):
    """Supported browser types."""
    CHROME = "chrome"
    FIREFOX = "firefox"
    SAFARI = "safari"
    EDGE = "edge"
    CHROMIUM = "chromium"


class BrowserProtocol(Enum):
    """Browser automation protocols."""
    CDP = "cdp"  # Chrome DevTools Protocol
    WEBDRIVER_BIDI = "webdriver-bidi"  # WebDriver BiDi (emerging standard)
    WEBDRIVER_CLASSIC = "webdriver-classic"  # Classic WebDriver (JSON Wire Protocol)


@dataclass
class BrowserCapabilities:
    """Browser capabilities and feature support."""
    browser_type: BrowserType
    protocols: Set[BrowserProtocol] = field(default_factory=set)
    supports_cdp: bool = False
    supports_webdriver_bidi: bool = False
    supports_stealth_mode: bool = False
    supports_network_interception: bool = False
    supports_javascript_dialogs: bool = True
    supports_file_chooser: bool = True
    supports_permissions: bool = False
    supports_geolocation: bool = False
    supports_webgl: bool = True
    supports_webrtc: bool = True
    max_concurrent_pages: int = 100
    quirks: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BrowserConfig:
    """Browser configuration."""
    browser_type: BrowserType
    headless: bool = True
    user_data_dir: Optional[Path] = None
    proxy: Optional[Dict[str, str]] = None
    user_agent: Optional[str] = None
    viewport: Optional[Dict[str, int]] = None
    args: List[str] = field(default_factory=list)
    ignore_default_args: bool = False
    env: Optional[Dict[str, str]] = None
    timeout: int = 30000
    slow_mo: int = 0
    devtools: bool = False


class BrowserCompatibilityError(Exception):
    """Base exception for browser compatibility issues."""
    pass


class UnsupportedBrowserError(BrowserCompatibilityError):
    """Raised when browser is not supported."""
    pass


class ProtocolNotAvailableError(BrowserCompatibilityError):
    """Raised when requested protocol is not available."""
    pass


class FeatureNotSupportedError(BrowserCompatibilityError):
    """Raised when a feature is not supported by the browser."""
    pass


class BrowserDetector:
    """Detects available browsers and their capabilities."""

    @staticmethod
    def detect_browser_type(executable_path: Optional[str] = None) -> BrowserType:
        """Detect browser type from executable path or system."""
        if executable_path:
            path = Path(executable_path).name.lower()
            if "chrome" in path or "chromium" in path:
                return BrowserType.CHROME
            elif "firefox" in path:
                return BrowserType.FIREFOX
            elif "safari" in path:
                return BrowserType.SAFARI
            elif "edge" in path or "msedge" in path:
                return BrowserType.EDGE
        
        # System detection
        system = platform.system()
        if system == "Darwin":
            # macOS - check for Safari
            safari_path = Path("/Applications/Safari.app/Contents/MacOS/Safari")
            if safari_path.exists():
                return BrowserType.SAFARI
        
        return BrowserType.CHROME  # Default to Chrome

    @staticmethod
    def detect_capabilities(browser_type: BrowserType) -> BrowserCapabilities:
        """Detect capabilities for a given browser type."""
        capabilities = BrowserCapabilities(browser_type=browser_type)
        
        if browser_type in (BrowserType.CHROME, BrowserType.CHROMIUM, BrowserType.EDGE):
            capabilities.protocols = {BrowserProtocol.CDP, BrowserProtocol.WEBDRIVER_BIDI}
            capabilities.supports_cdp = True
            capabilities.supports_webdriver_bidi = True
            capabilities.supports_stealth_mode = True
            capabilities.supports_network_interception = True
            capabilities.supports_permissions = True
            capabilities.supports_geolocation = True
            capabilities.quirks = {
                "needs_page_load_strategy": True,
                "supports_auto_download": True,
                "max_data_uri_size": 2 * 1024 * 1024 * 1024,  # 2GB
            }
        
        elif browser_type == BrowserType.FIREFOX:
            capabilities.protocols = {BrowserProtocol.WEBDRIVER_BIDI, BrowserProtocol.WEBDRIVER_CLASSIC}
            capabilities.supports_webdriver_bidi = True
            capabilities.supports_stealth_mode = False
            capabilities.supports_network_interception = True
            capabilities.supports_permissions = True
            capabilities.quirks = {
                "needs_page_load_strategy": False,
                "supports_auto_download": False,
                "max_data_uri_size": 512 * 1024 * 1024,  # 512MB
                "needs_geckodriver": True,
            }
        
        elif browser_type == BrowserType.SAFARI:
            capabilities.protocols = {BrowserProtocol.WEBDRIVER_CLASSIC}
            capabilities.supports_stealth_mode = False
            capabilities.supports_network_interception = False
            capabilities.supports_permissions = False
            capabilities.supports_geolocation = False
            capabilities.quirks = {
                "needs_page_load_strategy": False,
                "supports_auto_download": False,
                "max_data_uri_size": 100 * 1024 * 1024,  # 100MB
                "needs_safaridriver": True,
                "no_headless": True,
            }
        
        return capabilities

    @staticmethod
    def find_browser_executable(browser_type: BrowserType) -> Optional[str]:
        """Find browser executable path."""
        system = platform.system()
        
        if browser_type in (BrowserType.CHROME, BrowserType.CHROMIUM):
            if system == "Darwin":
                paths = [
                    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                    "/Applications/Chromium.app/Contents/MacOS/Chromium",
                ]
            elif system == "Windows":
                paths = [
                    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                ]
            else:  # Linux
                paths = [
                    "/usr/bin/google-chrome",
                    "/usr/bin/chromium",
                    "/usr/bin/chromium-browser",
                ]
        
        elif browser_type == BrowserType.FIREFOX:
            if system == "Darwin":
                paths = ["/Applications/Firefox.app/Contents/MacOS/firefox"]
            elif system == "Windows":
                paths = [r"C:\Program Files\Mozilla Firefox\firefox.exe"]
            else:
                paths = ["/usr/bin/firefox"]
        
        elif browser_type == BrowserType.SAFARI:
            if system == "Darwin":
                paths = ["/Applications/Safari.app/Contents/MacOS/Safari"]
            else:
                return None  # Safari only on macOS
        
        elif browser_type == BrowserType.EDGE:
            if system == "Darwin":
                paths = ["/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"]
            elif system == "Windows":
                paths = [
                    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                ]
            else:
                paths = ["/usr/bin/microsoft-edge"]
        
        else:
            return None
        
        for path in paths:
            if Path(path).exists():
                return path
        
        return None


class BrowserProtocolAdapter(ABC):
    """Abstract base class for browser protocol adapters."""

    @abstractmethod
    async def connect(self, config: BrowserConfig) -> None:
        """Connect to browser."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from browser."""
        pass

    @abstractmethod
    async def new_page(self, url: Optional[str] = None) -> Any:
        """Create new page/tab."""
        pass

    @abstractmethod
    async def close_page(self, page: Any) -> None:
        """Close page/tab."""
        pass

    @abstractmethod
    async def get_pages(self) -> List[Any]:
        """Get all open pages."""
        pass

    @abstractmethod
    async def execute_script(self, page: Any, script: str, *args: Any) -> Any:
        """Execute JavaScript in page context."""
        pass

    @abstractmethod
    async def navigate(self, page: Any, url: str, wait_until: str = "load") -> None:
        """Navigate to URL."""
        pass

    @abstractmethod
    async def go_back(self, page: Any) -> None:
        """Navigate back."""
        pass

    @abstractmethod
    async def go_forward(self, page: Any) -> None:
        """Navigate forward."""
        pass

    @abstractmethod
    async def reload(self, page: Any) -> None:
        """Reload page."""
        pass

    @abstractmethod
    async def set_viewport(self, page: Any, width: int, height: int) -> None:
        """Set viewport size."""
        pass

    @abstractmethod
    async def screenshot(self, page: Any, path: Optional[str] = None) -> bytes:
        """Take screenshot."""
        pass

    @abstractmethod
    async def pdf(self, page: Any, path: Optional[str] = None) -> bytes:
        """Generate PDF."""
        pass


class PlaywrightAdapter(BrowserProtocolAdapter):
    """Adapter for Playwright (supports CDP and WebDriver BiDi via Playwright)."""

    def __init__(self) -> None:
        self._browser: Optional["Browser"] = None
        self._context: Optional["BrowserContext"] = None
        self._playwright = None
        self._capabilities: Optional[BrowserCapabilities] = None

    async def connect(self, config: BrowserConfig) -> None:
        """Connect using Playwright."""
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise BrowserCompatibilityError(
                "Playwright not installed. Install with: pip install playwright"
            )

        self._playwright = await async_playwright().start()
        
        # Map browser type to Playwright browser name
        browser_mapping = {
            BrowserType.CHROME: "chromium",
            BrowserType.CHROMIUM: "chromium",
            BrowserType.FIREFOX: "firefox",
            BrowserType.SAFARI: "webkit",
            BrowserType.EDGE: "chromium",  # Edge uses Chromium
        }
        
        browser_name = browser_mapping.get(config.browser_type)
        if not browser_name:
            raise UnsupportedBrowserError(f"Unsupported browser: {config.browser_type}")

        # Get browser executable if needed
        executable_path = None
        if config.browser_type == BrowserType.EDGE:
            executable_path = BrowserDetector.find_browser_executable(BrowserType.EDGE)
        
        launch_options = {
            "headless": config.headless,
            "args": config.args,
            "ignore_default_args": config.ignore_default_args,
            "env": config.env,
            "timeout": config.timeout,
            "slow_mo": config.slow_mo,
            "devtools": config.devtools,
        }
        
        if executable_path:
            launch_options["executable_path"] = executable_path
        
        if config.user_data_dir:
            launch_options["user_data_dir"] = str(config.user_data_dir)
        
        browser_type_obj = getattr(self._playwright, browser_name)
        self._browser = await browser_type_obj.launch(**launch_options)
        
        # Create context with options
        context_options = {}
        if config.user_agent:
            context_options["user_agent"] = config.user_agent
        if config.viewport:
            context_options["viewport"] = config.viewport
        if config.proxy:
            context_options["proxy"] = config.proxy
        
        self._context = await self._browser.new_context(**context_options)
        
        # Detect capabilities
        self._capabilities = BrowserDetector.detect_capabilities(config.browser_type)

    async def disconnect(self) -> None:
        """Disconnect Playwright."""
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def new_page(self, url: Optional[str] = None) -> "Page":
        """Create new page."""
        if not self._context:
            raise BrowserCompatibilityError("Not connected to browser")
        
        page = await self._context.new_page()
        
        if url:
            await page.goto(url)
        
        return page

    async def close_page(self, page: "Page") -> None:
        """Close page."""
        await page.close()

    async def get_pages(self) -> List["Page"]:
        """Get all pages."""
        if not self._context:
            return []
        return self._context.pages

    async def execute_script(self, page: "Page", script: str, *args: Any) -> Any:
        """Execute script in page."""
        return await page.evaluate(script, *args)

    async def navigate(self, page: "Page", url: str, wait_until: str = "load") -> None:
        """Navigate to URL."""
        await page.goto(url, wait_until=wait_until)

    async def go_back(self, page: "Page") -> None:
        """Navigate back."""
        await page.go_back()

    async def go_forward(self, page: "Page") -> None:
        """Navigate forward."""
        await page.go_forward()

    async def reload(self, page: "Page") -> None:
        """Reload page."""
        await page.reload()

    async def set_viewport(self, page: "Page", width: int, height: int) -> None:
        """Set viewport size."""
        await page.set_viewport_size({"width": width, "height": height})

    async def screenshot(self, page: "Page", path: Optional[str] = None) -> bytes:
        """Take screenshot."""
        return await page.screenshot(path=path)

    async def pdf(self, page: "Page", path: Optional[str] = None) -> bytes:
        """Generate PDF."""
        if self._capabilities and self._capabilities.browser_type == BrowserType.FIREFOX:
            # Firefox doesn't support PDF generation via Playwright
            raise FeatureNotSupportedError("PDF generation not supported in Firefox")
        return await page.pdf(path=path)


class SeleniumWebDriverAdapter(BrowserProtocolAdapter):
    """Adapter for Selenium WebDriver (supports WebDriver BiDi and classic)."""

    def __init__(self) -> None:
        self._driver: Optional["WebDriver"] = None
        self._capabilities: Optional[BrowserCapabilities] = None

    async def connect(self, config: BrowserConfig) -> None:
        """Connect using Selenium WebDriver."""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options as ChromeOptions
            from selenium.webdriver.firefox.options import Options as FirefoxOptions
            from selenium.webdriver.safari.options import Options as SafariOptions
            from selenium.webdriver.edge.options import Options as EdgeOptions
            from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
        except ImportError:
            raise BrowserCompatibilityError(
                "Selenium not installed. Install with: pip install selenium"
            )

        self._capabilities = BrowserDetector.detect_capabilities(config.browser_type)
        
        # Configure options based on browser type
        if config.browser_type in (BrowserType.CHROME, BrowserType.CHROMIUM):
            options = ChromeOptions()
            if config.headless:
                options.add_argument("--headless=new")
            if config.user_data_dir:
                options.add_argument(f"--user-data-dir={config.user_data_dir}")
            if config.proxy:
                options.add_argument(f"--proxy-server={config.proxy.get('server')}")
            
            # Enable BiDi if supported
            if self._capabilities.supports_webdriver_bidi:
                options.set_capability("webSocketUrl", True)
            
            # Add custom args
            for arg in config.args:
                options.add_argument(arg)
            
            self._driver = webdriver.Chrome(options=options)
        
        elif config.browser_type == BrowserType.FIREFOX:
            options = FirefoxOptions()
            if config.headless:
                options.add_argument("--headless")
            if config.user_data_dir:
                options.profile = webdriver.FirefoxProfile(str(config.user_data_dir))
            
            # Firefox supports BiDi natively
            if self._capabilities.supports_webdriver_bidi:
                options.set_capability("webSocketUrl", True)
            
            for arg in config.args:
                options.add_argument(arg)
            
            self._driver = webdriver.Firefox(options=options)
        
        elif config.browser_type == BrowserType.SAFARI:
            if platform.system() != "Darwin":
                raise UnsupportedBrowserError("Safari is only available on macOS")
            
            options = SafariOptions()
            # Safari doesn't support headless mode
            if config.headless:
                logger.warning("Safari doesn't support headless mode, ignoring")
            
            self._driver = webdriver.Safari(options=options)
        
        elif config.browser_type == BrowserType.EDGE:
            options = EdgeOptions()
            if config.headless:
                options.add_argument("--headless=new")
            if config.user_data_dir:
                options.add_argument(f"--user-data-dir={config.user_data_dir}")
            
            # Edge (Chromium) supports BiDi
            if self._capabilities.supports_webdriver_bidi:
                options.set_capability("webSocketUrl", True)
            
            for arg in config.args:
                options.add_argument(arg)
            
            self._driver = webdriver.Edge(options=options)
        
        else:
            raise UnsupportedBrowserError(f"Unsupported browser: {config.browser_type}")
        
        # Set timeouts
        self._driver.implicitly_wait(config.timeout / 1000)
        self._driver.set_page_load_timeout(config.timeout / 1000)

    async def disconnect(self) -> None:
        """Disconnect Selenium WebDriver."""
        if self._driver:
            self._driver.quit()

    async def new_page(self, url: Optional[str] = None) -> "WebDriver":
        """Create new tab/window."""
        if not self._driver:
            raise BrowserCompatibilityError("Not connected to browser")
        
        # For Selenium, we work with the current window
        # Open new tab using JavaScript
        self._driver.execute_script("window.open('about:blank', '_blank');")
        self._driver.switch_to.window(self._driver.window_handles[-1])
        
        if url:
            self._driver.get(url)
        
        return self._driver

    async def close_page(self, page: "WebDriver") -> None:
        """Close current tab."""
        if self._driver:
            self._driver.close()
            # Switch to previous tab if available
            if self._driver.window_handles:
                self._driver.switch_to.window(self._driver.window_handles[-1])

    async def get_pages(self) -> List["WebDriver"]:
        """Get all window handles."""
        if not self._driver:
            return []
        return [self._driver]  # Selenium doesn't expose multiple page objects easily

    async def execute_script(self, page: "WebDriver", script: str, *args: Any) -> Any:
        """Execute JavaScript."""
        return page.execute_script(script, *args)

    async def navigate(self, page: "WebDriver", url: str, wait_until: str = "load") -> None:
        """Navigate to URL."""
        page.get(url)
        # Selenium doesn't have built-in wait_until, so we add a simple wait
        if wait_until == "load":
            page.execute_script("return document.readyState")

    async def go_back(self, page: "WebDriver") -> None:
        """Navigate back."""
        page.back()

    async def go_forward(self, page: "WebDriver") -> None:
        """Navigate forward."""
        page.forward()

    async def reload(self, page: "WebDriver") -> None:
        """Reload page."""
        page.refresh()

    async def set_viewport(self, page: "WebDriver", width: int, height: int) -> None:
        """Set viewport size."""
        page.set_window_size(width, height)

    async def screenshot(self, page: "WebDriver", path: Optional[str] = None) -> bytes:
        """Take screenshot."""
        screenshot_data = page.get_screenshot_as_png()
        if path:
            with open(path, "wb") as f:
                f.write(screenshot_data)
        return screenshot_data

    async def pdf(self, page: "WebDriver", path: Optional[str] = None) -> bytes:
        """Generate PDF (not widely supported in Selenium)."""
        raise FeatureNotSupportedError(
            "PDF generation not supported via Selenium WebDriver"
        )


class BrowserCompatibilityLayer:
    """Main compatibility layer that abstracts browser differences."""

    def __init__(self) -> None:
        self._adapter: Optional[BrowserProtocolAdapter] = None
        self._config: Optional[BrowserConfig] = None
        self._capabilities: Optional[BrowserCapabilities] = None
        self._fallback_chain: List[BrowserProtocolAdapter] = []

    async def initialize(
        self,
        config: BrowserConfig,
        preferred_protocol: Optional[BrowserProtocol] = None
    ) -> None:
        """Initialize browser with compatibility layer.
        
        Args:
            config: Browser configuration
            preferred_protocol: Preferred automation protocol (will fallback if not available)
        """
        self._config = config
        self._capabilities = BrowserDetector.detect_capabilities(config.browser_type)
        
        # Determine which adapter to use based on protocol preference
        if preferred_protocol == BrowserProtocol.CDP:
            if self._capabilities.supports_cdp:
                self._adapter = PlaywrightAdapter()
            else:
                raise ProtocolNotAvailableError(
                    f"CDP not supported by {config.browser_type}"
                )
        elif preferred_protocol == BrowserProtocol.WEBDRIVER_BIDI:
            if self._capabilities.supports_webdriver_bidi:
                # Try Selenium first for BiDi, then Playwright
                try:
                    self._adapter = SeleniumWebDriverAdapter()
                except Exception:
                    self._adapter = PlaywrightAdapter()
            else:
                raise ProtocolNotAvailableError(
                    f"WebDriver BiDi not supported by {config.browser_type}"
                )
        else:
            # Auto-select best adapter
            self._adapter = self._select_best_adapter()
        
        # Initialize the adapter
        try:
            await self._adapter.connect(config)
        except Exception as e:
            logger.warning(f"Failed to initialize with {type(self._adapter).__name__}: {e}")
            # Try fallback adapters
            await self._try_fallbacks(config)

    def _select_best_adapter(self) -> BrowserProtocolAdapter:
        """Select the best available adapter for the browser."""
        if self._capabilities is None:
            return PlaywrightAdapter()
        
        # Prefer Playwright for its better API and cross-browser support
        if self._capabilities.supports_cdp or self._capabilities.supports_webdriver_bidi:
            return PlaywrightAdapter()
        
        # Fallback to Selenium
        return SeleniumWebDriverAdapter()

    async def _try_fallbacks(self, config: BrowserConfig) -> None:
        """Try fallback adapters if primary fails."""
        adapters_to_try = []
        
        if isinstance(self._adapter, PlaywrightAdapter):
            adapters_to_try.append(SeleniumWebDriverAdapter())
        else:
            adapters_to_try.append(PlaywrightAdapter())
        
        for adapter in adapters_to_try:
            try:
                await adapter.connect(config)
                self._adapter = adapter
                logger.info(f"Successfully fell back to {type(adapter).__name__}")
                return
            except Exception as e:
                logger.warning(f"Fallback to {type(adapter).__name__} failed: {e}")
                continue
        
        raise BrowserCompatibilityError(
            f"Failed to initialize any adapter for {config.browser_type}"
        )

    async def shutdown(self) -> None:
        """Shutdown browser and cleanup."""
        if self._adapter:
            await self._adapter.disconnect()

    def get_capabilities(self) -> BrowserCapabilities:
        """Get detected browser capabilities."""
        if self._capabilities is None:
            raise BrowserCompatibilityError("Browser not initialized")
        return self._capabilities

    def check_feature_support(self, feature: str) -> bool:
        """Check if a specific feature is supported."""
        if self._capabilities is None:
            return False
        
        feature_map = {
            "cdp": self._capabilities.supports_cdp,
            "webdriver_bidi": self._capabilities.supports_webdriver_bidi,
            "stealth": self._capabilities.supports_stealth_mode,
            "network_interception": self._capabilities.supports_network_interception,
            "javascript_dialogs": self._capabilities.supports_javascript_dialogs,
            "file_chooser": self._capabilities.supports_file_chooser,
            "permissions": self._capabilities.supports_permissions,
            "geolocation": self._capabilities.supports_geolocation,
            "webgl": self._capabilities.supports_webgl,
            "webrtc": self._capabilities.supports_webrtc,
        }
        
        return feature_map.get(feature, False)

    async def new_page(self, url: Optional[str] = None) -> Any:
        """Create new page with compatibility handling."""
        if not self._adapter:
            raise BrowserCompatibilityError("Browser not initialized")
        
        page = await self._adapter.new_page(url)
        
        # Apply browser-specific quirks
        if self._capabilities and self._capabilities.quirks.get("needs_page_load_strategy"):
            # Set page load strategy for Chrome/Edge
            if hasattr(page, "set_default_navigation_timeout"):
                page.set_default_navigation_timeout(self._config.timeout)
        
        return page

    async def execute_with_fallback(
        self,
        page: Any,
        script: str,
        *args: Any,
        fallback_value: Any = None
    ) -> Any:
        """Execute script with fallback for unsupported features."""
        try:
            return await self._adapter.execute_script(page, script, *args)
        except Exception as e:
            logger.warning(f"Script execution failed: {e}")
            return fallback_value

    async def navigate_with_retry(
        self,
        page: Any,
        url: str,
        max_retries: int = 3,
        wait_until: str = "load"
    ) -> bool:
        """Navigate with retry logic for unreliable connections."""
        for attempt in range(max_retries):
            try:
                await self._adapter.navigate(page, url, wait_until)
                return True
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Navigation failed after {max_retries} attempts: {e}")
                    return False
                logger.warning(f"Navigation attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
        return False


class CrossBrowserTestSuite:
    """Test suite for cross-browser compatibility testing."""

    def __init__(self) -> None:
        self.results: Dict[BrowserType, Dict[str, Any]] = {}
        self.compatibility_layer = BrowserCompatibilityLayer()

    async def run_test(
        self,
        browser_type: BrowserType,
        test_name: str,
        test_func: callable,
        config: Optional[BrowserConfig] = None
    ) -> Dict[str, Any]:
        """Run a single test on specified browser."""
        if config is None:
            config = BrowserConfig(browser_type=browser_type, headless=True)
        
        result = {
            "browser": browser_type.value,
            "test": test_name,
            "success": False,
            "error": None,
            "duration": 0,
            "capabilities": None,
        }
        
        try:
            import time
            start_time = time.time()
            
            await self.compatibility_layer.initialize(config)
            capabilities = self.compatibility_layer.get_capabilities()
            result["capabilities"] = capabilities.__dict__
            
            # Run the test function
            test_result = await test_func(self.compatibility_layer)
            result["success"] = test_result.get("success", False)
            result["data"] = test_result.get("data")
            
            result["duration"] = time.time() - start_time
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Test {test_name} failed on {browser_type.value}: {e}")
        finally:
            await self.compatibility_layer.shutdown()
        
        if browser_type not in self.results:
            self.results[browser_type] = {}
        self.results[browser_type][test_name] = result
        
        return result

    async def run_all_browsers(
        self,
        test_name: str,
        test_func: callable,
        browsers: Optional[List[BrowserType]] = None
    ) -> Dict[BrowserType, Dict[str, Any]]:
        """Run test across multiple browsers."""
        if browsers is None:
            browsers = [
                BrowserType.CHROME,
                BrowserType.FIREFOX,
                BrowserType.EDGE,
            ]
            
            # Add Safari only on macOS
            if platform.system() == "Darwin":
                browsers.append(BrowserType.SAFARI)
        
        results = {}
        for browser_type in browsers:
            try:
                result = await self.run_test(browser_type, test_name, test_func)
                results[browser_type] = result
            except Exception as e:
                logger.error(f"Failed to run test on {browser_type.value}: {e}")
                results[browser_type] = {
                    "browser": browser_type.value,
                    "test": test_name,
                    "success": False,
                    "error": str(e),
                }
        
        return results

    def generate_report(self) -> Dict[str, Any]:
        """Generate compatibility report."""
        report = {
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "browsers_tested": list(self.results.keys()),
            },
            "details": self.results,
            "compatibility_matrix": {},
        }
        
        # Generate compatibility matrix
        for browser_type, tests in self.results.items():
            browser_name = browser_type.value
            report["compatibility_matrix"][browser_name] = {}
            
            for test_name, result in tests.items():
                report["compatibility_matrix"][browser_name][test_name] = result["success"]
                report["summary"]["total_tests"] += 1
                if result["success"]:
                    report["summary"]["passed"] += 1
                else:
                    report["summary"]["failed"] += 1
        
        return report


# Utility functions for common compatibility issues
def get_browser_specific_delay(browser_type: BrowserType) -> float:
    """Get recommended delay between actions for browser stability."""
    delays = {
        BrowserType.CHROME: 0.1,
        BrowserType.FIREFOX: 0.2,
        BrowserType.SAFARI: 0.3,
        BrowserType.EDGE: 0.1,
        BrowserType.CHROMIUM: 0.1,
    }
    return delays.get(browser_type, 0.1)


def get_user_agent_string(browser_type: BrowserType, platform: str = "desktop") -> str:
    """Get realistic user agent string for browser."""
    agents = {
        BrowserType.CHROME: {
            "desktop": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "mobile": "Mozilla/5.0 (Linux; Android 10; SM-G981B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
        },
        BrowserType.FIREFOX: {
            "desktop": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "mobile": "Mozilla/5.0 (Android 10; Mobile; rv:121.0) Gecko/121.0 Firefox/121.0",
        },
        BrowserType.SAFARI: {
            "desktop": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
            "mobile": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1",
        },
        BrowserType.EDGE: {
            "desktop": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
            "mobile": "Mozilla/5.0 (Linux; Android 10; HD1913) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36 EdgA/120.0.0.0",
        },
    }
    
    return agents.get(browser_type, {}).get(platform, agents[BrowserType.CHROME]["desktop"])


def handle_browser_quirk(
    browser_type: BrowserType,
    quirk_name: str,
    default_value: Any = None
) -> Any:
    """Handle browser-specific quirks and return appropriate value."""
    quirks = {
        BrowserType.CHROME: {
            "max_data_uri_length": 2 * 1024 * 1024 * 1024,  # 2GB
            "supports_beforeunload": True,
            "needs_click_delay": False,
            "supports_shadow_dom": True,
        },
        BrowserType.FIREFOX: {
            "max_data_uri_length": 512 * 1024 * 1024,  # 512MB
            "supports_beforeunload": False,
            "needs_click_delay": True,
            "supports_shadow_dom": True,
        },
        BrowserType.SAFARI: {
            "max_data_uri_length": 100 * 1024 * 1024,  # 100MB
            "supports_beforeunload": False,
            "needs_click_delay": True,
            "supports_shadow_dom": False,
        },
        BrowserType.EDGE: {
            "max_data_uri_length": 2 * 1024 * 1024 * 1024,  # 2GB
            "supports_beforeunload": True,
            "needs_click_delay": False,
            "supports_shadow_dom": True,
        },
    }
    
    return quirks.get(browser_type, {}).get(quirk_name, default_value)


# Integration with existing nexus modules
def create_compatible_page_adapter(page: Any, browser_type: BrowserType) -> Any:
    """Create a compatible page adapter for existing nexus modules.
    
    This function wraps an existing page object (from Playwright or Selenium)
    to provide a consistent interface for nexus modules.
    """
    from nexus.actor.page import PageActor
    
    class CompatiblePageAdapter:
        """Adapter to make different page objects work with nexus."""
        
        def __init__(self, page: Any, browser_type: BrowserType):
            self._page = page
            self._browser_type = browser_type
            self._delay = get_browser_specific_delay(browser_type)
        
        async def goto(self, url: str, **kwargs) -> None:
            """Navigate to URL with browser-specific handling."""
            if hasattr(self._page, 'goto'):  # Playwright
                await self._page.goto(url, **kwargs)
            elif hasattr(self._page, 'get'):  # Selenium
                self._page.get(url)
                # Add delay for Safari/Firefox stability
                if self._delay > 0:
                    await asyncio.sleep(self._delay)
        
        async def evaluate(self, expression: str, *args) -> Any:
            """Evaluate JavaScript with fallbacks."""
            try:
                if hasattr(self._page, 'evaluate'):  # Playwright
                    return await self._page.evaluate(expression, *args)
                elif hasattr(self._page, 'execute_script'):  # Selenium
                    return self._page.execute_script(expression, *args)
            except Exception as e:
                logger.warning(f"Script evaluation failed: {e}")
                return None
        
        async def screenshot(self, **kwargs) -> bytes:
            """Take screenshot with format handling."""
            if hasattr(self._page, 'screenshot'):  # Playwright
                return await self._page.screenshot(**kwargs)
            elif hasattr(self._page, 'get_screenshot_as_png'):  # Selenium
                return self._page.get_screenshot_as_png()
            return b''
        
        # Proxy other attributes
        def __getattr__(self, name: str) -> Any:
            return getattr(self._page, name)
    
    return CompatiblePageAdapter(page, browser_type)


# Example usage and testing
async def example_cross_browser_test():
    """Example of using the cross-browser compatibility layer."""
    suite = CrossBrowserTestSuite()
    
    async def test_navigation(compatibility: BrowserCompatibilityLayer) -> Dict[str, Any]:
        """Test basic navigation."""
        page = await compatibility.new_page()
        success = await compatibility.navigate_with_retry(page, "https://example.com")
        
        # Get page title
        title = await compatibility.execute_with_fallback(
            page,
            "return document.title",
            fallback_value="Unknown"
        )
        
        await compatibility.shutdown()
        
        return {
            "success": success,
            "data": {"title": title}
        }
    
    # Run test on Chrome and Firefox
    results = await suite.run_all_browsers(
        "navigation_test",
        test_navigation,
        browsers=[BrowserType.CHROME, BrowserType.FIREFOX]
    )
    
    # Generate report
    report = suite.generate_report()
    print(f"Test Results: {report['summary']}")
    
    return report


if __name__ == "__main__":
    # Example usage
    asyncio.run(example_cross_browser_test())