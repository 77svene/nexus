"""
Firefox adapter for nexus multi-engine support.
Implements Firefox DevTools Protocol translation layer.
"""

import asyncio
import json
import logging
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from urllib.parse import urlparse

import aiohttp
import websockets
from websockets.exceptions import ConnectionClosed

from nexus.actor.page import Page
from nexus.actor.element import Element
from nexus.actor.mouse import Mouse
from nexus.engines.base import BrowserEngine, BrowserEngineError

logger = logging.getLogger(__name__)

# Firefox DevTools Protocol constants
FIREFOX_DEBUG_PORT = 9222
FIREFOX_WEBSOCKET_TIMEOUT = 30
FIREFOX_STARTUP_TIMEOUT = 60
FIREFOX_PREFERENCE_OVERRIDES = {
    "devtools.debugger.remote-enabled": True,
    "devtools.debugger.prompt-connection": False,
    "devtools.debugger.unix-domain-socket": "/tmp/firefox-debug.sock",
    "marionette.enabled": True,
    "network.cookie.cookieBehavior": 0,
    "privacy.trackingprotection.enabled": False,
    "browser.safebrowsing.malware.enabled": False,
    "browser.safebrowsing.phishing.enabled": False,
    "dom.webdriver.enabled": True,
    "useAutomationExtension": False,
    "dom.ipc.processCount": 8,
    "fission.autostart": True,
    "gfx.webrender.all": True,
    "layers.acceleration.force-enabled": True,
    "media.autoplay.default": 0,
    "media.autoplay.enabled": True,
    "network.http.speculative-parallel-limit": 6,
    "network.dns.disablePrefetch": False,
    "network.prefetch-next": True,
    "browser.cache.disk.enable": True,
    "browser.cache.memory.enable": True,
    "browser.cache.memory.capacity": 65536,
    "network.http.pipelining": True,
    "network.http.proxy.pipelining": True,
    "network.http.pipelining.maxrequests": 8,
    "nglayout.initialpaint.delay": 0,
    "browser.startup.homepage": "about:blank",
    "browser.startup.firstrunSkipsHomepage": False,
    "browser.sessionstore.resume_from_crash": False,
    "browser.shell.checkDefaultBrowser": False,
    "browser.bookmarks.restore_default_bookmarks": False,
    "dom.disable_beforeunload": True,
    "dom.disable_window_move_resize": True,
    "dom.disable_window_flip": True,
    "dom.disable_window_open_feature.status": True,
    "dom.disable_window_open_feature.menubar": True,
    "dom.disable_window_open_feature.personalbar": True,
    "dom.disable_window_open_feature.scrollbars": True,
    "dom.disable_window_open_feature.toolbar": True,
    "dom.disable_open_during_load": True,
    "dom.successive_dialog_time_limit": 0,
    "extensions.autoDisableScopes": 0,
    "extensions.enabledScopes": 15,
    "xpinstall.signatures.required": False,
    "extensions.install.requireBuiltInCerts": False,
    "extensions.checkUpdateSecurity": False,
    "extensions.update.enabled": False,
    "extensions.update.autoUpdateDefault": False,
    "app.update.enabled": False,
    "app.update.silent": True,
    "datareporting.healthreport.uploadEnabled": False,
    "datareporting.policy.dataSubmissionEnabled": False,
    "toolkit.telemetry.enabled": False,
    "toolkit.telemetry.archive.enabled": False,
    "toolkit.telemetry.unified": False,
    "toolkit.telemetry.newProfilePing.enabled": False,
    "toolkit.telemetry.shutdownPingSender.enabled": False,
    "toolkit.telemetry.updatePing.enabled": False,
    "toolkit.telemetry.bhrPing.enabled": False,
    "toolkit.telemetry.firstShutdownPing.enabled": False,
    "breakpad.reportURL": "",
    "browser.tabs.crashSubmit.sendReport": False,
    "browser.crashReports.unsubmittedCheck.enabled": False,
    "browser.crashReports.unsubmittedCheck.autoSubmit": False,
    "network.dns.disablePrefetchFromHTTPS": True,
    "network.predictor.enabled": False,
    "network.http.speculative-parallel-limit": 0,
    "browser.urlbar.speculativeConnect.enabled": False,
    "network.captive-portal-service.enabled": False,
    "browser.safebrowsing.blockedURIs.enabled": False,
    "browser.safebrowsing.downloads.enabled": False,
    "browser.safebrowsing.downloads.remote.enabled": False,
    "browser.safebrowsing.passwords.enabled": False,
    "browser.safebrowsing.provider.google4.dataSharing.enabled": False,
    "browser.safebrowsing.provider.google4.lists": "",
    "browser.safebrowsing.provider.google4.updateURL": "",
    "browser.safebrowsing.provider.google.reportURL": "",
    "browser.safebrowsing.provider.google.gethashURL": "",
    "browser.safebrowsing.provider.google.updateURL": "",
    "browser.safebrowsing.provider.google.reportMalwareMistakeURL": "",
    "browser.safebrowsing.provider.google.reportPhishMistakeURL": "",
    "browser.safebrowsing.provider.google4.reportMalwareMistakeURL": "",
    "browser.safebrowsing.provider.google4.reportPhishMistakeURL": "",
    "browser.safebrowsing.allowOverride": False,
    "browser.newtabpage.enabled": False,
    "browser.newtabpage.enhanced": False,
    "browser.newtabpage.introShown": True,
    "browser.aboutHomeSnippets.updateUrl": "",
    "browser.startup.homepage_override.mstone": "ignore",
    "browser.startup.homepage_override.buildID": "",
    "browser.startup.page": 0,
    "browser.newtab.preload": False,
    "browser.newtabpage.directory.ping": "",
    "browser.newtabpage.directory.source": "data:text/plain,{}",
    "browser.library.activity-stream.enabled": False,
    "browser.topsites.contile.enabled": False,
    "browser.urlbar.quicksuggest.dataCollection.enabled": False,
    "browser.urlbar.suggest.quicksuggest.nonsponsored": False,
    "browser.urlbar.suggest.quicksuggest.sponsored": False,
    "browser.urlbar.suggest.topsites": False,
    "browser.urlbar.suggest.engines": False,
    "browser.urlbar.suggest.history": False,
    "browser.urlbar.suggest.bookmark": False,
    "browser.urlbar.suggest.openpage": False,
    "browser.urlbar.suggest.remotetab": False,
    "browser.urlbar.suggest.searches": False,
    "browser.urlbar.autoFill": False,
    "browser.urlbar.autoFill.adaptiveHistory.enabled": False,
    "browser.urlbar.suggest.quicksuggest.onboardingDialogChoice": "",
    "browser.urlbar.quicksuggest.enabled": False,
    "browser.urlbar.quicksuggest.rankingImproved": False,
    "browser.urlbar.quicksuggest.rankingExperimentGroup": "",
    "browser.urlbar.quicksuggest.rankingModel": "",
    "browser.urlbar.quicksuggest.rankingModelVersion": "",
    "browser.urlbar.quicksuggest.rankingModelRevision": "",
    "browser.urlbar.quicksuggest.rankingModelTrainingDate": "",
    "browser.urlbar.quicksuggest.rankingModelTrainingId": "",
    "browser.urlbar.quicksuggest.rankingModelTrainingVersion": "",
    "browser.urlbar.quicksuggest.rankingModelTrainingRevision": "",
    "browser.urlbar.quicksuggest.rankingModelTrainingDate": "",
    "browser.urlbar.quicksuggest.rankingModelTrainingId": "",
    "browser.urlbar.quicksuggest.rankingModelTrainingVersion": "",
    "browser.urlbar.quicksuggest.rankingModelTrainingRevision": "",
}

class FirefoxAdapter(BrowserEngine):
    """
    Firefox browser engine adapter implementing Firefox DevTools Protocol.
    Provides unified API compatible with nexus framework.
    """
    
    ENGINE_NAME = "firefox"
    PROTOCOL_VERSION = "1.3"
    
    def __init__(
        self,
        executable_path: Optional[str] = None,
        profile_path: Optional[str] = None,
        headless: bool = False,
        debugging_port: int = FIREFOX_DEBUG_PORT,
        websocket_timeout: int = FIREFOX_WEBSOCKET_TIMEOUT,
        startup_timeout: int = FIREFOX_STARTUP_TIMEOUT,
        extra_args: Optional[List[str]] = None,
        env_vars: Optional[Dict[str, str]] = None,
        preferences: Optional[Dict[str, Any]] = None,
        mobile_emulation: Optional[Dict[str, Any]] = None,
        proxy: Optional[Dict[str, str]] = None,
        user_agent: Optional[str] = None,
        viewport: Optional[Dict[str, int]] = None,
        **kwargs
    ):
        """
        Initialize Firefox adapter.
        
        Args:
            executable_path: Path to Firefox executable
            profile_path: Path to Firefox profile directory
            headless: Run in headless mode
            debugging_port: Port for Firefox DevTools Protocol
            websocket_timeout: Timeout for WebSocket connections
            startup_timeout: Timeout for Firefox startup
            extra_args: Additional command line arguments
            env_vars: Environment variables for Firefox process
            preferences: Firefox preferences to set
            mobile_emulation: Mobile device emulation settings
            proxy: Proxy configuration
            user_agent: Custom user agent string
            viewport: Viewport size {width, height}
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        
        self.executable_path = executable_path or self._find_firefox_executable()
        self.profile_path = profile_path
        self.headless = headless
        self.debugging_port = debugging_port
        self.websocket_timeout = websocket_timeout
        self.startup_timeout = startup_timeout
        self.extra_args = extra_args or []
        self.env_vars = env_vars or {}
        self.preferences = preferences or {}
        self.mobile_emulation = mobile_emulation
        self.proxy = proxy
        self.user_agent = user_agent
        self.viewport = viewport or {"width": 1280, "height": 720}
        
        self.process = None
        self.temp_profile_dir = None
        self.websocket = None
        self.session = None
        self.pages = {}
        self.targets = {}
        self.command_id = 0
        self.pending_commands = {}
        self.event_handlers = {}
        self._shutdown_event = asyncio.Event()
        self._message_queue = asyncio.Queue()
        
        # Merge preferences with defaults
        self.preferences = {**FIREFOX_PREFERENCE_OVERRIDES, **self.preferences}
        
        # Apply mobile emulation if specified
        if self.mobile_emulation:
            self._apply_mobile_emulation()
        
        # Apply proxy settings
        if self.proxy:
            self._apply_proxy_settings()
        
        # Apply user agent
        if self.user_agent:
            self.preferences["general.useragent.override"] = self.user_agent
        
        logger.info(f"FirefoxAdapter initialized with port {self.debugging_port}")
    
    def _find_firefox_executable(self) -> str:
        """Find Firefox executable on the system."""
        possible_paths = [
            "/usr/bin/firefox",
            "/usr/bin/firefox-esr",
            "/usr/local/bin/firefox",
            "/Applications/Firefox.app/Contents/MacOS/firefox",
            "C:\\Program Files\\Mozilla Firefox\\firefox.exe",
            "C:\\Program Files (x86)\\Mozilla Firefox\\firefox.exe",
        ]
        
        for path in possible_paths:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path
        
        # Try to find in PATH
        for path in os.environ.get("PATH", "").split(os.pathsep):
            firefox_path = os.path.join(path, "firefox")
            if os.path.isfile(firefox_path) and os.access(firefox_path, os.X_OK):
                return firefox_path
        
        raise BrowserEngineError("Firefox executable not found")
    
    def _apply_mobile_emulation(self):
        """Apply mobile device emulation settings."""
        if not self.mobile_emulation:
            return
        
        device_name = self.mobile_emulation.get("device_name")
        if device_name:
            # Common device configurations
            devices = {
                "iPhone 12": {"width": 390, "height": 844, "pixel_ratio": 3},
                "iPhone 12 Pro": {"width": 390, "height": 844, "pixel_ratio": 3},
                "iPhone 12 Pro Max": {"width": 428, "height": 926, "pixel_ratio": 3},
                "iPhone 12 Mini": {"width": 360, "height": 780, "pixel_ratio": 3},
                "iPhone SE": {"width": 375, "height": 667, "pixel_ratio": 2},
                "Pixel 5": {"width": 393, "height": 851, "pixel_ratio": 2.625},
                "Pixel 4": {"width": 393, "height": 851, "pixel_ratio": 2.625},
                "Galaxy S20": {"width": 360, "height": 800, "pixel_ratio": 3},
                "Galaxy S20 Ultra": {"width": 412, "height": 915, "pixel_ratio": 3.5},
                "iPad Pro": {"width": 1024, "height": 1366, "pixel_ratio": 2},
                "iPad Mini": {"width": 768, "height": 1024, "pixel_ratio": 2},
            }
            
            if device_name in devices:
                device = devices[device_name]
                self.viewport = {"width": device["width"], "height": device["height"]}
                self.preferences["layout.css.devPixelsPerPx"] = device["pixel_ratio"]
                
                # Set user agent for mobile
                if "iPhone" in device_name or "iPad" in device_name:
                    self.user_agent = (
                        "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) "
                        "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                        "Version/14.0 Mobile/15E148 Safari/604.1"
                    )
                elif "Pixel" in device_name:
                    self.user_agent = (
                        "Mozilla/5.0 (Linux; Android 11; Pixel 5) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/90.0.4430.91 Mobile Safari/537.36"
                    )
                
                self.preferences["general.useragent.override"] = self.user_agent
        else:
            # Custom device settings
            if "width" in self.mobile_emulation and "height" in self.mobile_emulation:
                self.viewport = {
                    "width": self.mobile_emulation["width"],
                    "height": self.mobile_emulation["height"]
                }
            if "pixel_ratio" in self.mobile_emulation:
                self.preferences["layout.css.devPixelsPerPx"] = self.mobile_emulation["pixel_ratio"]
            if "user_agent" in self.mobile_emulation:
                self.user_agent = self.mobile_emulation["user_agent"]
                self.preferences["general.useragent.override"] = self.user_agent
    
    def _apply_proxy_settings(self):
        """Apply proxy configuration."""
        if not self.proxy:
            return
        
        proxy_type = self.proxy.get("type", "http").lower()
        host = self.proxy.get("host", "")
        port = self.proxy.get("port", "")
        
        if proxy_type == "http":
            self.preferences["network.proxy.type"] = 1
            self.preferences["network.proxy.http"] = host
            self.preferences["network.proxy.http_port"] = int(port)
            self.preferences["network.proxy.ssl"] = host
            self.preferences["network.proxy.ssl_port"] = int(port)
        elif proxy_type == "socks":
            self.preferences["network.proxy.type"] = 1
            self.preferences["network.proxy.socks"] = host
            self.preferences["network.proxy.socks_port"] = int(port)
            self.preferences["network.proxy.socks_version"] = 5
        elif proxy_type == "socks4":
            self.preferences["network.proxy.type"] = 1
            self.preferences["network.proxy.socks"] = host
            self.preferences["network.proxy.socks_port"] = int(port)
            self.preferences["network.proxy.socks_version"] = 4
    
    async def start(self) -> None:
        """Start Firefox browser with remote debugging enabled."""
        try:
            # Create temporary profile directory
            self.temp_profile_dir = tempfile.mkdtemp(prefix="firefox_profile_")
            logger.debug(f"Created temporary profile: {self.temp_profile_dir}")
            
            # Write preferences to user.js
            prefs_file = os.path.join(self.temp_profile_dir, "user.js")
            with open(prefs_file, "w") as f:
                for key, value in self.preferences.items():
                    if isinstance(value, str):
                        f.write(f'user_pref("{key}", "{value}");\n')
                    elif isinstance(value, bool):
                        f.write(f'user_pref("{key}", {str(value).lower()});\n')
                    else:
                        f.write(f'user_pref("{key}", {value});\n')
            
            # Prepare command line arguments
            cmd_args = [
                self.executable_path,
                "--profile", self.temp_profile_dir,
                "--remote-debugging-port", str(self.debugging_port),
                "--no-remote",
                "--new-instance",
            ]
            
            if self.headless:
                cmd_args.append("--headless")
            
            # Add viewport size
            cmd_args.extend([
                "--width", str(self.viewport["width"]),
                "--height", str(self.viewport["height"]),
            ])
            
            # Add extra arguments
            cmd_args.extend(self.extra_args)
            
            # Set up environment
            env = os.environ.copy()
            env.update(self.env_vars)
            env["MOZ_HEADLESS"] = "1" if self.headless else "0"
            env["MOZ_REMOTE_DEBUGGING_PORT"] = str(self.debugging_port)
            
            # Start Firefox process
            logger.info(f"Starting Firefox: {' '.join(cmd_args)}")
            self.process = subprocess.Popen(
                cmd_args,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            
            # Wait for Firefox to start and debugging port to be available
            await self._wait_for_firefox_start()
            
            # Create HTTP session for REST API
            self.session = aiohttp.ClientSession()
            
            # Get available targets/pages
            await self._discover_targets()
            
            logger.info(f"Firefox started successfully on port {self.debugging_port}")
            
        except Exception as e:
            await self.stop()
            raise BrowserEngineError(f"Failed to start Firefox: {e}")
    
    async def _wait_for_firefox_start(self) -> None:
        """Wait for Firefox to start and debugging port to be available."""
        start_time = time.time()
        while time.time() - start_time < self.startup_timeout:
            # Check if process is still running
            if self.process.poll() is not None:
                stdout = self.process.stdout.read().decode() if self.process.stdout else ""
                stderr = self.process.stderr.read().decode() if self.process.stderr else ""
                raise BrowserEngineError(
                    f"Firefox process exited with code {self.process.returncode}\n"
                    f"stdout: {stdout}\nstderr: {stderr}"
                )
            
            # Try to connect to debugging port
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(("localhost", self.debugging_port))
                sock.close()
                
                if result == 0:
                    # Port is available, wait a bit more for Firefox to fully initialize
                    await asyncio.sleep(2)
                    return
            except socket.error:
                pass
            
            await asyncio.sleep(0.5)
        
        raise BrowserEngineError(f"Firefox failed to start within {self.startup_timeout} seconds")
    
    async def _discover_targets(self) -> None:
        """Discover available browser targets/pages."""
        try:
            async with self.session.get(
                f"http://localhost:{self.debugging_port}/json/list"
            ) as response:
                if response.status == 200:
                    targets = await response.json()
                    for target in targets:
                        target_id = target.get("id")
                        if target_id:
                            self.targets[target_id] = target
                            logger.debug(f"Discovered target: {target.get('title', 'Unknown')}")
                else:
                    logger.warning(f"Failed to discover targets: HTTP {response.status}")
        except Exception as e:
            logger.warning(f"Error discovering targets: {e}")
    
    async def stop(self) -> None:
        """Stop Firefox browser and clean up resources."""
        self._shutdown_event.set()
        
        # Close WebSocket connection
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception:
                pass
            self.websocket = None
        
        # Close HTTP session
        if self.session:
            try:
                await self.session.close()
            except Exception:
                pass
            self.session = None
        
        # Terminate Firefox process
        if self.process:
            try:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=2)
            except Exception as e:
                logger.warning(f"Error terminating Firefox process: {e}")
            self.process = None
        
        # Clean up temporary profile
        if self.temp_profile_dir and os.path.exists(self.temp_profile_dir):
            try:
                import shutil
                shutil.rmtree(self.temp_profile_dir, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Error removing temporary profile: {e}")
            self.temp_profile_dir = None
        
        # Clear internal state
        self.pages.clear()
        self.targets.clear()
        self.pending_commands.clear()
        self.event_handlers.clear()
        
        logger.info("Firefox stopped")
    
    async def new_page(self, url: Optional[str] = None) -> Page:
        """Create a new browser tab/page."""
        try:
            # Create new target via HTTP API
            create_url = f"http://localhost:{self.debugging_port}/json/new"
            if url:
                create_url += f"?{url}"
            
            async with self.session.put(create_url) as response:
                if response.status == 200:
                    target_info = await response.json()
                    target_id = target_info.get("id")
                    
                    if not target_id:
                        raise BrowserEngineError("Failed to get target ID for new page")
                    
                    # Store target info
                    self.targets[target_id] = target_info
                    
                    # Connect to the new page via WebSocket
                    page = await self._connect_to_target(target_id)
                    
                    # Navigate to URL if provided
                    if url and not url.startswith("http"):
                        url = f"http://{url}"
                    if url:
                        await page.goto(url)
                    
                    return page
                else:
                    raise BrowserEngineError(f"Failed to create new page: HTTP {response.status}")
        except Exception as e:
            raise BrowserEngineError(f"Failed to create new page: {e}")
    
    async def _connect_to_target(self, target_id: str) -> Page:
        """Connect to a specific browser target via WebSocket."""
        target_info = self.targets.get(target_id)
        if not target_info:
            raise BrowserEngineError(f"Target {target_id} not found")
        
        websocket_url = target_info.get("webSocketDebuggerUrl")
        if not websocket_url:
            raise BrowserEngineError(f"No WebSocket URL for target {target_id}")
        
        # Create FirefoxPage instance
        page = FirefoxPage(
            adapter=self,
            target_id=target_id,
            websocket_url=websocket_url,
            viewport=self.viewport,
        )
        
        await page.connect()
        self.pages[target_id] = page
        
        return page
    
    async def pages(self) -> List[Page]:
        """Get all open pages/tabs."""
        # Refresh targets list
        await self._discover_targets()
        
        # Return existing pages or create new ones for discovered targets
        pages = []
        for target_id, target_info in self.targets.items():
            if target_info.get("type") == "page":
                if target_id in self.pages:
                    pages.append(self.pages[target_id])
                else:
                    try:
                        page = await self._connect_to_target(target_id)
                        pages.append(page)
                    except Exception as e:
                        logger.warning(f"Failed to connect to target {target_id}: {e}")
        
        return pages
    
    async def close_page(self, page: Page) -> None:
        """Close a specific page/tab."""
        if isinstance(page, FirefoxPage):
            target_id = page.target_id
            try:
                # Close via HTTP API
                async with self.session.get(
                    f"http://localhost:{self.debugging_port}/json/close/{target_id}"
                ) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to close page via API: HTTP {response.status}")
            except Exception as e:
                logger.warning(f"Error closing page via API: {e}")
            
            # Disconnect WebSocket
            await page.disconnect()
            
            # Remove from internal state
            self.pages.pop(target_id, None)
            self.targets.pop(target_id, None)
    
    async def get_browser_version(self) -> Dict[str, str]:
        """Get browser version information."""
        try:
            async with self.session.get(
                f"http://localhost:{self.debugging_port}/json/version"
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise BrowserEngineError(f"Failed to get version: HTTP {response.status}")
        except Exception as e:
            raise BrowserEngineError(f"Failed to get browser version: {e}")
    
    async def execute_cdp_command(self, method: str, params: Optional[Dict] = None) -> Any:
        """Execute a CDP command (translated to Firefox DevTools Protocol)."""
        # Translate CDP method to Firefox DevTools Protocol equivalent
        translated_method = self._translate_cdp_method(method)
        translated_params = self._translate_cdp_params(method, params)
        
        # Use the first available page to execute command
        pages = await self.pages()
        if not pages:
            raise BrowserEngineError("No pages available to execute command")
        
        page = pages[0]
        return await page.send_command(translated_method, translated_params)
    
    def _translate_cdp_method(self, cdp_method: str) -> str:
        """Translate Chrome DevTools Protocol method to Firefox equivalent."""
        # Common CDP to Firefox DevTools Protocol mappings
        translations = {
            "Page.navigate": "Page.navigate",
            "Page.reload": "Page.reload",
            "Runtime.evaluate": "Runtime.evaluate",
            "DOM.getDocument": "DOM.getDocument",
            "DOM.querySelector": "DOM.querySelector",
            "DOM.querySelectorAll": "DOM.querySelectorAll",
            "DOM.getBoxModel": "DOM.getBoxModel",
            "DOM.setAttributeValue": "DOM.setAttribute",
            "DOM.removeAttribute": "DOM.removeAttribute",
            "DOM.setNodeValue": "DOM.setNodeValue",
            "DOM.getOuterHTML": "DOM.getOuterHTML",
            "DOM.setOuterHTML": "DOM.setOuterHTML",
            "DOM.getAttributes": "DOM.getAttributes",
            "DOM.moveTo": "DOM.moveTo",
            "DOM.resolveNode": "DOM.resolveNode",
            "DOM.describeNode": "DOM.describeNode",
            "DOM.scrollIntoViewIfNeeded": "DOM.scrollIntoViewIfNeeded",
            "DOM.focus": "DOM.focus",
            "Input.dispatchMouseEvent": "Input.dispatchMouseEvent",
            "Input.dispatchKeyEvent": "Input.dispatchKeyEvent",
            "Input.dispatchTouchEvent": "Input.dispatchTouchEvent",
            "Input.insertText": "Input.insertText",
            "Input.dispatchMouseEvent": "Input.dispatchMouseEvent",
            "Network.enable": "Network.enable",
            "Network.disable": "Network.disable",
            "Network.setExtraHTTPHeaders": "Network.setExtraHTTPHeaders",
            "Network.setCacheDisabled": "Network.setCacheDisabled",
            "Network.emulateNetworkConditions": "Network.emulateNetworkConditions",
            "Network.setCookie": "Network.setCookie",
            "Network.deleteCookies": "Network.deleteCookies",
            "Network.clearBrowserCookies": "Network.clearBrowserCookies",
            "Network.clearBrowserCache": "Network.clearBrowserCache",
            "Network.getCookies": "Network.getCookies",
            "Network.getRequestPostData": "Network.getRequestPostData",
            "Network.getResponseBody": "Network.getResponseBody",
            "Emulation.setDeviceMetricsOverride": "Emulation.setDeviceMetricsOverride",
            "Emulation.setTouchEmulationEnabled": "Emulation.setTouchEmulationEnabled",
            "Emulation.setEmulatedMedia": "Emulation.setEmulatedMedia",
            "Emulation.setUserAgentOverride": "Emulation.setUserAgentOverride",
            "Page.captureScreenshot": "Page.captureScreenshot",
            "Page.captureSnapshot": "Page.captureSnapshot",
            "Page.printToPDF": "Page.printToPDF",
            "Page.addScriptToEvaluateOnNewDocument": "Page.addScriptToEvaluateOnNewDocument",
            "Page.removeScriptToEvaluateOnNewDocument": "Page.removeScriptToEvaluateOnNewDocument",
            "Page.setInterceptFileChooserDialog": "Page.setInterceptFileChooserDialog",
            "Page.handleFileChooser": "Page.handleFileChooser",
            "Page.setGeolocationOverride": "Page.setGeolocationOverride",
            "Page.clearGeolocationOverride": "Page.clearGeolocationOverride",
            "Page.setDeviceOrientationOverride": "Page.setDeviceOrientationOverride",
            "Page.clearDeviceOrientationOverride": "Page.clearDeviceOrientationOverride",
            "Performance.enable": "Performance.enable",
            "Performance.disable": "Performance.disable",
            "Performance.getMetrics": "Performance.getMetrics",
            "Log.enable": "Log.enable",
            "Log.disable": "Log.disable",
            "Log.clear": "Log.clear",
            "Log.startViolationsReport": "Log.startViolationsReport",
            "Log.stopViolationsReport": "Log.stopViolationsReport",
        }
        
        return translations.get(cdp_method, cdp_method)
    
    def _translate_cdp_params(self, cdp_method: str, params: Optional[Dict]) -> Optional[Dict]:
        """Translate CDP parameters to Firefox DevTools Protocol format."""
        if not params:
            return params
        
        # Some parameters need translation
        if cdp_method == "DOM.setAttributeValue":
            # Firefox uses "DOM.setAttribute" with different parameter names
            if "nodeId" in params and "name" in params and "value" in params:
                return {
                    "nodeId": params["nodeId"],
                    "name": params["name"],
                    "value": params["value"]
                }
        
        elif cdp_method == "Emulation.setDeviceMetricsOverride":
            # Firefox uses same structure but might need adjustments
            return {
                "width": params.get("width", 0),
                "height": params.get("height", 0),
                "deviceScaleFactor": params.get("deviceScaleFactor", 1),
                "mobile": params.get("mobile", False),
                "screenWidth": params.get("screenWidth", params.get("width", 0)),
                "screenHeight": params.get("screenHeight", params.get("height", 0)),
                "positionX": params.get("positionX", 0),
                "positionY": params.get("positionY", 0),
                "dontSetVisibleSize": params.get("dontSetVisibleSize", False),
                "screenOrientation": params.get("screenOrientation"),
            }
        
        elif cdp_method == "Runtime.evaluate":
            # Firefox might have slightly different parameters
            result = {
                "expression": params.get("expression", ""),
                "objectGroup": params.get("objectGroup", "console"),
                "includeCommandLineAPI": params.get("includeCommandLineAPI", False),
                "silent": params.get("silent", False),
                "returnByValue": params.get("returnByValue", False),
                "generatePreview": params.get("generatePreview", False),
                "userGesture": params.get("userGesture", False),
                "awaitPromise": params.get("awaitPromise", False),
                "contextId": params.get("contextId"),
                "throwOnSideEffect": params.get("throwOnSideEffect"),
                "timeout": params.get("timeout"),
            }
            return {k: v for k, v in result.items() if v is not None}
        
        return params
    
    async def set_viewport(self, width: int, height: int) -> None:
        """Set viewport size for all pages."""
        self.viewport = {"width": width, "height": height}
        
        # Apply to all existing pages
        for page in self.pages.values():
            if isinstance(page, FirefoxPage):
                await page.set_viewport(width, height)
    
    async def get_cookies(self) -> List[Dict[str, Any]]:
        """Get all cookies."""
        pages = await self.pages()
        if not pages:
            return []
        
        page = pages[0]
        return await page.get_cookies()
    
    async def set_cookie(self, cookie: Dict[str, Any]) -> None:
        """Set a cookie."""
        pages = await self.pages()
        if not pages:
            raise BrowserEngineError("No pages available")
        
        page = pages[0]
        await page.set_cookie(cookie)
    
    async def clear_cookies(self) -> None:
        """Clear all cookies."""
        pages = await self.pages()
        if not pages:
            return
        
        page = pages[0]
        await page.clear_cookies()
    
    async def get_cache_storage(self) -> Dict[str, Any]:
        """Get cache storage information."""
        # Firefox doesn't expose cache storage via DevTools Protocol in the same way
        # This is a simplified implementation
        return {"note": "Cache storage API not fully supported in Firefox adapter"}
    
    async def clear_cache(self) -> None:
        """Clear browser cache."""
        try:
            # Use CDP command translation
            await self.execute_cdp_command("Network.clearBrowserCache")
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")


class FirefoxPage(Page):
    """
    Firefox page implementation using Firefox DevTools Protocol.
    """
    
    def __init__(
        self,
        adapter: FirefoxAdapter,
        target_id: str,
        websocket_url: str,
        viewport: Dict[str, int],
    ):
        """
        Initialize Firefox page.
        
        Args:
            adapter: Parent FirefoxAdapter instance
            target_id: Target ID for this page
            websocket_url: WebSocket URL for communication
            viewport: Viewport dimensions
        """
        super().__init__()
        self.adapter = adapter
        self.target_id = target_id
        self.websocket_url = websocket_url
        self.viewport = viewport
        self.websocket = None
        self.command_id = 0
        self.pending_commands = {}
        self.event_handlers = {}
        self._message_queue = asyncio.Queue()
        self._shutdown_event = asyncio.Event()
        self._listener_task = None
        self._dom_enabled = False
        self._network_enabled = False
        self._page_enabled = False
        self._runtime_enabled = False
        self._input_enabled = False
    
    async def connect(self) -> None:
        """Connect to the page via WebSocket."""
        try:
            self.websocket = await websockets.connect(
                self.websocket_url,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=10,
            )
            
            # Start message listener
            self._listener_task = asyncio.create_task(self._message_listener())
            
            # Enable necessary domains
            await self.enable_domains()
            
            # Set viewport
            await self.set_viewport(self.viewport["width"], self.viewport["height"])
            
            logger.debug(f"Connected to page {self.target_id}")
            
        except Exception as e:
            raise BrowserEngineError(f"Failed to connect to page: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from the page."""
        self._shutdown_event.set()
        
        # Cancel listener task
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
            self._listener_task = None
        
        # Close WebSocket
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception:
                pass
            self.websocket = None
        
        # Clear pending commands
        for future in self.pending_commands.values():
            if not future.done():
                future.cancel()
        self.pending_commands.clear()
        
        logger.debug(f"Disconnected from page {self.target_id}")
    
    async def enable_domains(self) -> None:
        """Enable necessary DevTools Protocol domains."""
        try:
            # Enable DOM domain
            await self.send_command("DOM.enable")
            self._dom_enabled = True
            
            # Enable Network domain
            await self.send_command("Network.enable")
            self._network_enabled = True
            
            # Enable Page domain
            await self.send_command("Page.enable")
            self._page_enabled = True
            
            # Enable Runtime domain
            await self.send_command("Runtime.enable")
            self._runtime_enabled = True
            
            # Enable Input domain
            await self.send_command("Input.enable")
            self._input_enabled = True
            
        except Exception as e:
            logger.warning(f"Failed to enable some domains: {e}")
    
    async def send_command(self, method: str, params: Optional[Dict] = None) -> Any:
        """Send a command to the page and wait for response."""
        if not self.websocket:
            raise BrowserEngineError("Page not connected")
        
        self.command_id += 1
        command = {
            "id": self.command_id,
            "method": method,
            "params": params or {},
        }
        
        future = asyncio.Future()
        self.pending_commands[self.command_id] = future
        
        try:
            await self.websocket.send(json.dumps(command))
            
            # Wait for response with timeout
            try:
                response = await asyncio.wait_for(future, timeout=30)
                
                # Check for error in response
                if "error" in response:
                    error = response["error"]
                    raise BrowserEngineError(
                        f"Command {method} failed: {error.get('message', 'Unknown error')}"
                    )
                
                return response.get("result", {})
                
            except asyncio.TimeoutError:
                raise BrowserEngineError(f"Command {method} timed out")
                
        except Exception as e:
            self.pending_commands.pop(self.command_id, None)
            raise BrowserEngineError(f"Failed to send command {method}: {e}")
    
    async def _message_listener(self) -> None:
        """Listen for incoming WebSocket messages."""
        while not self._shutdown_event.is_set():
            try:
                message = await asyncio.wait_for(
                    self.websocket.recv(),
                    timeout=1.0
                )
                await self._handle_message(json.loads(message))
            except asyncio.TimeoutError:
                continue
            except ConnectionClosed:
                logger.debug("WebSocket connection closed")
                break
            except Exception as e:
                logger.error(f"Error in message listener: {e}")
                break
    
    async def _handle_message(self, message: Dict) -> None:
        """Handle incoming WebSocket message."""
        # Handle command responses
        if "id" in message:
            command_id = message["id"]
            future = self.pending_commands.pop(command_id, None)
            if future and not future.done():
                future.set_result(message)
        
        # Handle events
        elif "method" in message:
            method = message["method"]
            params = message.get("params", {})
            
            # Call registered event handlers
            if method in self.event_handlers:
                for handler in self.event_handlers[method]:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(params)
                        else:
                            handler(params)
                    except Exception as e:
                        logger.error(f"Error in event handler for {method}: {e}")
    
    def on(self, event: str, handler: callable) -> None:
        """Register an event handler."""
        if event not in self.event_handlers:
            self.event_handlers[event] = []
        self.event_handlers[event].append(handler)
    
    def off(self, event: str, handler: Optional[callable] = None) -> None:
        """Unregister an event handler."""
        if event not in self.event_handlers:
            return
        
        if handler is None:
            self.event_handlers[event] = []
        else:
            self.event_handlers[event] = [
                h for h in self.event_handlers[event] if h != handler
            ]
    
    async def goto(self, url: str, wait_until: str = "load", timeout: int = 30000) -> None:
        """Navigate to a URL."""
        if not url.startswith(("http://", "https://", "file://", "about:")):
            url = f"http://{url}"
        
        # Set up load event listener
        load_future = asyncio.Future()
        
        def load_handler(params):
            if not load_future.done():
                load_future.set_result(params)
        
        self.on("Page.loadEventFired", load_handler)
        
        try:
            # Navigate
            await self.send_command("Page.navigate", {"url": url})
            
            # Wait for load event
            try:
                await asyncio.wait_for(load_future, timeout=timeout/1000)
            except asyncio.TimeoutError:
                raise BrowserEngineError(f"Navigation to {url} timed out")
            
        finally:
            self.off("Page.loadEventFired", load_handler)
    
    async def reload(self, ignore_cache: bool = False, timeout: int = 30000) -> None:
        """Reload the current page."""
        # Set up load event listener
        load_future = asyncio.Future()
        
        def load_handler(params):
            if not load_future.done():
                load_future.set_result(params)
        
        self.on("Page.loadEventFired", load_handler)
        
        try:
            # Reload
            await self.send_command("Page.reload", {"ignoreCache": ignore_cache})
            
            # Wait for load event
            try:
                await asyncio.wait_for(load_future, timeout=timeout/1000)
            except asyncio.TimeoutError:
                raise BrowserEngineError("Reload timed out")
            
        finally:
            self.off("Page.loadEventFired", load_handler)
    
    async def go_back(self, timeout: int = 30000) -> None:
        """Navigate back in history."""
        # Get navigation history
        history = await self.send_command("Page.getNavigationHistory")
        entries = history.get("entries", [])
        current_index = history.get("currentIndex", 0)
        
        if current_index > 0:
            entry_id = entries[current_index - 1].get("id")
            if entry_id:
                # Set up load event listener
                load_future = asyncio.Future()
                
                def load_handler(params):
                    if not load_future.done():
                        load_future.set_result(params)
                
                self.on("Page.loadEventFired", load_handler)
                
                try:
                    # Navigate back
                    await self.send_command("Page.navigateToHistoryEntry", {"entryId": entry_id})
                    
                    # Wait for load event
                    try:
                        await asyncio.wait_for(load_future, timeout=timeout/1000)
                    except asyncio.TimeoutError:
                        raise BrowserEngineError("Back navigation timed out")
                    
                finally:
                    self.off("Page.loadEventFired", load_handler)
    
    async def go_forward(self, timeout: int = 30000) -> None:
        """Navigate forward in history."""
        # Get navigation history
        history = await self.send_command("Page.getNavigationHistory")
        entries = history.get("entries", [])
        current_index = history.get("currentIndex", 0)
        
        if current_index < len(entries) - 1:
            entry_id = entries[current_index + 1].get("id")
            if entry_id:
                # Set up load event listener
                load_future = asyncio.Future()
                
                def load_handler(params):
                    if not load_future.done():
                        load_future.set_result(params)
                
                self.on("Page.loadEventFired", load_handler)
                
                try:
                    # Navigate forward
                    await self.send_command("Page.navigateToHistoryEntry", {"entryId": entry_id})
                    
                    # Wait for load event
                    try:
                        await asyncio.wait_for(load_future, timeout=timeout/1000)
                    except asyncio.TimeoutError:
                        raise BrowserEngineError("Forward navigation timed out")
                    
                finally:
                    self.off("Page.loadEventFired", load_handler)
    
    async def get_url(self) -> str:
        """Get current page URL."""
        result = await self.send_command("Runtime.evaluate", {
            "expression": "window.location.href",
            "returnByValue": True,
        })
        return result.get("result", {}).get("value", "")
    
    async def get_title(self) -> str:
        """Get current page title."""
        result = await self.send_command("Runtime.evaluate", {
            "expression": "document.title",
            "returnByValue": True,
        })
        return result.get("result", {}).get("value", "")
    
    async def set_viewport(self, width: int, height: int) -> None:
        """Set viewport size."""
        self.viewport = {"width": width, "height": height}
        await self.send_command("Emulation.setDeviceMetricsOverride", {
            "width": width,
            "height": height,
            "deviceScaleFactor": 1,
            "mobile": False,
        })
    
    async def get_content(self) -> str:
        """Get page HTML content."""
        result = await self.send_command("Runtime.evaluate", {
            "expression": "document.documentElement.outerHTML",
            "returnByValue": True,
        })
        return result.get("result", {}).get("value", "")
    
    async def set_content(self, html: str) -> None:
        """Set page HTML content."""
        await self.send_command("Runtime.evaluate", {
            "expression": f"document.open(); document.write({json.dumps(html)}); document.close();",
        })
    
    async def query_selector(self, selector: str) -> Optional[Element]:
        """Find element by CSS selector."""
        # Get document root
        doc = await self.send_command("DOM.getDocument")
        root_node_id = doc.get("root", {}).get("nodeId")
        
        # Query selector
        result = await self.send_command("DOM.querySelector", {
            "nodeId": root_node_id,
            "selector": selector,
        })
        
        node_id = result.get("nodeId")
        if node_id and node_id != 0:
            return FirefoxElement(self, node_id, selector)
        
        return None
    
    async def query_selector_all(self, selector: str) -> List[Element]:
        """Find all elements by CSS selector."""
        # Get document root
        doc = await self.send_command("DOM.getDocument")
        root_node_id = doc.get("root", {}).get("nodeId")
        
        # Query selector all
        result = await self.send_command("DOM.querySelectorAll", {
            "nodeId": root_node_id,
            "selector": selector,
        })
        
        node_ids = result.get("nodeIds", [])
        elements = []
        for node_id in node_ids:
            if node_id and node_id != 0:
                elements.append(FirefoxElement(self, node_id, selector))
        
        return elements
    
    async def evaluate(self, expression: str, await_promise: bool = False) -> Any:
        """Evaluate JavaScript expression."""
        result = await self.send_command("Runtime.evaluate", {
            "expression": expression,
            "awaitPromise": await_promise,
            "returnByValue": True,
        })
        
        if "exceptionDetails" in result:
            exception = result["exceptionDetails"]
            raise BrowserEngineError(
                f"JavaScript evaluation failed: {exception.get('text', 'Unknown error')}"
            )
        
        return result.get("result", {}).get("value")
    
    async def get_cookies(self) -> List[Dict[str, Any]]:
        """Get all cookies for this page."""
        result = await self.send_command("Network.getCookies")
        return result.get("cookies", [])
    
    async def set_cookie(self, cookie: Dict[str, Any]) -> None:
        """Set a cookie."""
        await self.send_command("Network.setCookie", cookie)
    
    async def clear_cookies(self) -> None:
        """Clear all cookies."""
        await self.send_command("Network.clearBrowserCookies")
    
    async def screenshot(
        self,
        format: str = "png",
        quality: int = 80,
        clip: Optional[Dict] = None,
    ) -> bytes:
        """Take a screenshot."""
        params = {"format": format}
        
        if format == "jpeg" or format == "webp":
            params["quality"] = quality
        
        if clip:
            params["clip"] = {
                "x": clip.get("x", 0),
                "y": clip.get("y", 0),
                "width": clip.get("width", self.viewport["width"]),
                "height": clip.get("height", self.viewport["height"]),
                "scale": clip.get("scale", 1),
            }
        
        result = await self.send_command("Page.captureScreenshot", params)
        
        import base64
        return base64.b64decode(result.get("data", ""))
    
    async def pdf(self, **kwargs) -> bytes:
        """Generate PDF of the page."""
        # Firefox doesn't support PDF generation via DevTools Protocol
        raise BrowserEngineError("PDF generation not supported in Firefox adapter")
    
    async def wait_for_selector(
        self,
        selector: str,
        timeout: int = 30000,
        visible: bool = False,
        hidden: bool = False,
    ) -> Optional[Element]:
        """Wait for element matching selector."""
        start_time = time.time()
        
        while (time.time() - start_time) * 1000 < timeout:
            element = await self.query_selector(selector)
            
            if element:
                if visible:
                    is_visible = await element.is_visible()
                    if not is_visible:
                        await asyncio.sleep(0.1)
                        continue
                
                if hidden:
                    is_visible = await element.is_visible()
                    if is_visible:
                        await asyncio.sleep(0.1)
                        continue
                
                return element
            
            await asyncio.sleep(0.1)
        
        return None
    
    async def wait_for_function(
        self,
        function: str,
        timeout: int = 30000,
        polling: Union[str, int] = "raf",
    ) -> Any:
        """Wait for function to return truthy value."""
        start_time = time.time()
        
        while (time.time() - start_time) * 1000 < timeout:
            try:
                result = await self.evaluate(function)
                if result:
                    return result
            except Exception:
                pass
            
            if polling == "raf":
                await asyncio.sleep(0.016)  # ~60fps
            elif isinstance(polling, int):
                await asyncio.sleep(polling / 1000)
            else:
                await asyncio.sleep(0.1)
        
        raise BrowserEngineError(f"Function did not return truthy value within {timeout}ms")
    
    async def get_element_by_id(self, element_id: str) -> Optional[Element]:
        """Get element by ID."""
        return await self.query_selector(f"#{element_id}")
    
    async def get_elements_by_tag_name(self, tag_name: str) -> List[Element]:
        """Get elements by tag name."""
        return await self.query_selector_all(tag_name)
    
    async def get_elements_by_class_name(self, class_name: str) -> List[Element]:
        """Get elements by class name."""
        return await self.query_selector_all(f".{class_name}")
    
    async def get_elements_by_name(self, name: str) -> List[Element]:
        """Get elements by name attribute."""
        return await self.query_selector_all(f'[name="{name}"]')
    
    async def execute_script(self, script: str, *args) -> Any:
        """Execute JavaScript script with arguments."""
        # Convert arguments to JSON and inject into script
        args_json = json.dumps(args)
        full_script = f"""
        (function() {{
            const args = {args_json};
            {script}
        }})();
        """
        
        return await self.evaluate(full_script)
    
    async def add_script_tag(
        self,
        url: Optional[str] = None,
        path: Optional[str] = None,
        content: Optional[str] = None,
        type: str = "text/javascript",
    ) -> None:
        """Add script tag to page."""
        if url:
            script = f"""
            const script = document.createElement('script');
            script.src = {json.dumps(url)};
            script.type = {json.dumps(type)};
            document.head.appendChild(script);
            """
        elif path:
            with open(path, "r") as f:
                file_content = f.read()
            script = f"""
            const script = document.createElement('script');
            script.textContent = {json.dumps(file_content)};
            script.type = {json.dumps(type)};
            document.head.appendChild(script);
            """
        elif content:
            script = f"""
            const script = document.createElement('script');
            script.textContent = {json.dumps(content)};
            script.type = {json.dumps(type)};
            document.head.appendChild(script);
            """
        else:
            raise ValueError("Either url, path, or content must be provided")
        
        await self.evaluate(script)
    
    async def add_style_tag(
        self,
        url: Optional[str] = None,
        path: Optional[str] = None,
        content: Optional[str] = None,
    ) -> None:
        """Add style tag to page."""
        if url:
            script = f"""
            const link = document.createElement('link');
            link.rel = 'stylesheet';
            link.href = {json.dumps(url)};
            document.head.appendChild(link);
            """
        elif path:
            with open(path, "r") as f:
                file_content = f.read()
            script = f"""
            const style = document.createElement('style');
            style.textContent = {json.dumps(file_content)};
            document.head.appendChild(style);
            """
        elif content:
            script = f"""
            const style = document.createElement('style');
            style.textContent = {json.dumps(content)};
            document.head.appendChild(style);
            """
        else:
            raise ValueError("Either url, path, or content must be provided")
        
        await self.evaluate(script)
    
    async def get_mouse(self) -> Mouse:
        """Get mouse controller for this page."""
        return FirefoxMouse(self)
    
    async def close(self) -> None:
        """Close the page."""
        await self.adapter.close_page(self)


class FirefoxElement(Element):
    """Firefox element implementation."""
    
    def __init__(self, page: FirefoxPage, node_id: int, selector: str):
        """
        Initialize Firefox element.
        
        Args:
            page: Parent FirefoxPage instance
            node_id: DOM node ID
            selector: CSS selector used to find this element
        """
        super().__init__()
        self.page = page
        self.node_id = node_id
        self.selector = selector
        self._remote_object_id = None
    
    async def _get_remote_object_id(self) -> str:
        """Get remote object ID for this element."""
        if not self._remote_object_id:
            result = await self.page.send_command("DOM.resolveNode", {
                "nodeId": self.node_id,
            })
            self._remote_object_id = result.get("object", {}).get("objectId")
        
        return self._remote_object_id
    
    async def get_attribute(self, name: str) -> Optional[str]:
        """Get element attribute value."""
        result = await self.page.send_command("DOM.getAttributes", {
            "nodeId": self.node_id,
        })
        
        attributes = result.get("attributes", [])
        for i in range(0, len(attributes), 2):
            if attributes[i] == name:
                return attributes[i + 1]
        
        return None
    
    async def set_attribute(self, name: str, value: str) -> None:
        """Set element attribute value."""
        await self.page.send_command("DOM.setAttributeValue", {
            "nodeId": self.node_id,
            "name": name,
            "value": value,
        })
    
    async def remove_attribute(self, name: str) -> None:
        """Remove element attribute."""
        await self.page.send_command("DOM.removeAttribute", {
            "nodeId": self.node_id,
            "name": name,
        })
    
    async def get_text(self) -> str:
        """Get element text content."""
        result = await self.page.send_command("Runtime.evaluate", {
            "expression": f"""
                (function() {{
                    const node = document.querySelector({json.dumps(self.selector)});
                    return node ? node.textContent : '';
                }})();
            """,
            "returnByValue": True,
        })
        return result.get("result", {}).get("value", "")
    
    async def set_text(self, text: str) -> None:
        """Set element text content."""
        await self.page.send_command("Runtime.evaluate", {
            "expression": f"""
                (function() {{
                    const node = document.querySelector({json.dumps(self.selector)});
                    if (node) node.textContent = {json.dumps(text)};
                }})();
            """,
        })
    
    async def get_inner_html(self) -> str:
        """Get element inner HTML."""
        result = await self.page.send_command("DOM.getOuterHTML", {
            "nodeId": self.node_id,
        })
        return result.get("outerHTML", "")
    
    async def set_inner_html(self, html: str) -> None:
        """Set element inner HTML."""
        await self.page.send_command("DOM.setOuterHTML", {
            "nodeId": self.node_id,
            "outerHTML": html,
        })
    
    async def get_value(self) -> str:
        """Get element value (for input elements)."""
        result = await self.page.send_command("Runtime.evaluate", {
            "expression": f"""
                (function() {{
                    const node = document.querySelector({json.dumps(self.selector)});
                    return node ? (node.value || '') : '';
                }})();
            """,
            "returnByValue": True,
        })
        return result.get("result", {}).get("value", "")
    
    async def set_value(self, value: str) -> None:
        """Set element value (for input elements)."""
        await self.page.send_command("Runtime.evaluate", {
            "expression": f"""
                (function() {{
                    const node = document.querySelector({json.dumps(self.selector)});
                    if (node) {{
                        node.value = {json.dumps(value)};
                        node.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        node.dispatchEvent(new Event('change', {{ bubbles: true }}));
                    }}
                }})();
            """,
        })
    
    async def click(self) -> None:
        """Click the element."""
        # Scroll element into view
        await self.scroll_into_view()
        
        # Get element center coordinates
        box = await self.get_bounding_box()
        if not box:
            raise BrowserEngineError("Could not get element bounding box")
        
        x = box["x"] + box["width"] / 2
        y = box["y"] + box["height"] / 2
        
        # Dispatch mouse events
        await self.page.send_command("Input.dispatchMouseEvent", {
            "type": "mousePressed",
            "x": x,
            "y": y,
            "button": "left",
            "clickCount": 1,
        })
        
        await self.page.send_command("Input.dispatchMouseEvent", {
            "type": "mouseReleased",
            "x": x,
            "y": y,
            "button": "left",
            "clickCount": 1,
        })
    
    async def double_click(self) -> None:
        """Double click the element."""
        # Scroll element into view
        await self.scroll_into_view()
        
        # Get element center coordinates
        box = await self.get_bounding_box()
        if not box:
            raise BrowserEngineError("Could not get element bounding box")
        
        x = box["x"] + box["width"] / 2
        y = box["y"] + box["height"] / 2
        
        # Dispatch mouse events with double click
        await self.page.send_command("Input.dispatchMouseEvent", {
            "type": "mousePressed",
            "x": x,
            "y": y,
            "button": "left",
            "clickCount": 2,
        })
        
        await self.page.send_command("Input.dispatchMouseEvent", {
            "type": "mouseReleased",
            "x": x,
            "y": y,
            "button": "left",
            "clickCount": 2,
        })
    
    async def right_click(self) -> None:
        """Right click the element."""
        # Scroll element into view
        await self.scroll_into_view()
        
        # Get element center coordinates
        box = await self.get_bounding_box()
        if not box:
            raise BrowserEngineError("Could not get element bounding box")
        
        x = box["x"] + box["width"] / 2
        y = box["y"] + box["height"] / 2
        
        # Dispatch mouse events
        await self.page.send_command("Input.dispatchMouseEvent", {
            "type": "mousePressed",
            "x": x,
            "y": y,
            "button": "right",
            "clickCount": 1,
        })
        
        await self.page.send_command("Input.dispatchMouseEvent", {
            "type": "mouseReleased",
            "x": x,
            "y": y,
            "button": "right",
            "clickCount": 1,
        })
    
    async def hover(self) -> None:
        """Hover over the element."""
        # Scroll element into view
        await self.scroll_into_view()
        
        # Get element center coordinates
        box = await self.get_bounding_box()
        if not box:
            raise BrowserEngineError("Could not get element bounding box")
        
        x = box["x"] + box["width"] / 2
        y = box["y"] + box["height"] / 2
        
        # Dispatch mouse move event
        await self.page.send_command("Input.dispatchMouseEvent", {
            "type": "mouseMoved",
            "x": x,
            "y": y,
        })
    
    async def focus(self) -> None:
        """Focus the element."""
        await self.page.send_command("DOM.focus", {
            "nodeId": self.node_id,
        })
    
    async def blur(self) -> None:
        """Remove focus from the element."""
        await self.page.send_command("Runtime.evaluate", {
            "expression": f"""
                (function() {{
                    const node = document.querySelector({json.dumps(self.selector)});
                    if (node) node.blur();
                }})();
            """,
        })
    
    async def type_text(self, text: str, delay: int = 0) -> None:
        """Type text into the element."""
        # Focus the element first
        await self.focus()
        
        # Type each character with optional delay
        for char in text:
            await self.page.send_command("Input.dispatchKeyEvent", {
                "type": "keyDown",
                "text": char,
            })
            
            await self.page.send_command("Input.dispatchKeyEvent", {
                "type": "keyUp",
                "text": char,
            })
            
            if delay > 0:
                await asyncio.sleep(delay / 1000)
    
    async def press_key(self, key: str) -> None:
        """Press a key."""
        # Map common key names to Firefox key codes
        key_mapping = {
            "Enter": "\r",
            "Tab": "\t",
            "Backspace": "\b",
            "Delete": "\x7f",
            "ArrowLeft": "ArrowLeft",
            "ArrowRight": "ArrowRight",
            "ArrowUp": "ArrowUp",
            "ArrowDown": "ArrowDown",
            "Home": "Home",
            "End": "End",
            "PageUp": "PageUp",
            "PageDown": "PageDown",
            "Escape": "Escape",
            "F1": "F1",
            "F2": "F2",
            "F3": "F3",
            "F4": "F4",
            "F5": "F5",
            "F6": "F6",
            "F7": "F7",
            "F8": "F8",
            "F9": "F9",
            "F10": "F10",
            "F11": "F11",
            "F12": "F12",
        }
        
        key_text = key_mapping.get(key, key)
        
        await self.page.send_command("Input.dispatchKeyEvent", {
            "type": "keyDown",
            "key": key_text,
        })
        
        await self.page.send_command("Input.dispatchKeyEvent", {
            "type": "keyUp",
            "key": key_text,
        })
    
    async def select_option(self, *values) -> None:
        """Select option(s) in a select element."""
        values_json = json.dumps(list(values))
        await self.page.send_command("Runtime.evaluate", {
            "expression": f"""
                (function() {{
                    const select = document.querySelector({json.dumps(self.selector)});
                    if (!select) return;
                    
                    const values = {values_json};
                    for (let i = 0; i < select.options.length; i++) {{
                        const option = select.options[i];
                        option.selected = values.includes(option.value) || 
                                          values.includes(option.textContent);
                    }}
                    
                    select.dispatchEvent(new Event('change', {{ bubbles: true }}));
                }})();
            """,
        })
    
    async def is_selected(self) -> bool:
        """Check if element is selected."""
        result = await self.page.send_command("Runtime.evaluate", {
            "expression": f"""
                (function() {{
                    const node = document.querySelector({json.dumps(self.selector)});
                    return node ? (node.selected || node.checked || false) : false;
                }})();
            """,
            "returnByValue": True,
        })
        return result.get("result", {}).get("value", False)
    
    async def is_checked(self) -> bool:
        """Check if checkbox/radio is checked."""
        return await self.is_selected()
    
    async def check(self) -> None:
        """Check a checkbox/radio element."""
        await self.page.send_command("Runtime.evaluate", {
            "expression": f"""
                (function() {{
                    const node = document.querySelector({json.dumps(self.selector)});
                    if (node && node.type === 'checkbox' || node.type === 'radio') {{
                        if (!node.checked) {{
                            node.checked = true;
                            node.dispatchEvent(new Event('change', {{ bubbles: true }}));
                        }}
                    }}
                }})();
            """,
        })
    
    async def uncheck(self) -> None:
        """Uncheck a checkbox element."""
        await self.page.send_command("Runtime.evaluate", {
            "expression": f"""
                (function() {{
                    const node = document.querySelector({json.dumps(self.selector)});
                    if (node && node.type === 'checkbox') {{
                        if (node.checked) {{
                            node.checked = false;
                            node.dispatchEvent(new Event('change', {{ bubbles: true }}));
                        }}
                    }}
                }})();
            """,
        })
    
    async def get_bounding_box(self) -> Optional[Dict[str, float]]:
        """Get element bounding box."""
        try:
            result = await self.page.send_command("DOM.getBoxModel", {
                "nodeId": self.node_id,
            })
            
            model = result.get("model", {})
            content = model.get("content", [])
            
            if len(content) >= 8:
                # content contains [x1, y1, x2, y2, x3, y3, x4, y4]
                x_coords = [content[i] for i in range(0, 8, 2)]
                y_coords = [content[i] for i in range(1, 8, 2)]
                
                x = min(x_coords)
                y = min(y_coords)
                width = max(x_coords) - x
                height = max(y_coords) - y
                
                return {"x": x, "y": y, "width": width, "height": height}
        except Exception:
            pass
        
        return None
    
    async def scroll_into_view(self) -> None:
        """Scroll element into view."""
        await self.page.send_command("DOM.scrollIntoViewIfNeeded", {
            "nodeId": self.node_id,
        })
    
    async def is_visible(self) -> bool:
        """Check if element is visible."""
        result = await self.page.send_command("Runtime.evaluate", {
            "expression": f"""
                (function() {{
                    const node = document.querySelector({json.dumps(self.selector)});
                    if (!node) return false;
                    
                    const style = window.getComputedStyle(node);
                    return style.display !== 'none' && 
                           style.visibility !== 'hidden' && 
                           style.opacity !== '0' &&
                           node.offsetWidth > 0 &&
                           node.offsetHeight > 0;
                }})();
            """,
            "returnByValue": True,
        })
        return result.get("result", {}).get("value", False)
    
    async def is_hidden(self) -> bool:
        """Check if element is hidden."""
        return not await self.is_visible()
    
    async def is_enabled(self) -> bool:
        """Check if element is enabled."""
        result = await self.page.send_command("Runtime.evaluate", {
            "expression": f"""
                (function() {{
                    const node = document.querySelector({json.dumps(self.selector)});
                    return node ? (!node.disabled) : false;
                }})();
            """,
            "returnByValue": True,
        })
        return result.get("result", {}).get("value", False)
    
    async def is_disabled(self) -> bool:
        """Check if element is disabled."""
        return not await self.is_enabled()
    
    async def get_element_by_id(self, element_id: str) -> Optional[Element]:
        """Get child element by ID."""
        selector = f"{self.selector} #{element_id}"
        return await self.page.query_selector(selector)
    
    async def get_elements_by_tag_name(self, tag_name: str) -> List[Element]:
        """Get child elements by tag name."""
        selector = f"{self.selector} {tag_name}"
        return await self.page.query_selector_all(selector)
    
    async def get_elements_by_class_name(self, class_name: str) -> List[Element]:
        """Get child elements by class name."""
        selector = f"{self.selector} .{class_name}"
        return await self.page.query_selector_all(selector)
    
    async def get_parent(self) -> Optional[Element]:
        """Get parent element."""
        result = await self.page.send_command("Runtime.evaluate", {
            "expression": f"""
                (function() {{
                    const node = document.querySelector({json.dumps(self.selector)});
                    if (!node || !node.parentElement) return null;
                    
                    // Generate a unique selector for the parent
                    const parent = node.parentElement;
                    const path = [];
                    let current = parent;
                    
                    while (current && current.nodeType === Node.ELEMENT_NODE) {{
                        let selector = current.tagName.toLowerCase();
                        if (current.id) {{
                            selector = '#' + current.id;
                            path.unshift(selector);
                            break;
                        }} else if (current.className && typeof current.className === 'string') {{
                            selector += '.' + current.className.trim().replace(/\\s+/g, '.');
                        }}
                        
                        const siblings = Array.from(current.parentNode.children).filter(
                            e => e.tagName === current.tagName
                        );
                        if (siblings.length > 1) {{
                            const index = siblings.indexOf(current) + 1;
                            selector += ':nth-child(' + index + ')';
                        }}
                        
                        path.unshift(selector);
                        current = current.parentElement;
                    }}
                    
                    return path.join(' > ');
                }})();
            """,
            "returnByValue": True,
        })
        
        parent_selector = result.get("result", {}).get("value")
        if parent_selector:
            return await self.page.query_selector(parent_selector)
        
        return None
    
    async def get_children(self) -> List[Element]:
        """Get child elements."""
        selector = f"{self.selector} > *"
        return await self.page.query_selector_all(selector)
    
    async def get_sibling_next(self) -> Optional[Element]:
        """Get next sibling element."""
        result = await self.page.send_command("Runtime.evaluate", {
            "expression": f"""
                (function() {{
                    const node = document.querySelector({json.dumps(self.selector)});
                    if (!node) return null;
                    
                    const next = node.nextElementSibling;
                    if (!next) return null;
                    
                    // Generate a unique selector for the sibling
                    let selector = next.tagName.toLowerCase();
                    if (next.id) {{
                        selector = '#' + next.id;
                    }} else if (next.className && typeof next.className === 'string') {{
                        selector += '.' + next.className.trim().replace(/\\s+/g, '.');
                    }}
                    
                    const siblings = Array.from(next.parentNode.children).filter(
                        e => e.tagName === next.tagName
                    );
                    if (siblings.length > 1) {{
                        const index = siblings.indexOf(next) + 1;
                        selector += ':nth-child(' + index + ')';
                    }}
                    
                    return selector;
                }})();
            """,
            "returnByValue": True,
        })
        
        sibling_selector = result.get("result", {}).get("value")
        if sibling_selector:
            return await self.page.query_selector(sibling_selector)
        
        return None
    
    async def get_sibling_previous(self) -> Optional[Element]:
        """Get previous sibling element."""
        result = await self.page.send_command("Runtime.evaluate", {
            "expression": f"""
                (function() {{
                    const node = document.querySelector({json.dumps(self.selector)});
                    if (!node) return null;
                    
                    const prev = node.previousElementSibling;
                    if (!prev) return null;
                    
                    // Generate a unique selector for the sibling
                    let selector = prev.tagName.toLowerCase();
                    if (prev.id) {{
                        selector = '#' + prev.id;
                    }} else if (prev.className && typeof prev.className === 'string') {{
                        selector += '.' + prev.className.trim().replace(/\\s+/g, '.');
                    }}
                    
                    const siblings = Array.from(prev.parentNode.children).filter(
                        e => e.tagName === prev.tagName
                    );
                    if (siblings.length > 1) {{
                        const index = siblings.indexOf(prev) + 1;
                        selector += ':nth-child(' + index + ')';
                    }}
                    
                    return selector;
                }})();
            """,
            "returnByValue": True,
        })
        
        sibling_selector = result.get("result", {}).get("value")
        if sibling_selector:
            return await self.page.query_selector(sibling_selector)
        
        return None
    
    async def screenshot(self, **kwargs) -> bytes:
        """Take screenshot of this element."""
        box = await self.get_bounding_box()
        if not box:
            raise BrowserEngineError("Could not get element bounding box for screenshot")
        
        return await self.page.screenshot(
            clip={
                "x": box["x"],
                "y": box["y"],
                "width": box["width"],
                "height": box["height"],
            },
            **kwargs,
        )


class FirefoxMouse(Mouse):
    """Firefox mouse implementation."""
    
    def __init__(self, page: FirefoxPage):
        """
        Initialize Firefox mouse controller.
        
        Args:
            page: Parent FirefoxPage instance
        """
        super().__init__()
        self.page = page
    
    async def move(self, x: float, y: float, steps: int = 1) -> None:
        """Move mouse to coordinates."""
        if steps > 1:
            # Get current position (approximate)
            current_x, current_y = 0, 0
            
            # Calculate step increments
            dx = (x - current_x) / steps
            dy = (y - current_y) / steps
            
            for i in range(steps):
                step_x = current_x + dx * (i + 1)
                step_y = current_y + dy * (i + 1)
                
                await self.page.send_command("Input.dispatchMouseEvent", {
                    "type": "mouseMoved",
                    "x": step_x,
                    "y": step_y,
                })
                
                await asyncio.sleep(0.01)  # Small delay between steps
        else:
            await self.page.send_command("Input.dispatchMouseEvent", {
                "type": "mouseMoved",
                "x": x,
                "y": y,
            })
    
    async def down(self, button: str = "left", click_count: int = 1) -> None:
        """Press mouse button."""
        await self.page.send_command("Input.dispatchMouseEvent", {
            "type": "mousePressed",
            "x": 0,  # Will be overridden by move
            "y": 0,  # Will be overridden by move
            "button": button,
            "clickCount": click_count,
        })
    
    async def up(self, button: str = "left", click_count: int = 1) -> None:
        """Release mouse button."""
        await self.page.send_command("Input.dispatchMouseEvent", {
            "type": "mouseReleased",
            "x": 0,  # Will be overridden by move
            "y": 0,  # Will be overridden by move
            "button": button,
            "clickCount": click_count,
        })
    
    async def click(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        button: str = "left",
        click_count: int = 1,
        delay: int = 0,
    ) -> None:
        """Click at coordinates."""
        if x is not None and y is not None:
            await self.move(x, y)
        
        await self.down(button, click_count)
        
        if delay > 0:
            await asyncio.sleep(delay / 1000)
        
        await self.up(button, click_count)
    
    async def double_click(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        button: str = "left",
    ) -> None:
        """Double click at coordinates."""
        await self.click(x, y, button, click_count=2)
    
    async def triple_click(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        button: str = "left",
    ) -> None:
        """Triple click at coordinates."""
        await self.click(x, y, button, click_count=3)
    
    async def wheel(self, delta_x: float = 0, delta_y: float = 0) -> None:
        """Scroll mouse wheel."""
        await self.page.send_command("Input.dispatchMouseEvent", {
            "type": "mouseWheel",
            "x": 0,
            "y": 0,
            "deltaX": delta_x,
            "deltaY": delta_y,
        })
    
    async def drag(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        steps: int = 1,
    ) -> None:
        """Drag from start to end coordinates."""
        await self.move(start_x, start_y)
        await self.down()
        
        if steps > 1:
            dx = (end_x - start_x) / steps
            dy = (end_y - start_y) / steps
            
            for i in range(steps):
                step_x = start_x + dx * (i + 1)
                step_y = start_y + dy * (i + 1)
                
                await self.move(step_x, step_y)
                await asyncio.sleep(0.01)
        else:
            await self.move(end_x, end_y)
        
        await self.up()
    
    async def drag_and_drop(
        self,
        source_x: float,
        source_y: float,
        target_x: float,
        target_y: float,
        steps: int = 10,
    ) -> None:
        """Drag and drop from source to target."""
        await self.drag(source_x, source_y, target_x, target_y, steps)


# Factory function for creating Firefox adapter
def create_firefox_adapter(**kwargs) -> FirefoxAdapter:
    """Create and return a Firefox adapter instance."""
    return FirefoxAdapter(**kwargs)


# Export for use in nexus package
__all__ = ["FirefoxAdapter", "FirefoxPage", "FirefoxElement", "FirefoxMouse", "create_firefox_adapter"]