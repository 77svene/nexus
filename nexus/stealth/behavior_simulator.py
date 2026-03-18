"""
nexus/stealth/behavior_simulator.py

Stealth Mode with Fingerprint Rotation — Comprehensive anti-detection capabilities
including browser fingerprint randomization, WebGL/WebRTC spoofing, and human-like
interaction patterns. Integrates with existing nexus actor modules.
"""

import asyncio
import json
import random
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from playwright.async_api import Page, BrowserContext, ElementHandle

from ..actor.mouse import Mouse
from ..actor.page import Page as ActorPage


class BehaviorProfile(Enum):
    """Different user behavior profiles for varied interaction patterns."""
    CASUAL_USER = "casual_user"
    POWER_USER = "power_user"
    MOBILE_USER = "mobile_user"
    ELDERLY_USER = "elderly_user"
    GAMER = "gamer"


@dataclass
class Fingerprint:
    """Complete browser fingerprint configuration."""
    user_agent: str
    viewport: Dict[str, int]
    screen_resolution: Dict[str, int]
    device_pixel_ratio: float
    platform: str
    webgl_vendor: str
    webgl_renderer: str
    webgl_extensions: List[str]
    webrtc_ip_handling: str
    canvas_hash: str
    audio_hash: str
    fonts: List[str]
    plugins: List[Dict[str, str]]
    languages: List[str]
    timezone: str
    hardware_concurrency: int
    device_memory: int
    max_touch_points: int
    color_depth: int
    pixel_ratio: float
    do_not_track: Optional[bool]
    cookie_enabled: bool
    local_storage: bool
    session_storage: bool
    indexed_db: bool
    web_sql: bool
    webdriver: bool = False
    chrome_app: bool = True
    permissions: Dict[str, str] = field(default_factory=dict)


@dataclass
class ProxyConfig:
    """Proxy configuration with health status."""
    server: str
    username: Optional[str] = None
    password: Optional[str] = None
    protocol: str = "http"
    country: Optional[str] = None
    city: Optional[str] = None
    latency_ms: float = 0.0
    last_checked: float = 0.0
    healthy: bool = True
    fail_count: int = 0


class FingerprintGenerator:
    """Generates realistic browser fingerprints."""
    
    # Common user agents by browser and OS
    USER_AGENTS = {
        "chrome_windows": [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        ],
        "chrome_mac": [
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        ],
        "firefox_windows": [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
        ],
        "safari_mac": [
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
        ],
    }
    
    # Common screen resolutions
    VIEWPORTS = [
        {"width": 1920, "height": 1080},
        {"width": 1366, "height": 768},
        {"width": 1440, "height": 900},
        {"width": 1536, "height": 864},
        {"width": 1280, "height": 720},
    ]
    
    # WebGL configurations
    WEBGL_VENDORS = [
        "Google Inc. (NVIDIA)",
        "Google Inc. (Intel)",
        "Google Inc. (AMD)",
        "Mozilla",
    ]
    
    WEBGL_RENDERERS = [
        "ANGLE (NVIDIA, NVIDIA GeForce RTX 3080 Direct3D11 vs_5_0 ps_5_0)",
        "ANGLE (Intel, Intel(R) UHD Graphics 630 Direct3D11 vs_5_0 ps_5_0)",
        "ANGLE (AMD, AMD Radeon RX 6800 XT Direct3D11 vs_5_0 ps_5_0)",
        "Mali-G78 MP24",
        "Apple GPU",
    ]
    
    # Common fonts
    FONTS = [
        "Arial", "Verdana", "Helvetica", "Times New Roman", "Courier New",
        "Georgia", "Palatino", "Garamond", "Comic Sans MS", "Trebuchet MS",
        "Arial Black", "Impact", "Lucida Console", "Tahoma", "Lucida Sans Unicode",
    ]
    
    # Common plugins
    PLUGINS = [
        {"name": "Chrome PDF Plugin", "filename": "internal-pdf-viewer"},
        {"name": "Chrome PDF Viewer", "filename": "mhjfbmdgcfjbbpaeojofohoefgiehjai"},
        {"name": "Native Client", "filename": "internal-nacl-plugin"},
    ]
    
    # Common languages
    LANGUAGES = [
        ["en-US", "en"],
        ["en-GB", "en"],
        ["es-ES", "es"],
        ["fr-FR", "fr"],
        ["de-DE", "de"],
    ]
    
    # Timezones
    TIMEZONES = [
        "America/New_York", "America/Los_Angeles", "America/Chicago",
        "Europe/London", "Europe/Paris", "Europe/Berlin",
        "Asia/Tokyo", "Asia/Shanghai", "Australia/Sydney",
    ]
    
    @classmethod
    def generate(cls, profile: BehaviorProfile = BehaviorProfile.CASUAL_USER) -> Fingerprint:
        """Generate a complete fingerprint based on behavior profile."""
        # Select browser type based on profile
        if profile == BehaviorProfile.MOBILE_USER:
            browser_type = random.choice(["chrome_windows", "chrome_mac"])
            viewport = {"width": 375, "height": 812}  # iPhone X
            device_pixel_ratio = 3.0
            max_touch_points = 5
        elif profile == BehaviorProfile.GAMER:
            browser_type = random.choice(["chrome_windows", "firefox_windows"])
            viewport = {"width": 2560, "height": 1440}  # 2K gaming
            device_pixel_ratio = 1.0
            max_touch_points = 0
        else:
            browser_type = random.choice(list(cls.USER_AGENTS.keys()))
            viewport = random.choice(cls.VIEWPORTS)
            device_pixel_ratio = random.choice([1.0, 1.25, 1.5, 2.0])
            max_touch_points = random.randint(0, 10)
        
        user_agent = random.choice(cls.USER_AGENTS[browser_type])
        
        # Generate WebGL hash
        webgl_vendor = random.choice(cls.WEBGL_VENDORS)
        webgl_renderer = random.choice(cls.WEBGL_RENDERERS)
        
        # Generate canvas and audio hashes (simulated)
        canvas_hash = str(uuid.uuid4()).replace("-", "")[:32]
        audio_hash = str(uuid.uuid4()).replace("-", "")[:32]
        
        # Select fonts (subset of available)
        num_fonts = random.randint(10, len(cls.FONTS))
        fonts = random.sample(cls.FONTS, num_fonts)
        
        # Select plugins (subset)
        num_plugins = random.randint(0, len(cls.PLUGINS))
        plugins = random.sample(cls.PLUGINS, num_plugins) if num_plugins > 0 else []
        
        # Select languages
        languages = random.choice(cls.LANGUAGES)
        
        # Hardware specs based on profile
        if profile == BehaviorProfile.GAMER:
            hardware_concurrency = random.choice([8, 12, 16, 24, 32])
            device_memory = random.choice([16, 32, 64])
        elif profile == BehaviorProfile.ELDERLY_USER:
            hardware_concurrency = random.choice([2, 4])
            device_memory = random.choice([4, 8])
        else:
            hardware_concurrency = random.choice([4, 8, 12, 16])
            device_memory = random.choice([8, 16, 32])
        
        return Fingerprint(
            user_agent=user_agent,
            viewport=viewport,
            screen_resolution={
                "width": viewport["width"],
                "height": viewport["height"] + random.randint(50, 150)
            },
            device_pixel_ratio=device_pixel_ratio,
            platform="Win32" if "Windows" in user_agent else "MacIntel" if "Mac" in user_agent else "Linux x86_64",
            webgl_vendor=webgl_vendor,
            webgl_renderer=webgl_renderer,
            webgl_extensions=["ANGLE_instanced_arrays", "EXT_blend_minmax", "EXT_color_buffer_half_float"],
            webrtc_ip_handling="default_public_interface_only",
            canvas_hash=canvas_hash,
            audio_hash=audio_hash,
            fonts=fonts,
            plugins=plugins,
            languages=languages,
            timezone=random.choice(cls.TIMEZONES),
            hardware_concurrency=hardware_concurrency,
            device_memory=device_memory,
            max_touch_points=max_touch_points,
            color_depth=random.choice([24, 32]),
            pixel_ratio=device_pixel_ratio,
            do_not_track=random.choice([None, True, False]),
            cookie_enabled=True,
            local_storage=True,
            session_storage=True,
            indexed_db=True,
            web_sql=False,
            webdriver=False,
            chrome_app=True,
            permissions={"notifications": "prompt", "geolocation": "prompt"}
        )
    
    @classmethod
    def to_playwright_args(cls, fingerprint: Fingerprint) -> Dict[str, Any]:
        """Convert fingerprint to Playwright browser context arguments."""
        return {
            "user_agent": fingerprint.user_agent,
            "viewport": fingerprint.viewport,
            "screen": fingerprint.screen_resolution,
            "device_scale_factor": fingerprint.device_pixel_ratio,
            "is_mobile": fingerprint.max_touch_points > 0,
            "has_touch": fingerprint.max_touch_points > 0,
            "locale": fingerprint.languages[0] if fingerprint.languages else "en-US",
            "timezone_id": fingerprint.timezone,
            "color_scheme": "light",
            "reduced_motion": "no-preference",
            "forced_colors": "none",
        }


class HumanInteractionSimulator:
    """Simulates human-like interaction patterns."""
    
    def __init__(self, profile: BehaviorProfile = BehaviorProfile.CASUAL_USER):
        self.profile = profile
        self._setup_profile_parameters()
    
    def _setup_profile_parameters(self):
        """Setup parameters based on behavior profile."""
        if self.profile == BehaviorProfile.CASUAL_USER:
            self.mouse_speed = (0.3, 0.8)  # seconds
            self.typing_speed = (0.05, 0.15)  # seconds per character
            self.typing_error_rate = 0.02
            self.scroll_speed = (0.5, 1.5)
            self.click_delay = (0.1, 0.3)
            self.hover_duration = (0.5, 2.0)
            
        elif self.profile == BehaviorProfile.POWER_USER:
            self.mouse_speed = (0.1, 0.4)
            self.typing_speed = (0.02, 0.08)
            self.typing_error_rate = 0.005
            self.scroll_speed = (0.2, 0.8)
            self.click_delay = (0.05, 0.15)
            self.hover_duration = (0.2, 0.8)
            
        elif self.profile == BehaviorProfile.MOBILE_USER:
            self.mouse_speed = (0.4, 1.0)  # Touch is slower
            self.typing_speed = (0.1, 0.3)  # Mobile typing is slower
            self.typing_error_rate = 0.05  # More errors on mobile
            self.scroll_speed = (0.8, 2.0)
            self.click_delay = (0.2, 0.5)
            self.hover_duration = (0.3, 1.0)
            
        elif self.profile == BehaviorProfile.ELDERLY_USER:
            self.mouse_speed = (0.8, 2.0)
            self.typing_speed = (0.15, 0.4)
            self.typing_error_rate = 0.08
            self.scroll_speed = (1.0, 3.0)
            self.click_delay = (0.3, 0.8)
            self.hover_duration = (1.0, 3.0)
            
        elif self.profile == BehaviorProfile.GAMER:
            self.mouse_speed = (0.05, 0.2)
            self.typing_speed = (0.01, 0.05)
            self.typing_error_rate = 0.01
            self.scroll_speed = (0.1, 0.5)
            self.click_delay = (0.02, 0.1)
            self.hover_duration = (0.1, 0.5)
    
    def generate_mouse_path(
        self, 
        start: Tuple[float, float], 
        end: Tuple[float, float],
        num_points: int = None
    ) -> List[Tuple[float, float]]:
        """Generate human-like mouse movement path using Bezier curves."""
        if num_points is None:
            # More points for longer distances
            distance = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            num_points = max(10, min(50, int(distance / 20)))
        
        # Add some randomness to control points
        control1 = (
            start[0] + (end[0] - start[0]) * 0.3 + random.uniform(-50, 50),
            start[1] + (end[1] - start[1]) * 0.3 + random.uniform(-50, 50)
        )
        control2 = (
            start[0] + (end[0] - start[0]) * 0.7 + random.uniform(-50, 50),
            start[1] + (end[1] - start[1]) * 0.7 + random.uniform(-50, 50)
        )
        
        # Generate points along cubic Bezier curve
        points = []
        for i in range(num_points):
            t = i / (num_points - 1)
            # Cubic Bezier formula
            x = (1-t)**3 * start[0] + 3*(1-t)**2*t * control1[0] + 3*(1-t)*t**2 * control2[0] + t**3 * end[0]
            y = (1-t)**3 * start[1] + 3*(1-t)**2*t * control1[1] + 3*(1-t)*t**2 * control2[1] + t**3 * end[1]
            
            # Add micro-movements (human imperfection)
            if 0 < i < num_points - 1:
                x += random.uniform(-2, 2)
                y += random.uniform(-2, 2)
            
            points.append((x, y))
        
        return points
    
    async def human_mouse_move(
        self, 
        mouse: Mouse, 
        target_x: float, 
        target_y: float,
        current_x: float = None,
        current_y: float = None
    ):
        """Perform human-like mouse movement."""
        if current_x is None or current_y is None:
            # Get current position (simplified - in real implementation would track position)
            current_x, current_y = 0, 0
        
        path = self.generate_mouse_path((current_x, current_y), (target_x, target_y))
        
        # Move along path with variable speed
        for i, (x, y) in enumerate(path):
            # Vary speed: slower at start/end, faster in middle
            progress = i / len(path)
            speed_factor = 1.0 - abs(progress - 0.5) * 1.5  # Bell curve
            delay = random.uniform(*self.mouse_speed) * speed_factor
            
            await mouse.move(x, y, steps=1)
            await asyncio.sleep(delay)
    
    async def human_type(self, page: Page, text: str, element: ElementHandle = None):
        """Type text with human-like patterns including errors and corrections."""
        if element:
            await element.click()
            await asyncio.sleep(random.uniform(0.1, 0.3))
        
        for char in text:
            # Random typing speed
            delay = random.uniform(*self.typing_speed)
            
            # Occasionally make a typo
            if random.random() < self.typing_error_rate:
                # Type wrong character
                wrong_char = random.choice("qwertyuiopasdfghjklzxcvbnm")
                await page.keyboard.press(wrong_char)
                await asyncio.sleep(delay * 2)
                
                # Realize mistake and delete
                await page.keyboard.press("Backspace")
                await asyncio.sleep(delay)
            
            # Type correct character
            await page.keyboard.press(char)
            await asyncio.sleep(delay)
            
            # Occasionally pause (thinking)
            if random.random() < 0.05:
                await asyncio.sleep(random.uniform(0.5, 2.0))
    
    async def human_scroll(self, page: Page, direction: str = "down", amount: int = None):
        """Scroll with human-like patterns."""
        if amount is None:
            amount = random.randint(100, 500)
        
        # Scroll in chunks with pauses
        chunks = random.randint(2, 5)
        chunk_size = amount / chunks
        
        for _ in range(chunks):
            if direction == "down":
                await page.mouse.wheel(0, chunk_size)
            else:
                await page.mouse.wheel(0, -chunk_size)
            
            # Pause between scroll chunks
            await asyncio.sleep(random.uniform(*self.scroll_speed))
    
    async def human_click(self, page: Page, element: ElementHandle):
        """Click with human-like delay and movement."""
        # Get element bounding box
        box = await element.bounding_box()
        if not box:
            return
        
        # Move to element with human-like path
        target_x = box["x"] + box["width"] / 2 + random.uniform(-5, 5)
        target_y = box["y"] + box["height"] / 2 + random.uniform(-5, 5)
        
        await self.human_mouse_move(page.mouse, target_x, target_y)
        
        # Hover before clicking
        await asyncio.sleep(random.uniform(*self.hover_duration))
        
        # Click with variable pressure/duration
        await page.mouse.down()
        await asyncio.sleep(random.uniform(0.05, 0.15))  # Click duration
        await page.mouse.up()
        
        # Delay after click
        await asyncio.sleep(random.uniform(*self.click_delay))


class ProxyManager:
    """Manages proxy rotation with health checks."""
    
    def __init__(self, proxies: List[ProxyConfig] = None):
        self.proxies = proxies or []
        self.current_index = 0
        self.health_check_interval = 300  # 5 minutes
        self.max_fails = 3
    
    async def check_proxy_health(self, proxy: ProxyConfig) -> bool:
        """Check if proxy is healthy by making a test request."""
        try:
            # In a real implementation, this would make a test request
            # For now, simulate with random success/failure
            await asyncio.sleep(random.uniform(0.1, 0.5))
            
            # Simulate occasional failures
            if random.random() < 0.1:  # 10% failure rate
                proxy.fail_count += 1
                if proxy.fail_count >= self.max_fails:
                    proxy.healthy = False
                return False
            
            proxy.latency_ms = random.uniform(50, 500)
            proxy.last_checked = time.time()
            proxy.fail_count = 0
            proxy.healthy = True
            return True
            
        except Exception:
            proxy.fail_count += 1
            if proxy.fail_count >= self.max_fails:
                proxy.healthy = False
            return False
    
    async def get_next_proxy(self) -> Optional[ProxyConfig]:
        """Get next healthy proxy in rotation."""
        if not self.proxies:
            return None
        
        # Check if we need to health check current proxy
        current_proxy = self.proxies[self.current_index]
        if time.time() - current_proxy.last_checked > self.health_check_interval:
            await self.check_proxy_health(current_proxy)
        
        # Find next healthy proxy
        attempts = 0
        while attempts < len(self.proxies):
            self.current_index = (self.current_index + 1) % len(self.proxies)
            proxy = self.proxies[self.current_index]
            
            if proxy.healthy:
                return proxy
            
            # Check health if not recently checked
            if time.time() - proxy.last_checked > self.health_check_interval:
                if await self.check_proxy_health(proxy):
                    return proxy
            
            attempts += 1
        
        return None  # No healthy proxies
    
    def add_proxy(self, proxy: ProxyConfig):
        """Add a new proxy to rotation."""
        self.proxies.append(proxy)
    
    def remove_unhealthy_proxies(self):
        """Remove proxies that have failed too many times."""
        self.proxies = [p for p in self.proxies if p.healthy or p.fail_count < self.max_fails]


class BehaviorSimulator:
    """
    Main stealth behavior simulator that integrates fingerprint rotation,
    human-like interactions, and proxy management.
    
    Integrates with existing nexus actor modules.
    """
    
    def __init__(
        self,
        profile: BehaviorProfile = BehaviorProfile.CASUAL_USER,
        proxies: List[ProxyConfig] = None,
        fingerprint_rotation_interval: int = 300,  # 5 minutes
    ):
        self.profile = profile
        self.fingerprint_rotation_interval = fingerprint_rotation_interval
        
        # Core components
        self.fingerprint_generator = FingerprintGenerator()
        self.interaction_simulator = HumanInteractionSimulator(profile)
        self.proxy_manager = ProxyManager(proxies)
        
        # State
        self.current_fingerprint: Optional[Fingerprint] = None
        self.fingerprint_created_at: float = 0
        self.session_id: str = str(uuid.uuid4())
        
        # Integration with existing modules
        self._actor_page: Optional[ActorPage] = None
        self._playwright_page: Optional[Page] = None
    
    async def initialize(self, context: BrowserContext, page: Page):
        """Initialize with Playwright context and page."""
        self._playwright_page = page
        
        # Generate and apply initial fingerprint
        await self.rotate_fingerprint()
        
        # Apply fingerprint to context
        await self.apply_fingerprint_to_context(context)
        
        # Apply fingerprint overrides via JavaScript
        await self.apply_fingerprint_overrides()
    
    async def rotate_fingerprint(self):
        """Rotate to a new fingerprint."""
        self.current_fingerprint = self.fingerprint_generator.generate(self.profile)
        self.fingerprint_created_at = time.time()
        
        # Log fingerprint rotation (for debugging)
        print(f"[Stealth] Rotated fingerprint: {self.current_fingerprint.user_agent[:50]}...")
    
    async def apply_fingerprint_to_context(self, context: BrowserContext):
        """Apply fingerprint settings to browser context."""
        if not self.current_fingerprint:
            return
        
        # Note: Some fingerprint properties can only be set at context creation
        # This method would typically be called before context creation
        # For existing contexts, we use JavaScript overrides
        pass
    
    async def apply_fingerprint_overrides(self):
        """Apply fingerprint overrides via JavaScript injection."""
        if not self.current_fingerprint or not self._playwright_page:
            return
        
        fingerprint = self.current_fingerprint
        
        # JavaScript to override fingerprint properties
        override_script = """
        // Override navigator properties
        Object.defineProperty(navigator, 'webdriver', {
            get: () => false,
        });
        
        Object.defineProperty(navigator, 'languages', {
            get: () => %(languages)s,
        });
        
        Object.defineProperty(navigator, 'plugins', {
            get: () => %(plugins)s,
        });
        
        Object.defineProperty(navigator, 'platform', {
            get: () => '%(platform)s',
        });
        
        Object.defineProperty(navigator, 'hardwareConcurrency', {
            get: () => %(hardware_concurrency)d,
        });
        
        Object.defineProperty(navigator, 'deviceMemory', {
            get: () => %(device_memory)d,
        });
        
        Object.defineProperty(navigator, 'maxTouchPoints', {
            get: () => %(max_touch_points)d,
        });
        
        Object.defineProperty(navigator, 'cookieEnabled', {
            get: () => %(cookie_enabled)s,
        });
        
        Object.defineProperty(navigator, 'doNotTrack', {
            get: () => %(do_not_track)s,
        });
        
        // Override screen properties
        Object.defineProperty(screen, 'width', {
            get: () => %(screen_width)d,
        });
        
        Object.defineProperty(screen, 'height', {
            get: () => %(screen_height)d,
        });
        
        Object.defineProperty(screen, 'colorDepth', {
            get: () => %(color_depth)d,
        });
        
        Object.defineProperty(screen, 'pixelDepth', {
            get: () => %(color_depth)d,
        });
        
        // Override WebGL
        const getParameter = WebGLRenderingContext.prototype.getParameter;
        WebGLRenderingContext.prototype.getParameter = function(parameter) {
            if (parameter === 37445) {
                return '%(webgl_vendor)s';
            }
            if (parameter === 37446) {
                return '%(webgl_renderer)s';
            }
            return getParameter.call(this, parameter);
        };
        
        // Override Canvas fingerprinting
        const originalToDataURL = HTMLCanvasElement.prototype.toDataURL;
        HTMLCanvasElement.prototype.toDataURL = function(type) {
            if (type === 'image/png' && this.width === 16 && this.height === 16) {
                // Likely fingerprinting attempt
                return 'data:image/png;base64,%(canvas_hash)s';
            }
            return originalToDataURL.apply(this, arguments);
        };
        
        // Override AudioContext fingerprinting
        const originalCreateOscillator = AudioContext.prototype.createOscillator;
        AudioContext.prototype.createOscillator = function() {
            const oscillator = originalCreateOscillator.call(this);
            const originalConnect = oscillator.connect;
            oscillator.connect = function() {
                // Add noise to audio fingerprint
                const analyser = arguments[0];
                if (analyser instanceof AnalyserNode) {
                    const data = new Float32Array(analyser.frequencyBinCount);
                    for (let i = 0; i < data.length; i++) {
                        data[i] = Math.random() * 0.0001;
                    }
                }
                return originalConnect.apply(this, arguments);
            };
            return oscillator;
        };
        
        // Override WebRTC
        if (window.RTCPeerConnection) {
            const originalRTC = window.RTCPeerConnection;
            window.RTCPeerConnection = function(config, constraints) {
                if (config && config.iceServers) {
                    // Filter out STUN/TURN servers that could leak IP
                    config.iceServers = config.iceServers.filter(server => {
                        return !server.urls.some(url => 
                            url.includes('stun:') || url.includes('turn:')
                        );
                    });
                }
                return new originalRTC(config, constraints);
            };
        }
        
        // Override permissions API
        const originalQuery = navigator.permissions.query;
        navigator.permissions.query = function(descriptor) {
            if (descriptor.name === 'notifications') {
                return Promise.resolve({ state: '%(notifications_permission)s' });
            }
            if (descriptor.name === 'geolocation') {
                return Promise.resolve({ state: '%(geolocation_permission)s' });
            }
            return originalQuery.call(this, descriptor);
        };
        """ % {
            'languages': json.dumps(fingerprint.languages),
            'plugins': json.dumps(fingerprint.plugins),
            'platform': fingerprint.platform,
            'hardware_concurrency': fingerprint.hardware_concurrency,
            'device_memory': fingerprint.device_memory,
            'max_touch_points': fingerprint.max_touch_points,
            'cookie_enabled': str(fingerprint.cookie_enabled).lower(),
            'do_not_track': 'null' if fingerprint.do_not_track is None else str(fingerprint.do_not_track).lower(),
            'screen_width': fingerprint.screen_resolution['width'],
            'screen_height': fingerprint.screen_resolution['height'],
            'color_depth': fingerprint.color_depth,
            'webgl_vendor': fingerprint.webgl_vendor,
            'webgl_renderer': fingerprint.webgl_renderer,
            'canvas_hash': fingerprint.canvas_hash,
            'notifications_permission': fingerprint.permissions.get('notifications', 'prompt'),
            'geolocation_permission': fingerprint.permissions.get('geolocation', 'prompt'),
        }
        
        await self._playwright_page.add_init_script(override_script)
    
    async def check_and_rotate_fingerprint(self):
        """Check if fingerprint needs rotation and rotate if necessary."""
        if not self.current_fingerprint:
            await self.rotate_fingerprint()
            return
        
        time_elapsed = time.time() - self.fingerprint_created_at
        if time_elapsed > self.fingerprint_rotation_interval:
            await self.rotate_fingerprint()
    
    async def get_proxy_for_next_request(self) -> Optional[Dict[str, str]]:
        """Get proxy configuration for next request."""
        proxy = await self.proxy_manager.get_next_proxy()
        if not proxy:
            return None
        
        proxy_config = {
            "server": proxy.server,
        }
        
        if proxy.username and proxy.password:
            proxy_config["username"] = proxy.username
            proxy_config["password"] = proxy.password
        
        return proxy_config
    
    async def human_navigate(self, url: str):
        """Navigate to URL with human-like behavior."""
        if not self._playwright_page:
            raise RuntimeError("Page not initialized")
        
        # Check if we need to rotate fingerprint
        await self.check_and_rotate_fingerprint()
        
        # Simulate human-like typing of URL
        await self._playwright_page.goto("about:blank")
        await asyncio.sleep(random.uniform(0.5, 1.5))
        
        # Type URL character by character
        await self.interaction_simulator.human_type(self._playwright_page, url)
        
        # Press enter with delay
        await asyncio.sleep(random.uniform(0.2, 0.5))
        await self._playwright_page.keyboard.press("Enter")
    
    async def human_click_element(self, selector: str):
        """Click element with human-like behavior."""
        if not self._playwright_page:
            raise RuntimeError("Page not initialized")
        
        element = await self._playwright_page.query_selector(selector)
        if not element:
            raise ValueError(f"Element not found: {selector}")
        
        await self.interaction_simulator.human_click(self._playwright_page, element)
    
    async def human_type_in_element(self, selector: str, text: str):
        """Type in element with human-like behavior."""
        if not self._playwright_page:
            raise RuntimeError("Page not initialized")
        
        element = await self._playwright_page.query_selector(selector)
        if not element:
            raise ValueError(f"Element not found: {selector}")
        
        await self.interaction_simulator.human_type(self._playwright_page, text, element)
    
    async def human_scroll_page(self, direction: str = "down", amount: int = None):
        """Scroll page with human-like behavior."""
        if not self._playwright_page:
            raise RuntimeError("Page not initialized")
        
        await self.interaction_simulator.human_scroll(self._playwright_page, direction, amount)
    
    def get_fingerprint_info(self) -> Dict[str, Any]:
        """Get current fingerprint information."""
        if not self.current_fingerprint:
            return {}
        
        return {
            "user_agent": self.current_fingerprint.user_agent,
            "viewport": self.current_fingerprint.viewport,
            "platform": self.current_fingerprint.platform,
            "webgl_vendor": self.current_fingerprint.webgl_vendor,
            "webgl_renderer": self.current_fingerprint.webgl_renderer,
            "timezone": self.current_fingerprint.timezone,
            "languages": self.current_fingerprint.languages,
            "session_id": self.session_id,
            "created_at": self.fingerprint_created_at,
        }
    
    async def close(self):
        """Clean up resources."""
        # In a real implementation, would clean up any resources
        pass


# Factory function for easy integration
async def create_stealth_session(
    context: BrowserContext,
    page: Page,
    profile: BehaviorProfile = BehaviorProfile.CASUAL_USER,
    proxies: List[ProxyConfig] = None,
) -> BehaviorSimulator:
    """
    Create and initialize a stealth behavior simulator session.
    
    Args:
        context: Playwright browser context
        page: Playwright page
        profile: Behavior profile to use
        proxies: List of proxy configurations
    
    Returns:
        Initialized BehaviorSimulator instance
    """
    simulator = BehaviorSimulator(
        profile=profile,
        proxies=proxies,
    )
    
    await simulator.initialize(context, page)
    return simulator


# Integration with existing ActorPage
class StealthActorPage:
    """
    Wrapper around existing ActorPage that adds stealth capabilities.
    """
    
    def __init__(self, actor_page: ActorPage, simulator: BehaviorSimulator):
        self._actor_page = actor_page
        self._simulator = simulator
        self._page = actor_page.page  # Access underlying Playwright page
    
    async def goto(self, url: str, **kwargs):
        """Navigate with stealth."""
        await self._simulator.human_navigate(url)
    
    async def click(self, selector: str, **kwargs):
        """Click with stealth."""
        await self._simulator.human_click_element(selector)
    
    async def type(self, selector: str, text: str, **kwargs):
        """Type with stealth."""
        await self._simulator.human_type_in_element(selector, text)
    
    async def scroll(self, direction: str = "down", amount: int = None, **kwargs):
        """Scroll with stealth."""
        await self._simulator.human_scroll_page(direction, amount)
    
    def __getattr__(self, name):
        """Delegate other methods to original ActorPage."""
        return getattr(self._actor_page, name)


# Example usage with existing modules
async def example_integration():
    """Example of how to integrate with existing nexus modules."""
    from playwright.async_api import async_playwright
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        
        # Create stealth simulator
        simulator = await create_stealth_session(
            context=context,
            page=page,
            profile=BehaviorProfile.CASUAL_USER,
        )
        
        # Use with existing ActorPage
        from nexus.actor.page import Page as ActorPage
        actor_page = ActorPage(page)
        stealth_page = StealthActorPage(actor_page, simulator)
        
        # Now use stealth_page for all operations
        await stealth_page.goto("https://example.com")
        await stealth_page.click("a")
        await stealth_page.type("input[name='q']", "search query")
        
        # Get fingerprint info
        fingerprint_info = simulator.get_fingerprint_info()
        print(f"Using fingerprint: {fingerprint_info['user_agent'][:50]}...")
        
        await simulator.close()
        await browser.close()


if __name__ == "__main__":
    asyncio.run(example_integration())