"""nexus/stealth/fingerprint_manager.py

Comprehensive anti-detection system for browser automation with fingerprint rotation,
human-like behavior simulation, and proxy management. Integrates with existing
nexus actor and agent modules.
"""

import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
import math
from datetime import datetime, timedelta

import numpy as np
from playwright.async_api import Page, BrowserContext, Browser
import aiohttp

from nexus.actor.mouse import Mouse
from nexus.actor.page import Page as ActorPage


@dataclass
class FingerprintProfile:
    """Complete browser fingerprint profile."""
    user_agent: str
    platform: str
    screen_width: int
    screen_height: int
    color_depth: int
    pixel_ratio: float
    timezone: str
    language: str
    languages: List[str]
    webgl_vendor: str
    webgl_renderer: str
    canvas_hash: str
    audio_hash: str
    fonts: List[str]
    plugins: List[Dict[str, str]]
    webrtc_ips: List[str]
    hardware_concurrency: int
    device_memory: int
    touch_support: bool
    do_not_track: Optional[str]
    cookie_enabled: bool
    session_id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:16])


@dataclass
class BehaviorProfile:
    """Human-like behavior configuration."""
    mouse_speed_range: Tuple[float, float] = (0.3, 1.2)
    mouse_acceleration_range: Tuple[float, float] = (0.8, 1.5)
    typing_speed_range: Tuple[float, float] = (0.05, 0.15)
    typing_error_rate: float = 0.02
    scroll_behavior: str = "smooth"  # "smooth", "instant", "natural"
    click_delay_range: Tuple[float, float] = (0.1, 0.4)
    page_load_wait_range: Tuple[float, float] = (1.5, 4.0)
    idle_time_range: Tuple[float, float] = (0.5, 2.0)
    focus_behavior: str = "natural"  # "natural", "aggressive", "passive"
    attention_span: float = 0.7  # 0-1, likelihood to stay focused


@dataclass
class ProxyConfig:
    """Proxy configuration with health monitoring."""
    url: str
    protocol: str  # "http", "https", "socks5"
    username: Optional[str] = None
    password: Optional[str] = None
    country: Optional[str] = None
    latency: float = 0.0
    last_check: Optional[datetime] = None
    success_rate: float = 1.0
    consecutive_failures: int = 0
    max_failures: int = 3


class FingerprintGenerator:
    """Generates realistic browser fingerprints with consistency."""
    
    # Realistic data pools
    USER_AGENTS = [
        # Chrome
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        # Firefox
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
        # Safari
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
        # Edge
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    ]
    
    PLATFORMS = ["Win32", "MacIntel", "Linux x86_64", "Linux armv8l"]
    
    SCREEN_RESOLUTIONS = [
        (1920, 1080), (1366, 768), (1536, 864), (1440, 900),
        (1280, 720), (2560, 1440), (1680, 1050), (1600, 900)
    ]
    
    TIMEZONES = [
        "America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles",
        "Europe/London", "Europe/Paris", "Europe/Berlin", "Asia/Tokyo",
        "Asia/Shanghai", "Australia/Sydney"
    ]
    
    LANGUAGES = [
        ["en-US", "en"], ["en-GB", "en"], ["fr-FR", "fr"], ["de-DE", "de"],
        ["es-ES", "es"], ["ja-JP", "ja"], ["zh-CN", "zh"], ["pt-BR", "pt"]
    ]
    
    WEBGL_VENDORS = [
        "Google Inc. (NVIDIA)", "Google Inc. (Intel)", "Google Inc. (AMD)",
        "Mozilla (NVIDIA)", "Mozilla (Intel)", "WebKit"
    ]
    
    WEBGL_RENDERERS = [
        "ANGLE (NVIDIA, NVIDIA GeForce RTX 3080 Direct3D11 vs_5_0 ps_5_0, D3D11)",
        "ANGLE (Intel, Intel(R) UHD Graphics 630 Direct3D11 vs_5_0 ps_5_0, D3D11)",
        "ANGLE (AMD, AMD Radeon RX 6800 XT Direct3D11 vs_5_0 ps_5_0, D3D11)",
        "Mali-G78 MP24", "Apple GPU"
    ]
    
    FONTS = [
        "Arial", "Helvetica", "Times New Roman", "Courier New", "Verdana",
        "Georgia", "Palatino", "Garamond", "Comic Sans MS", "Trebuchet MS",
        "Impact", "Lucida Console", "Tahoma", "Lucida Sans Unicode"
    ]
    
    @classmethod
    def generate(cls, seed: Optional[int] = None) -> FingerprintProfile:
        """Generate a consistent fingerprint profile."""
        if seed is not None:
            random.seed(seed)
        
        # Select consistent values
        user_agent = random.choice(cls.USER_AGENTS)
        platform = cls._platform_from_ua(user_agent)
        screen_w, screen_h = random.choice(cls.SCREEN_RESOLUTIONS)
        
        # Generate consistent hashes based on profile
        profile_hash = hashlib.md5(f"{user_agent}{platform}{screen_w}".encode()).hexdigest()
        
        return FingerprintProfile(
            user_agent=user_agent,
            platform=platform,
            screen_width=screen_w,
            screen_height=screen_h,
            color_depth=random.choice([24, 32]),
            pixel_ratio=random.choice([1, 1.5, 2, 2.5]),
            timezone=random.choice(cls.TIMEZONES),
            language=random.choice(cls.LANGUAGES)[0],
            languages=random.choice(cls.LANGUAGES),
            webgl_vendor=random.choice(cls.WEBGL_VENDORS),
            webgl_renderer=random.choice(cls.WEBGL_RENDERERS),
            canvas_hash=hashlib.md5(f"canvas_{profile_hash}".encode()).hexdigest()[:16],
            audio_hash=hashlib.md5(f"audio_{profile_hash}".encode()).hexdigest()[:16],
            fonts=random.sample(cls.FONTS, k=random.randint(5, 12)),
            plugins=cls._generate_plugins(user_agent),
            webrtc_ips=cls._generate_webrtc_ips(),
            hardware_concurrency=random.choice([2, 4, 6, 8, 12, 16]),
            device_memory=random.choice([2, 4, 8, 16, 32]),
            touch_support=random.choice([True, False]),
            do_not_track=random.choice(["1", None]),
            cookie_enabled=True
        )
    
    @staticmethod
    def _platform_from_ua(user_agent: str) -> str:
        """Extract platform from user agent."""
        if "Windows" in user_agent:
            return "Win32"
        elif "Macintosh" in user_agent:
            return "MacIntel"
        elif "Linux" in user_agent:
            return "Linux x86_64"
        return "Win32"
    
    @staticmethod
    def _generate_plugins(user_agent: str) -> List[Dict[str, str]]:
        """Generate realistic plugin list."""
        plugins = []
        if "Chrome" in user_agent:
            plugins.extend([
                {"name": "Chrome PDF Plugin", "filename": "internal-pdf-viewer"},
                {"name": "Chrome PDF Viewer", "filename": "mhjfbmdgcfjbbpaeojofohoefgiehjai"},
                {"name": "Native Client", "filename": "internal-nacl-plugin"}
            ])
        elif "Firefox" in user_agent:
            plugins.extend([
                {"name": "PDF Viewer", "filename": "pdf.js"},
                {"name": "OpenH264 Video Codec", "filename": "gmp-gmpopenh264"}
            ])
        return plugins
    
    @staticmethod
    def _generate_webrtc_ips() -> List[str]:
        """Generate realistic WebRTC IP addresses."""
        # Local IPs
        ips = ["192.168.1." + str(random.randint(2, 254))]
        # Sometimes include public IP
        if random.random() > 0.7:
            ips.append(f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}")
        return ips


class HumanBehaviorSimulator:
    """Simulates human-like interaction patterns."""
    
    def __init__(self, profile: BehaviorProfile):
        self.profile = profile
        self._last_action_time = time.time()
        self._attention_span = profile.attention_span
    
    async def human_mouse_move(self, mouse: Mouse, target_x: int, target_y: int, 
                               page_width: int, page_height: int) -> None:
        """Move mouse in a human-like curved path with variable speed."""
        # Get current position
        current_x, current_y = await mouse.position()
        
        # Calculate distance
        distance = math.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)
        
        # Generate control points for bezier curve
        control_points = self._generate_control_points(
            current_x, current_y, target_x, target_y, page_width, page_height
        )
        
        # Calculate number of steps based on distance and speed
        base_steps = max(10, int(distance / 50))
        speed_factor = random.uniform(*self.profile.mouse_speed_range)
        steps = int(base_steps * speed_factor)
        
        # Move along bezier curve
        for i in range(steps + 1):
            t = i / steps
            x, y = self._bezier_point(t, control_points)
            
            # Add slight randomness
            x += random.uniform(-2, 2)
            y += random.uniform(-2, 2)
            
            # Ensure within bounds
            x = max(0, min(x, page_width))
            y = max(0, min(y, page_height))
            
            await mouse.move(x, y)
            
            # Variable delay between movements
            delay = random.uniform(0.005, 0.03) * (1 + (1 - t) * 0.5)
            await asyncio.sleep(delay)
        
        # Small pause after movement
        await asyncio.sleep(random.uniform(0.05, 0.15))
    
    def _generate_control_points(self, x1: int, y1: int, x2: int, y2: int,
                                 width: int, height: int) -> List[Tuple[float, float]]:
        """Generate control points for bezier curve mouse movement."""
        # Calculate midpoint
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        # Add randomness to control points
        cp1_x = mid_x + random.uniform(-width * 0.1, width * 0.1)
        cp1_y = mid_y + random.uniform(-height * 0.1, height * 0.1)
        cp2_x = mid_x + random.uniform(-width * 0.1, width * 0.1)
        cp2_y = mid_y + random.uniform(-height * 0.1, height * 0.1)
        
        return [(x1, y1), (cp1_x, cp1_y), (cp2_x, cp2_y), (x2, y2)]
    
    @staticmethod
    def _bezier_point(t: float, points: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Calculate point on bezier curve."""
        n = len(points) - 1
        x = sum(
            points[i][0] * (math.factorial(n) / (math.factorial(i) * math.factorial(n - i))) *
            (t ** i) * ((1 - t) ** (n - i))
            for i in range(n + 1)
        )
        y = sum(
            points[i][1] * (math.factorial(n) / (math.factorial(i) * math.factorial(n - i))) *
            (t ** i) * ((1 - t) ** (n - i))
            for i in range(n + 1)
        )
        return x, y
    
    async def human_type(self, page: Page, selector: str, text: str) -> None:
        """Type text with human-like timing and occasional errors."""
        element = await page.query_selector(selector)
        if not element:
            raise ValueError(f"Element not found: {selector}")
        
        await element.focus()
        await asyncio.sleep(random.uniform(0.1, 0.3))
        
        for i, char in enumerate(text):
            # Occasionally make a typo
            if random.random() < self.profile.typing_error_rate:
                # Type wrong character
                wrong_char = random.choice("abcdefghijklmnopqrstuvwxyz")
                await element.type(wrong_char, delay=0)
                await asyncio.sleep(random.uniform(0.1, 0.3))
                # Delete it
                await page.keyboard.press("Backspace")
                await asyncio.sleep(random.uniform(0.05, 0.15))
            
            # Type correct character
            await element.type(char, delay=0)
            
            # Variable typing speed
            delay = random.uniform(*self.profile.typing_speed_range)
            
            # Occasionally pause longer (thinking)
            if random.random() < 0.1:
                delay += random.uniform(0.2, 0.8)
            
            await asyncio.sleep(delay)
        
        # Small pause after typing
        await asyncio.sleep(random.uniform(0.1, 0.3))
    
    async def human_scroll(self, page: Page, direction: str = "down", amount: int = None) -> None:
        """Scroll with human-like behavior."""
        if amount is None:
            amount = random.randint(100, 500)
        
        # Break scroll into smaller steps
        steps = random.randint(3, 8)
        step_size = amount / steps
        
        for i in range(steps):
            # Variable scroll amount per step
            current_step = step_size * random.uniform(0.8, 1.2)
            
            if direction == "down":
                await page.mouse.wheel(0, current_step)
            else:
                await page.mouse.wheel(0, -current_step)
            
            # Variable delay between scrolls
            delay = random.uniform(0.05, 0.2)
            if i == steps // 2:
                # Pause in middle (reading)
                delay += random.uniform(0.3, 0.8)
            
            await asyncio.sleep(delay)
    
    async def random_idle(self) -> None:
        """Simulate random idle time (reading, thinking)."""
        idle_time = random.uniform(*self.profile.idle_time_range)
        await asyncio.sleep(idle_time)
    
    def should_continue(self) -> bool:
        """Determine if user should continue based on attention span."""
        return random.random() < self._attention_span


class ProxyManager:
    """Manages proxy rotation with health checks."""
    
    def __init__(self, proxies: List[ProxyConfig]):
        self.proxies = proxies
        self.current_index = 0
        self._health_check_interval = 300  # 5 minutes
        self._last_health_check = {}
    
    async def get_next_proxy(self) -> ProxyConfig:
        """Get next healthy proxy in rotation."""
        # Filter healthy proxies
        healthy_proxies = [
            p for p in self.proxies 
            if p.consecutive_failures < p.max_failures
        ]
        
        if not healthy_proxies:
            raise Exception("No healthy proxies available")
        
        # Rotate through proxies
        proxy = healthy_proxies[self.current_index % len(healthy_proxies)]
        self.current_index += 1
        
        # Check if health check needed
        if (proxy.last_check is None or 
            (datetime.now() - proxy.last_check).seconds > self._health_check_interval):
            await self.check_proxy_health(proxy)
        
        return proxy
    
    async def check_proxy_health(self, proxy: ProxyConfig) -> bool:
        """Check proxy health with test request."""
        try:
            start_time = time.time()
            
            # Configure proxy for aiohttp
            proxy_url = proxy.url
            if proxy.username and proxy.password:
                # Add credentials to URL
                protocol, rest = proxy_url.split("://")
                proxy_url = f"{protocol}://{proxy.username}:{proxy.password}@{rest}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://httpbin.org/ip",
                    proxy=proxy_url,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        latency = time.time() - start_time
                        proxy.latency = latency
                        proxy.last_check = datetime.now()
                        proxy.success_rate = min(1.0, proxy.success_rate + 0.1)
                        proxy.consecutive_failures = 0
                        return True
                    else:
                        proxy.consecutive_failures += 1
                        proxy.success_rate = max(0.0, proxy.success_rate - 0.2)
                        return False
        except Exception:
            proxy.consecutive_failures += 1
            proxy.success_rate = max(0.0, proxy.success_rate - 0.3)
            return False
    
    async def rotate_proxy(self, browser: Browser) -> BrowserContext:
        """Create new browser context with rotated proxy."""
        proxy = await self.get_next_proxy()
        
        # Create context with proxy
        context = await browser.new_context(
            proxy={
                "server": proxy.url,
                "username": proxy.username,
                "password": proxy.password
            } if proxy.username else {"server": proxy.url}
        )
        
        return context


class FingerprintManager:
    """Main class for managing stealth operations and fingerprint rotation."""
    
    def __init__(self, 
                 behavior_profile: Optional[BehaviorProfile] = None,
                 proxy_configs: Optional[List[ProxyConfig]] = None,
                 rotation_interval: int = 300):
        
        self.behavior_profile = behavior_profile or BehaviorProfile()
        self.proxy_manager = ProxyManager(proxy_configs) if proxy_configs else None
        self.rotation_interval = rotation_interval
        
        self._current_fingerprint: Optional[FingerprintProfile] = None
        self._behavior_simulator = HumanBehaviorSimulator(self.behavior_profile)
        self._last_rotation = time.time()
        self._fingerprint_history: List[FingerprintProfile] = []
    
    async def apply_fingerprint(self, context: BrowserContext, 
                                fingerprint: Optional[FingerprintProfile] = None) -> FingerprintProfile:
        """Apply fingerprint to browser context."""
        if fingerprint is None:
            fingerprint = FingerprintGenerator.generate()
        
        self._current_fingerprint = fingerprint
        self._fingerprint_history.append(fingerprint)
        
        # Apply basic fingerprint settings
        await context.set_extra_http_headers({
            "Accept-Language": ",".join(fingerprint.languages),
        })
        
        # Inject fingerprint override script
        await self._inject_fingerprint_script(context, fingerprint)
        
        return fingerprint
    
    async def _inject_fingerprint_script(self, context: BrowserContext, 
                                        fingerprint: FingerprintProfile) -> None:
        """Inject JavaScript to override fingerprint properties."""
        script = """
        // Override navigator properties
        Object.defineProperty(navigator, 'userAgent', {
            get: () => '%s'
        });
        Object.defineProperty(navigator, 'platform', {
            get: () => '%s'
        });
        Object.defineProperty(navigator, 'hardwareConcurrency', {
            get: () => %d
        });
        Object.defineProperty(navigator, 'deviceMemory', {
            get: () => %d
        });
        Object.defineProperty(navigator, 'language', {
            get: () => '%s'
        });
        Object.defineProperty(navigator, 'languages', {
            get: () => %s
        });
        Object.defineProperty(navigator, 'cookieEnabled', {
            get: () => %s
        });
        Object.defineProperty(navigator, 'doNotTrack', {
            get: () => %s
        });
        
        // Override screen properties
        Object.defineProperty(screen, 'width', {
            get: () => %d
        });
        Object.defineProperty(screen, 'height', {
            get: () => %d
        });
        Object.defineProperty(screen, 'colorDepth', {
            get: () => %d
        });
        Object.defineProperty(screen, 'pixelDepth', {
            get: () => %d
        });
        
        // Override WebGL
        const getParameter = WebGLRenderingContext.prototype.getParameter;
        WebGLRenderingContext.prototype.getParameter = function(parameter) {
            if (parameter === 37445) {
                return '%s';
            }
            if (parameter === 37446) {
                return '%s';
            }
            return getParameter.call(this, parameter);
        };
        
        // Override canvas fingerprint
        const originalToDataURL = HTMLCanvasElement.prototype.toDataURL;
        HTMLCanvasElement.prototype.toDataURL = function(type, quality) {
            const context = this.getContext('2d');
            if (context) {
                // Add subtle noise to canvas
                const imageData = context.getImageData(0, 0, this.width, this.height);
                for (let i = 0; i < imageData.data.length; i += 4) {
                    imageData.data[i] += Math.random() * 2 - 1;     // R
                    imageData.data[i + 1] += Math.random() * 2 - 1; // G
                    imageData.data[i + 2] += Math.random() * 2 - 1; // B
                }
                context.putImageData(imageData, 0, 0);
            }
            return originalToDataURL.call(this, type, quality);
        };
        
        // Override audio fingerprint
        const originalCreateOscillator = AudioContext.prototype.createOscillator;
        AudioContext.prototype.createOscillator = function() {
            const oscillator = originalCreateOscillator.call(this);
            const originalConnect = oscillator.connect.bind(oscillator);
            oscillator.connect = function(destination) {
                // Add subtle noise to audio
                const gainNode = this.context.createGain();
                gainNode.gain.value = 0.999 + Math.random() * 0.002;
                originalConnect(gainNode);
                gainNode.connect(destination);
                return destination;
            };
            return oscillator;
        };
        
        // Override WebRTC
        const originalRTCPeerConnection = window.RTCPeerConnection;
        window.RTCPeerConnection = function(...args) {
            const pc = new originalRTCPeerConnection(...args);
            const originalCreateOffer = pc.createOffer.bind(pc);
            pc.createOffer = async function(options) {
                const offer = await originalCreateOffer(options);
                // Modify SDP to hide real IP
                offer.sdp = offer.sdp.replace(
                    /c=IN IP4 \\d+\\.\\d+\\.\\d+\\.\\d+/g,
                    'c=IN IP4 0.0.0.0'
                );
                return offer;
            };
            return pc;
        };
        
        // Override timezone
        const originalDateTimeFormat = Intl.DateTimeFormat;
        Intl.DateTimeFormat = function(...args) {
            if (!args[1] || !args[1].timeZone) {
                args[1] = { ...args[1], timeZone: '%s' };
            }
            return new originalDateTimeFormat(...args);
        };
        """ % (
            fingerprint.user_agent,
            fingerprint.platform,
            fingerprint.hardware_concurrency,
            fingerprint.device_memory,
            fingerprint.language,
            json.dumps(fingerprint.languages),
            "true" if fingerprint.cookie_enabled else "false",
            fingerprint.do_not_track if fingerprint.do_not_track else "null",
            fingerprint.screen_width,
            fingerprint.screen_height,
            fingerprint.color_depth,
            fingerprint.color_depth,
            fingerprint.webgl_vendor,
            fingerprint.webgl_renderer,
            fingerprint.timezone
        )
        
        await context.add_init_script(script)
    
    async def create_stealth_context(self, browser: Browser) -> Tuple[BrowserContext, FingerprintProfile]:
        """Create a new browser context with full stealth configuration."""
        # Rotate proxy if available
        if self.proxy_manager:
            context = await self.proxy_manager.rotate_proxy(browser)
        else:
            context = await browser.new_context()
        
        # Apply fingerprint
        fingerprint = await self.apply_fingerprint(context)
        
        # Set viewport
        await context.set_viewport_size({
            "width": fingerprint.screen_width,
            "height": fingerprint.screen_height
        })
        
        return context, fingerprint
    
    async def should_rotate(self) -> bool:
        """Check if fingerprint should be rotated."""
        return (time.time() - self._last_rotation) > self.rotation_interval
    
    async def rotate_fingerprint(self, browser: Browser) -> Tuple[BrowserContext, FingerprintProfile]:
        """Rotate to new fingerprint and context."""
        self._last_rotation = time.time()
        return await self.create_stealth_context(browser)
    
    def get_behavior_simulator(self) -> HumanBehaviorSimulator:
        """Get the behavior simulator for human-like interactions."""
        return self._behavior_simulator
    
    def get_current_fingerprint(self) -> Optional[FingerprintProfile]:
        """Get current fingerprint profile."""
        return self._current_fingerprint
    
    def get_fingerprint_stats(self) -> Dict[str, Any]:
        """Get statistics about fingerprint usage."""
        if not self._fingerprint_history:
            return {}
        
        return {
            "total_fingerprints": len(self._fingerprint_history),
            "current_session": self._current_fingerprint.session_id if self._current_fingerprint else None,
            "platforms_used": list(set(fp.platform for fp in self._fingerprint_history)),
            "user_agents_used": list(set(fp.user_agent for fp in self._fingerprint_history)),
            "screen_resolutions_used": list(set(
                (fp.screen_width, fp.screen_height) for fp in self._fingerprint_history
            ))
        }


# Integration with existing ActorPage
class StealthActorPage(ActorPage):
    """Extended ActorPage with stealth capabilities."""
    
    def __init__(self, page: Page, fingerprint_manager: FingerprintManager):
        super().__init__(page)
        self.fingerprint_manager = fingerprint_manager
        self.behavior_simulator = fingerprint_manager.get_behavior_simulator()
    
    async def stealth_click(self, selector: str, **kwargs) -> None:
        """Click with human-like mouse movement."""
        element = await self.page.query_selector(selector)
        if not element:
            raise ValueError(f"Element not found: {selector}")
        
        # Get element position
        box = await element.bounding_box()
        if not box:
            raise ValueError(f"Could not get bounding box for: {selector}")
        
        # Calculate target position (with slight randomness)
        target_x = box["x"] + box["width"] * random.uniform(0.3, 0.7)
        target_y = box["y"] + box["height"] * random.uniform(0.3, 0.7)
        
        # Move mouse humanly
        await self.behavior_simulator.human_mouse_move(
            self.mouse, 
            target_x, 
            target_y,
            self.page.viewport_size["width"],
            self.page.viewport_size["height"]
        )
        
        # Click with human delay
        await asyncio.sleep(random.uniform(*self.fingerprint_manager.behavior_profile.click_delay_range))
        await element.click(**kwargs)
    
    async def stealth_type(self, selector: str, text: str, **kwargs) -> None:
        """Type with human-like patterns."""
        await self.behavior_simulator.human_type(self.page, selector, text)
    
    async def stealth_scroll(self, direction: str = "down", amount: int = None) -> None:
        """Scroll with human-like behavior."""
        await self.behavior_simulator.human_scroll(self.page, direction, amount)
    
    async def stealth_navigate(self, url: str) -> None:
        """Navigate with human-like behavior."""
        # Random idle before navigation
        if random.random() < 0.3:
            await self.behavior_simulator.random_idle()
        
        await self.page.goto(url)
        
        # Wait for page load with variable timing
        wait_time = random.uniform(*self.fingerprint_manager.behavior_profile.page_load_wait_range)
        await asyncio.sleep(wait_time)
        
        # Random scroll after load
        if random.random() < 0.5:
            await self.stealth_scroll("down", random.randint(100, 300))


# Factory function for easy integration
def create_stealth_manager(
    behavior_profile: Optional[BehaviorProfile] = None,
    proxy_list: Optional[List[str]] = None,
    rotation_interval: int = 300
) -> FingerprintManager:
    """Create a FingerprintManager with optional proxy support."""
    proxy_configs = []
    if proxy_list:
        for proxy_url in proxy_list:
            # Parse proxy URL
            if "://" in proxy_url:
                protocol, rest = proxy_url.split("://", 1)
                if "@" in rest:
                    auth, host_port = rest.split("@", 1)
                    username, password = auth.split(":", 1)
                    proxy_configs.append(ProxyConfig(
                        url=f"{protocol}://{host_port}",
                        protocol=protocol,
                        username=username,
                        password=password
                    ))
                else:
                    proxy_configs.append(ProxyConfig(
                        url=proxy_url,
                        protocol=protocol
                    ))
    
    return FingerprintManager(
        behavior_profile=behavior_profile,
        proxy_configs=proxy_configs if proxy_configs else None,
        rotation_interval=rotation_interval
    )