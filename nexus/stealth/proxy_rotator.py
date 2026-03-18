"""
Stealth Mode with Proxy Rotation and Fingerprint Randomization
"""

import asyncio
import random
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
import logging
import json
from datetime import datetime, timedelta

import aiohttp
from playwright.async_api import BrowserContext, Page, ProxySettings

logger = logging.getLogger(__name__)


class ProxyType(Enum):
    HTTP = "http"
    HTTPS = "https"
    SOCKS4 = "socks4"
    SOCKS5 = "socks5"


@dataclass
class ProxyConfig:
    """Configuration for a single proxy"""
    url: str
    proxy_type: ProxyType = ProxyType.HTTP
    username: Optional[str] = None
    password: Optional[str] = None
    country: Optional[str] = None
    city: Optional[str] = None
    asn: Optional[str] = None
    speed: float = 1.0  # 0.0 to 1.0, higher is faster
    last_used: Optional[datetime] = None
    failure_count: int = 0
    success_count: int = 0
    is_healthy: bool = True
    last_health_check: Optional[datetime] = None
    
    @property
    def proxy_settings(self) -> ProxySettings:
        """Convert to Playwright proxy settings"""
        settings = {"server": self.url}
        if self.username and self.password:
            settings["username"] = self.username
            settings["password"] = self.password
        return settings
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        total = self.success_count + self.failure_count
        if total == 0:
            return 1.0
        return self.success_count / total


@dataclass
class FingerprintProfile:
    """Browser fingerprint profile for anti-detection"""
    user_agent: str
    platform: str
    languages: List[str]
    screen_width: int
    screen_height: int
    color_depth: int
    pixel_ratio: float
    timezone: str
    webgl_vendor: str
    webgl_renderer: str
    canvas_hash: str
    audio_hash: str
    fonts: List[str]
    plugins: List[Dict[str, str]]
    hardware_concurrency: int
    device_memory: int
    touch_support: bool
    webrtc_ips: List[str]
    
    @classmethod
    def generate_random(cls, user_type: str = "normal") -> "FingerprintProfile":
        """Generate a random fingerprint profile"""
        # Base user agents by platform
        user_agents = {
            "windows": [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0",
            ],
            "mac": [
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
            ],
            "linux": [
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
            ],
        }
        
        platform = random.choice(["windows", "mac", "linux"])
        user_agent = random.choice(user_agents[platform])
        
        # Common screen resolutions
        resolutions = [
            (1920, 1080), (1366, 768), (1536, 864), (1440, 900),
            (1280, 720), (2560, 1440), (1680, 1050), (1600, 900),
        ]
        
        # WebGL vendors/renderers
        webgl_configs = [
            ("Google Inc. (NVIDIA)", "ANGLE (NVIDIA, NVIDIA GeForce RTX 3080 Direct3D11 vs_5_0 ps_5_0)"),
            ("Google Inc. (AMD)", "ANGLE (AMD, AMD Radeon RX 6800 XT Direct3D11 vs_5_0 ps_5_0)"),
            ("Google Inc. (Intel)", "ANGLE (Intel, Intel(R) UHD Graphics 630 Direct3D11 vs_5_0 ps_5_0)"),
            ("WebKit", "WebKit WebGL"),
        ]
        
        # Generate canvas and audio hashes (simulated)
        canvas_hash = hashlib.md5(str(random.random()).encode()).hexdigest()[:16]
        audio_hash = hashlib.md5(str(random.random()).encode()).hexdigest()[:16]
        
        # Common fonts
        all_fonts = [
            "Arial", "Verdana", "Helvetica", "Times New Roman", "Courier New",
            "Georgia", "Palatino", "Garamond", "Bookman", "Trebuchet MS",
            "Arial Black", "Impact", "Comic Sans MS", "Lucida Console",
            "Monaco", "Consolas", "Andale Mono", "Courier",
        ]
        num_fonts = random.randint(5, 15)
        fonts = random.sample(all_fonts, num_fonts)
        
        # Common plugins (reduced in modern browsers)
        plugins = []
        if random.random() > 0.7:  # 30% chance of having plugins
            plugins = [
                {"name": "Chrome PDF Plugin", "filename": "internal-pdf-viewer"},
                {"name": "Chrome PDF Viewer", "filename": "mhjfbmdgcfjbbpaeojofohoefgiehjai"},
                {"name": "Native Client", "filename": "internal-nacl-plugin"},
            ]
        
        # Timezones
        timezones = [
            "America/New_York", "America/Chicago", "America/Denver",
            "America/Los_Angeles", "Europe/London", "Europe/Paris",
            "Asia/Tokyo", "Australia/Sydney",
        ]
        
        # WebRTC IP simulation (local IPs)
        webrtc_ips = [
            f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}",
            f"10.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 255)}",
        ]
        
        # Adjust based on user type
        if user_type == "mobile":
            screen_width = random.choice([375, 390, 414, 360, 412])
            screen_height = random.choice([667, 812, 896, 780, 736])
            device_memory = random.choice([2, 4])
            hardware_concurrency = random.choice([2, 4, 6])
            touch_support = True
        elif user_type == "tablet":
            screen_width = random.choice([768, 810, 800, 1024])
            screen_height = random.choice([1024, 1080, 1280, 1366])
            device_memory = random.choice([4, 8])
            hardware_concurrency = random.choice([4, 8])
            touch_support = True
        else:  # desktop
            screen_width, screen_height = random.choice(resolutions)
            device_memory = random.choice([4, 8, 16, 32])
            hardware_concurrency = random.choice([4, 8, 12, 16])
            touch_support = random.random() > 0.8  # 20% of desktops have touch
        
        webgl_vendor, webgl_renderer = random.choice(webgl_configs)
        
        return cls(
            user_agent=user_agent,
            platform=platform,
            languages=["en-US", "en"],
            screen_width=screen_width,
            screen_height=screen_height,
            color_depth=24,
            pixel_ratio=random.choice([1, 1.25, 1.5, 2]),
            timezone=random.choice(timezones),
            webgl_vendor=webgl_vendor,
            webgl_renderer=webgl_renderer,
            canvas_hash=canvas_hash,
            audio_hash=audio_hash,
            fonts=fonts,
            plugins=plugins,
            hardware_concurrency=hardware_concurrency,
            device_memory=device_memory,
            touch_support=touch_support,
            webrtc_ips=webrtc_ips,
        )


@dataclass
class BehaviorProfile:
    """Human-like behavior patterns"""
    typing_speed: float = 0.1  # seconds per character
    typing_variance: float = 0.05  # variance in typing speed
    mouse_speed: float = 0.5  # seconds to move across screen
    mouse_variance: float = 0.2  # variance in mouse speed
    click_delay: float = 0.1  # delay before click
    scroll_speed: float = 0.3  # seconds per scroll
    thinking_time: float = 1.0  # time to "think" between actions
    mistake_rate: float = 0.02  # probability of making a typo
    correction_delay: float = 0.5  # delay before correcting typo
    
    @classmethod
    def generate_random(cls) -> "BehaviorProfile":
        """Generate random behavior profile"""
        return cls(
            typing_speed=random.uniform(0.05, 0.15),
            typing_variance=random.uniform(0.02, 0.08),
            mouse_speed=random.uniform(0.3, 0.8),
            mouse_variance=random.uniform(0.1, 0.3),
            click_delay=random.uniform(0.05, 0.2),
            scroll_speed=random.uniform(0.2, 0.5),
            thinking_time=random.uniform(0.5, 2.0),
            mistake_rate=random.uniform(0.01, 0.05),
            correction_delay=random.uniform(0.3, 0.8),
        )


class ProxyRotator:
    """
    Manages proxy rotation with health checks and intelligent selection
    """
    
    def __init__(
        self,
        proxies: Optional[List[ProxyConfig]] = None,
        health_check_url: str = "https://httpbin.org/ip",
        health_check_interval: int = 300,  # 5 minutes
        max_failures: int = 3,
        rotation_strategy: str = "round_robin",  # round_robin, random, smart
    ):
        self.proxies: List[ProxyConfig] = proxies or []
        self.health_check_url = health_check_url
        self.health_check_interval = health_check_interval
        self.max_failures = max_failures
        self.rotation_strategy = rotation_strategy
        
        self._current_index = 0
        self._last_health_check = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "proxy_rotations": 0,
        }
    
    async def initialize(self):
        """Initialize the proxy rotator"""
        self._session = aiohttp.ClientSession()
        await self.health_check_all()
    
    async def close(self):
        """Close the proxy rotator"""
        if self._session:
            await self._session.close()
    
    async def add_proxy(self, proxy: ProxyConfig):
        """Add a new proxy to the rotation"""
        async with self._lock:
            self.proxies.append(proxy)
            logger.info(f"Added proxy: {proxy.url}")
    
    async def remove_proxy(self, proxy_url: str):
        """Remove a proxy from rotation"""
        async with self._lock:
            self.proxies = [p for p in self.proxies if p.url != proxy_url]
            logger.info(f"Removed proxy: {proxy_url}")
    
    async def health_check(self, proxy: ProxyConfig) -> bool:
        """Check if a proxy is healthy"""
        if not self._session:
            return False
        
        try:
            proxy_config = {"http": proxy.url, "https": proxy.url}
            if proxy.username and proxy.password:
                auth = aiohttp.BasicAuth(proxy.username, proxy.password)
                proxy_config["auth"] = auth
            
            start_time = time.time()
            async with self._session.get(
                self.health_check_url,
                proxy=proxy.url,
                timeout=aiohttp.ClientTimeout(total=10),
                ssl=False,
            ) as response:
                elapsed = time.time() - start_time
                
                if response.status == 200:
                    proxy.is_healthy = True
                    proxy.failure_count = 0
                    proxy.success_count += 1
                    proxy.speed = min(1.0, 1.0 / max(0.1, elapsed))
                    proxy.last_health_check = datetime.now()
                    logger.debug(f"Proxy {proxy.url} is healthy (speed: {proxy.speed:.2f})")
                    return True
                else:
                    raise Exception(f"HTTP {response.status}")
        
        except Exception as e:
            proxy.failure_count += 1
            proxy.is_healthy = proxy.failure_count < self.max_failures
            proxy.last_health_check = datetime.now()
            logger.warning(f"Proxy {proxy.url} health check failed: {e}")
            return False
    
    async def health_check_all(self):
        """Check health of all proxies"""
        logger.info("Running health check for all proxies")
        tasks = [self.health_check(proxy) for proxy in self.proxies]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        healthy_count = sum(1 for r in results if r is True)
        logger.info(f"Health check complete: {healthy_count}/{len(self.proxies)} proxies healthy")
        self._last_health_check = datetime.now()
    
    def get_next_proxy(self) -> Optional[ProxyConfig]:
        """Get next proxy based on rotation strategy"""
        healthy_proxies = [p for p in self.proxies if p.is_healthy]
        
        if not healthy_proxies:
            logger.error("No healthy proxies available")
            return None
        
        if self.rotation_strategy == "round_robin":
            proxy = healthy_proxies[self._current_index % len(healthy_proxies)]
            self._current_index += 1
        
        elif self.rotation_strategy == "random":
            proxy = random.choice(healthy_proxies)
        
        elif self.rotation_strategy == "smart":
            # Weight by success rate and speed
            weights = []
            for p in healthy_proxies:
                weight = p.success_rate * p.speed
                # Reduce weight for recently used proxies
                if p.last_used:
                    minutes_since_use = (datetime.now() - p.last_used).total_seconds() / 60
                    if minutes_since_use < 5:
                        weight *= 0.5
                weights.append(weight)
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
                proxy = random.choices(healthy_proxies, weights=weights, k=1)[0]
            else:
                proxy = random.choice(healthy_proxies)
        
        else:
            proxy = healthy_proxies[0]
        
        proxy.last_used = datetime.now()
        self.stats["proxy_rotations"] += 1
        return proxy
    
    def mark_success(self, proxy: ProxyConfig):
        """Mark a proxy request as successful"""
        proxy.success_count += 1
        self.stats["successful_requests"] += 1
        self.stats["total_requests"] += 1
    
    def mark_failure(self, proxy: ProxyConfig):
        """Mark a proxy request as failed"""
        proxy.failure_count += 1
        self.stats["failed_requests"] += 1
        self.stats["total_requests"] += 1
        
        if proxy.failure_count >= self.max_failures:
            proxy.is_healthy = False
            logger.warning(f"Proxy {proxy.url} marked as unhealthy after {proxy.failure_count} failures")
    
    async def get_proxy_for_context(self) -> Optional[ProxySettings]:
        """Get proxy settings for a new browser context"""
        # Check if we need to run health checks
        if (not self._last_health_check or 
            (datetime.now() - self._last_health_check).total_seconds() > self.health_check_interval):
            await self.health_check_all()
        
        proxy = self.get_next_proxy()
        if not proxy:
            return None
        
        return proxy.proxy_settings
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rotation statistics"""
        healthy_count = sum(1 for p in self.proxies if p.is_healthy)
        return {
            **self.stats,
            "total_proxies": len(self.proxies),
            "healthy_proxies": healthy_count,
            "unhealthy_proxies": len(self.proxies) - healthy_count,
            "average_success_rate": sum(p.success_rate for p in self.proxies) / max(1, len(self.proxies)),
        }


class StealthManager:
    """
    Comprehensive stealth manager integrating fingerprint rotation,
    proxy rotation, and human-like behavior
    """
    
    def __init__(
        self,
        proxy_rotator: Optional[ProxyRotator] = None,
        fingerprint_profiles: Optional[List[FingerprintProfile]] = None,
        behavior_profile: Optional[BehaviorProfile] = None,
        user_types: List[str] = ["normal", "mobile", "tablet"],
    ):
        self.proxy_rotator = proxy_rotator
        self.fingerprint_profiles = fingerprint_profiles or []
        self.behavior_profile = behavior_profile or BehaviorProfile.generate_random()
        self.user_types = user_types
        
        self._current_fingerprint_index = 0
        self._fingerprint_cache: Dict[str, FingerprintProfile] = {}
        
        # Track used fingerprints to avoid repetition
        self._used_fingerprints: Set[str] = set()
        
        logger.info("StealthManager initialized")
    
    async def initialize(self):
        """Initialize the stealth manager"""
        if self.proxy_rotator:
            await self.proxy_rotator.initialize()
        
        # Generate initial fingerprint profiles if none provided
        if not self.fingerprint_profiles:
            for _ in range(10):  # Start with 10 profiles
                user_type = random.choice(self.user_types)
                profile = FingerprintProfile.generate_random(user_type)
                self.fingerprint_profiles.append(profile)
    
    async def close(self):
        """Close the stealth manager"""
        if self.proxy_rotator:
            await self.proxy_rotator.close()
    
    def get_fingerprint_profile(self, user_type: Optional[str] = None) -> FingerprintProfile:
        """Get a fingerprint profile, optionally for specific user type"""
        if user_type:
            # Generate new profile for this user type
            profile = FingerprintProfile.generate_random(user_type)
            self.fingerprint_profiles.append(profile)
            return profile
        
        # Rotate through existing profiles
        if not self.fingerprint_profiles:
            profile = FingerprintProfile.generate_random()
            self.fingerprint_profiles.append(profile)
            return profile
        
        profile = self.fingerprint_profiles[self._current_fingerprint_index % len(self.fingerprint_profiles)]
        self._current_fingerprint_index += 1
        return profile
    
    async def create_stealth_context(
        self,
        browser,
        user_type: Optional[str] = None,
        proxy: Optional[ProxyConfig] = None,
    ) -> Tuple[BrowserContext, FingerprintProfile]:
        """Create a stealth browser context with fingerprint and proxy"""
        # Get proxy settings
        proxy_settings = None
        if self.proxy_rotator:
            proxy_settings = await self.proxy_rotator.get_proxy_for_context()
        elif proxy:
            proxy_settings = proxy.proxy_settings
        
        # Get fingerprint profile
        fingerprint = self.get_fingerprint_profile(user_type)
        
        # Create context with proxy
        context_options = {
            "user_agent": fingerprint.user_agent,
            "viewport": {
                "width": fingerprint.screen_width,
                "height": fingerprint.screen_height,
            },
            "screen": {
                "width": fingerprint.screen_width,
                "height": fingerprint.screen_height,
            },
            "device_scale_factor": fingerprint.pixel_ratio,
            "is_mobile": user_type in ["mobile", "tablet"],
            "has_touch": fingerprint.touch_support,
            "locale": fingerprint.languages[0] if fingerprint.languages else "en-US",
            "timezone_id": fingerprint.timezone,
        }
        
        if proxy_settings:
            context_options["proxy"] = proxy_settings
        
        context = await browser.new_context(**context_options)
        
        # Inject fingerprint overrides
        await self._inject_fingerprint_overrides(context, fingerprint)
        
        # Inject behavior scripts
        await self._inject_behavior_scripts(context)
        
        logger.info(f"Created stealth context with fingerprint: {fingerprint.user_agent[:50]}...")
        return context, fingerprint
    
    async def _inject_fingerprint_overrides(self, context: BrowserContext, fingerprint: FingerprintProfile):
        """Inject JavaScript to override browser fingerprint"""
        script = """
        // Override WebGL
        const getParameter = WebGLRenderingContext.prototype.getParameter;
        WebGLRenderingContext.prototype.getParameter = function(parameter) {
            if (parameter === 37445) return '{webgl_vendor}';
            if (parameter === 37446) return '{webgl_renderer}';
            return getParameter.call(this, parameter);
        };
        
        // Override Canvas fingerprint
        const toDataURL = HTMLCanvasElement.prototype.toDataURL;
        HTMLCanvasElement.prototype.toDataURL = function(type) {
            if (type === 'image/png' && this.width === 16 && this.height === 16) {
                // This is likely a fingerprinting attempt
                return 'data:image/png;base64,{canvas_hash}';
            }
            return toDataURL.apply(this, arguments);
        };
        
        // Override AudioContext fingerprint
        const createOscillator = AudioContext.prototype.createOscillator;
        AudioContext.prototype.createOscillator = function() {
            const oscillator = createOscillator.call(this);
            const originalConnect = oscillator.connect;
            oscillator.connect = function() {
                // Add subtle noise to audio fingerprint
                const noise = Math.random() * 0.0001;
                arguments[0].gain.value += noise;
                return originalConnect.apply(this, arguments);
            };
            return oscillator;
        };
        
        // Override WebRTC IP leak
        const originalRTCPeerConnection = window.RTCPeerConnection;
        window.RTCPeerConnection = function(config, constraints) {
            if (config && config.iceServers) {
                // Filter out STUN servers that could leak IP
                config.iceServers = config.iceServers.filter(server => {
                    if (server.urls && typeof server.urls === 'string') {
                        return !server.urls.includes('stun:');
                    }
                    return true;
                });
            }
            return new originalRTCPeerConnection(config, constraints);
        };
        
        // Override plugins
        Object.defineProperty(navigator, 'plugins', {
            get: () => {plugins_json}
        });
        
        // Override languages
        Object.defineProperty(navigator, 'languages', {
            get: () => {languages_json}
        });
        
        // Override hardware concurrency
        Object.defineProperty(navigator, 'hardwareConcurrency', {
            get: () => {hardware_concurrency}
        });
        
        // Override device memory
        Object.defineProperty(navigator, 'deviceMemory', {
            get: () => {device_memory}
        });
        
        // Override platform
        Object.defineProperty(navigator, 'platform', {
            get: () => '{platform}'
        });
        
        // Override screen properties
        Object.defineProperty(screen, 'width', {
            get: () => {screen_width}
        });
        Object.defineProperty(screen, 'height', {
            get: () => {screen_height}
        });
        Object.defineProperty(screen, 'colorDepth', {
            get: () => {color_depth}
        });
        
        // Override timezone
        const originalDateTimeFormat = Intl.DateTimeFormat;
        Intl.DateTimeFormat = function() {
            const instance = new originalDateTimeFormat(...arguments);
            const resolvedOptions = instance.resolvedOptions();
            resolvedOptions.timeZone = '{timezone}';
            return instance;
        };
        Intl.DateTimeFormat.prototype = originalDateTimeFormat.prototype;
        Intl.DateTimeFormat.supportedLocalesOf = originalDateTimeFormat.supportedLocalesOf;
        """.format(
            webgl_vendor=fingerprint.webgl_vendor,
            webgl_renderer=fingerprint.webgl_renderer,
            canvas_hash=fingerprint.canvas_hash,
            plugins_json=json.dumps(fingerprint.plugins),
            languages_json=json.dumps(fingerprint.languages),
            hardware_concurrency=fingerprint.hardware_concurrency,
            device_memory=fingerprint.device_memory,
            platform=fingerprint.platform,
            screen_width=fingerprint.screen_width,
            screen_height=fingerprint.screen_height,
            color_depth=fingerprint.color_depth,
            timezone=fingerprint.timezone,
        )
        
        await context.add_init_script(script)
    
    async def _inject_behavior_scripts(self, context: BrowserContext):
        """Inject scripts for human-like behavior"""
        script = """
        // Human-like mouse movement
        let lastMouseX = 0;
        let lastMouseY = 0;
        let mouseMovementTimeout = null;
        
        function simulateHumanMouseMovement(targetX, targetY) {
            const startX = lastMouseX;
            const startY = lastMouseY;
            const distance = Math.sqrt(Math.pow(targetX - startX, 2) + Math.pow(targetY - startY, 2));
            const duration = Math.max(100, Math.min(1000, distance * {mouse_speed}));
            const steps = Math.max(10, Math.floor(duration / 16));
            
            let step = 0;
            const animate = () => {
                step++;
                const progress = step / steps;
                // Add some easing and randomness
                const easeProgress = progress < 0.5 
                    ? 2 * progress * progress 
                    : 1 - Math.pow(-2 * progress + 2, 2) / 2;
                
                const currentX = startX + (targetX - startX) * easeProgress + (Math.random() - 0.5) * 2;
                const currentY = startY + (targetY - startY) * easeProgress + (Math.random() - 0.5) * 2;
                
                // Move mouse
                const event = new MouseEvent('mousemove', {
                    clientX: currentX,
                    clientY: currentY,
                    bubbles: true,
                });
                document.dispatchEvent(event);
                
                lastMouseX = currentX;
                lastMouseY = currentY;
                
                if (step < steps) {
                    requestAnimationFrame(animate);
                }
            };
            
            requestAnimationFrame(animate);
        }
        
        // Override click to add delay
        const originalClick = HTMLElement.prototype.click;
        HTMLElement.prototype.click = function() {
            const delay = {click_delay} * 1000 + Math.random() * 50;
            setTimeout(() => {
                originalClick.call(this);
            }, delay);
        };
        
        // Human-like typing
        const originalInputValueSetter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
        Object.defineProperty(HTMLInputElement.prototype, 'value', {
            set: function(value) {
                if (this.type === 'text' || this.type === 'password' || this.type === 'email') {
                    // Simulate typing
                    const input = this;
                    const originalValue = input.value;
                    let currentIndex = 0;
                    
                    const typeChar = () => {
                        if (currentIndex < value.length) {
                            const char = value[currentIndex];
                            const delay = {typing_speed} * 1000 + (Math.random() - 0.5) * {typing_variance} * 1000;
                            
                            // Occasionally make a typo
                            if (Math.random() < {mistake_rate}) {
                                const typoChar = String.fromCharCode(char.charCodeAt(0) + (Math.random() > 0.5 ? 1 : -1));
                                originalValueSetter.call(input, originalValue + typoChar);
                                input.dispatchEvent(new Event('input', { bubbles: true }));
                                
                                // Correct the typo after a delay
                                setTimeout(() => {
                                    originalValueSetter.call(input, originalValue + char);
                                    input.dispatchEvent(new Event('input', { bubbles: true }));
                                    currentIndex++;
                                    setTimeout(typeChar, delay);
                                }, {correction_delay} * 1000);
                            } else {
                                originalValueSetter.call(input, originalValue + char);
                                input.dispatchEvent(new Event('input', { bubbles: true }));
                                currentIndex++;
                                setTimeout(typeChar, delay);
                            }
                        }
                    };
                    
                    typeChar();
                } else {
                    originalValueSetter.call(this, value);
                }
            }
        });
        
        // Random scroll behavior
        let scrollTimeout = null;
        window.addEventListener('scroll', () => {
            if (scrollTimeout) clearTimeout(scrollTimeout);
            scrollTimeout = setTimeout(() => {
                // Add slight random scroll after user stops scrolling
                if (Math.random() > 0.7) {
                    const scrollAmount = (Math.random() - 0.5) * 50;
                    window.scrollBy({
                        top: scrollAmount,
                        behavior: 'smooth'
                    });
                }
            }, 100);
        });
        
        // Track mouse position
        document.addEventListener('mousemove', (e) => {
            lastMouseX = e.clientX;
            lastMouseY = e.clientY;
        });
        """.format(
            mouse_speed=self.behavior_profile.mouse_speed,
            click_delay=self.behavior_profile.click_delay,
            typing_speed=self.behavior_profile.typing_speed,
            typing_variance=self.behavior_profile.typing_variance,
            mistake_rate=self.behavior_profile.mistake_rate,
            correction_delay=self.behavior_profile.correction_delay,
        )
        
        await context.add_init_script(script)
    
    async def human_like_typing(self, page: Page, selector: str, text: str):
        """Type text with human-like delays and occasional mistakes"""
        element = await page.query_selector(selector)
        if not element:
            raise ValueError(f"Element not found: {selector}")
        
        await element.click()
        await asyncio.sleep(self.behavior_profile.click_delay)
        
        for char in text:
            # Type the character
            await page.keyboard.press(char)
            
            # Random delay between keystrokes
            delay = self.behavior_profile.typing_speed
            if random.random() < 0.3:  # 30% chance of longer pause
                delay *= 2
            elif random.random() < 0.1:  # 10% chance of very short pause
                delay *= 0.5
            
            await asyncio.sleep(delay + random.uniform(-self.behavior_profile.typing_variance, 
                                                      self.behavior_profile.typing_variance))
            
            # Occasionally make and correct a typo
            if random.random() < self.behavior_profile.mistake_rate:
                typo_char = chr(ord(char) + random.choice([-1, 1]))
                await page.keyboard.press(typo_char)
                await asyncio.sleep(self.behavior_profile.correction_delay)
                await page.keyboard.press("Backspace")
                await page.keyboard.press(char)
    
    async def human_like_mouse_movement(self, page: Page, x: int, y: int):
        """Move mouse with human-like trajectory"""
        # Get current mouse position (simplified)
        current_x, current_y = 0, 0
        
        # Calculate distance and duration
        distance = ((x - current_x) ** 2 + (y - current_y) ** 2) ** 0.5
        duration = min(1000, max(100, distance * self.behavior_profile.mouse_speed))
        
        # Generate bezier curve points
        steps = int(duration / 16)
        points = []
        
        for i in range(steps + 1):
            t = i / steps
            # Add some randomness to the path
            random_x = random.uniform(-10, 10) * (1 - t) * t
            random_y = random.uniform(-10, 10) * (1 - t) * t
            
            point_x = current_x + (x - current_x) * t + random_x
            point_y = current_y + (y - current_y) * t + random_y
            points.append((point_x, point_y))
        
        # Move through points
        for point_x, point_y in points:
            await page.mouse.move(point_x, point_y)
            await asyncio.sleep(0.016)  # ~60fps
    
    def rotate_behavior_profile(self):
        """Rotate to a new behavior profile"""
        self.behavior_profile = BehaviorProfile.generate_random()
        logger.info("Rotated behavior profile")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stealth manager statistics"""
        stats = {
            "fingerprint_profiles": len(self.fingerprint_profiles),
            "current_behavior_profile": {
                "typing_speed": self.behavior_profile.typing_speed,
                "mouse_speed": self.behavior_profile.mouse_speed,
                "mistake_rate": self.behavior_profile.mistake_rate,
            },
        }
        
        if self.proxy_rotator:
            stats["proxy_stats"] = self.proxy_rotator.get_stats()
        
        return stats


# Factory function for easy integration
async def create_stealth_manager(
    proxy_list: Optional[List[str]] = None,
    proxy_type: ProxyType = ProxyType.HTTP,
    **kwargs
) -> StealthManager:
    """
    Factory function to create a stealth manager with proxy rotation
    
    Args:
        proxy_list: List of proxy URLs
        proxy_type: Type of proxies
        **kwargs: Additional arguments for StealthManager
    
    Returns:
        Configured StealthManager instance
    """
    proxy_rotator = None
    if proxy_list:
        proxies = []
        for proxy_url in proxy_list:
            # Parse proxy URL for authentication
            if "@" in proxy_url:
                auth_part, server_part = proxy_url.split("@", 1)
                if ":" in auth_part:
                    username, password = auth_part.split(":", 1)
                    proxy_url = f"{proxy_type.value}://{server_part}"
                else:
                    username = auth_part
                    password = None
                    proxy_url = f"{proxy_type.value}://{server_part}"
            else:
                username = None
                password = None
                proxy_url = f"{proxy_type.value}://{proxy_url}"
            
            proxy_config = ProxyConfig(
                url=proxy_url,
                proxy_type=proxy_type,
                username=username,
                password=password,
            )
            proxies.append(proxy_config)
        
        proxy_rotator = ProxyRotator(proxies=proxies)
    
    manager = StealthManager(proxy_rotator=proxy_rotator, **kwargs)
    await manager.initialize()
    
    return manager


# Integration with existing Page class
async def apply_stealth_to_page(page: Page, stealth_manager: StealthManager):
    """
    Apply stealth measures to an existing page
    
    This can be called after creating a page to add stealth capabilities
    """
    # Get a fingerprint profile
    fingerprint = stealth_manager.get_fingerprint_profile()
    
    # Inject fingerprint overrides
    await stealth_manager._inject_fingerprint_overrides(page.context, fingerprint)
    
    # Inject behavior scripts
    await stealth_manager._inject_behavior_scripts(page.context)
    
    logger.info(f"Applied stealth to page with fingerprint: {fingerprint.user_agent[:50]}...")


# Example usage
async def example_usage():
    """Example of how to use the stealth manager"""
    # Create stealth manager with proxies
    proxies = [
        "http://proxy1.example.com:8080",
        "http://user:pass@proxy2.example.com:8080",
        "socks5://proxy3.example.com:1080",
    ]
    
    stealth_manager = await create_stealth_manager(
        proxy_list=proxies,
        proxy_type=ProxyType.HTTP,
        user_types=["normal", "mobile"],
    )
    
    # Get stats
    stats = stealth_manager.get_stats()
    print(f"Stealth manager stats: {stats}")
    
    # Use with browser (example with Playwright)
    from playwright.async_api import async_playwright
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        
        # Create stealth context
        context, fingerprint = await stealth_manager.create_stealth_context(
            browser,
            user_type="normal",
        )
        
        page = await context.new_page()
        
        # Use human-like interactions
        await page.goto("https://example.com")
        await stealth_manager.human_like_typing(page, "input[name='q']", "search query")
        await stealth_manager.human_like_mouse_movement(page, 100, 100)
        
        await browser.close()
    
    await stealth_manager.close()


# Import hashlib for fingerprint generation
import hashlib