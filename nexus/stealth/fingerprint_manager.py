"""
nexus/stealth/fingerprint_manager.py

Adaptive Stealth & Anti-Detection System
Dynamically adjusts browser fingerprinting, mouse movements, typing patterns,
and request timing to mimic human behavior and avoid bot detection systems.
"""

import asyncio
import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import numpy as np
from datetime import datetime, timedelta
import json
import hashlib
from pathlib import Path

from ..actor.mouse import MouseController
from ..actor.page import PageController
from ..actor.utils import BezierCurve, Point


class FingerprintType(Enum):
    """Types of browser fingerprints to rotate"""
    CHROME_DESKTOP = "chrome_desktop"
    FIREFOX_DESKTOP = "firefox_desktop"
    SAFARI_DESKTOP = "safari_desktop"
    CHROME_MOBILE = "chrome_mobile"
    SAFARI_MOBILE = "safari_mobile"
    EDGE_DESKTOP = "edge_desktop"


class BehaviorProfile(Enum):
    """Human behavior profiles for different use cases"""
    CASUAL_BROWSING = "casual_browsing"
    RESEARCH_INTENSIVE = "research_intensive"
    ECOMMERCE_SHOPPING = "ecommerce_shopping"
    SOCIAL_MEDIA = "social_media"
    PROFESSIONAL_WORK = "professional_work"


@dataclass
class HumanTiming:
    """Human-like timing patterns"""
    typing_speed_wpm: Tuple[int, int] = (40, 80)  # Words per minute range
    typing_error_rate: float = 0.02  # 2% error rate
    typing_correction_delay: Tuple[float, float] = (0.3, 0.8)  # Seconds
    mouse_move_duration: Tuple[float, float] = (0.1, 0.5)  # Seconds
    click_delay: Tuple[float, float] = (0.05, 0.15)  # Seconds
    scroll_speed: Tuple[float, float] = (0.5, 2.0)  # Seconds per viewport
    page_load_wait: Tuple[float, float] = (1.0, 3.0)  # Seconds
    micro_pause_frequency: float = 0.1  # 10% chance of micro-pause
    micro_pause_duration: Tuple[float, float] = (0.05, 0.2)  # Seconds
    request_delay: Tuple[float, float] = (0.5, 2.0)  # Seconds between requests


@dataclass
class BrowserFingerprint:
    """Complete browser fingerprint configuration"""
    fingerprint_type: FingerprintType
    user_agent: str
    viewport: Dict[str, int]
    screen_resolution: Dict[str, int]
    platform: str
    language: str
    languages: List[str]
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
    webrtc_ip_handling: str
    do_not_track: Optional[bool] = None
    cookie_enabled: bool = True
    session_storage: bool = True
    local_storage: bool = True
    indexed_db: bool = True
    web_sql: bool = False
    color_depth: int = 24
    pixel_ratio: float = 1.0
    
    def to_playwright_context_options(self) -> Dict[str, Any]:
        """Convert to Playwright context options"""
        return {
            "user_agent": self.user_agent,
            "viewport": self.viewport,
            "screen": self.screen_resolution,
            "locale": self.language,
            "timezone_id": self.timezone,
            "color_scheme": "light",
            "reduced_motion": "no",
            "forced_colors": "none",
            "accept_downloads": True,
            "has_touch": self.touch_support,
            "is_mobile": "mobile" in self.fingerprint_type.value,
            "java_script_enabled": True,
            "bypass_csp": False,
            "ignore_https_errors": False,
            "extra_http_headers": {
                "Accept-Language": f"{self.language},{self.languages[0]};q=0.9,en;q=0.8",
                "Accept-Encoding": "gzip, deflate, br",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Cache-Control": "max-age=0"
            }
        }


@dataclass
class BehavioralBiometrics:
    """Behavioral biometrics for human-like interaction"""
    mouse_movement_pattern: List[Dict[str, Any]] = field(default_factory=list)
    typing_pattern: List[Dict[str, Any]] = field(default_factory=list)
    scroll_pattern: List[Dict[str, Any]] = field(default_factory=list)
    click_pattern: List[Dict[str, Any]] = field(default_factory=list)
    focus_pattern: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mouse_movement": self.mouse_movement_pattern,
            "typing": self.typing_pattern,
            "scroll": self.scroll_pattern,
            "click": self.click_pattern,
            "focus": self.focus_pattern
        }


class FingerprintManager:
    """
    Manages adaptive stealth and anti-detection by dynamically adjusting
    browser fingerprinting and human-like behavior patterns.
    """
    
    # Common user agents for rotation
    USER_AGENTS = {
        FingerprintType.CHROME_DESKTOP: [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ],
        FingerprintType.FIREFOX_DESKTOP: [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (X11; Linux i686; rv:121.0) Gecko/20100101 Firefox/121.0"
        ],
        FingerprintType.SAFARI_DESKTOP: [
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15"
        ],
        FingerprintType.CHROME_MOBILE: [
            "Mozilla/5.0 (Linux; Android 14; SM-S911B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.6099.210 Mobile Safari/537.36",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/120.0.6099.119 Mobile/15E148 Safari/604.1"
        ],
        FingerprintType.SAFARI_MOBILE: [
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1"
        ],
        FingerprintType.EDGE_DESKTOP: [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0"
        ]
    }
    
    # Common screen resolutions
    SCREEN_RESOLUTIONS = [
        {"width": 1920, "height": 1080},
        {"width": 1366, "height": 768},
        {"width": 1536, "height": 864},
        {"width": 1440, "height": 900},
        {"width": 1280, "height": 720},
        {"width": 2560, "height": 1440},
        {"width": 1680, "height": 1050},
        {"width": 1600, "height": 900}
    ]
    
    # Mobile screen resolutions
    MOBILE_RESOLUTIONS = [
        {"width": 390, "height": 844},  # iPhone 14
        {"width": 414, "height": 896},  # iPhone 11
        {"width": 375, "height": 812},  # iPhone X
        {"width": 412, "height": 915},  # Pixel 7
        {"width": 360, "height": 800},  # Galaxy S21
        {"width": 393, "height": 873}   # Pixel 6
    ]
    
    # Common timezones
    TIMEZONES = [
        "America/New_York", "America/Los_Angeles", "America/Chicago",
        "Europe/London", "Europe/Paris", "Europe/Berlin",
        "Asia/Tokyo", "Asia/Shanghai", "Asia/Kolkata",
        "Australia/Sydney", "Pacific/Auckland"
    ]
    
    # Common languages
    LANGUAGES = [
        ("en-US", ["en-US", "en"]),
        ("en-GB", ["en-GB", "en"]),
        ("es-ES", ["es-ES", "es", "en"]),
        ("fr-FR", ["fr-FR", "fr", "en"]),
        ("de-DE", ["de-DE", "de", "en"]),
        ("ja-JP", ["ja-JP", "ja", "en"]),
        ("zh-CN", ["zh-CN", "zh", "en"])
    ]
    
    # WebGL configurations
    WEBGL_CONFIGS = [
        {"vendor": "Google Inc. (NVIDIA)", "renderer": "ANGLE (NVIDIA, NVIDIA GeForce RTX 3080 Direct3D11 vs_5_0 ps_5_0, D3D11)"},
        {"vendor": "Google Inc. (AMD)", "renderer": "ANGLE (AMD, AMD Radeon RX 6800 XT Direct3D11 vs_5_0 ps_5_0, D3D11)"},
        {"vendor": "Google Inc. (Intel)", "renderer": "ANGLE (Intel, Intel(R) UHD Graphics 630 Direct3D11 vs_5_0 ps_5_0, D3D11)"},
        {"vendor": "Apple", "renderer": "Apple M1"},
        {"vendor": "Mozilla", "renderer": "Mali-G78"}
    ]
    
    # Common fonts
    COMMON_FONTS = [
        "Arial", "Verdana", "Helvetica", "Times New Roman", "Courier New",
        "Georgia", "Palatino", "Garamond", "Bookman", "Trebuchet MS",
        "Arial Black", "Impact", "Comic Sans MS", "Lucida Console",
        "Monaco", "Consolas", "Courier"
    ]
    
    def __init__(
        self,
        behavior_profile: BehaviorProfile = BehaviorProfile.CASUAL_BROWSING,
        fingerprint_rotation_interval: int = 3600,  # 1 hour in seconds
        proxy_manager: Optional[Any] = None,
        storage_path: Optional[str] = None
    ):
        """
        Initialize the FingerprintManager.
        
        Args:
            behavior_profile: The human behavior profile to emulate
            fingerprint_rotation_interval: How often to rotate fingerprints (seconds)
            proxy_manager: Optional proxy manager for residential proxies
            storage_path: Path to store fingerprint and behavior data
        """
        self.behavior_profile = behavior_profile
        self.rotation_interval = fingerprint_rotation_interval
        self.proxy_manager = proxy_manager
        self.storage_path = Path(storage_path) if storage_path else Path.home() / ".nexus" / "stealth"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Current state
        self.current_fingerprint: Optional[BrowserFingerprint] = None
        self.current_biometrics: Optional[BehavioralBiometrics] = None
        self.last_rotation: Optional[datetime] = None
        self.fingerprint_history: List[Dict[str, Any]] = []
        self.behavior_stats: Dict[str, Any] = self._load_behavior_stats()
        
        # Timing configuration based on behavior profile
        self.timing = self._get_timing_for_profile(behavior_profile)
        
        # Initialize with a fingerprint
        self.rotate_fingerprint()
    
    def _get_timing_for_profile(self, profile: BehaviorProfile) -> HumanTiming:
        """Get timing configuration for the specified behavior profile"""
        base_timing = HumanTiming()
        
        if profile == BehaviorProfile.CASUAL_BROWSING:
            base_timing.typing_speed_wpm = (35, 65)
            base_timing.mouse_move_duration = (0.15, 0.6)
            base_timing.scroll_speed = (0.8, 2.5)
            base_timing.micro_pause_frequency = 0.15
            
        elif profile == BehaviorProfile.RESEARCH_INTENSIVE:
            base_timing.typing_speed_wpm = (50, 90)
            base_timing.mouse_move_duration = (0.1, 0.4)
            base_timing.scroll_speed = (0.5, 1.5)
            base_timing.page_load_wait = (2.0, 5.0)
            base_timing.micro_pause_frequency = 0.08
            
        elif profile == BehaviorProfile.ECOMMERCE_SHOPPING:
            base_timing.typing_speed_wpm = (40, 70)
            base_timing.mouse_move_duration = (0.2, 0.7)
            base_timing.scroll_speed = (0.6, 2.0)
            base_timing.click_delay = (0.08, 0.2)
            base_timing.micro_pause_frequency = 0.12
            
        elif profile == BehaviorProfile.SOCIAL_MEDIA:
            base_timing.typing_speed_wpm = (45, 85)
            base_timing.mouse_move_duration = (0.08, 0.3)
            base_timing.scroll_speed = (0.3, 1.0)
            base_timing.request_delay = (0.3, 1.5)
            base_timing.micro_pause_frequency = 0.18
            
        elif profile == BehaviorProfile.PROFESSIONAL_WORK:
            base_timing.typing_speed_wpm = (60, 100)
            base_timing.typing_error_rate = 0.01
            base_timing.mouse_move_duration = (0.05, 0.25)
            base_timing.scroll_speed = (0.4, 1.2)
            base_timing.micro_pause_frequency = 0.05
        
        return base_timing
    
    def _load_behavior_stats(self) -> Dict[str, Any]:
        """Load behavior statistics from storage"""
        stats_file = self.storage_path / "behavior_stats.json"
        if stats_file.exists():
            try:
                with open(stats_file, "r") as f:
                    return json.load(f)
            except:
                pass
        return {
            "total_sessions": 0,
            "total_interactions": 0,
            "avg_session_duration": 0,
            "fingerprint_usage": {},
            "behavior_patterns": {}
        }
    
    def _save_behavior_stats(self):
        """Save behavior statistics to storage"""
        stats_file = self.storage_path / "behavior_stats.json"
        with open(stats_file, "w") as f:
            json.dump(self.behavior_stats, f, indent=2)
    
    def _generate_fingerprint_hash(self, fingerprint: BrowserFingerprint) -> str:
        """Generate a unique hash for a fingerprint"""
        fingerprint_str = json.dumps({
            "ua": fingerprint.user_agent,
            "viewport": fingerprint.viewport,
            "platform": fingerprint.platform,
            "webgl": f"{fingerprint.webgl_vendor}:{fingerprint.webgl_renderer}",
            "canvas": fingerprint.canvas_hash
        }, sort_keys=True)
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]
    
    def rotate_fingerprint(self, force: bool = False) -> BrowserFingerprint:
        """
        Rotate to a new fingerprint.
        
        Args:
            force: Force rotation even if interval hasn't passed
            
        Returns:
            The new browser fingerprint
        """
        now = datetime.now()
        
        # Check if rotation is needed
        if (not force and self.last_rotation and 
            (now - self.last_rotation).total_seconds() < self.rotation_interval):
            return self.current_fingerprint
        
        # Select fingerprint type (weighted by common usage)
        fingerprint_weights = {
            FingerprintType.CHROME_DESKTOP: 0.45,
            FingerprintType.FIREFOX_DESKTOP: 0.20,
            FingerprintType.SAFARI_DESKTOP: 0.15,
            FingerprintType.EDGE_DESKTOP: 0.10,
            FingerprintType.CHROME_MOBILE: 0.07,
            FingerprintType.SAFARI_MOBILE: 0.03
        }
        
        fingerprint_type = random.choices(
            list(fingerprint_weights.keys()),
            weights=list(fingerprint_weights.values())
        )[0]
        
        # Generate fingerprint components
        user_agent = random.choice(self.USER_AGENTS[fingerprint_type])
        
        if "mobile" in fingerprint_type.value:
            viewport = random.choice(self.MOBILE_RESOLUTIONS)
            screen = viewport.copy()
            touch_support = True
            device_memory = random.choice([2, 4, 6, 8])
            hardware_concurrency = random.choice([4, 6, 8])
        else:
            viewport = random.choice(self.SCREEN_RESOLUTIONS)
            screen = viewport.copy()
            touch_support = random.random() < 0.1  # 10% chance of touch on desktop
            device_memory = random.choice([4, 8, 16, 32])
            hardware_concurrency = random.choice([4, 8, 12, 16, 24])
        
        # Adjust viewport for mobile
        if "mobile" in fingerprint_type.value:
            viewport = {
                "width": int(viewport["width"] * 0.9),
                "height": int(viewport["height"] * 0.9)
            }
        
        language_config = random.choice(self.LANGUAGES)
        webgl_config = random.choice(self.WEBGL_CONFIGS)
        
        # Generate canvas and audio hashes
        canvas_hash = hashlib.md5(f"canvas_{random.random()}".encode()).hexdigest()
        audio_hash = hashlib.md5(f"audio_{random.random()}".encode()).hexdigest()
        
        # Select fonts (subset of common fonts)
        num_fonts = random.randint(8, 15)
        fonts = random.sample(self.COMMON_FONTS, num_fonts)
        
        # Generate plugins
        plugins = self._generate_plugins(fingerprint_type)
        
        fingerprint = BrowserFingerprint(
            fingerprint_type=fingerprint_type,
            user_agent=user_agent,
            viewport=viewport,
            screen_resolution=screen,
            platform=self._get_platform(fingerprint_type),
            language=language_config[0],
            languages=language_config[1],
            timezone=random.choice(self.TIMEZONES),
            webgl_vendor=webgl_config["vendor"],
            webgl_renderer=webgl_config["renderer"],
            canvas_hash=canvas_hash,
            audio_hash=audio_hash,
            fonts=fonts,
            plugins=plugins,
            hardware_concurrency=hardware_concurrency,
            device_memory=device_memory,
            touch_support=touch_support,
            webrtc_ip_handling="default_public_interface_only",
            do_not_track=random.choice([True, False, None]),
            color_depth=random.choice([24, 32]),
            pixel_ratio=random.choice([1.0, 1.5, 2.0, 2.5, 3.0])
        )
        
        # Update state
        self.current_fingerprint = fingerprint
        self.last_rotation = now
        
        # Record in history
        fingerprint_hash = self._generate_fingerprint_hash(fingerprint)
        self.fingerprint_history.append({
            "hash": fingerprint_hash,
            "timestamp": now.isoformat(),
            "type": fingerprint_type.value,
            "user_agent": user_agent[:50] + "..." if len(user_agent) > 50 else user_agent
        })
        
        # Keep only last 100 fingerprints in history
        if len(self.fingerprint_history) > 100:
            self.fingerprint_history = self.fingerprint_history[-100:]
        
        # Update stats
        self.behavior_stats["fingerprint_usage"][fingerprint_type.value] = \
            self.behavior_stats["fingerprint_usage"].get(fingerprint_type.value, 0) + 1
        self._save_behavior_stats()
        
        return fingerprint
    
    def _get_platform(self, fingerprint_type: FingerprintType) -> str:
        """Get platform string based on fingerprint type"""
        if "mobile" in fingerprint_type.value:
            if "iphone" in fingerprint_type.value or "safari_mobile" in fingerprint_type.value:
                return "iPhone"
            else:
                return "Linux armv8l"
        else:
            platforms = ["Win32", "MacIntel", "Linux x86_64"]
            weights = [0.6, 0.3, 0.1]
            return random.choices(platforms, weights=weights)[0]
    
    def _generate_plugins(self, fingerprint_type: FingerprintType) -> List[Dict[str, str]]:
        """Generate realistic browser plugins"""
        plugins = []
        
        # Common plugins for desktop
        if "mobile" not in fingerprint_type.value:
            plugins.extend([
                {"name": "Chrome PDF Plugin", "filename": "internal-pdf-viewer", "description": "Portable Document Format"},
                {"name": "Chrome PDF Viewer", "filename": "mhjfbmdgcfjbbpaeojofohoefgiehjai", "description": ""},
                {"name": "Native Client", "filename": "internal-nacl-plugin", "description": ""}
            ])
            
            # Add random additional plugins
            additional_plugins = [
                {"name": "Widevine Content Decryption Module", "filename": "widevinecdmadapter.dll", "description": "Enables Widevine licenses for playback of HTML audio/video content."},
                {"name": "Microsoft Edge PDF Plugin", "filename": "pdf.dll", "description": "Portable Document Format"},
                {"name": "WebKit built-in PDF", "filename": "internal-pdf-plugin", "description": "Portable Document Format"}
            ]
            
            if random.random() < 0.7:  # 70% chance to have additional plugins
                plugins.extend(random.sample(additional_plugins, random.randint(1, len(additional_plugins))))
        
        return plugins
    
    async def apply_fingerprint_to_context(self, context: Any) -> None:
        """
        Apply the current fingerprint to a browser context.
        
        Args:
            context: Playwright browser context
        """
        if not self.current_fingerprint:
            self.rotate_fingerprint()
        
        fingerprint = self.current_fingerprint
        
        # Set viewport and user agent
        await context.set_viewport_size(fingerprint.viewport)
        await context.set_extra_http_headers({
            "User-Agent": fingerprint.user_agent
        })
        
        # Add script to override fingerprint properties
        stealth_script = self._generate_stealth_script(fingerprint)
        await context.add_init_script(stealth_script)
        
        # Set timezone and locale
        await context.set_geolocation({"latitude": 40.7128, "longitude": -74.0060})  # NYC
        await context.set_permissions(["geolocation"], origin="*")
    
    def _generate_stealth_script(self, fingerprint: BrowserFingerprint) -> str:
        """Generate JavaScript to apply fingerprint overrides"""
        return f"""
        // Override navigator properties
        Object.defineProperty(navigator, 'userAgent', {{
            get: () => '{fingerprint.user_agent}'
        }});
        
        Object.defineProperty(navigator, 'platform', {{
            get: () => '{fingerprint.platform}'
        }});
        
        Object.defineProperty(navigator, 'language', {{
            get: () => '{fingerprint.language}'
        }});
        
        Object.defineProperty(navigator, 'languages', {{
            get: () => {json.dumps(fingerprint.languages)}
        }});
        
        Object.defineProperty(navigator, 'hardwareConcurrency', {{
            get: () => {fingerprint.hardware_concurrency}
        }});
        
        Object.defineProperty(navigator, 'deviceMemory', {{
            get: () => {fingerprint.device_memory}
        }});
        
        Object.defineProperty(navigator, 'maxTouchPoints', {{
            get: () => {1 if fingerprint.touch_support else 0}
        }});
        
        // Override screen properties
        Object.defineProperty(screen, 'width', {{
            get: () => {fingerprint.screen_resolution['width']}
        }});
        
        Object.defineProperty(screen, 'height', {{
            get: () => {fingerprint.screen_resolution['height']}
        }});
        
        Object.defineProperty(screen, 'availWidth', {{
            get: () => {fingerprint.screen_resolution['width']}
        }});
        
        Object.defineProperty(screen, 'availHeight', {{
            get: () => {fingerprint.screen_resolution['height'] - 40}
        }});
        
        Object.defineProperty(screen, 'colorDepth', {{
            get: () => {fingerprint.color_depth}
        }});
        
        Object.defineProperty(screen, 'pixelDepth', {{
            get: () => {fingerprint.color_depth}
        }});
        
        // Override WebGL
        const getParameter = WebGLRenderingContext.prototype.getParameter;
        WebGLRenderingContext.prototype.getParameter = function(parameter) {{
            if (parameter === 37445) {{
                return '{fingerprint.webgl_vendor}';
            }}
            if (parameter === 37446) {{
                return '{fingerprint.webgl_renderer}';
            }}
            return getParameter.call(this, parameter);
        }};
        
        // Override canvas fingerprint
        const originalToDataURL = HTMLCanvasElement.prototype.toDataURL;
        HTMLCanvasElement.prototype.toDataURL = function(type) {{
            if (type === 'image/png' && this.width === 16 && this.height === 16) {{
                return 'data:image/png;base64,{fingerprint.canvas_hash}';
            }}
            return originalToDataURL.apply(this, arguments);
        }};
        
        // Override audio fingerprint
        const originalCreateOscillator = AudioContext.prototype.createOscillator;
        AudioContext.prototype.createOscillator = function() {{
            const oscillator = originalCreateOscillator.call(this);
            const originalConnect = oscillator.connect;
            oscillator.connect = function() {{
                // Add slight noise to audio fingerprint
                const noise = 0.00001 * (Math.random() - 0.5);
                if (arguments[0] instanceof AudioParam) {{
                    arguments[0].value += noise;
                }}
                return originalConnect.apply(this, arguments);
            }};
            return oscillator;
        }};
        
        // Override plugins
        Object.defineProperty(navigator, 'plugins', {{
            get: () => {{
                const plugins = {json.dumps(fingerprint.plugins)};
                const pluginArray = [];
                plugins.forEach((plugin, i) => {{
                    const pluginObj = {{}};
                    Object.defineProperties(pluginObj, {{
                        name: {{ value: plugin.name, enumerable: true }},
                        filename: {{ value: plugin.filename, enumerable: true }},
                        description: {{ value: plugin.description, enumerable: true }},
                        length: {{ value: 1, enumerable: true }}
                    }});
                    pluginArray.push(pluginObj);
                }});
                return pluginArray;
            }}
        }});
        
        // Override fonts
        const originalFonts = document.fonts;
        Object.defineProperty(document, 'fonts', {{
            get: () => {{
                const fonts = new Set([{', '.join(f'"{font}"' for font in fingerprint.fonts)}]);
                const fontSet = new Set();
                fonts.forEach(font => fontSet.add(font));
                return fontSet;
            }}
        }});
        
        // WebRTC leak prevention
        const originalRTCPeerConnection = window.RTCPeerConnection;
        window.RTCPeerConnection = function(config, constraints) {{
            if (config && config.iceServers) {{
                config.iceServers = [];
            }}
            return new originalRTCPeerConnection(config, constraints);
        }};
        window.RTCPeerConnection.prototype = originalRTCPeerConnection.prototype;
        
        // Console log to verify script is running
        console.log('Stealth mode activated');
        """
    
    def generate_human_mouse_movement(
        self,
        start: Point,
        end: Point,
        duration: Optional[float] = None,
        include_curves: bool = True
    ) -> List[Tuple[Point, float]]:
        """
        Generate human-like mouse movement between two points.
        
        Args:
            start: Starting point (x, y)
            end: Ending point (x, y)
            duration: Total duration in seconds (random if None)
            include_curves: Whether to include Bezier curves for natural movement
            
        Returns:
            List of (point, timestamp) tuples for the movement path
        """
        if duration is None:
            duration = random.uniform(*self.timing.mouse_move_duration)
        
        # Calculate distance for speed adjustment
        distance = math.sqrt((end.x - start.x) ** 2 + (end.y - start.y) ** 2)
        
        # Adjust duration based on distance (longer distances take longer)
        base_duration = duration
        duration = base_duration * (1 + distance / 1000)
        
        # Generate control points for Bezier curve
        if include_curves and distance > 50:
            # Add 1-3 control points for natural curve
            num_control_points = random.randint(1, 3)
            control_points = []
            
            for i in range(num_control_points):
                # Control points along the line with random offset
                t = (i + 1) / (num_control_points + 1)
                base_x = start.x + (end.x - start.x) * t
                base_y = start.y + (end.y - start.y) * t
                
                # Add perpendicular offset for curve
                offset = random.uniform(-distance * 0.2, distance * 0.2)
                angle = math.atan2(end.y - start.y, end.x - start.x) + math.pi / 2
                
                control_x = base_x + offset * math.cos(angle)
                control_y = base_y + offset * math.sin(angle)
                control_points.append(Point(control_x, control_y))
            
            # Create Bezier curve
            bezier = BezierCurve([start] + control_points + [end])
            
            # Generate points along the curve with variable speed
            num_points = max(10, int(distance / 10))  # More points for longer distances
            points = []
            
            for i in range(num_points):
                t = i / (num_points - 1)
                
                # Apply easing function for human-like acceleration/deceleration
                # Ease in-out with slight randomness
                if t < 0.5:
                    eased_t = 2 * t * t
                else:
                    eased_t = 1 - math.pow(-2 * t + 2, 2) / 2
                
                # Add micro-variations
                eased_t += random.uniform(-0.02, 0.02)
                eased_t = max(0, min(1, eased_t))
                
                point = bezier.get_point(eased_t)
                
                # Add small random jitter (human hand tremor)
                jitter_x = random.gauss(0, 1.5)
                jitter_y = random.gauss(0, 1.5)
                point = Point(point.x + jitter_x, point.y + jitter_y)
                
                # Calculate timestamp
                timestamp = duration * t
                
                # Add micro-pauses occasionally
                if random.random() < self.timing.micro_pause_frequency * 0.1:
                    timestamp += random.uniform(*self.timing.micro_pause_duration)
                
                points.append((point, timestamp))
        else:
            # Simple linear movement with jitter
            num_points = max(5, int(distance / 20))
            points = []
            
            for i in range(num_points):
                t = i / (num_points - 1)
                x = start.x + (end.x - start.x) * t
                y = start.y + (end.y - start.y) * t
                
                # Add jitter
                x += random.gauss(0, 2)
                y += random.gauss(0, 2)
                
                timestamp = duration * t
                points.append((Point(x, y), timestamp))
        
        return points
    
    def generate_human_typing_pattern(
        self,
        text: str,
        include_errors: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate human-like typing pattern for the given text.
        
        Args:
            text: Text to type
            include_errors: Whether to include typing errors and corrections
            
        Returns:
            List of typing events with character, delay, and optional error
        """
        pattern = []
        current_time = 0.0
        
        # Calculate base typing speed (characters per second)
        wpm = random.uniform(*self.timing.typing_speed_wpm)
        cps = wpm * 5 / 60  # Convert WPM to characters per second (assuming 5 chars per word)
        
        i = 0
        while i < len(text):
            char = text[i]
            
            # Base delay for this character
            base_delay = 1.0 / cps
            
            # Adjust delay based on character type
            if char in ' .,!?;:\n\t':
                # Longer delay for punctuation and whitespace
                delay = base_delay * random.uniform(1.5, 3.0)
            elif char.isupper():
                # Slightly longer for uppercase (shift key)
                delay = base_delay * random.uniform(1.1, 1.3)
            else:
                delay = base_delay
            
            # Add random variation
            delay *= random.uniform(0.8, 1.2)
            
            # Occasionally add micro-pause
            if random.random() < self.timing.micro_pause_frequency:
                delay += random.uniform(*self.timing.micro_pause_duration)
            
            # Check for typing errors
            error = None
            if include_errors and random.random() < self.timing.typing_error_rate:
                # Generate a common typing error
                error_char = self._get_typo_char(char)
                error = {
                    "type": "error",
                    "character": error_char,
                    "delay": delay * 0.5  # Faster for error
                }
                
                # Add correction after error
                correction_delay = random.uniform(*self.timing.typing_correction_delay)
                pattern.append({
                    "character": error_char,
                    "delay": error["delay"],
                    "event_type": "keypress"
                })
                pattern.append({
                    "character": "Backspace",
                    "delay": correction_delay,
                    "event_type": "keypress"
                })
                current_time += error["delay"] + correction_delay
            
            pattern.append({
                "character": char,
                "delay": delay,
                "event_type": "keypress"
            })
            
            current_time += delay
            i += 1
        
        return pattern
    
    def _get_typo_char(self, char: str) -> str:
        """Get a common typo for the given character"""
        # QWERTY keyboard layout typos
        typos = {
            'a': ['s', 'q', 'z'],
            'b': ['v', 'g', 'h', 'n'],
            'c': ['x', 'd', 'f', 'v'],
            'd': ['s', 'e', 'f', 'c', 'x'],
            'e': ['w', 'r', 'd', 's'],
            'f': ['d', 'r', 'g', 'v', 'c'],
            'g': ['f', 't', 'h', 'b', 'v'],
            'h': ['g', 'y', 'j', 'n', 'b'],
            'i': ['u', 'o', 'k', 'j'],
            'j': ['h', 'u', 'k', 'm', 'n'],
            'k': ['j', 'i', 'l', 'm'],
            'l': ['k', 'o', 'p'],
            'm': ['n', 'j', 'k'],
            'n': ['b', 'h', 'j', 'm'],
            'o': ['i', 'p', 'l'],
            'p': ['o', 'l'],
            'q': ['w', 'a'],
            'r': ['e', 't', 'f', 'd'],
            's': ['a', 'w', 'd', 'z', 'x'],
            't': ['r', 'y', 'g', 'f'],
            'u': ['y', 'i', 'j', 'h'],
            'v': ['c', 'f', 'g', 'b'],
            'w': ['q', 'e', 's', 'a'],
            'x': ['z', 's', 'd', 'c'],
            'y': ['t', 'u', 'h', 'g'],
            'z': ['a', 's', 'x']
        }
        
        char_lower = char.lower()
        if char_lower in typos:
            typo = random.choice(typos[char_lower])
            return typo.upper() if char.isupper() else typo
        
        return char
    
    def generate_scroll_pattern(
        self,
        total_height: int,
        viewport_height: int,
        direction: str = "down"
    ) -> List[Dict[str, Any]]:
        """
        Generate human-like scroll pattern.
        
        Args:
            total_height: Total page height in pixels
            viewport_height: Viewport height in pixels
            direction: Scroll direction ('down' or 'up')
            
        Returns:
            List of scroll events with amount and delay
        """
        pattern = []
        current_position = 0
        target_position = total_height if direction == "down" else 0
        
        # Calculate number of scroll chunks
        avg_scroll_chunk = viewport_height * random.uniform(0.3, 0.8)
        num_chunks = int(total_height / avg_scroll_chunk)
        
        for i in range(num_chunks):
            # Determine scroll amount for this chunk
            if direction == "down":
                # Variable scroll amount
                scroll_amount = random.uniform(
                    viewport_height * 0.2,
                    viewport_height * 0.9
                )
                
                # Occasionally do a large scroll
                if random.random() < 0.1:  # 10% chance
                    scroll_amount = viewport_height * random.uniform(1.0, 2.0)
                
                current_position = min(current_position + scroll_amount, total_height)
            else:
                scroll_amount = random.uniform(
                    viewport_height * 0.2,
                    viewport_height * 0.9
                )
                current_position = max(current_position - scroll_amount, 0)
            
            # Calculate delay based on scroll speed
            base_delay = scroll_amount / viewport_height * random.uniform(*self.timing.scroll_speed)
            
            # Add random variation
            delay = base_delay * random.uniform(0.7, 1.3)
            
            # Occasionally pause while scrolling
            if random.random() < 0.2:  # 20% chance
                delay += random.uniform(0.1, 0.5)
            
            pattern.append({
                "position": current_position,
                "amount": scroll_amount,
                "delay": delay,
                "direction": direction
            })
            
            # Check if we've reached the target
            if direction == "down" and current_position >= total_height:
                break
            elif direction == "up" and current_position <= 0:
                break
        
        return pattern
    
    def generate_request_delay(self) -> float:
        """Generate human-like delay between requests"""
        return random.uniform(*self.timing.request_delay)
    
    def generate_page_load_wait(self) -> float:
        """Generate human-like wait time after page load"""
        return random.uniform(*self.timing.page_load_wait)
    
    def get_proxy(self) -> Optional[Dict[str, str]]:
        """Get a proxy configuration if proxy manager is available"""
        if self.proxy_manager:
            return self.proxy_manager.get_proxy()
        return None
    
    def should_rotate_fingerprint(self) -> bool:
        """Check if fingerprint should be rotated"""
        if not self.last_rotation:
            return True
        
        elapsed = (datetime.now() - self.last_rotation).total_seconds()
        return elapsed >= self.rotation_interval
    
    async def create_stealth_context(
        self,
        browser: Any,
        proxy: Optional[Dict[str, str]] = None
    ) -> Any:
        """
        Create a new browser context with stealth settings.
        
        Args:
            browser: Playwright browser instance
            proxy: Optional proxy configuration
            
        Returns:
            Configured browser context
        """
        # Rotate fingerprint if needed
        if self.should_rotate_fingerprint():
            self.rotate_fingerprint()
        
        fingerprint = self.current_fingerprint
        context_options = fingerprint.to_playwright_context_options()
        
        # Add proxy if provided
        if proxy:
            context_options["proxy"] = proxy
        elif self.proxy_manager:
            proxy = self.proxy_manager.get_proxy()
            if proxy:
                context_options["proxy"] = proxy
        
        # Create context
        context = await browser.new_context(**context_options)
        
        # Apply stealth scripts
        await self.apply_fingerprint_to_context(context)
        
        # Initialize behavioral biometrics
        self.current_biometrics = BehavioralBiometrics()
        
        # Update session stats
        self.behavior_stats["total_sessions"] += 1
        self._save_behavior_stats()
        
        return context
    
    def record_interaction(
        self,
        interaction_type: str,
        details: Dict[str, Any]
    ):
        """Record an interaction for behavior analysis"""
        if not self.current_biometrics:
            return
        
        timestamp = datetime.now().isoformat()
        
        if interaction_type == "mouse_move":
            self.current_biometrics.mouse_movement_pattern.append({
                "timestamp": timestamp,
                **details
            })
        elif interaction_type == "keypress":
            self.current_biometrics.typing_pattern.append({
                "timestamp": timestamp,
                **details
            })
        elif interaction_type == "scroll":
            self.current_biometrics.scroll_pattern.append({
                "timestamp": timestamp,
                **details
            })
        elif interaction_type == "click":
            self.current_biometrics.click_pattern.append({
                "timestamp": timestamp,
                **details
            })
        
        # Update stats
        self.behavior_stats["total_interactions"] += 1
    
    def get_behavior_report(self) -> Dict[str, Any]:
        """Get a report of current behavior patterns"""
        report = {
            "fingerprint": {
                "type": self.current_fingerprint.fingerprint_type.value if self.current_fingerprint else None,
                "hash": self._generate_fingerprint_hash(self.current_fingerprint) if self.current_fingerprint else None,
                "last_rotation": self.last_rotation.isoformat() if self.last_rotation else None
            },
            "behavior_profile": self.behavior_profile.value,
            "timing_config": {
                "typing_speed_wpm": self.timing.typing_speed_wpm,
                "mouse_move_duration": self.timing.mouse_move_duration,
                "scroll_speed": self.timing.scroll_speed
            },
            "stats": self.behavior_stats,
            "current_biometrics": self.current_biometrics.to_dict() if self.current_biometrics else None
        }
        
        return report


class StealthMouseController:
    """Enhanced mouse controller with stealth features"""
    
    def __init__(
        self,
        mouse_controller: MouseController,
        fingerprint_manager: FingerprintManager
    ):
        self.mouse = mouse_controller
        self.fingerprint_manager = fingerprint_manager
        self.last_position = Point(0, 0)
    
    async def move_to(
        self,
        x: float,
        y: float,
        steps: Optional[int] = None,
        human_like: bool = True
    ):
        """Move mouse to position with human-like movement"""
        if human_like:
            # Generate human-like path
            start = self.last_position
            end = Point(x, y)
            movement_path = self.fingerprint_manager.generate_human_mouse_movement(
                start, end
            )
            
            # Execute movement along path
            for point, timestamp in movement_path:
                await self.mouse.move(point.x, point.y, steps=1)
                await asyncio.sleep(timestamp / len(movement_path))
                
                # Record interaction
                self.fingerprint_manager.record_interaction(
                    "mouse_move",
                    {"x": point.x, "y": point.y, "timestamp": timestamp}
                )
        else:
            await self.mouse.move(x, y, steps=steps)
        
        self.last_position = Point(x, y)
    
    async def click(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        button: str = "left",
        click_count: int = 1,
        human_like: bool = True
    ):
        """Click with human-like timing"""
        if x is not None and y is not None:
            await self.move_to(x, y, human_like=human_like)
        
        # Add delay before click
        if human_like:
            delay = random.uniform(*self.fingerprint_manager.timing.click_delay)
            await asyncio.sleep(delay)
        
        await self.mouse.click(button=button, click_count=click_count)
        
        # Record interaction
        self.fingerprint_manager.record_interaction(
            "click",
            {"x": x, "y": y, "button": button, "timestamp": time.time()}
        )
    
    async def scroll(
        self,
        delta_x: float = 0,
        delta_y: float = 0,
        human_like: bool = True
    ):
        """Scroll with human-like patterns"""
        if human_like and delta_y != 0:
            # Generate scroll pattern
            direction = "down" if delta_y > 0 else "up"
            pattern = self.fingerprint_manager.generate_scroll_pattern(
                total_height=abs(delta_y),
                viewport_height=800,  # Default viewport height
                direction=direction
            )
            
            for scroll_event in pattern:
                scroll_amount = scroll_event["amount"]
                if direction == "up":
                    scroll_amount = -scroll_amount
                
                await self.mouse.scroll(delta_x=0, delta_y=scroll_amount)
                await asyncio.sleep(scroll_event["delay"])
                
                # Record interaction
                self.fingerprint_manager.record_interaction(
                    "scroll",
                    {"amount": scroll_amount, "delay": scroll_event["delay"]}
                )
        else:
            await self.mouse.scroll(delta_x=delta_x, delta_y=delta_y)


class StealthPageController:
    """Enhanced page controller with stealth features"""
    
    def __init__(
        self,
        page_controller: PageController,
        fingerprint_manager: FingerprintManager
    ):
        self.page = page_controller
        self.fingerprint_manager = fingerprint_manager
        self.stealth_mouse = StealthMouseController(
            page_controller.mouse,
            fingerprint_manager
        )
    
    async def goto(self, url: str, **kwargs):
        """Navigate to URL with stealth features"""
        # Add request delay
        delay = self.fingerprint_manager.generate_request_delay()
        await asyncio.sleep(delay)
        
        # Navigate
        response = await self.page.goto(url, **kwargs)
        
        # Wait for page load with human-like timing
        wait_time = self.fingerprint_manager.generate_page_load_wait()
        await asyncio.sleep(wait_time)
        
        return response
    
    async def type_text(
        self,
        selector: str,
        text: str,
        human_like: bool = True,
        include_errors: bool = True
    ):
        """Type text with human-like pattern"""
        if human_like:
            # Generate typing pattern
            pattern = self.fingerprint_manager.generate_human_typing_pattern(
                text, include_errors=include_errors
            )
            
            # Focus on element
            await self.page.click(selector)
            await asyncio.sleep(0.1)
            
            # Type according to pattern
            for event in pattern:
                char = event["character"]
                delay = event["delay"]
                
                if char == "Backspace":
                    await self.page.keyboard.press("Backspace")
                else:
                    await self.page.keyboard.type(char, delay=0)  # We handle delay ourselves
                
                await asyncio.sleep(delay)
                
                # Record interaction
                self.fingerprint_manager.record_interaction(
                    "keypress",
                    {"character": char, "delay": delay}
                )
        else:
            await self.page.type(selector, text)
    
    async def scroll_page(
        self,
        amount: float,
        direction: str = "down",
        human_like: bool = True
    ):
        """Scroll page with human-like pattern"""
        if human_like:
            # Get page height
            page_height = await self.page.evaluate("document.body.scrollHeight")
            viewport_height = await self.page.evaluate("window.innerHeight")
            
            # Generate scroll pattern
            pattern = self.fingerprint_manager.generate_scroll_pattern(
                total_height=page_height,
                viewport_height=viewport_height,
                direction=direction
            )
            
            # Execute scroll pattern
            for scroll_event in pattern:
                scroll_amount = scroll_event["amount"]
                if direction == "up":
                    scroll_amount = -scroll_amount
                
                await self.page.evaluate(f"window.scrollBy(0, {scroll_amount})")
                await asyncio.sleep(scroll_event["delay"])
                
                # Record interaction
                self.fingerprint_manager.record_interaction(
                    "scroll",
                    {"amount": scroll_amount, "delay": scroll_event["delay"]}
                )
        else:
            await self.page.evaluate(f"window.scrollBy(0, {amount})")


# Factory function for easy integration
def create_stealth_manager(
    behavior_profile: BehaviorProfile = BehaviorProfile.CASUAL_BROWSING,
    **kwargs
) -> FingerprintManager:
    """
    Create a FingerprintManager with the specified behavior profile.
    
    Args:
        behavior_profile: The human behavior profile to emulate
        **kwargs: Additional arguments for FingerprintManager
        
    Returns:
        Configured FingerprintManager instance
    """
    return FingerprintManager(behavior_profile=behavior_profile, **kwargs)


# Integration with existing modules
def enhance_actor_with_stealth(
    actor: Any,
    fingerprint_manager: Optional[FingerprintManager] = None
) -> Tuple[Any, FingerprintManager]:
    """
    Enhance an existing actor with stealth capabilities.
    
    Args:
        actor: The actor to enhance (should have mouse and page attributes)
        fingerprint_manager: Optional existing fingerprint manager
        
    Returns:
        Tuple of (enhanced_actor, fingerprint_manager)
    """
    if fingerprint_manager is None:
        fingerprint_manager = create_stealth_manager()
    
    # Create stealth controllers
    stealth_mouse = StealthMouseController(actor.mouse, fingerprint_manager)
    stealth_page = StealthPageController(actor.page, fingerprint_manager)
    
    # Replace controllers
    actor.mouse = stealth_mouse
    actor.page = stealth_page
    actor.fingerprint_manager = fingerprint_manager
    
    return actor, fingerprint_manager


# Example usage
if __name__ == "__main__":
    # Create a stealth manager
    manager = create_stealth_manager(
        behavior_profile=BehaviorProfile.RESEARCH_INTENSIVE,
        fingerprint_rotation_interval=1800  # 30 minutes
    )
    
    # Generate a fingerprint
    fingerprint = manager.rotate_fingerprint()
    print(f"Generated fingerprint: {fingerprint.fingerprint_type.value}")
    print(f"User Agent: {fingerprint.user_agent}")
    
    # Generate human-like mouse movement
    start = Point(0, 0)
    end = Point(500, 300)
    movement = manager.generate_human_mouse_movement(start, end)
    print(f"Generated {len(movement)} points for mouse movement")
    
    # Generate typing pattern
    text = "Hello, this is a test of human-like typing."
    typing_pattern = manager.generate_human_typing_pattern(text)
    print(f"Generated {len(typing_pattern)} typing events")
    
    # Generate scroll pattern
    scroll_pattern = manager.generate_scroll_pattern(
        total_height=2000,
        viewport_height=800
    )
    print(f"Generated {len(scroll_pattern)} scroll events")
    
    # Get behavior report
    report = manager.get_behavior_report()
    print(f"Behavior report: {json.dumps(report, indent=2)}")