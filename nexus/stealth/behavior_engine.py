"""
Adaptive Stealth & Anti-Detection Engine for nexus
Simulates human behavioral patterns to evade bot detection systems
"""

import asyncio
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path

# Import existing modules for integration
from nexus.actor.mouse import MouseActor
from nexus.actor.page import PageActor
from nexus.actor.element import ElementActor


class StealthProfile(Enum):
    """Predefined stealth behavior profiles"""
    HUMAN_CASUAL = "casual"
    HUMAN_FOCUSED = "focused"
    HUMAN_DISTRACTED = "distracted"
    BOT_AVOIDANCE = "avoidance"
    CUSTOM = "custom"


@dataclass
class FingerprintProfile:
    """Browser fingerprint configuration"""
    user_agent: str = ""
    viewport: Tuple[int, int] = (1920, 1080)
    platform: str = "Win32"
    language: str = "en-US"
    timezone: str = "America/New_York"
    webgl_vendor: str = "Google Inc."
    webgl_renderer: str = "ANGLE (Intel(R) UHD Graphics 630 Direct3D11 vs_5_0 ps_5_0)"
    canvas_hash: str = ""
    audio_hash: str = ""
    fonts: List[str] = field(default_factory=lambda: [
        "Arial", "Verdana", "Helvetica", "Times New Roman", "Courier New"
    ])
    plugins: List[str] = field(default_factory=lambda: [
        "PDF Viewer", "Chrome PDF Viewer", "Chromium PDF Viewer"
    ])
    hardware_concurrency: int = 8
    device_memory: int = 8
    touch_support: bool = False
    max_touch_points: int = 0


@dataclass
class BehaviorProfile:
    """Human behavior simulation parameters"""
    typing_speed: Tuple[float, float] = (0.05, 0.15)  # seconds between keystrokes
    typing_error_rate: float = 0.02  # probability of typo
    mouse_speed: Tuple[float, float] = (0.3, 0.8)  # seconds for mouse movement
    mouse_curvature: float = 0.3  # curve intensity (0-1)
    scroll_speed: Tuple[float, float] = (0.2, 0.5)  # seconds per scroll
    micro_pause_probability: float = 0.1  # random pause probability
    micro_pause_duration: Tuple[float, float] = (0.1, 0.3)  # pause duration
    attention_span: float = 30.0  # seconds before distraction
    distraction_probability: float = 0.05
    click_delay: Tuple[float, float] = (0.1, 0.3)  # delay before click


class BehaviorEngine:
    """
    Adaptive Stealth & Anti-Detection Engine
    Dynamically adjusts browser behavior to mimic human patterns
    """
    
    def __init__(
        self,
        page_actor: Optional[PageActor] = None,
        mouse_actor: Optional[MouseActor] = None,
        profile: StealthProfile = StealthProfile.HUMAN_CASUAL,
        proxy_pool: Optional[List[str]] = None,
        fingerprint_rotation_interval: int = 300  # seconds
    ):
        self.page_actor = page_actor
        self.mouse_actor = mouse_actor
        self.profile = profile
        self.proxy_pool = proxy_pool or []
        self.fingerprint_rotation_interval = fingerprint_rotation_interval
        
        # Initialize behavior profiles
        self.behavior_profiles = self._init_behavior_profiles()
        self.current_behavior = self.behavior_profiles[profile]
        
        # Fingerprint management
        self.fingerprint_profiles = self._generate_fingerprint_profiles()
        self.current_fingerprint = random.choice(self.fingerprint_profiles)
        self.last_fingerprint_rotation = time.time()
        
        # Behavioral state tracking
        self.typing_state = TypingState()
        self.mouse_state = MouseState()
        self.scroll_state = ScrollState()
        
        # Detection evasion metrics
        self.detection_score = 0.0
        self.action_history = []
        
        # Load residential proxies if available
        self.residential_proxies = self._load_residential_proxies()
        
    def _init_behavior_profiles(self) -> Dict[StealthProfile, BehaviorProfile]:
        """Initialize predefined behavior profiles"""
        return {
            StealthProfile.HUMAN_CASUAL: BehaviorProfile(
                typing_speed=(0.08, 0.2),
                typing_error_rate=0.03,
                mouse_speed=(0.4, 1.0),
                mouse_curvature=0.4,
                scroll_speed=(0.3, 0.6),
                micro_pause_probability=0.15,
                micro_pause_duration=(0.2, 0.5),
                attention_span=45.0,
                distraction_probability=0.08,
                click_delay=(0.15, 0.4)
            ),
            StealthProfile.HUMAN_FOCUSED: BehaviorProfile(
                typing_speed=(0.05, 0.12),
                typing_error_rate=0.01,
                mouse_speed=(0.2, 0.6),
                mouse_curvature=0.2,
                scroll_speed=(0.15, 0.4),
                micro_pause_probability=0.05,
                micro_pause_duration=(0.1, 0.2),
                attention_span=120.0,
                distraction_probability=0.02,
                click_delay=(0.08, 0.2)
            ),
            StealthProfile.HUMAN_DISTRACTED: BehaviorProfile(
                typing_speed=(0.1, 0.25),
                typing_error_rate=0.05,
                mouse_speed=(0.5, 1.2),
                mouse_curvature=0.5,
                scroll_speed=(0.4, 0.8),
                micro_pause_probability=0.25,
                micro_pause_duration=(0.3, 0.8),
                attention_span=20.0,
                distraction_probability=0.15,
                click_delay=(0.2, 0.5)
            ),
            StealthProfile.BOT_AVOIDANCE: BehaviorProfile(
                typing_speed=(0.06, 0.18),
                typing_error_rate=0.02,
                mouse_speed=(0.3, 0.9),
                mouse_curvature=0.35,
                scroll_speed=(0.25, 0.55),
                micro_pause_probability=0.12,
                micro_pause_duration=(0.15, 0.4),
                attention_span=60.0,
                distraction_probability=0.06,
                click_delay=(0.12, 0.35)
            )
        }
    
    def _generate_fingerprint_profiles(self, count: int = 10) -> List[FingerprintProfile]:
        """Generate diverse fingerprint profiles for rotation"""
        profiles = []
        
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ]
        
        viewports = [
            (1920, 1080), (1366, 768), (1536, 864), (1440, 900), (1280, 720)
        ]
        
        for i in range(count):
            profile = FingerprintProfile(
                user_agent=random.choice(user_agents),
                viewport=random.choice(viewports),
                platform=random.choice(["Win32", "MacIntel", "Linux x86_64"]),
                language=random.choice(["en-US", "en-GB", "fr-FR", "de-DE", "es-ES"]),
                timezone=random.choice([
                    "America/New_York", "America/Los_Angeles", "Europe/London", 
                    "Europe/Paris", "Asia/Tokyo"
                ]),
                webgl_vendor="Google Inc.",
                webgl_renderer=f"ANGLE ({random.choice(['Intel', 'NVIDIA', 'AMD'])} Direct3D11)",
                canvas_hash=hashlib.md5(f"canvas_{i}_{random.random()}".encode()).hexdigest()[:16],
                audio_hash=hashlib.md5(f"audio_{i}_{random.random()}".encode()).hexdigest()[:16],
                fonts=random.sample([
                    "Arial", "Verdana", "Helvetica", "Times New Roman", "Courier New",
                    "Georgia", "Palatino", "Garamond", "Bookman", "Trebuchet MS"
                ], k=random.randint(4, 8)),
                hardware_concurrency=random.choice([4, 8, 12, 16]),
                device_memory=random.choice([4, 8, 16, 32]),
                touch_support=random.choice([True, False]),
                max_touch_points=random.choice([0, 1, 5, 10])
            )
            profiles.append(profile)
        
        return profiles
    
    def _load_residential_proxies(self) -> List[Dict[str, Any]]:
        """Load residential proxy configurations"""
        # This would typically load from a configuration file or API
        # For now, return empty list - integration point for proxy services
        return []
    
    async def type_text(
        self,
        text: str,
        element: Optional[ElementActor] = None,
        human_like: bool = True
    ) -> List[float]:
        """
        Type text with human-like patterns including variable speed, errors, and pauses
        Returns list of delays between keystrokes
        """
        if not human_like:
            # Fast, non-human typing
            delays = [0.01] * len(text)
            if element:
                await element.type(text)
            return delays
        
        delays = []
        current_text = ""
        
        for i, char in enumerate(text):
            # Calculate typing delay with variation
            base_delay = random.uniform(*self.current_behavior.typing_speed)
            
            # Add micro-pauses (thinking pauses)
            if random.random() < self.current_behavior.micro_pause_probability:
                pause = random.uniform(*self.current_behavior.micro_pause_duration)
                delays.append(pause)
                await asyncio.sleep(pause)
            
            # Simulate typing errors
            if random.random() < self.current_behavior.typing_error_rate and i > 0:
                # Type wrong character
                wrong_char = random.choice("abcdefghijklmnopqrstuvwxyz")
                delays.append(base_delay)
                current_text += wrong_char
                
                if element:
                    await element.type(wrong_char)
                
                await asyncio.sleep(base_delay)
                
                # Backspace after error
                backspace_delay = random.uniform(0.1, 0.2)
                delays.append(backspace_delay)
                current_text = current_text[:-1]
                
                if element:
                    await element.press("Backspace")
                
                await asyncio.sleep(backspace_delay)
            
            # Type correct character
            delays.append(base_delay)
            current_text += char
            
            if element:
                await element.type(char)
            
            # Update typing state
            self.typing_state.update(char, base_delay)
            
            # Random distraction (look away, etc.)
            if random.random() < self.current_behavior.distraction_probability:
                distraction_time = random.uniform(0.5, 2.0)
                delays.append(distraction_time)
                await asyncio.sleep(distraction_time)
        
        return delays
    
    async def move_mouse(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        duration: Optional[float] = None,
        curve_intensity: Optional[float] = None
    ) -> List[Tuple[float, float, float]]:
        """
        Move mouse along curved, human-like path
        Returns list of (x, y, timestamp) coordinates
        """
        if duration is None:
            duration = random.uniform(*self.current_behavior.mouse_speed)
        
        if curve_intensity is None:
            curve_intensity = self.current_behavior.mouse_curvature
        
        # Calculate distance for number of steps
        distance = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        steps = max(10, int(distance / 5))  # At least 10 steps, roughly 5px per step
        
        # Generate control points for Bezier curve
        control_points = self._generate_control_points(
            start_x, start_y, end_x, end_y, curve_intensity
        )
        
        coordinates = []
        start_time = time.time()
        
        for i in range(steps + 1):
            t = i / steps
            
            # Apply easing function for natural acceleration/deceleration
            t_eased = self._ease_in_out_cubic(t)
            
            # Calculate position along Bezier curve
            x, y = self._bezier_point(t_eased, control_points)
            
            # Add micro-variations (hand tremor)
            x += random.gauss(0, 0.5)
            y += random.gauss(0, 0.5)
            
            timestamp = start_time + (duration * t)
            coordinates.append((x, y, timestamp))
            
            # Move mouse if actor is available
            if self.mouse_actor:
                await self.mouse_actor.move(x, y)
            
            # Small delay between movements
            await asyncio.sleep(duration / steps)
        
        # Update mouse state
        self.mouse_state.update(end_x, end_y, duration)
        
        return coordinates
    
    def _generate_control_points(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        curve_intensity: float
    ) -> List[Tuple[float, float]]:
        """Generate control points for curved mouse movement"""
        # Calculate perpendicular vector for curve
        dx = end_x - start_x
        dy = end_y - start_y
        length = math.sqrt(dx**2 + dy**2)
        
        if length == 0:
            return [(start_x, start_y), (end_x, end_y)]
        
        # Normalize and rotate 90 degrees for perpendicular
        perp_x = -dy / length
        perp_y = dx / length
        
        # Generate random curve offset
        curve_offset = random.uniform(-curve_intensity * 100, curve_intensity * 100)
        
        # Create control points
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        
        control1_x = mid_x + perp_x * curve_offset * 0.5
        control1_y = mid_y + perp_y * curve_offset * 0.5
        
        control2_x = mid_x - perp_x * curve_offset * 0.3
        control2_y = mid_y - perp_y * curve_offset * 0.3
        
        return [
            (start_x, start_y),
            (control1_x, control1_y),
            (control2_x, control2_y),
            (end_x, end_y)
        ]
    
    def _bezier_point(
        self,
        t: float,
        points: List[Tuple[float, float]]
    ) -> Tuple[float, float]:
        """Calculate point on Bezier curve"""
        n = len(points) - 1
        x = 0.0
        y = 0.0
        
        for i, (px, py) in enumerate(points):
            # Bernstein polynomial
            coeff = math.comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
            x += coeff * px
            y += coeff * py
        
        return x, y
    
    def _ease_in_out_cubic(self, t: float) -> float:
        """Cubic easing function for natural acceleration/deceleration"""
        if t < 0.5:
            return 4 * t * t * t
        else:
            return 1 - math.pow(-2 * t + 2, 3) / 2
    
    async def scroll_page(
        self,
        direction: str = "down",
        amount: Optional[int] = None,
        smooth: bool = True
    ) -> List[float]:
        """
        Scroll page with human-like patterns
        Returns list of scroll delays
        """
        if amount is None:
            amount = random.randint(100, 500)
        
        delays = []
        
        if smooth:
            # Break scroll into smaller steps
            steps = random.randint(3, 8)
            step_amount = amount / steps
            
            for _ in range(steps):
                delay = random.uniform(*self.current_behavior.scroll_speed)
                delays.append(delay)
                
                if self.page_actor:
                    if direction == "down":
                        await self.page_actor.scroll_down(step_amount)
                    else:
                        await self.page_actor.scroll_up(step_amount)
                
                await asyncio.sleep(delay)
                
                # Random pause during scrolling
                if random.random() < 0.2:
                    pause = random.uniform(0.1, 0.3)
                    delays.append(pause)
                    await asyncio.sleep(pause)
        else:
            # Single scroll action
            delay = random.uniform(*self.current_behavior.scroll_speed)
            delays.append(delay)
            
            if self.page_actor:
                if direction == "down":
                    await self.page_actor.scroll_down(amount)
                else:
                    await self.page_actor.scroll_up(amount)
            
            await asyncio.sleep(delay)
        
        # Update scroll state
        self.scroll_state.update(direction, amount)
        
        return delays
    
    async def click_element(
        self,
        element: ElementActor,
        human_like: bool = True
    ) -> float:
        """
        Click element with human-like delay and movement
        Returns delay before click
        """
        if not human_like:
            await element.click()
            return 0.0
        
        # Get element position
        box = await element.bounding_box()
        if not box:
            await element.click()
            return 0.0
        
        # Calculate click position (slightly random within element)
        click_x = box["x"] + random.uniform(box["width"] * 0.2, box["width"] * 0.8)
        click_y = box["y"] + random.uniform(box["height"] * 0.2, box["height"] * 0.8)
        
        # Move mouse to element with human-like movement
        current_pos = await self.mouse_actor.position() if self.mouse_actor else (0, 0)
        await self.move_mouse(
            current_pos[0], current_pos[1],
            click_x, click_y
        )
        
        # Add delay before click
        click_delay = random.uniform(*self.current_behavior.click_delay)
        await asyncio.sleep(click_delay)
        
        # Click
        await element.click()
        
        return click_delay
    
    async def rotate_fingerprint(self) -> FingerprintProfile:
        """Rotate to a new fingerprint profile"""
        # Select new fingerprint (avoid recent ones if possible)
        available_profiles = [
            p for p in self.fingerprint_profiles 
            if p != self.current_fingerprint
        ]
        
        if not available_profiles:
            available_profiles = self.fingerprint_profiles
        
        self.current_fingerprint = random.choice(available_profiles)
        self.last_fingerprint_rotation = time.time()
        
        # Apply fingerprint to browser context
        if self.page_actor:
            await self._apply_fingerprint(self.current_fingerprint)
        
        return self.current_fingerprint
    
    async def _apply_fingerprint(self, profile: FingerprintProfile):
        """Apply fingerprint profile to browser context"""
        # This would integrate with nexus's context management
        # For now, this is a placeholder for the actual implementation
        pass
    
    async def rotate_proxy(self) -> Optional[str]:
        """Rotate to a new residential proxy"""
        if not self.residential_proxies:
            return None
        
        proxy = random.choice(self.residential_proxies)
        # Proxy rotation would be handled by the browser context
        return proxy.get("url")
    
    def should_rotate_fingerprint(self) -> bool:
        """Check if fingerprint should be rotated based on interval"""
        elapsed = time.time() - self.last_fingerprint_rotation
        return elapsed >= self.fingerprint_rotation_interval
    
    def update_detection_score(self, score_delta: float):
        """Update bot detection risk score"""
        self.detection_score = max(0.0, min(1.0, self.detection_score + score_delta))
        
        # Adjust behavior based on detection risk
        if self.detection_score > 0.7:
            self.current_behavior = self.behavior_profiles[StealthProfile.BOT_AVOIDANCE]
        elif self.detection_score > 0.4:
            self.current_behavior = self.behavior_profiles[StealthProfile.HUMAN_FOCUSED]
    
    def get_stealth_metrics(self) -> Dict[str, Any]:
        """Get current stealth performance metrics"""
        return {
            "detection_score": self.detection_score,
            "current_profile": self.profile.value,
            "current_fingerprint": self.current_fingerprint.user_agent[:50] + "...",
            "typing_stats": self.typing_state.get_stats(),
            "mouse_stats": self.mouse_state.get_stats(),
            "scroll_stats": self.scroll_state.get_stats(),
            "actions_count": len(self.action_history)
        }


@dataclass
class TypingState:
    """Track typing behavior state"""
    characters_typed: int = 0
    total_time: float = 0.0
    errors_corrected: int = 0
    last_character: str = ""
    last_delay: float = 0.0
    
    def update(self, character: str, delay: float):
        self.characters_typed += 1
        self.total_time += delay
        self.last_character = character
        self.last_delay = delay
    
    def get_stats(self) -> Dict[str, Any]:
        if self.characters_typed == 0:
            return {"wpm": 0, "avg_delay": 0, "error_rate": 0}
        
        wpm = (self.characters_typed / 5) / (self.total_time / 60)  # Words per minute
        avg_delay = self.total_time / self.characters_typed
        
        return {
            "wpm": round(wpm, 1),
            "avg_delay": round(avg_delay, 3),
            "error_rate": round(self.errors_corrected / self.characters_typed, 3)
        }


@dataclass
class MouseState:
    """Track mouse movement state"""
    total_distance: float = 0.0
    total_time: float = 0.0
    movements: int = 0
    last_position: Tuple[float, float] = (0.0, 0.0)
    
    def update(self, x: float, y: float, duration: float):
        if self.movements > 0:
            distance = math.sqrt(
                (x - self.last_position[0])**2 + 
                (y - self.last_position[1])**2
            )
            self.total_distance += distance
        
        self.total_time += duration
        self.movements += 1
        self.last_position = (x, y)
    
    def get_stats(self) -> Dict[str, Any]:
        if self.movements == 0:
            return {"avg_speed": 0, "total_distance": 0, "movements": 0}
        
        avg_speed = self.total_distance / self.total_time if self.total_time > 0 else 0
        
        return {
            "avg_speed": round(avg_speed, 1),
            "total_distance": round(self.total_distance, 1),
            "movements": self.movements
        }


@dataclass
class ScrollState:
    """Track scrolling behavior state"""
    total_scrolls: int = 0
    total_distance: int = 0
    directions: Dict[str, int] = field(default_factory=lambda: {"up": 0, "down": 0})
    
    def update(self, direction: str, amount: int):
        self.total_scrolls += 1
        self.total_distance += amount
        self.directions[direction] = self.directions.get(direction, 0) + 1
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_scrolls": self.total_scrolls,
            "total_distance": self.total_distance,
            "direction_ratio": self.directions.get("down", 0) / max(1, self.total_scrolls)
        }


class StealthOrchestrator:
    """
    High-level orchestrator for stealth operations
    Coordinates fingerprint rotation, proxy switching, and behavior adaptation
    """
    
    def __init__(
        self,
        behavior_engine: BehaviorEngine,
        rotation_strategy: str = "time_based"
    ):
        self.behavior_engine = behavior_engine
        self.rotation_strategy = rotation_strategy
        self.last_action_time = time.time()
        self.session_start = time.time()
        
        # Anti-detection patterns
        self.anti_detection_patterns = self._load_anti_detection_patterns()
    
    def _load_anti_detection_patterns(self) -> Dict[str, Any]:
        """Load known bot detection patterns to avoid"""
        return {
            "webdriver_navigator": True,  # navigator.webdriver = true
            "chrome_runtime": False,  # window.chrome.runtime
            "permissions_api": True,  # Permissions API behavior
            "connection_rtt": True,  # Network Information API
            "hardware_concurrency": True,  # navigator.hardwareConcurrency
            "device_memory": True,  # navigator.deviceMemory
            "plugins_length": True,  # navigator.plugins.length
            "languages_length": True,  # navigator.languages.length
        }
    
    async def perform_stealth_action(
        self,
        action_type: str,
        **kwargs
    ) -> Any:
        """
        Perform action with full stealth integration
        Handles fingerprint rotation, proxy switching, and behavior adaptation
        """
        # Check if fingerprint rotation is needed
        if self.behavior_engine.should_rotate_fingerprint():
            await self.behavior_engine.rotate_fingerprint()
        
        # Add natural delay between actions
        delay = self._calculate_natural_delay()
        await asyncio.sleep(delay)
        
        # Perform the requested action
        result = None
        if action_type == "type":
            result = await self.behavior_engine.type_text(**kwargs)
        elif action_type == "move_mouse":
            result = await self.behavior_engine.move_mouse(**kwargs)
        elif action_type == "scroll":
            result = await self.behavior_engine.scroll_page(**kwargs)
        elif action_type == "click":
            result = await self.behavior_engine.click_element(**kwargs)
        
        # Update detection score based on action
        self._update_detection_risk(action_type)
        
        # Record action
        self.behavior_engine.action_history.append({
            "type": action_type,
            "timestamp": time.time(),
            "detection_score": self.behavior_engine.detection_score
        })
        
        self.last_action_time = time.time()
        
        return result
    
    def _calculate_natural_delay(self) -> float:
        """Calculate natural delay between actions based on context"""
        base_delay = random.uniform(0.1, 0.5)
        
        # Increase delay if actions are happening too quickly
        time_since_last = time.time() - self.last_action_time
        if time_since_last < 0.5:
            base_delay += random.uniform(0.2, 0.8)
        
        # Random "thinking" pauses
        if random.random() < 0.1:
            base_delay += random.uniform(1.0, 3.0)
        
        return base_delay
    
    def _update_detection_risk(self, action_type: str):
        """Update detection risk based on action patterns"""
        # This would analyze patterns and adjust detection score
        # For now, simple implementation
        pass
    
    def get_session_report(self) -> Dict[str, Any]:
        """Generate comprehensive stealth session report"""
        session_duration = time.time() - self.session_start
        
        return {
            "session_duration": round(session_duration, 1),
            "actions_performed": len(self.behavior_engine.action_history),
            "detection_score": self.behavior_engine.detection_score,
            "stealth_metrics": self.behavior_engine.get_stealth_metrics(),
            "fingerprint_rotations": len([
                a for a in self.behavior_engine.action_history 
                if a.get("type") == "fingerprint_rotation"
            ]),
            "risk_level": self._calculate_risk_level()
        }
    
    def _calculate_risk_level(self) -> str:
        """Calculate overall risk level"""
        score = self.behavior_engine.detection_score
        if score < 0.3:
            return "LOW"
        elif score < 0.6:
            return "MEDIUM"
        elif score < 0.8:
            return "HIGH"
        else:
            return "CRITICAL"


# Integration helpers for existing nexus modules
def create_stealth_engine(
    page_actor: Optional[PageActor] = None,
    mouse_actor: Optional[MouseActor] = None,
    profile: StealthProfile = StealthProfile.HUMAN_CASUAL,
    **kwargs
) -> BehaviorEngine:
    """Factory function to create BehaviorEngine with existing actors"""
    return BehaviorEngine(
        page_actor=page_actor,
        mouse_actor=mouse_actor,
        profile=profile,
        **kwargs
    )


async def enhance_actor_with_stealth(
    actor: Union[PageActor, MouseActor, ElementActor],
    stealth_engine: BehaviorEngine
) -> None:
    """Enhance existing actor with stealth capabilities"""
    # This would monkey-patch or wrap actor methods with stealth behavior
    # Implementation depends on the specific actor structure
    pass


# Export main classes
__all__ = [
    "BehaviorEngine",
    "StealthOrchestrator",
    "StealthProfile",
    "FingerprintProfile",
    "BehaviorProfile",
    "create_stealth_engine",
    "enhance_actor_with_stealth"
]