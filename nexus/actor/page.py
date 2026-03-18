"""Page class for page-level operations with stealth capabilities."""

from typing import TYPE_CHECKING, TypeVar, Dict, Optional, Any, List, Tuple, AsyncGenerator, Union, Type
import asyncio
import time
import psutil
import random
import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
import os
from urllib.parse import urlparse
from pydantic import BaseModel, ValidationError, create_model
from pydantic.fields import FieldInfo

from nexus import logger
from nexus.actor.utils import get_key_info
from nexus.dom.serializer.serializer import DOMTreeSerializer
from nexus.dom.service import DomService
from nexus.llm.messages import SystemMessage, UserMessage

T = TypeVar('T', bound=BaseModel)


class RateLimitState(Enum):
    """State of rate limiting for a domain."""
    NORMAL = "normal"
    BACKOFF = "backoff"
    CAPTCHA = "captcha"


@dataclass
class StealthProfile:
    """Behavior profile for stealth operations."""
    name: str
    mouse_speed_range: Tuple[float, float] = (0.1, 0.8)
    typing_speed_range: Tuple[float, float] = (0.05, 0.15)
    error_rate: float = 0.02
    pause_frequency: float = 0.1
    scroll_behavior: str = "natural"
    viewport_variance: float = 0.1
    timezone: str = "America/New_York"
    language: str = "en-US"
    platform: str = "Win32"
    hardware_concurrency: int = 8
    device_memory: int = 8
    max_touch_points: int = 0
    do_not_track: bool = True
    cookie_enabled: bool = True
    webgl_vendor: str = "Google Inc. (NVIDIA)"
    webgl_renderer: str = "ANGLE (NVIDIA, NVIDIA GeForce RTX 3080 Direct3D11 vs_5_0 ps_5_0, D3D11)"
    canvas_hash: str = ""
    audio_hash: str = ""
    fonts: List[str] = field(default_factory=lambda: [
        "Arial", "Verdana", "Times New Roman", "Courier New", "Georgia", 
        "Palatino Linotype", "Book Antiqua", "Lucida Console", "Monaco"
    ])
    plugins: List[Dict[str, str]] = field(default_factory=lambda: [
        {"name": "Chrome PDF Plugin", "filename": "internal-pdf-viewer"},
        {"name": "Chrome PDF Viewer", "filename": "mhjfbmdgcfjbbpaeojofohoefgiehjai"},
        {"name": "Native Client", "filename": "internal-nacl-plugin"}
    ])


@dataclass
class Fingerprint:
    """Complete browser fingerprint."""
    user_agent: str
    viewport: Dict[str, int]
    screen: Dict[str, int]
    device_pixel_ratio: float
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
    max_touch_points: int
    do_not_track: bool
    cookie_enabled: bool
    color_depth: int
    pixel_ratio: float
    touch_support: bool
    audio_context: Dict[str, Any]
    web_rtc_ips: List[str]
    battery_info: Dict[str, Any]
    connection_type: str
    headers_order: List[str]


@dataclass
class DomainRateLimiter:
    """Token bucket rate limiter for a specific domain."""
    tokens: float = 10.0
    max_tokens: float = 10.0
    refill_rate: float = 2.0  # tokens per second
    last_refill: float = 0.0
    state: RateLimitState = RateLimitState.NORMAL
    backoff_until: float = 0.0
    consecutive_errors: int = 0
    last_response_time: float = 0.0
    response_times: list = None
    
    def __post_init__(self):
        self.response_times = []
        self.last_refill = time.time()
    
    def refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.max_tokens, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
    
    def acquire(self, tokens: float = 1.0) -> bool:
        """Try to acquire tokens. Returns True if successful."""
        if self.state == RateLimitState.CAPTCHA:
            return False
        
        if self.state == RateLimitState.BACKOFF:
            if time.time() < self.backoff_until:
                return False
            else:
                self.state = RateLimitState.NORMAL
                self.consecutive_errors = 0
        
        self.refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def record_response(self, status_code: int, response_time: float, headers: Dict[str, str] = None):
        """Record response and adjust rate limiting based on response patterns."""
        self.last_response_time = time.time()
        self.response_times.append(response_time)
        if len(self.response_times) > 10:
            self.response_times.pop(0)
        
        # Check for rate limit headers
        if headers:
            if 'Retry-After' in headers:
                try:
                    retry_after = int(headers['Retry-After'])
                    self.backoff_until = time.time() + retry_after
                    self.state = RateLimitState.BACKOFF
                    logger.warning(f"Rate limited by Retry-After header: {retry_after}s")
                except ValueError:
                    pass
            
            # Check for common rate limit headers
            rate_limit_headers = ['X-RateLimit-Remaining', 'X-Rate-Limit-Remaining']
            for header in rate_limit_headers:
                if header in headers:
                    try:
                        remaining = int(headers[header])
                        if remaining == 0:
                            self.backoff_until = time.time() + 60  # Default 60s backoff
                            self.state = RateLimitState.BACKOFF
                            logger.warning(f"Rate limited by {header} header")
                    except ValueError:
                        pass
        
        # Check for HTTP status codes indicating rate limiting
        if status_code == 429:  # Too Many Requests
            self.consecutive_errors += 1
            backoff_time = min(300, 2 ** self.consecutive_errors)  # Exponential backoff, max 5 minutes
            self.backoff_until = time.time() + backoff_time
            self.state = RateLimitState.BACKOFF
            logger.warning(f"Rate limited (429). Backing off for {backoff_time}s")
        elif status_code == 403 and 'captcha' in str(headers).lower():
            self.state = RateLimitState.CAPTCHA
            logger.error("CAPTCHA detected. Stopping requests to this domain.")
        elif 200 <= status_code < 300:
            self.consecutive_errors = 0
            # Gradually increase rate on success
            self.refill_rate = min(10.0, self.refill_rate * 1.1)
        elif status_code >= 500:
            self.consecutive_errors += 1
            if self.consecutive_errors > 3:
                self.backoff_until = time.time() + 30
                self.state = RateLimitState.BACKOFF
    
    def adjust_for_system_resources(self, cpu_percent: float, memory_percent: float):
        """Adjust rate limiting based on system resource usage."""
        # Reduce rate if system is under heavy load
        if cpu_percent > 80 or memory_percent > 80:
            self.refill_rate = max(0.5, self.refill_rate * 0.8)
            self.max_tokens = max(2.0, self.max_tokens * 0.9)
        elif cpu_percent < 30 and memory_percent < 50:
            # Increase rate if system has capacity
            self.refill_rate = min(20.0, self.refill_rate * 1.2)
            self.max_tokens = min(50.0, self.max_tokens * 1.1)


class FingerprintGenerator:
    """Generates realistic browser fingerprints with randomization."""
    
    USER_AGENTS = [
        # Chrome on Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        # Chrome on Mac
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        # Chrome on Linux
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        # Firefox on Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        # Firefox on Mac
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
        # Safari on Mac
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    ]
    
    VIEWPORTS = [
        {"width": 1920, "height": 1080},
        {"width": 1366, "height": 768},
        {"width": 1440, "height": 900},
        {"width": 1536, "height": 864},
        {"width": 1280, "height": 720},
        {"width": 2560, "height": 1440},
        {"width": 1680, "height": 1050},
        {"width": 1600, "height": 900},
    ]
    
    SCREEN_RESOLUTIONS = [
        {"width": 1920, "height": 1080},
        {"width": 1366, "height": 768},
        {"width": 1440, "height": 900},
        {"width": 1536, "height": 864},
        {"width": 1280, "height": 720},
        {"width": 2560, "height": 1440},
        {"width": 1680, "height": 1050},
        {"width": 1600, "height": 900},
        {"width": 3840, "height": 2160},
        {"width": 2560, "height": 1080},
    ]
    
    TIMEZONES = [
        "America/New_York",
        "America/Chicago",
        "America/Denver",
        "America/Los_Angeles",
        "Europe/London",
        "Europe/Paris",
        "Europe/Berlin",
        "Asia/Tokyo",
        "Asia/Shanghai",
        "Australia/Sydney",
    ]
    
    LANGUAGES = [
        "en-US",
        "en-GB",
        "en-CA",
        "en-AU",
        "es-ES",
        "es-MX",
        "fr-FR",
        "fr-CA",
        "de-DE",
        "it-IT",
        "pt-BR",
        "ja-JP",
        "zh-CN",
        "zh-TW",
        "ko-KR",
    ]
    
    @classmethod
    def generate(cls, profile: StealthProfile = None) -> Fingerprint:
        """Generate a complete browser fingerprint."""
        if profile is None:
            profile = StealthProfile(name="default")
        
        user_agent = random.choice(cls.USER_AGENTS)
        viewport = random.choice(cls.VIEWPORTS)
        screen = random.choice(cls.SCREEN_RESOLUTIONS)
        
        # Add variance to viewport based on profile
        viewport_variance = profile.viewport_variance
        viewport = {
            "width": int(viewport["width"] * random.uniform(1 - viewport_variance, 1 + viewport_variance)),
            "height": int(viewport["height"] * random.uniform(1 - viewport_variance, 1 + viewport_variance)),
        }
        
        # Ensure viewport doesn't exceed screen
        viewport["width"] = min(viewport["width"], screen["width"])
        viewport["height"] = min(viewport["height"], screen["height"])
        
        device_pixel_ratio = random.choice([1, 1.25, 1.5, 2])
        
        # Generate WebGL hashes
        canvas_hash = hashlib.md5(f"canvas_{random.randint(1000000, 9999999)}".encode()).hexdigest()
        audio_hash = hashlib.md5(f"audio_{random.randint(1000000, 9999999)}".encode()).hexdigest()
        
        # Generate audio context fingerprint
        audio_context = {
            "sampleRate": random.choice([44100, 48000]),
            "channelCount": random.choice([2, 6]),
            "latencyHint": random.choice(["interactive", "playback", "balanced"]),
        }
        
        # Generate WebRTC IPs (simulated)
        web_rtc_ips = [
            f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}",
            f"10.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 255)}",
        ]
        
        # Generate battery info
        battery_info = {
            "charging": random.choice([True, False]),
            "chargingTime": random.randint(0, 3600) if random.random() > 0.5 else 0,
            "dischargingTime": random.randint(3600, 28800) if random.random() > 0.5 else float('inf'),
            "level": random.uniform(0.1, 1.0),
        }
        
        # Connection type
        connection_type = random.choice(["wifi", "cellular", "ethernet", "unknown"])
        
        # Headers order (simulated)
        headers_order = [
            "Host",
            "Connection",
            "Cache-Control",
            "sec-ch-ua",
            "sec-ch-ua-mobile",
            "sec-ch-ua-platform",
            "Upgrade-Insecure-Requests",
            "User-Agent",
            "Accept",
            "Sec-Fetch-Site",
            "Sec-Fetch-Mode",
            "Sec-Fetch-User",
            "Sec-Fetch-Dest",
            "Referer",
            "Accept-Encoding",
            "Accept-Language",
        ]
        random.shuffle(headers_order)
        
        return Fingerprint(
            user_agent=user_agent,
            viewport=viewport,
            screen=screen,
            device_pixel_ratio=device_pixel_ratio,
            platform=profile.platform,
            language=profile.language,
            languages=[profile.language] + random.sample(cls.LANGUAGES, min(3, len(cls.LANGUAGES))),
            timezone=profile.timezone,
            webgl_vendor=profile.webgl_vendor,
            webgl_renderer=profile.webgl_renderer,
            canvas_hash=canvas_hash,
            audio_hash=audio_hash,
            fonts=profile.fonts,
            plugins=profile.plugins,
            hardware_concurrency=profile.hardware_concurrency,
            device_memory=profile.device_memory,
            max_touch_points=profile.max_touch_points,
            do_not_track=profile.do_not_track,
            cookie_enabled=profile.cookie_enabled,
            color_depth=random.choice([24, 32]),
            pixel_ratio=device_pixel_ratio,
            touch_support=profile.max_touch_points > 0,
            audio_context=audio_context,
            web_rtc_ips=web_rtc_ips,
            battery_info=battery_info,
            connection_type=connection_type,
            headers_order=headers_order,
        )


@dataclass
class PaginationState:
    """State for pagination tracking."""
    current_page: int = 1
    total_pages: Optional[int] = None
    next_page_selectors: List[str] = field(default_factory=list)
    page_hashes: Dict[int, str] = field(default_factory=dict)
    last_page_hash: Optional[str] = None
    has_more_pages: bool = True
    pagination_type: Optional[str] = None  # "numbered", "next_button", "infinite_scroll", "load_more"


@dataclass
class ExtractionState:
    """State for incremental extraction."""
    url: str
    schema_hash: str
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    page_hashes: Dict[str, str] = field(default_factory=dict)
    last_extraction_time: float = 0.0
    change_detection_hash: Optional[str] = None


class Page:
    """Page class for page-level operations with stealth capabilities."""
    
    def __init__(self, page, dom_service: DomService, llm, config: Dict[str, Any] = None):
        self.page = page
        self.dom_service = dom_service
        self.llm = llm
        self.config = config or {}
        self.fingerprint = FingerprintGenerator.generate()
        self.rate_limiters: Dict[str, DomainRateLimiter] = defaultdict(DomainRateLimiter)
        self.pagination_states: Dict[str, PaginationState] = {}
        self.extraction_states: Dict[str, ExtractionState] = {}
        self._setup_stealth()
    
    def _setup_stealth(self):
        """Setup stealth mode with the current fingerprint."""
        # This would inject stealth scripts and set the fingerprint
        # Implementation depends on the underlying browser automation library
        pass
    
    async def extract(self, prompt: str = None, **kwargs) -> Dict[str, Any]:
        """Extract data from the current page using LLM."""
        # Get DOM content
        dom_content = await self.dom_service.get_content()
        
        # Create extraction prompt
        if prompt is None:
            prompt = "Extract all structured data from this page. Return as JSON."
        
        messages = [
            SystemMessage(content="You are a data extraction assistant. Extract structured data from the provided HTML content."),
            UserMessage(content=f"HTML Content:\n{dom_content}\n\nExtraction Task: {prompt}")
        ]
        
        # Call LLM
        response = await self.llm.chat(messages, **kwargs)
        
        try:
            # Parse JSON response
            data = json.loads(response.content)
            return data
        except json.JSONDecodeError:
            logger.error("Failed to parse LLM response as JSON")
            return {}
    
    async def extract_structured_data(
        self, 
        schema: Union[Type[T], Dict[str, Any]], 
        prompt: str = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ) -> T:
        """
        Extract structured data with schema validation and automatic retry for missing fields.
        
        Args:
            schema: Pydantic model or JSON schema dict for validation
            prompt: Custom extraction prompt
            max_retries: Maximum number of retries for missing fields
            retry_delay: Delay between retries in seconds
            **kwargs: Additional arguments for LLM call
        
        Returns:
            Validated data instance
        """
        # Convert JSON schema to Pydantic model if needed
        if isinstance(schema, dict):
            schema = self._json_schema_to_pydantic(schema)
        
        schema_name = schema.__name__ if hasattr(schema, '__name__') else str(schema)
        logger.info(f"Extracting structured data with schema: {schema_name}")
        
        last_error = None
        extracted_data = {}
        
        for attempt in range(max_retries):
            try:
                # Extract data
                if attempt == 0:
                    # First attempt: use original prompt
                    current_prompt = prompt or f"Extract data according to this schema: {schema_name}"
                else:
                    # Subsequent attempts: ask for missing fields
                    missing_fields = self._get_missing_fields(extracted_data, schema)
                    if not missing_fields:
                        break
                    current_prompt = f"{prompt}\n\nIMPORTANT: The following fields are missing or invalid: {', '.join(missing_fields)}. Please extract them."
                
                raw_data = await self.extract(prompt=current_prompt, **kwargs)
                
                # Validate against schema
                if isinstance(raw_data, dict):
                    extracted_data = raw_data
                elif isinstance(raw_data, list) and len(raw_data) > 0:
                    extracted_data = raw_data[0] if isinstance(raw_data[0], dict) else {}
                else:
                    extracted_data = {}
                
                # Try to create instance
                instance = schema(**extracted_data)
                logger.info(f"Successfully extracted and validated data on attempt {attempt + 1}")
                return instance
                
            except ValidationError as e:
                last_error = e
                logger.warning(f"Validation failed on attempt {attempt + 1}: {e}")
                
                if attempt < max_retries - 1:
                    # Extract missing fields from validation error
                    missing_fields = []
                    for error in e.errors():
                        if error['type'] == 'value_error.missing':
                            field_name = error['loc'][0] if error['loc'] else 'unknown'
                            missing_fields.append(field_name)
                    
                    if missing_fields:
                        logger.info(f"Retrying for missing fields: {missing_fields}")
                        await asyncio.sleep(retry_delay)
                    else:
                        # No missing fields, but other validation errors
                        logger.error(f"Non-missing field validation errors: {e}")
                        break
                else:
                    logger.error(f"Failed to extract valid data after {max_retries} attempts")
                    raise last_error
        
        # If we get here, all retries failed
        if last_error:
            raise last_error
        else:
            raise ValueError(f"Failed to extract data matching schema {schema_name}")
    
    def _json_schema_to_pydantic(self, json_schema: Dict[str, Any]) -> Type[BaseModel]:
        """Convert JSON schema to Pydantic model dynamically."""
        properties = json_schema.get('properties', {})
        required = json_schema.get('required', [])
        
        # Create field definitions
        field_definitions = {}
        for field_name, field_schema in properties.items():
            field_type = self._json_type_to_python_type(field_schema.get('type', 'string'))
            field_description = field_schema.get('description', '')
            
            # Handle default values
            default = ... if field_name in required else field_schema.get('default')
            
            # Create Field with metadata
            field_definitions[field_name] = (
                field_type, 
                FieldInfo(default=default, description=field_description)
            )
        
        # Create dynamic model
        model_name = json_schema.get('title', 'DynamicModel')
        return create_model(model_name, **field_definitions)
    
    def _json_type_to_python_type(self, json_type: str) -> type:
        """Convert JSON schema type to Python type."""
        type_mapping = {
            'string': str,
            'integer': int,
            'number': float,
            'boolean': bool,
            'array': list,
            'object': dict,
            'null': type(None),
        }
        return type_mapping.get(json_type, Any)
    
    def _get_missing_fields(self, data: Dict[str, Any], schema: Type[BaseModel]) -> List[str]:
        """Identify missing fields in extracted data compared to schema."""
        try:
            # Try to create instance to see what's missing
            schema(**data)
            return []
        except ValidationError as e:
            missing_fields = []
            for error in e.errors():
                if error['type'] == 'value_error.missing':
                    field_name = error['loc'][0] if error['loc'] else 'unknown'
                    missing_fields.append(field_name)
            return missing_fields
    
    async def handle_pagination(
        self,
        schema: Union[Type[T], Dict[str, Any]],
        prompt: str = None,
        max_pages: int = 10,
        pagination_selectors: List[str] = None,
        stop_on_duplicate: bool = True,
        **kwargs
    ) -> AsyncGenerator[T, None]:
        """
        Automatically handle pagination and extract data from multiple pages.
        
        Args:
            schema: Pydantic model or JSON schema for validation
            prompt: Extraction prompt
            max_pages: Maximum number of pages to scrape
            pagination_selectors: CSS selectors for pagination elements
            stop_on_duplicate: Stop if we detect duplicate content
            **kwargs: Additional arguments
        
        Yields:
            Extracted data from each page
        """
        url = self.page.url
        pagination_key = hashlib.md5(url.encode()).hexdigest()
        
        # Initialize or get pagination state
        if pagination_key not in self.pagination_states:
            self.pagination_states[pagination_key] = PaginationState()
        
        state = self.pagination_states[pagination_key]
        
        # Detect pagination type if not already detected
        if state.pagination_type is None:
            state.pagination_type = await self._detect_pagination_type()
            logger.info(f"Detected pagination type: {state.pagination_type}")
        
        # Default pagination selectors
        if pagination_selectors is None:
            pagination_selectors = self._get_default_pagination_selectors(state.pagination_type)
        
        page_count = 0
        seen_hashes = set()
        
        while page_count < max_pages and state.has_more_pages:
            page_count += 1
            logger.info(f"Processing page {page_count}")
            
            # Check for duplicate content
            if stop_on_duplicate:
                page_hash = await self._get_page_content_hash()
                if page_hash in seen_hashes:
                    logger.info(f"Duplicate content detected on page {page_count}. Stopping.")
                    break
                seen_hashes.add(page_hash)
                state.page_hashes[state.current_page] = page_hash
            
            # Extract data from current page
            try:
                data = await self.extract_structured_data(schema, prompt, **kwargs)
                yield data
            except Exception as e:
                logger.error(f"Failed to extract data from page {page_count}: {e}")
                break
            
            # Try to navigate to next page
            if page_count < max_pages:
                has_next = await self._navigate_to_next_page(pagination_selectors, state)
                if not has_next:
                    logger.info("No more pages found")
                    state.has_more_pages = False
                    break
                
                # Wait for page to load
                await self._wait_for_page_load()
                state.current_page += 1
        
        # Clean up pagination state if we're done
        if not state.has_more_pages or page_count >= max_pages:
            if pagination_key in self.pagination_states:
                del self.pagination_states[pagination_key]
    
    async def _detect_pagination_type(self) -> str:
        """Detect the type of pagination on the page."""
        # Check for numbered pagination
        numbered_selectors = [
            'nav.pagination',
            'ul.pagination',
            '.pagination',
            '.pager',
            '[class*="pagination"]',
            '[class*="pager"]',
        ]
        
        for selector in numbered_selectors:
            try:
                elements = await self.page.query_selector_all(selector)
                if elements:
                    # Check if it contains numbered links
                    for element in elements:
                        text = await element.text_content()
                        if any(char.isdigit() for char in text):
                            return "numbered"
            except:
                continue
        
        # Check for next button
        next_selectors = [
            'a:has-text("Next")',
            'button:has-text("Next")',
            '[rel="next"]',
            '.next',
            '.next-page',
            '[aria-label*="next" i]',
            '[class*="next"]',
        ]
        
        for selector in next_selectors:
            try:
                element = await self.page.query_selector(selector)
                if element:
                    return "next_button"
            except:
                continue
        
        # Check for infinite scroll
        scroll_selectors = [
            '[class*="infinite"]',
            '[class*="scroll"]',
            '[data-infinite]',
        ]
        
        for selector in scroll_selectors:
            try:
                element = await self.page.query_selector(selector)
                if element:
                    return "infinite_scroll"
            except:
                continue
        
        # Check for load more button
        load_more_selectors = [
            'button:has-text("Load More")',
            'button:has-text("Show More")',
            '.load-more',
            '[class*="load-more"]',
        ]
        
        for selector in load_more_selectors:
            try:
                element = await self.page.query_selector(selector)
                if element:
                    return "load_more"
            except:
                continue
        
        return "unknown"
    
    def _get_default_pagination_selectors(self, pagination_type: str) -> List[str]:
        """Get default selectors based on pagination type."""
        if pagination_type == "numbered":
            return [
                'a:has-text("Next")',
                'button:has-text("Next")',
                '.pagination .next',
                '.pagination a:last-child',
                'li.active + li a',
                '[aria-label="Next page"]',
            ]
        elif pagination_type == "next_button":
            return [
                'a:has-text("Next")',
                'button:has-text("Next")',
                '[rel="next"]',
                '.next',
                '.next-page',
                '[aria-label*="next" i]',
            ]
        elif pagination_type == "load_more":
            return [
                'button:has-text("Load More")',
                'button:has-text("Show More")',
                '.load-more',
                '[class*="load-more"]',
            ]
        else:
            return [
                'a:has-text("Next")',
                'button:has-text("Next")',
                '[rel="next"]',
                '.next',
            ]
    
    async def _navigate_to_next_page(self, selectors: List[str], state: PaginationState) -> bool:
        """Navigate to the next page using provided selectors."""
        for selector in selectors:
            try:
                element = await self.page.query_selector(selector)
                if element:
                    # Check if element is visible and enabled
                    is_visible = await element.is_visible()
                    is_enabled = await element.is_enabled()
                    
                    if is_visible and is_enabled:
                        logger.info(f"Clicking next page with selector: {selector}")
                        await element.click()
                        return True
            except Exception as e:
                logger.debug(f"Selector {selector} failed: {e}")
                continue
        
        return False
    
    async def _wait_for_page_load(self, timeout: float = 10.0):
        """Wait for page to load after navigation."""
        try:
            # Wait for network to be idle
            await self.page.wait_for_load_state("networkidle", timeout=timeout)
        except:
            # Fallback: wait for a short time
            await asyncio.sleep(2)
    
    async def _get_page_content_hash(self) -> str:
        """Get hash of main page content for duplicate detection."""
        try:
            # Get main content area (common selectors)
            content_selectors = [
                'main',
                'article',
                '.content',
                '.main-content',
                '#content',
                '#main',
                '[role="main"]',
            ]
            
            content = ""
            for selector in content_selectors:
                element = await self.page.query_selector(selector)
                if element:
                    content = await element.inner_html()
                    break
            
            if not content:
                # Fallback to body
                body = await self.page.query_selector('body')
                if body:
                    content = await body.inner_html()
            
            # Create hash
            return hashlib.md5(content.encode()).hexdigest()
        except Exception as e:
            logger.warning(f"Failed to get page content hash: {e}")
            return hashlib.md5(str(time.time()).encode()).hexdigest()
    
    async def incremental_scrape(
        self,
        url: str,
        schema: Union[Type[T], Dict[str, Any]],
        prompt: str = None,
        state_file: str = None,
        force_refresh: bool = False,
        **kwargs
    ) -> T:
        """
        Incremental scraping that only fetches new or changed data.
        
        Args:
            url: URL to scrape
            schema: Pydantic model or JSON schema
            prompt: Extraction prompt
            state_file: File to store extraction state
            force_refresh: Force re-extraction even if data hasn't changed
            **kwargs: Additional arguments
        
        Returns:
            Extracted data (cached if unchanged)
        """
        # Create extraction key
        schema_hash = self._get_schema_hash(schema)
        extraction_key = f"{url}_{schema_hash}"
        
        # Load or initialize state
        state = self._load_extraction_state(extraction_key, state_file)
        
        # Check if we need to re-extract
        should_extract = force_refresh or state.extracted_data is None
        
        if not should_extract:
            # Check if page has changed
            current_hash = await self._get_page_content_hash()
            if state.change_detection_hash != current_hash:
                logger.info(f"Page content changed for {url}")
                should_extract = True
            else:
                logger.info(f"Using cached data for {url}")
                # Return cached data as instance
                if isinstance(schema, dict):
                    schema = self._json_schema_to_pydantic(schema)
                return schema(**state.extracted_data)
        
        if should_extract:
            # Navigate to URL if not already there
            if self.page.url != url:
                await self.page.goto(url)
                await self._wait_for_page_load()
            
            # Extract data
            logger.info(f"Extracting data from {url}")
            data = await self.extract_structured_data(schema, prompt, **kwargs)
            
            # Update state
            state.url = url
            state.schema_hash = schema_hash
            state.extracted_data = data.dict() if hasattr(data, 'dict') else data
            state.change_detection_hash = await self._get_page_content_hash()
            state.last_extraction_time = time.time()
            
            # Save state
            self._save_extraction_state(extraction_key, state, state_file)
            
            return data
    
    def _get_schema_hash(self, schema: Union[Type[T], Dict[str, Any]]) -> str:
        """Get hash of schema for state tracking."""
        if isinstance(schema, dict):
            schema_str = json.dumps(schema, sort_keys=True)
        else:
            # For Pydantic model, use its schema
            schema_str = json.dumps(schema.schema(), sort_keys=True)
        
        return hashlib.md5(schema_str.encode()).hexdigest()
    
    def _load_extraction_state(self, key: str, state_file: str = None) -> ExtractionState:
        """Load extraction state from file or memory."""
        if state_file and os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                    if key in data:
                        state_data = data[key]
                        return ExtractionState(**state_data)
            except Exception as e:
                logger.warning(f"Failed to load extraction state: {e}")
        
        # Return new state
        return ExtractionState(url="", schema_hash="")
    
    def _save_extraction_state(self, key: str, state: ExtractionState, state_file: str = None):
        """Save extraction state to file."""
        if state_file:
            try:
                # Load existing state
                data = {}
                if os.path.exists(state_file):
                    with open(state_file, 'r') as f:
                        data = json.load(f)
                
                # Update with new state
                data[key] = {
                    'url': state.url,
                    'schema_hash': state.schema_hash,
                    'extracted_data': state.extracted_data,
                    'page_hashes': state.page_hashes,
                    'last_extraction_time': state.last_extraction_time,
                    'change_detection_hash': state.change_detection_hash,
                }
                
                # Save
                with open(state_file, 'w') as f:
                    json.dump(data, f, indent=2)
                    
            except Exception as e:
                logger.warning(f"Failed to save extraction state: {e}")
    
    async def get_html(self, selector: str = "body") -> str:
        """Get HTML content of an element."""
        try:
            element = await self.page.query_selector(selector)
            if element:
                return await element.inner_html()
        except Exception as e:
            logger.warning(f"Failed to get HTML for selector {selector}: {e}")
        return ""
    
    async def get_page_hash(self, selector: str = "body") -> str:
        """Get hash of page content."""
        html = await self.get_html(selector)
        return hashlib.md5(html.encode()).hexdigest()
    
    # Existing methods would be preserved here
    # ... (all other existing methods from the original file)