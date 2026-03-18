"""
nexus/extraction/data_pipeline.py

Advanced Data Extraction Pipeline with schema validation, automatic pagination handling,
and incremental scraping capabilities. Integrates with the existing nexus codebase.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Callable, Awaitable
from urllib.parse import urlparse, urljoin

import jsonschema
from jsonschema import ValidationError, Draft7Validator

from nexus.actor.page import Page
from nexus.actor.element import Element


logger = logging.getLogger(__name__)


class PaginationType(Enum):
    """Types of pagination patterns the pipeline can detect and handle."""
    NEXT_BUTTON = "next_button"
    LOAD_MORE = "load_more"
    INFINITE_SCROLL = "infinite_scroll"
    PAGE_NUMBERS = "page_numbers"
    URL_PATTERN = "url_pattern"
    CUSTOM = "custom"


class ExtractionStatus(Enum):
    """Status of an extraction operation."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class ExtractionField:
    """Configuration for extracting a single field from a page."""
    name: str
    selector: str
    attribute: Optional[str] = None  # If None, extracts text content
    multiple: bool = False  # If True, extracts all matching elements
    required: bool = True
    default: Any = None
    transform: Optional[Callable[[str], Any]] = None
    retry_count: int = 3
    retry_delay: float = 0.5


@dataclass
class PaginationConfig:
    """Configuration for handling pagination."""
    type: PaginationType
    selector: Optional[str] = None  # For button-based pagination
    url_pattern: Optional[str] = None  # For URL-based pagination
    max_pages: int = 100
    scroll_pause: float = 2.0  # For infinite scroll
    scroll_increment: int = 500  # Pixels to scroll each time
    custom_handler: Optional[Callable[[Page], Awaitable[bool]]] = None


@dataclass
class ExtractionSchema:
    """Schema definition for data extraction."""
    fields: List[ExtractionField]
    container_selector: Optional[str] = None  # Selector for repeating items
    item_selector: Optional[str] = None  # Selector for individual items within container
    validation_schema: Optional[Dict[str, Any]] = None  # JSON Schema for validation
    pagination: Optional[PaginationConfig] = None


@dataclass
class ExtractionResult:
    """Result of a data extraction operation."""
    data: List[Dict[str, Any]]
    status: ExtractionStatus
    pages_scraped: int = 0
    items_extracted: int = 0
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class IncrementalState:
    """State tracking for incremental scraping."""
    last_url: Optional[str] = None
    last_item_ids: Set[str] = field(default_factory=set)
    last_timestamp: Optional[datetime] = None
    fingerprint_hashes: Set[str] = field(default_factory=set)


class DataExtractor:
    """Handles the actual extraction of data from pages using configured rules."""
    
    def __init__(self, schema: ExtractionSchema):
        self.schema = schema
        self._validator = None
        if schema.validation_schema:
            self._validator = Draft7Validator(schema.validation_schema)
    
    async def extract_from_page(self, page: Page) -> List[Dict[str, Any]]:
        """Extract data from a single page based on the schema."""
        extracted_items = []
        
        try:
            # If container selector is specified, extract multiple items
            if self.schema.container_selector:
                containers = await page.get_elements(self.schema.container_selector)
                
                for container in containers:
                    item_data = await self._extract_item(container)
                    if item_data:
                        extracted_items.append(item_data)
            else:
                # Extract single item from page
                item_data = await self._extract_item(page)
                if item_data:
                    extracted_items.append(item_data)
        
        except Exception as e:
            logger.error(f"Error extracting data from page: {e}")
            raise
        
        return extracted_items
    
    async def _extract_item(self, context: Union[Page, Element]) -> Optional[Dict[str, Any]]:
        """Extract a single item from a page or container element."""
        item_data = {}
        errors = []
        
        for field_config in self.schema.fields:
            value = await self._extract_field(context, field_config)
            
            if value is None and field_config.required:
                errors.append(f"Required field '{field_config.name}' not found")
                if field_config.default is not None:
                    value = field_config.default
            elif value is not None and field_config.transform:
                try:
                    value = field_config.transform(value)
                except Exception as e:
                    logger.warning(f"Transform failed for field {field_config.name}: {e}")
            
            if value is not None:
                item_data[field_config.name] = value
        
        # Validate against schema if provided
        if self._validator and item_data:
            try:
                self._validator.validate(item_data)
            except ValidationError as e:
                logger.warning(f"Schema validation failed: {e.message}")
                errors.append(f"Validation error: {e.message}")
        
        # Return item if we have at least some data
        return item_data if item_data else None
    
    async def _extract_field(self, context: Union[Page, Element], field_config: ExtractionField) -> Any:
        """Extract a single field with retry logic."""
        for attempt in range(field_config.retry_count):
            try:
                if field_config.multiple:
                    elements = await context.get_elements(field_config.selector)
                    values = []
                    for element in elements:
                        value = await self._get_element_value(element, field_config.attribute)
                        if value:
                            values.append(value)
                    return values if values else None
                else:
                    element = await context.get_element(field_config.selector)
                    if element:
                        return await self._get_element_value(element, field_config.attribute)
                
                # If we get here, element wasn't found
                if attempt < field_config.retry_count - 1:
                    await asyncio.sleep(field_config.retry_delay)
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for field {field_config.name}: {e}")
                if attempt < field_config.retry_count - 1:
                    await asyncio.sleep(field_config.retry_delay)
        
        return None
    
    async def _get_element_value(self, element: Element, attribute: Optional[str]) -> Optional[str]:
        """Get value from an element based on attribute configuration."""
        try:
            if attribute is None:
                return await element.get_text()
            elif attribute == "html":
                return await element.get_attribute("outerHTML")
            else:
                return await element.get_attribute(attribute)
        except Exception as e:
            logger.warning(f"Failed to get element value: {e}")
            return None


class PaginationHandler:
    """Handles different types of pagination patterns."""
    
    def __init__(self, page: Page, config: PaginationConfig):
        self.page = page
        self.config = config
        self._current_page = 1
        self._visited_urls: Set[str] = set()
    
    async def has_next_page(self) -> bool:
        """Check if there's a next page available."""
        if self._current_page >= self.config.max_pages:
            return False
        
        try:
            if self.config.type == PaginationType.NEXT_BUTTON:
                return await self._handle_next_button()
            elif self.config.type == PaginationType.LOAD_MORE:
                return await self._handle_load_more()
            elif self.config.type == PaginationType.INFINITE_SCROLL:
                return await self._handle_infinite_scroll()
            elif self.config.type == PaginationType.PAGE_NUMBERS:
                return await self._handle_page_numbers()
            elif self.config.type == PaginationType.URL_PATTERN:
                return await self._handle_url_pattern()
            elif self.config.type == PaginationType.CUSTOM and self.config.custom_handler:
                return await self.config.custom_handler(self.page)
        except Exception as e:
            logger.error(f"Pagination error: {e}")
        
        return False
    
    async def go_to_next_page(self) -> bool:
        """Navigate to the next page."""
        try:
            if self.config.type == PaginationType.NEXT_BUTTON:
                next_button = await self.page.get_element(self.config.selector)
                if next_button:
                    await next_button.click()
                    await asyncio.sleep(1)  # Wait for page load
                    self._current_page += 1
                    return True
            
            elif self.config.type == PaginationType.LOAD_MORE:
                load_more = await self.page.get_element(self.config.selector)
                if load_more:
                    await load_more.click()
                    await asyncio.sleep(self.config.scroll_pause)
                    self._current_page += 1
                    return True
            
            elif self.config.type == PaginationType.PAGE_NUMBERS:
                next_page_num = self._current_page + 1
                next_page_selector = f"{self.config.selector}:contains('{next_page_num}')"
                next_page = await self.page.get_element(next_page_selector)
                if next_page:
                    await next_page.click()
                    await asyncio.sleep(1)
                    self._current_page = next_page_num
                    return True
            
            elif self.config.type == PaginationType.URL_PATTERN:
                next_url = self.config.url_pattern.format(page=self._current_page + 1)
                if next_url not in self._visited_urls:
                    await self.page.goto(next_url)
                    self._visited_urls.add(next_url)
                    self._current_page += 1
                    return True
            
            elif self.config.type == PaginationType.INFINITE_SCROLL:
                # Infinite scroll doesn't navigate, just loads more content
                self._current_page += 1
                return True
                
        except Exception as e:
            logger.error(f"Failed to navigate to next page: {e}")
        
        return False
    
    async def _handle_next_button(self) -> bool:
        """Handle next button pagination."""
        if not self.config.selector:
            return False
        
        next_button = await self.page.get_element(self.config.selector)
        if next_button:
            is_disabled = await next_button.get_attribute("disabled")
            aria_disabled = await next_button.get_attribute("aria-disabled")
            
            if is_disabled or aria_disabled == "true":
                return False
            
            # Check if button is visible and clickable
            is_visible = await next_button.is_visible()
            return is_visible
        
        return False
    
    async def _handle_load_more(self) -> bool:
        """Handle load more button pagination."""
        if not self.config.selector:
            return False
        
        load_more = await self.page.get_element(self.config.selector)
        if load_more:
            is_visible = await load_more.is_visible()
            return is_visible
        
        return False
    
    async def _handle_infinite_scroll(self) -> bool:
        """Handle infinite scroll pagination."""
        # Get current scroll position
        scroll_height = await self.page.evaluate("document.body.scrollHeight")
        current_scroll = await self.page.evaluate("window.pageYOffset + window.innerHeight")
        
        # Scroll to bottom
        await self.page.evaluate(f"window.scrollTo(0, {scroll_height})")
        await asyncio.sleep(self.config.scroll_pause)
        
        # Check if new content loaded
        new_scroll_height = await self.page.evaluate("document.body.scrollHeight")
        return new_scroll_height > scroll_height
    
    async def _handle_page_numbers(self) -> bool:
        """Handle page number pagination."""
        if not self.config.selector:
            return False
        
        next_page_num = self._current_page + 1
        next_page_selector = f"{self.config.selector}[data-page='{next_page_num}'], " \
                            f"{self.config.selector}:contains('{next_page_num}')"
        
        next_page = await self.page.get_element(next_page_selector)
        return next_page is not None
    
    async def _handle_url_pattern(self) -> bool:
        """Handle URL pattern pagination."""
        if not self.config.url_pattern:
            return False
        
        next_url = self.config.url_pattern.format(page=self._current_page + 1)
        return next_url not in self._visited_urls
    
    def reset(self):
        """Reset pagination state."""
        self._current_page = 1
        self._visited_urls.clear()


class IncrementalScraper:
    """Handles incremental scraping to only fetch new or changed data."""
    
    def __init__(self, state_file: Optional[str] = None):
        self.state_file = state_file
        self.state = self._load_state()
    
    def _load_state(self) -> IncrementalState:
        """Load incremental state from file."""
        if self.state_file:
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    state = IncrementalState()
                    state.last_url = data.get('last_url')
                    state.last_item_ids = set(data.get('last_item_ids', []))
                    if data.get('last_timestamp'):
                        state.last_timestamp = datetime.fromisoformat(data['last_timestamp'])
                    state.fingerprint_hashes = set(data.get('fingerprint_hashes', []))
                    return state
            except (FileNotFoundError, json.JSONDecodeError):
                pass
        
        return IncrementalState()
    
    def save_state(self):
        """Save incremental state to file."""
        if self.state_file:
            data = {
                'last_url': self.state.last_url,
                'last_item_ids': list(self.state.last_item_ids),
                'last_timestamp': self.state.last_timestamp.isoformat() if self.state.last_timestamp else None,
                'fingerprint_hashes': list(self.state.fingerprint_hashes)
            }
            
            try:
                with open(self.state_file, 'w') as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save incremental state: {e}")
    
    def generate_item_fingerprint(self, item: Dict[str, Any]) -> str:
        """Generate a fingerprint for an item to detect changes."""
        # Create a stable string representation for hashing
        sorted_items = sorted(item.items())
        content = json.dumps(sorted_items, sort_keys=True)
        return str(hash(content))
    
    def is_item_new_or_changed(self, item: Dict[str, Any], item_id: Optional[str] = None) -> bool:
        """Check if an item is new or has changed since last scrape."""
        fingerprint = self.generate_item_fingerprint(item)
        
        # Check by ID if provided
        if item_id and item_id in self.state.last_item_ids:
            # Item exists, check if changed
            if fingerprint in self.state.fingerprint_hashes:
                return False  # Item hasn't changed
            else:
                return True  # Item has changed
        
        # Check by fingerprint
        return fingerprint not in self.state.fingerprint_hashes
    
    def update_state(self, items: List[Dict[str, Any]], item_ids: Optional[List[str]] = None):
        """Update state with newly scraped items."""
        for i, item in enumerate(items):
            fingerprint = self.generate_item_fingerprint(item)
            self.state.fingerprint_hashes.add(fingerprint)
            
            if item_ids and i < len(item_ids):
                self.state.last_item_ids.add(item_ids[i])
        
        self.state.last_timestamp = datetime.now()
        self.save_state()


class DataPipeline:
    """
    Advanced data extraction pipeline with schema validation, automatic pagination,
    and incremental scraping capabilities.
    
    Example usage:
        schema = ExtractionSchema(
            fields=[
                ExtractionField(name="title", selector="h1"),
                ExtractionField(name="price", selector=".price", transform=lambda x: float(x.replace("$", ""))),
                ExtractionField(name="description", selector=".description", required=False)
            ],
            container_selector=".product-item",
            pagination=PaginationConfig(type=PaginationType.NEXT_BUTTON, selector=".next-page")
        )
        
        pipeline = DataPipeline(schema, incremental_state_file="state.json")
        result = await pipeline.run(page)
    """
    
    def __init__(
        self,
        schema: ExtractionSchema,
        incremental_state_file: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        self.schema = schema
        self.extractor = DataExtractor(schema)
        self.incremental_scraper = IncrementalScraper(incremental_state_file)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Statistics
        self._stats = {
            'total_items': 0,
            'new_items': 0,
            'changed_items': 0,
            'pages_scraped': 0,
            'errors': 0
        }
    
    async def run(
        self,
        page: Page,
        incremental: bool = False,
        item_id_field: Optional[str] = None
    ) -> ExtractionResult:
        """
        Run the data extraction pipeline.
        
        Args:
            page: The page to extract data from
            incremental: If True, only extract new/changed items
            item_id_field: Field name to use as item ID for incremental scraping
            
        Returns:
            ExtractionResult containing extracted data and metadata
        """
        result = ExtractionResult(
            data=[],
            status=ExtractionStatus.RUNNING,
            metadata={'incremental': incremental}
        )
        
        try:
            all_items = []
            
            # Handle pagination if configured
            if self.schema.pagination:
                pagination_handler = PaginationHandler(page, self.schema.pagination)
                
                while await pagination_handler.has_next_page():
                    # Extract items from current page
                    page_items = await self._extract_page_with_retry(page)
                    
                    # Filter for incremental scraping
                    if incremental:
                        page_items = self._filter_incremental_items(page_items, item_id_field)
                    
                    all_items.extend(page_items)
                    result.pages_scraped += 1
                    
                    # Navigate to next page
                    if not await pagination_handler.go_to_next_page():
                        break
                    
                    # Small delay between pages
                    await asyncio.sleep(0.5)
            else:
                # Single page extraction
                page_items = await self._extract_page_with_retry(page)
                
                if incremental:
                    page_items = self._filter_incremental_items(page_items, item_id_field)
                
                all_items.extend(page_items)
                result.pages_scraped = 1
            
            # Update incremental state
            if incremental and all_items:
                item_ids = [item.get(item_id_field) for item in all_items] if item_id_field else None
                self.incremental_scraper.update_state(all_items, item_ids)
            
            result.data = all_items
            result.items_extracted = len(all_items)
            result.status = ExtractionStatus.COMPLETED
            
            # Update metadata
            result.metadata.update({
                'total_items_scraped': self._stats['total_items'],
                'new_items': self._stats['new_items'],
                'changed_items': self._stats['changed_items'],
                'pages_scraped': self._stats['pages_scraped'],
                'errors': self._stats['errors']
            })
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            result.status = ExtractionStatus.FAILED
            result.errors.append(str(e))
        
        return result
    
    async def _extract_page_with_retry(self, page: Page) -> List[Dict[str, Any]]:
        """Extract data from a page with retry logic."""
        for attempt in range(self.max_retries):
            try:
                items = await self.extractor.extract_from_page(page)
                self._stats['total_items'] += len(items)
                self._stats['pages_scraped'] += 1
                return items
                
            except Exception as e:
                logger.warning(f"Extraction attempt {attempt + 1} failed: {e}")
                self._stats['errors'] += 1
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                    # Try refreshing the page
                    try:
                        await page.reload()
                    except:
                        pass
        
        # If all retries failed, return empty list
        return []
    
    def _filter_incremental_items(
        self,
        items: List[Dict[str, Any]],
        item_id_field: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Filter items for incremental scraping, keeping only new/changed items."""
        filtered_items = []
        
        for item in items:
            item_id = item.get(item_id_field) if item_id_field else None
            
            if self.incremental_scraper.is_item_new_or_changed(item, item_id):
                filtered_items.append(item)
                
                if item_id and item_id in self.incremental_scraper.state.last_item_ids:
                    self._stats['changed_items'] += 1
                else:
                    self._stats['new_items'] += 1
        
        return filtered_items
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current pipeline statistics."""
        return self._stats.copy()
    
    def reset_stats(self):
        """Reset pipeline statistics."""
        self._stats = {
            'total_items': 0,
            'new_items': 0,
            'changed_items': 0,
            'pages_scraped': 0,
            'errors': 0
        }


# Utility functions for common extraction patterns
def create_extraction_schema(
    fields: List[Dict[str, Any]],
    container_selector: Optional[str] = None,
    pagination_config: Optional[Dict[str, Any]] = None,
    validation_schema: Optional[Dict[str, Any]] = None
) -> ExtractionSchema:
    """
    Utility function to create an ExtractionSchema from a dictionary configuration.
    
    Example:
        schema = create_extraction_schema(
            fields=[
                {"name": "title", "selector": "h1"},
                {"name": "price", "selector": ".price", "transform": lambda x: float(x.replace("$", ""))}
            ],
            container_selector=".product",
            pagination_config={"type": "next_button", "selector": ".next"}
        )
    """
    extraction_fields = []
    for field_config in fields:
        extraction_fields.append(ExtractionField(**field_config))
    
    pagination = None
    if pagination_config:
        pagination = PaginationConfig(**pagination_config)
    
    return ExtractionSchema(
        fields=extraction_fields,
        container_selector=container_selector,
        pagination=pagination,
        validation_schema=validation_schema
    )


def auto_detect_pagination(page: Page) -> Optional[PaginationConfig]:
    """
    Attempt to automatically detect pagination patterns on a page.
    Returns a PaginationConfig if detected, None otherwise.
    """
    # Common pagination selectors
    common_selectors = [
        ".next", ".next-page", ".pagination .next",
        "a[rel='next']", "li.next a", "a.next",
        ".load-more", ".show-more", ".btn-load-more",
        "[data-page]", ".page-next"
    ]
    
    # This is a simplified detection - in production you'd want more sophisticated detection
    # including checking for infinite scroll, URL patterns, etc.
    
    for selector in common_selectors:
        # This would need to be implemented with actual element checking
        # For now, return a basic config
        pass
    
    return None


# Integration with existing nexus agent system
class DataPipelineAgent:
    """
    Wrapper to integrate DataPipeline with the nexus agent system.
    Can be used as a tool or skill within the agent framework.
    """
    
    def __init__(self, pipeline: DataPipeline):
        self.pipeline = pipeline
    
    async def extract_data(
        self,
        page: Page,
        incremental: bool = False,
        item_id_field: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract data using the pipeline and return in agent-compatible format.
        
        Returns:
            Dictionary with extraction results and metadata for the agent
        """
        result = await self.pipeline.run(page, incremental, item_id_field)
        
        return {
            "success": result.status == ExtractionStatus.COMPLETED,
            "data": result.data,
            "metadata": {
                "items_extracted": result.items_extracted,
                "pages_scraped": result.pages_scraped,
                "status": result.status.value,
                "errors": result.errors,
                **result.metadata
            }
        }
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return capabilities for agent discovery."""
        return {
            "name": "data_extraction",
            "description": "Extract structured data from web pages with schema validation",
            "supports_pagination": self.pipeline.schema.pagination is not None,
            "supports_incremental": True,
            "schema_fields": [f.name for f in self.pipeline.schema.fields]
        }