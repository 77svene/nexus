"""
Advanced Data Extraction Pipeline with pagination handling and incremental scraping.
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

import jsonschema
from jsonschema import validate, ValidationError

from nexus.actor.page import Page
from nexus.agent.views import AgentOutput

logger = logging.getLogger(__name__)


class ExtractionSchema:
    """JSON Schema wrapper for data validation and extraction rules."""
    
    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema
        self._compiled_schema = None
        
    @property
    def compiled(self):
        if self._compiled_schema is None:
            self._compiled_schema = jsonschema.Draft7Validator(self.schema)
        return self._compiled_schema
    
    def validate(self, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate data against schema. Returns (is_valid, error_message)."""
        try:
            validate(instance=data, schema=self.schema)
            return True, None
        except ValidationError as e:
            return False, str(e)
    
    def get_required_fields(self) -> List[str]:
        """Extract required fields from schema."""
        return self.schema.get("required", [])
    
    def get_field_paths(self) -> List[str]:
        """Get all field paths from schema properties."""
        def extract_paths(obj: Dict, prefix: str = "") -> List[str]:
            paths = []
            for key, value in obj.get("properties", {}).items():
                current_path = f"{prefix}.{key}" if prefix else key
                paths.append(current_path)
                if "properties" in value:
                    paths.extend(extract_paths(value, current_path))
            return paths
        
        return extract_paths(self.schema)


class PaginationHandler:
    """Handles automatic pagination detection and navigation."""
    
    def __init__(self, page: Page):
        self.page = page
        self._pagination_cache: Dict[str, Any] = {}
        
    async def detect_pagination_type(self) -> str:
        """Detect the type of pagination on the current page."""
        # Check for common pagination patterns
        patterns = {
            "next_button": [
                "a:has-text('Next')",
                "button:has-text('Next')",
                "[rel='next']",
                ".pagination .next:not(.disabled)",
                "a[aria-label='Next']"
            ],
            "load_more": [
                "button:has-text('Load More')",
                "button:has-text('Show More')",
                ".load-more",
                "[data-action='load-more']"
            ],
            "infinite_scroll": [
                "[data-infinite-scroll]",
                ".infinite-scroll",
                "[data-pagination='infinite']"
            ],
            "numbered_pages": [
                ".pagination a:not(.next):not(.prev)",
                "nav[aria-label='pagination'] a",
                ".page-numbers a"
            ]
        }
        
        for pagination_type, selectors in patterns.items():
            for selector in selectors:
                try:
                    element = await self.page.query_selector(selector)
                    if element:
                        self._pagination_cache["type"] = pagination_type
                        self._pagination_cache["selector"] = selector
                        return pagination_type
                except Exception:
                    continue
        
        # Default to no pagination
        self._pagination_cache["type"] = "none"
        return "none"
    
    async def get_next_page_url(self) -> Optional[str]:
        """Extract the next page URL based on detected pagination type."""
        pagination_type = self._pagination_cache.get("type")
        
        if pagination_type == "next_button":
            selector = self._pagination_cache.get("selector")
            if selector:
                try:
                    next_element = await self.page.query_selector(selector)
                    if next_element:
                        href = await next_element.get_attribute("href")
                        if href:
                            return self._resolve_url(href)
                except Exception as e:
                    logger.warning(f"Failed to get next page URL: {e}")
        
        elif pagination_type == "numbered_pages":
            # Find current page and get next page number
            try:
                current_page = await self.page.query_selector(".pagination .active, .current")
                if current_page:
                    next_sibling = await current_page.evaluate_handle("el => el.nextElementSibling")
                    if next_sibling:
                        href = await next_sibling.get_attribute("href")
                        if href:
                            return self._resolve_url(href)
            except Exception as e:
                logger.warning(f"Failed to get next numbered page: {e}")
        
        return None
    
    async def has_next_page(self) -> bool:
        """Check if there's a next page available."""
        pagination_type = self._pagination_cache.get("type", await self.detect_pagination_type())
        
        if pagination_type == "none":
            return False
        
        if pagination_type == "next_button":
            selector = self._pagination_cache.get("selector")
            if selector:
                try:
                    next_element = await self.page.query_selector(selector)
                    if next_element:
                        # Check if button is disabled
                        is_disabled = await next_element.evaluate("""
                            el => el.classList.contains('disabled') || 
                                  el.hasAttribute('disabled') || 
                                  el.getAttribute('aria-disabled') === 'true'
                        """)
                        return not is_disabled
                except Exception:
                    pass
        
        elif pagination_type == "load_more":
            selector = self._pagination_cache.get("selector")
            if selector:
                try:
                    load_more_btn = await self.page.query_selector(selector)
                    return load_more_btn is not None
                except Exception:
                    pass
        
        elif pagination_type == "infinite_scroll":
            # Check if we're at the bottom of the page
            try:
                scroll_position = await self.page.evaluate("window.scrollY")
                page_height = await self.page.evaluate("document.body.scrollHeight")
                viewport_height = await self.page.evaluate("window.innerHeight")
                return scroll_position + viewport_height < page_height - 100  # 100px threshold
            except Exception:
                pass
        
        return False
    
    async def navigate_to_next_page(self) -> bool:
        """Navigate to the next page. Returns True if successful."""
        pagination_type = self._pagination_cache.get("type", await self.detect_pagination_type())
        
        try:
            if pagination_type == "next_button":
                selector = self._pagination_cache.get("selector")
                if selector:
                    await self.page.click(selector)
                    await self.page.wait_for_load_state("networkidle")
                    return True
            
            elif pagination_type == "load_more":
                selector = self._pagination_cache.get("selector")
                if selector:
                    await self.page.click(selector)
                    # Wait for new content to load
                    await asyncio.sleep(2)  # Adjust based on typical load time
                    return True
            
            elif pagination_type == "infinite_scroll":
                # Scroll to bottom to trigger loading
                await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await asyncio.sleep(2)  # Wait for content to load
                return True
            
            elif pagination_type == "numbered_pages":
                next_url = await self.get_next_page_url()
                if next_url:
                    await self.page.goto(next_url)
                    await self.page.wait_for_load_state("networkidle")
                    return True
        
        except Exception as e:
            logger.error(f"Failed to navigate to next page: {e}")
        
        return False
    
    def _resolve_url(self, url: str) -> str:
        """Resolve relative URLs to absolute."""
        if url.startswith(("http://", "https://")):
            return url
        
        current_url = self.page.url
        parsed = urlparse(current_url)
        
        if url.startswith("/"):
            return f"{parsed.scheme}://{parsed.netloc}{url}"
        else:
            base_path = "/".join(parsed.path.split("/")[:-1])
            return f"{parsed.scheme}://{parsed.netloc}{base_path}/{url}"


class IncrementalScraper:
    """Handles incremental scraping by tracking seen items."""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path
        self._seen_hashes: Set[str] = set()
        self._load_seen_hashes()
    
    def _load_seen_hashes(self):
        """Load previously seen item hashes from storage."""
        if self.storage_path:
            try:
                with open(self.storage_path, "r") as f:
                    self._seen_hashes = set(json.load(f))
            except (FileNotFoundError, json.JSONDecodeError):
                self._seen_hashes = set()
    
    def _save_seen_hashes(self):
        """Save seen item hashes to storage."""
        if self.storage_path:
            try:
                with open(self.storage_path, "w") as f:
                    json.dump(list(self._seen_hashes), f)
            except Exception as e:
                logger.warning(f"Failed to save seen hashes: {e}")
    
    def generate_item_hash(self, item: Dict[str, Any], hash_fields: Optional[List[str]] = None) -> str:
        """Generate a hash for an item to track changes."""
        if hash_fields:
            # Only hash specified fields
            hash_data = {k: v for k, v in item.items() if k in hash_fields}
        else:
            hash_data = item
        
        # Sort keys for consistent hashing
        hash_str = json.dumps(hash_data, sort_keys=True, default=str)
        return hashlib.md5(hash_str.encode()).hexdigest()
    
    def is_item_new(self, item: Dict[str, Any], hash_fields: Optional[List[str]] = None) -> bool:
        """Check if an item is new (not seen before)."""
        item_hash = self.generate_item_hash(item, hash_fields)
        return item_hash not in self._seen_hashes
    
    def mark_item_seen(self, item: Dict[str, Any], hash_fields: Optional[List[str]] = None):
        """Mark an item as seen."""
        item_hash = self.generate_item_hash(item, hash_fields)
        self._seen_hashes.add(item_hash)
        self._save_seen_hashes()
    
    def filter_new_items(self, items: List[Dict[str, Any]], hash_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Filter list to only include new items."""
        new_items = []
        for item in items:
            if self.is_item_new(item, hash_fields):
                new_items.append(item)
                self.mark_item_seen(item, hash_fields)
        return new_items


class DataExtractionPipeline:
    """Main pipeline for structured data extraction with schema validation and pagination."""
    
    def __init__(
        self,
        page: Page,
        schema: ExtractionSchema,
        extraction_function: Callable[[Page], List[Dict[str, Any]]],
        incremental_scraper: Optional[IncrementalScraper] = None,
        max_pages: int = 10,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
        hash_fields: Optional[List[str]] = None
    ):
        self.page = page
        self.schema = schema
        self.extraction_function = extraction_function
        self.pagination_handler = PaginationHandler(page)
        self.incremental_scraper = incremental_scraper or IncrementalScraper()
        self.max_pages = max_pages
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.hash_fields = hash_fields
        
        self._extracted_data: List[Dict[str, Any]] = []
        self._failed_extractions: List[Dict[str, Any]] = []
    
    async def extract_with_retry(self, item_data: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """Extract data with retry logic for missing fields."""
        for attempt in range(self.retry_attempts):
            try:
                # Validate against schema
                is_valid, error = self.schema.validate(item_data)
                
                if is_valid:
                    return True, item_data, None
                
                # If validation failed, check if it's due to missing required fields
                required_fields = self.schema.get_required_fields()
                missing_fields = [field for field in required_fields if field not in item_data]
                
                if missing_fields and attempt < self.retry_attempts - 1:
                    logger.warning(f"Missing fields {missing_fields}, retrying... (attempt {attempt + 1})")
                    await asyncio.sleep(self.retry_delay)
                    # Re-extract this specific item
                    # This assumes extraction_function can handle partial re-extraction
                    # In practice, you might need a more sophisticated retry mechanism
                    continue
                else:
                    return False, item_data, error
                    
            except Exception as e:
                if attempt < self.retry_attempts - 1:
                    logger.warning(f"Extraction attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(self.retry_delay)
                else:
                    return False, item_data, str(e)
        
        return False, item_data, "Max retry attempts reached"
    
    async def extract_page_data(self) -> List[Dict[str, Any]]:
        """Extract data from current page with validation."""
        try:
            raw_items = await self.extraction_function(self.page)
            validated_items = []
            
            for item in raw_items:
                is_valid, validated_item, error = await self.extract_with_retry(item)
                
                if is_valid:
                    validated_items.append(validated_item)
                else:
                    logger.warning(f"Item validation failed: {error}")
                    self._failed_extractions.append({
                        "item": item,
                        "error": error,
                        "timestamp": datetime.now().isoformat()
                    })
            
            return validated_items
            
        except Exception as e:
            logger.error(f"Failed to extract page data: {e}")
            return []
    
    async def run(self, start_url: Optional[str] = None) -> AgentOutput:
        """Run the complete extraction pipeline."""
        if start_url:
            await self.page.goto(start_url)
            await self.page.wait_for_load_state("networkidle")
        
        current_page = 1
        total_items = 0
        new_items = 0
        
        logger.info(f"Starting extraction pipeline on {self.page.url}")
        
        while current_page <= self.max_pages:
            logger.info(f"Extracting data from page {current_page}")
            
            # Extract data from current page
            page_items = await self.extract_page_data()
            
            if page_items:
                # Filter for new items if incremental scraping is enabled
                if self.incremental_scraper:
                    new_page_items = self.incremental_scraper.filter_new_items(
                        page_items, 
                        self.hash_fields
                    )
                    new_items += len(new_page_items)
                    self._extracted_data.extend(new_page_items)
                else:
                    self._extracted_data.extend(page_items)
                
                total_items += len(page_items)
                logger.info(f"Extracted {len(page_items)} items from page {current_page}")
            
            # Check for next page
            if not await self.pagination_handler.has_next_page():
                logger.info("No more pages available")
                break
            
            # Navigate to next page
            if not await self.pagination_handler.navigate_to_next_page():
                logger.warning("Failed to navigate to next page")
                break
            
            current_page += 1
            
            # Small delay between pages to be respectful
            await asyncio.sleep(0.5)
        
        # Prepare output
        output = AgentOutput(
            result={
                "extracted_data": self._extracted_data,
                "failed_extractions": self._failed_extractions,
                "statistics": {
                    "total_pages_scraped": current_page,
                    "total_items_extracted": total_items,
                    "new_items_found": new_items,
                    "failed_items": len(self._failed_extractions),
                    "success_rate": (total_items - len(self._failed_extractions)) / max(total_items, 1)
                }
            },
            metadata={
                "schema_version": self.schema.schema.get("$schema", "unknown"),
                "extraction_timestamp": datetime.now().isoformat(),
                "source_url": self.page.url
            }
        )
        
        logger.info(f"Extraction completed: {len(self._extracted_data)} items extracted")
        return output


# Utility functions for common extraction patterns
async def extract_table_data(page: Page, table_selector: str = "table") -> List[Dict[str, Any]]:
    """Extract data from HTML tables."""
    items = []
    
    try:
        table = await page.query_selector(table_selector)
        if not table:
            return items
        
        # Extract headers
        headers = []
        header_elements = await table.query_selector_all("th")
        for header in header_elements:
            header_text = await header.inner_text()
            headers.append(header_text.strip().lower().replace(" ", "_"))
        
        # Extract rows
        rows = await table.query_selector_all("tr")
        for row in rows[1:]:  # Skip header row
            cells = await row.query_selector_all("td")
            if len(cells) == len(headers):
                row_data = {}
                for i, cell in enumerate(cells):
                    cell_text = await cell.inner_text()
                    row_data[headers[i]] = cell_text.strip()
                items.append(row_data)
    
    except Exception as e:
        logger.error(f"Failed to extract table data: {e}")
    
    return items


async def extract_list_data(page: Page, list_selector: str, item_selector: str) -> List[Dict[str, Any]]:
    """Extract data from lists (ul/ol)."""
    items = []
    
    try:
        list_element = await page.query_selector(list_selector)
        if not list_element:
            return items
        
        list_items = await list_element.query_selector_all(item_selector)
        for list_item in list_items:
            text = await list_item.inner_text()
            href = await list_item.get_attribute("href")
            
            item_data = {"text": text.strip()}
            if href:
                item_data["href"] = href
            
            items.append(item_data)
    
    except Exception as e:
        logger.error(f"Failed to extract list data: {e}")
    
    return items


async def extract_card_data(page: Page, card_selector: str, field_selectors: Dict[str, str]) -> List[Dict[str, Any]]:
    """Extract data from card-like elements."""
    items = []
    
    try:
        cards = await page.query_selector_all(card_selector)
        
        for card in cards:
            item_data = {}
            for field_name, selector in field_selectors.items():
                element = await card.query_selector(selector)
                if element:
                    if selector.endswith("@href"):
                        value = await element.get_attribute("href")
                    elif selector.endswith("@src"):
                        value = await element.get_attribute("src")
                    else:
                        value = await element.inner_text()
                    
                    item_data[field_name] = value.strip() if isinstance(value, str) else value
            
            items.append(item_data)
    
    except Exception as e:
        logger.error(f"Failed to extract card data: {e}")
    
    return items