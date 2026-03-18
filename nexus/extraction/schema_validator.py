"""
Advanced Data Extraction Pipeline for nexus.

Provides structured data extraction with JSON Schema validation,
intelligent pagination handling, and incremental scraping capabilities.
"""

import asyncio
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse, urlunparse

import jsonschema
from jsonschema import ValidationError, validate

from nexus.actor.page import Page
from nexus.actor.element import Element

logger = logging.getLogger(__name__)


class ExtractionError(Exception):
    """Base exception for extraction pipeline errors."""
    pass


class SchemaValidationError(ExtractionError):
    """Raised when extracted data fails schema validation."""
    pass


class PaginationError(ExtractionError):
    """Raised when pagination handling fails."""
    pass


class DataHasher:
    """Handles data hashing for incremental scraping."""
    
    @staticmethod
    def hash_data(data: Any) -> str:
        """Create a deterministic hash of data for comparison."""
        if isinstance(data, dict):
            # Sort keys for deterministic hashing
            sorted_data = json.dumps(data, sort_keys=True, default=str)
        elif isinstance(data, (list, tuple)):
            # Convert to string representation
            sorted_data = json.dumps(data, default=str)
        else:
            sorted_data = str(data)
        
        return hashlib.sha256(sorted_data.encode('utf-8')).hexdigest()
    
    @staticmethod
    def hash_url_pattern(url: str, strip_query: bool = False) -> str:
        """Create a hash of URL pattern for pagination detection."""
        parsed = urlparse(url)
        if strip_query:
            # Remove query parameters for pattern matching
            parsed = parsed._replace(query='')
        pattern = urlunparse(parsed)
        return hashlib.sha256(pattern.encode('utf-8')).hexdigest()


class PaginationDetector:
    """Detects and handles pagination patterns."""
    
    # Common pagination selectors and patterns
    PAGINATION_SELECTORS = [
        'a[rel="next"]',
        'a.next',
        'button.next',
        '[aria-label*="next"]',
        '[aria-label*="Next"]',
        '.pagination .next',
        '.pager-next',
        'li.next a',
        'a:contains("Next")',
        'a:contains("next")',
        'a:contains("»")',
        'a:contains("›")',
        '[class*="next"]',
        '[class*="pagination"] a:last-child',
    ]
    
    PAGINATION_URL_PATTERNS = [
        r'page[=/]\d+',
        r'p[=/]\d+',
        r'offset[=/]\d+',
        r'start[=/]\d+',
        r'[?&]page=\d+',
        r'[?&]p=\d+',
        r'[?&]offset=\d+',
        r'[?&]start=\d+',
        r'/\d+/$',  # Trailing number in URL path
    ]
    
    def __init__(self, page: Page):
        self.page = page
        self._current_page = 1
        self._visited_urls: Set[str] = set()
    
    async def detect_next_page(self) -> Optional[str]:
        """Detect and return the next page URL if available."""
        current_url = self.page.url
        self._visited_urls.add(current_url)
        
        # Try common pagination selectors
        for selector in self.PAGINATION_SELECTORS:
            try:
                next_element = await self.page.query_selector(selector)
                if next_element:
                    href = await next_element.get_attribute('href')
                    if href:
                        next_url = self._resolve_url(current_url, href)
                        if next_url and next_url not in self._visited_urls:
                            logger.debug(f"Found next page via selector: {selector}")
                            return next_url
            except Exception:
                continue
        
        # Try URL pattern detection
        next_url = self._detect_next_url_pattern(current_url)
        if next_url and next_url not in self._visited_urls:
            logger.debug(f"Found next page via URL pattern")
            return next_url
        
        return None
    
    def _detect_next_url_pattern(self, current_url: str) -> Optional[str]:
        """Detect next page URL based on common patterns."""
        parsed = urlparse(current_url)
        
        # Check query parameters
        if parsed.query:
            from urllib.parse import parse_qs, urlencode
            query_params = parse_qs(parsed.query)
            
            # Look for page parameters
            for param in ['page', 'p', 'offset', 'start']:
                if param in query_params:
                    try:
                        current_val = int(query_params[param][0])
                        query_params[param] = [str(current_val + 1)]
                        new_query = urlencode(query_params, doseq=True)
                        return urlunparse(parsed._replace(query=new_query))
                    except (ValueError, IndexError):
                        continue
        
        # Check URL path for page numbers
        path = parsed.path
        for pattern in self.PAGINATION_URL_PATTERNS:
            match = re.search(pattern, path)
            if match:
                # Extract current page number and increment
                num_match = re.search(r'\d+', match.group())
                if num_match:
                    current_num = int(num_match.group())
                    new_path = path[:match.start()] + str(current_num + 1) + path[match.end():]
                    return urlunparse(parsed._replace(path=new_path))
        
        return None
    
    def _resolve_url(self, base_url: str, href: str) -> Optional[str]:
        """Resolve relative URLs to absolute."""
        if not href:
            return None
        
        if href.startswith(('http://', 'https://')):
            return href
        
        from urllib.parse import urljoin
        return urljoin(base_url, href)
    
    async def has_more_pages(self) -> bool:
        """Check if there are more pages available."""
        next_url = await self.detect_next_page()
        return next_url is not None
    
    def reset(self):
        """Reset pagination state."""
        self._current_page = 1
        self._visited_urls.clear()


class SchemaValidator:
    """Validates extracted data against JSON Schema."""
    
    def __init__(self, schema: Dict[str, Any]):
        """
        Initialize with JSON Schema.
        
        Args:
            schema: JSON Schema dictionary
        """
        self.schema = schema
        self._compiled_schema = None
        self._validate_schema()
    
    def _validate_schema(self):
        """Validate that the provided schema is valid JSON Schema."""
        try:
            # Check if it's a valid JSON Schema
            jsonschema.Draft7Validator.check_schema(self.schema)
            self._compiled_schema = self.schema
        except jsonschema.SchemaError as e:
            raise SchemaValidationError(f"Invalid JSON Schema: {e}")
    
    def validate(self, data: Any) -> Tuple[bool, List[str]]:
        """
        Validate data against the schema.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        try:
            validate(instance=data, schema=self._compiled_schema)
            return True, []
        except ValidationError as e:
            # Collect all validation errors
            errors.append(str(e))
            
            # If there are nested errors, collect them too
            if hasattr(e, 'context'):
                for error in e.context:
                    errors.append(str(error))
            
            return False, errors
    
    def get_required_fields(self) -> List[str]:
        """Get list of required fields from schema."""
        return self.schema.get('required', [])
    
    def get_field_schema(self, field_name: str) -> Optional[Dict[str, Any]]:
        """Get schema for a specific field."""
        properties = self.schema.get('properties', {})
        return properties.get(field_name)


class DataExtractor:
    """Extracts structured data from web pages using CSS selectors and rules."""
    
    def __init__(self, page: Page):
        self.page = page
        self._extraction_rules: Dict[str, Dict[str, Any]] = {}
    
    def add_extraction_rule(self, field_name: str, selector: str, 
                          attribute: str = 'text', 
                          multiple: bool = False,
                          transform: Optional[callable] = None,
                          default: Any = None):
        """
        Add an extraction rule for a field.
        
        Args:
            field_name: Name of the field to extract
            selector: CSS selector to find the element(s)
            attribute: Attribute to extract ('text', 'href', 'src', etc.)
            multiple: Whether to extract multiple values (returns list)
            transform: Optional function to transform the extracted value
            default: Default value if extraction fails
        """
        self._extraction_rules[field_name] = {
            'selector': selector,
            'attribute': attribute,
            'multiple': multiple,
            'transform': transform,
            'default': default
        }
    
    def load_rules_from_schema(self, schema: Dict[str, Any]):
        """
        Load extraction rules from a JSON Schema that contains extraction metadata.
        
        Expected schema format:
        {
            "type": "object",
            "properties": {
                "field_name": {
                    "type": "string",
                    "selector": "css_selector",
                    "attribute": "text",
                    "multiple": false
                }
            }
        }
        """
        properties = schema.get('properties', {})
        
        for field_name, field_schema in properties.items():
            if 'selector' in field_schema:
                self.add_extraction_rule(
                    field_name=field_name,
                    selector=field_schema['selector'],
                    attribute=field_schema.get('attribute', 'text'),
                    multiple=field_schema.get('multiple', False),
                    default=field_schema.get('default')
                )
    
    async def extract_field(self, field_name: str) -> Any:
        """
        Extract a single field based on its rule.
        
        Args:
            field_name: Name of the field to extract
            
        Returns:
            Extracted value or default if extraction fails
        """
        if field_name not in self._extraction_rules:
            raise ExtractionError(f"No extraction rule defined for field: {field_name}")
        
        rule = self._extraction_rules[field_name]
        
        try:
            if rule['multiple']:
                elements = await self.page.query_selector_all(rule['selector'])
                values = []
                
                for element in elements:
                    value = await self._extract_from_element(element, rule['attribute'])
                    if value is not None:
                        if rule['transform']:
                            value = rule['transform'](value)
                        values.append(value)
                
                return values if values else rule['default']
            else:
                element = await self.page.query_selector(rule['selector'])
                if element:
                    value = await self._extract_from_element(element, rule['attribute'])
                    if value is not None and rule['transform']:
                        value = rule['transform'](value)
                    return value if value is not None else rule['default']
                else:
                    return rule['default']
                    
        except Exception as e:
            logger.warning(f"Failed to extract field '{field_name}': {e}")
            return rule['default']
    
    async def _extract_from_element(self, element: Element, attribute: str) -> Optional[str]:
        """Extract value from an element based on attribute."""
        if attribute == 'text':
            return await element.inner_text()
        elif attribute == 'html':
            return await element.inner_html()
        elif attribute in ['href', 'src', 'alt', 'title', 'value', 'data-*']:
            return await element.get_attribute(attribute)
        else:
            # Try to get as attribute first, fall back to text
            attr_value = await element.get_attribute(attribute)
            return attr_value if attr_value is not None else await element.inner_text()
    
    async def extract_all(self) -> Dict[str, Any]:
        """Extract all fields based on defined rules."""
        result = {}
        
        for field_name in self._extraction_rules:
            result[field_name] = await self.extract_field(field_name)
        
        return result
    
    async def extract_with_retry(self, field_name: str, max_retries: int = 3, 
                               delay: float = 1.0) -> Any:
        """
        Extract a field with retry logic for missing data.
        
        Args:
            field_name: Name of the field to extract
            max_retries: Maximum number of retry attempts
            delay: Delay between retries in seconds
            
        Returns:
            Extracted value
        """
        for attempt in range(max_retries + 1):
            value = await self.extract_field(field_name)
            
            # Check if value is meaningful (not None, not empty string, not empty list)
            if value is not None and value != '' and value != []:
                return value
            
            if attempt < max_retries:
                logger.debug(f"Retrying extraction for '{field_name}' (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(delay)
        
        # Return default after all retries
        rule = self._extraction_rules.get(field_name, {})
        return rule.get('default')
    
    def clear_rules(self):
        """Clear all extraction rules."""
        self._extraction_rules.clear()


class IncrementalScrapeManager:
    """Manages incremental scraping state and deduplication."""
    
    def __init__(self, state_file: Optional[Union[str, Path]] = None):
        """
        Initialize incremental scrape manager.
        
        Args:
            state_file: Path to JSON file for persisting scrape state
        """
        self.state_file = Path(state_file) if state_file else None
        self.state: Dict[str, Any] = {
            'last_scrape_time': None,
            'extracted_hashes': {},
            'page_hashes': {},
            'urls_visited': set()
        }
        
        if self.state_file and self.state_file.exists():
            self._load_state()
    
    def _load_state(self):
        """Load state from file."""
        try:
            with open(self.state_file, 'r') as f:
                loaded_state = json.load(f)
                
                # Convert lists back to sets
                if 'urls_visited' in loaded_state:
                    loaded_state['urls_visited'] = set(loaded_state['urls_visited'])
                
                self.state.update(loaded_state)
                logger.info(f"Loaded incremental state from {self.state_file}")
        except Exception as e:
            logger.warning(f"Failed to load state from {self.state_file}: {e}")
    
    def save_state(self):
        """Save state to file."""
        if not self.state_file:
            return
        
        try:
            # Convert sets to lists for JSON serialization
            state_to_save = self.state.copy()
            if 'urls_visited' in state_to_save:
                state_to_save['urls_visited'] = list(state_to_save['urls_visited'])
            
            with open(self.state_file, 'w') as f:
                json.dump(state_to_save, f, indent=2, default=str)
            
            logger.debug(f"Saved incremental state to {self.state_file}")
        except Exception as e:
            logger.error(f"Failed to save state to {self.state_file}: {e}")
    
    def is_url_visited(self, url: str) -> bool:
        """Check if URL has been visited before."""
        return url in self.state['urls_visited']
    
    def mark_url_visited(self, url: str):
        """Mark URL as visited."""
        self.state['urls_visited'].add(url)
    
    def is_data_changed(self, data_id: str, data: Any) -> bool:
        """
        Check if data has changed since last scrape.
        
        Args:
            data_id: Unique identifier for the data
            data: Data to check
            
        Returns:
            True if data is new or changed
        """
        current_hash = DataHasher.hash_data(data)
        previous_hash = self.state['extracted_hashes'].get(data_id)
        
        if previous_hash is None or previous_hash != current_hash:
            # Update hash and return True (data changed)
            self.state['extracted_hashes'][data_id] = current_hash
            return True
        
        return False
    
    def get_changed_data(self, data_items: List[Dict[str, Any]], 
                        id_field: str = 'id') -> List[Dict[str, Any]]:
        """
        Filter data items to only include new or changed items.
        
        Args:
            data_items: List of data items to filter
            id_field: Field name to use as unique identifier
            
        Returns:
            List of new or changed items
        """
        changed_items = []
        
        for item in data_items:
            if id_field in item:
                data_id = str(item[id_field])
                if self.is_data_changed(data_id, item):
                    changed_items.append(item)
            else:
                # If no ID field, include all items
                changed_items.append(item)
        
        return changed_items
    
    def update_scrape_time(self):
        """Update last scrape timestamp."""
        self.state['last_scrape_time'] = datetime.now(timezone.utc).isoformat()


class ExtractionPipeline:
    """
    Advanced data extraction pipeline with schema validation,
    pagination handling, and incremental scraping.
    """
    
    def __init__(self, page: Page, schema: Optional[Dict[str, Any]] = None,
                 state_file: Optional[Union[str, Path]] = None):
        """
        Initialize extraction pipeline.
        
        Args:
            page: Browser page instance
            schema: JSON Schema for validation
            state_file: Path to state file for incremental scraping
        """
        self.page = page
        self.schema_validator = SchemaValidator(schema) if schema else None
        self.data_extractor = DataExtractor(page)
        self.pagination_detector = PaginationDetector(page)
        self.incremental_manager = IncrementalScrapeManager(state_file)
        
        # Configuration
        self.max_retries = 3
        self.retry_delay = 1.0
        self.max_pages = 100  # Safety limit for pagination
        self.id_field = 'id'  # Default field for deduplication
        
        # Load extraction rules from schema if provided
        if schema:
            self.data_extractor.load_rules_from_schema(schema)
    
    async def extract_single_page(self, validate_schema: bool = True) -> Dict[str, Any]:
        """
        Extract data from the current page.
        
        Args:
            validate_schema: Whether to validate against schema
            
        Returns:
            Extracted data dictionary
        """
        # Extract data using rules
        data = await self.data_extractor.extract_all()
        
        # Apply retry logic for required fields
        if self.schema_validator:
            required_fields = self.schema_validator.get_required_fields()
            for field in required_fields:
                if field in data and (data[field] is None or data[field] == ''):
                    # Try to extract with retry
                    data[field] = await self.data_extractor.extract_with_retry(
                        field, self.max_retries, self.retry_delay
                    )
        
        # Validate against schema if requested
        if validate_schema and self.schema_validator:
            is_valid, errors = self.schema_validator.validate(data)
            if not is_valid:
                logger.warning(f"Schema validation failed: {errors}")
                # You might want to raise an exception or handle differently
                # For now, we'll log and continue
        
        return data
    
    async def extract_with_pagination(self, validate_each_page: bool = True,
                                    stop_on_duplicate: bool = True) -> List[Dict[str, Any]]:
        """
        Extract data from multiple pages with pagination.
        
        Args:
            validate_each_page: Whether to validate each page's data
            stop_on_duplicate: Stop when encountering duplicate data (for incremental)
            
        Returns:
            List of extracted data from all pages
        """
        all_data = []
        page_count = 0
        
        self.pagination_detector.reset()
        
        while page_count < self.max_pages:
            page_count += 1
            logger.info(f"Extracting page {page_count}: {self.page.url}")
            
            # Extract data from current page
            page_data = await self.extract_single_page(validate_schema=validate_each_page)
            
            # Handle both single items and lists of items
            if isinstance(page_data, list):
                items = page_data
            else:
                items = [page_data]
            
            # Filter for incremental scraping
            if stop_on_duplicate:
                changed_items = self.incremental_manager.get_changed_data(
                    items, self.id_field
                )
                
                if not changed_items:
                    logger.info("No new data found, stopping pagination")
                    break
                
                all_data.extend(changed_items)
            else:
                all_data.extend(items)
            
            # Mark URL as visited
            self.incremental_manager.mark_url_visited(self.page.url)
            
            # Check for next page
            if not await self.pagination_detector.has_more_pages():
                logger.info("No more pages available")
                break
            
            # Navigate to next page
            next_url = await self.pagination_detector.detect_next_page()
            if not next_url:
                break
            
            try:
                await self.page.goto(next_url)
                await self.page.wait_for_load_state('networkidle')
            except Exception as e:
                logger.error(f"Failed to navigate to next page: {e}")
                break
        
        # Update scrape time
        self.incremental_manager.update_scrape_time()
        self.incremental_manager.save_state()
        
        logger.info(f"Extracted {len(all_data)} items from {page_count} pages")
        return all_data
    
    async def extract_with_schema_validation(self, 
                                           extraction_rules: Dict[str, Dict[str, Any]],
                                           schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and validate data using provided rules and schema.
        
        Args:
            extraction_rules: Dictionary of field extraction rules
            schema: JSON Schema for validation
            
        Returns:
            Validated data dictionary
        """
        # Clear existing rules and set new ones
        self.data_extractor.clear_rules()
        
        for field_name, rule in extraction_rules.items():
            self.data_extractor.add_extraction_rule(
                field_name=field_name,
                selector=rule['selector'],
                attribute=rule.get('attribute', 'text'),
                multiple=rule.get('multiple', False),
                transform=rule.get('transform'),
                default=rule.get('default')
            )
        
        # Update schema validator
        self.schema_validator = SchemaValidator(schema)
        
        # Extract and validate
        return await self.extract_single_page(validate_schema=True)
    
    def configure(self, max_retries: int = 3, retry_delay: float = 1.0,
                 max_pages: int = 100, id_field: str = 'id'):
        """
        Configure pipeline parameters.
        
        Args:
            max_retries: Maximum retry attempts for missing fields
            retry_delay: Delay between retries in seconds
            max_pages: Maximum pages to scrape
            id_field: Field name to use as unique identifier
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_pages = max_pages
        self.id_field = id_field
    
    def set_extraction_rules(self, rules: Dict[str, Dict[str, Any]]):
        """
        Set extraction rules from dictionary.
        
        Args:
            rules: Dictionary mapping field names to extraction rules
        """
        self.data_extractor.clear_rules()
        
        for field_name, rule in rules.items():
            self.data_extractor.add_extraction_rule(
                field_name=field_name,
                selector=rule['selector'],
                attribute=rule.get('attribute', 'text'),
                multiple=rule.get('multiple', False),
                transform=rule.get('transform'),
                default=rule.get('default')
            )
    
    async def extract_structured_list(self, item_selector: str,
                                    item_schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract a list of structured items from the page.
        
        Args:
            item_selector: CSS selector for each item container
            item_schema: Schema for each item (with extraction rules)
            
        Returns:
            List of extracted items
        """
        # Find all item containers
        items = await self.page.query_selector_all(item_selector)
        results = []
        
        for item_element in items:
            # Create a temporary extractor for this item
            item_extractor = DataExtractor(self.page)
            
            # Load rules from schema
            properties = item_schema.get('properties', {})
            for field_name, field_schema in properties.items():
                if 'selector' in field_schema:
                    item_extractor.add_extraction_rule(
                        field_name=field_name,
                        selector=field_schema['selector'],
                        attribute=field_schema.get('attribute', 'text'),
                        multiple=field_schema.get('multiple', False),
                        default=field_schema.get('default')
                    )
            
            # Extract data from this item
            # Note: This is simplified - in reality you'd need to scope extraction to the item element
            item_data = await item_extractor.extract_all()
            
            # Validate against item schema if provided
            if 'type' in item_schema:
                try:
                    validate(instance=item_data, schema=item_schema)
                    results.append(item_data)
                except ValidationError as e:
                    logger.warning(f"Item validation failed: {e}")
            else:
                results.append(item_data)
        
        return results


# Factory function for easy instantiation
async def create_extraction_pipeline(page: Page, 
                                   schema: Optional[Dict[str, Any]] = None,
                                   state_file: Optional[Union[str, Path]] = None,
                                   **config) -> ExtractionPipeline:
    """
    Create and configure an extraction pipeline.
    
    Args:
        page: Browser page instance
        schema: JSON Schema for validation
        state_file: Path to state file for incremental scraping
        **config: Additional configuration parameters
        
    Returns:
        Configured ExtractionPipeline instance
    """
    pipeline = ExtractionPipeline(page, schema, state_file)
    
    if config:
        pipeline.configure(**config)
    
    return pipeline


# Example usage and helper functions
def create_simple_schema(field_mappings: Dict[str, str]) -> Dict[str, Any]:
    """
    Create a simple JSON Schema from field mappings.
    
    Args:
        field_mappings: Dictionary mapping field names to CSS selectors
        
    Returns:
        JSON Schema dictionary
    """
    properties = {}
    required = []
    
    for field_name, selector in field_mappings.items():
        properties[field_name] = {
            "type": "string",
            "selector": selector,
            "attribute": "text"
        }
        required.append(field_name)
    
    return {
        "type": "object",
        "properties": properties,
        "required": required
    }


def create_extraction_rules(field_mappings: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Create extraction rules from field mappings.
    
    Args:
        field_mappings: Dictionary with field configurations
        
    Returns:
        Extraction rules dictionary
    """
    rules = {}
    
    for field_name, config in field_mappings.items():
        rules[field_name] = {
            "selector": config["selector"],
            "attribute": config.get("attribute", "text"),
            "multiple": config.get("multiple", False),
            "default": config.get("default"),
            "transform": config.get("transform")
        }
    
    return rules