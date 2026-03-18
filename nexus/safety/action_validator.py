"""
Action Validation & Safety System for nexus
Prevents destructive actions with pre-execution validation, sandboxed testing, and rollback capabilities.
"""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk levels for browser actions."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ActionType(Enum):
    """Types of browser actions that can be validated."""
    CLICK = "click"
    TYPE = "type"
    NAVIGATE = "navigate"
    SUBMIT = "submit"
    EXECUTE_SCRIPT = "execute_script"
    UPLOAD_FILE = "upload_file"
    DOWNLOAD_FILE = "download_file"
    DELETE = "delete"
    FORM_SUBMIT = "form_submit"
    CLOSE = "close"
    RELOAD = "reload"
    GO_BACK = "go_back"
    GO_FORWARD = "go_forward"
    SCREENSHOT = "screenshot"
    SET_VIEWPORT = "set_viewport"
    SCROLL = "scroll"
    DRAG_AND_DROP = "drag_and_drop"
    SELECT_OPTION = "select_option"
    HOVER = "hover"
    FOCUS = "focus"
    BLUR = "blur"
    CLEAR = "clear"
    PRESS_KEY = "press_key"


@dataclass
class DOMSnapshot:
    """Snapshot of DOM state for rollback capabilities."""
    timestamp: float
    url: str
    html: str
    cookies: List[Dict] = field(default_factory=list)
    local_storage: Dict[str, str] = field(default_factory=dict)
    session_storage: Dict[str, str] = field(default_factory=dict)
    scroll_position: Dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0})
    form_data: Dict[str, Any] = field(default_factory=dict)
    hash: str = ""
    
    def __post_init__(self):
        """Generate hash for snapshot verification."""
        content = f"{self.url}:{self.html}:{json.dumps(self.cookies)}:{json.dumps(self.local_storage)}"
        self.hash = hashlib.md5(content.encode()).hexdigest()


@dataclass
class ActionContext:
    """Context for action validation and execution."""
    action_type: ActionType
    selector: Optional[str] = None
    value: Optional[Any] = None
    url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    user_confirmed: bool = False
    risk_override: Optional[RiskLevel] = None


@dataclass
class ValidationResult:
    """Result of action validation."""
    is_valid: bool
    risk_level: RiskLevel
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    suggested_alternatives: List[str] = field(default_factory=list)
    requires_confirmation: bool = False
    sandbox_test_passed: Optional[bool] = None
    execution_time_estimate: float = 0.0


class SafetyConfig:
    """Configuration for the safety system."""
    
    def __init__(
        self,
        enable_sandbox: bool = True,
        enable_rollback: bool = True,
        max_snapshots: int = 10,
        snapshot_retention_seconds: int = 3600,
        risk_thresholds: Optional[Dict[str, RiskLevel]] = None,
        blocked_domains: Optional[Set[str]] = None,
        blocked_selectors: Optional[Set[str]] = None,
        require_confirmation_above: RiskLevel = RiskLevel.HIGH,
        sandbox_timeout: float = 5.0,
        destructive_patterns: Optional[List[str]] = None,
    ):
        self.enable_sandbox = enable_sandbox
        self.enable_rollback = enable_rollback
        self.max_snapshots = max_snapshots
        self.snapshot_retention_seconds = snapshot_retention_seconds
        self.require_confirmation_above = require_confirmation_above
        self.sandbox_timeout = sandbox_timeout
        
        # Default risk thresholds
        self.risk_thresholds = risk_thresholds or {
            "navigate_away": RiskLevel.HIGH,
            "form_submit": RiskLevel.MEDIUM,
            "delete_action": RiskLevel.CRITICAL,
            "external_domain": RiskLevel.HIGH,
            "sensitive_data": RiskLevel.HIGH,
            "javascript_execution": RiskLevel.MEDIUM,
            "file_upload": RiskLevel.MEDIUM,
            "file_download": RiskLevel.MEDIUM,
        }
        
        # Default blocked patterns
        self.blocked_domains = blocked_domains or {
            "malware.com",
            "phishing-site.com",
        }
        
        self.blocked_selectors = blocked_selectors or {
            "button:contains('Delete All')",
            "[data-destructive='true']",
            ".danger-zone",
        }
        
        # Destructive patterns in selectors or values
        self.destructive_patterns = destructive_patterns or [
            "rm -rf",
            "DROP TABLE",
            "DELETE FROM",
            "format c:",
            "sudo",
            "admin",
            "password",
            "credit.card",
            "ssn",
            "social.security",
        ]


class DOMSnapshotManager:
    """Manages DOM snapshots for rollback capabilities."""
    
    def __init__(self, config: SafetyConfig):
        self.config = config
        self.snapshots: Dict[str, List[DOMSnapshot]] = {}  # page_id -> snapshots
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def start_cleanup_task(self):
        """Start background task to clean up old snapshots."""
        async def cleanup():
            while True:
                await asyncio.sleep(300)  # Clean up every 5 minutes
                self._cleanup_old_snapshots()
        
        self._cleanup_task = asyncio.create_task(cleanup())
    
    def _cleanup_old_snapshots(self):
        """Remove snapshots older than retention period."""
        current_time = time.time()
        for page_id in list(self.snapshots.keys()):
            self.snapshots[page_id] = [
                s for s in self.snapshots[page_id]
                if current_time - s.timestamp < self.config.snapshot_retention_seconds
            ]
            # Limit number of snapshots per page
            if len(self.snapshots[page_id]) > self.config.max_snapshots:
                self.snapshots[page_id] = self.snapshots[page_id][-self.config.max_snapshots:]
    
    async def take_snapshot(self, page, page_id: str = "default") -> DOMSnapshot:
        """Take a snapshot of the current DOM state."""
        try:
            # Get page content
            html = await page.content()
            url = page.url
            
            # Get storage data
            cookies = await page.context.cookies()
            local_storage = await page.evaluate("() => JSON.stringify(localStorage)")
            session_storage = await page.evaluate("() => JSON.stringify(sessionStorage)")
            
            # Get scroll position
            scroll_position = await page.evaluate("""() => ({
                x: window.scrollX,
                y: window.scrollY
            })""")
            
            # Get form data
            form_data = await page.evaluate("""() => {
                const forms = document.querySelectorAll('form');
                const formData = {};
                forms.forEach((form, index) => {
                    const inputs = form.querySelectorAll('input, select, textarea');
                    const formInputs = {};
                    inputs.forEach(input => {
                        if (input.name) {
                            formInputs[input.name] = input.value;
                        }
                    });
                    formData[`form_${index}`] = formInputs;
                });
                return formData;
            }""")
            
            snapshot = DOMSnapshot(
                timestamp=time.time(),
                url=url,
                html=html,
                cookies=cookies or [],
                local_storage=json.loads(local_storage) if local_storage else {},
                session_storage=json.loads(session_storage) if session_storage else {},
                scroll_position=scroll_position,
                form_data=form_data,
            )
            
            # Store snapshot
            if page_id not in self.snapshots:
                self.snapshots[page_id] = []
            self.snapshots[page_id].append(snapshot)
            
            logger.debug(f"DOM snapshot taken for page {page_id} at {url}")
            return snapshot
            
        except Exception as e:
            logger.error(f"Failed to take DOM snapshot: {e}")
            raise
    
    async def rollback_to_snapshot(self, page, snapshot: DOMSnapshot) -> bool:
        """Rollback page to a previous snapshot state."""
        try:
            # Verify snapshot integrity
            current_content = f"{page.url}:{await page.content()}"
            current_hash = hashlib.md5(current_content.encode()).hexdigest()
            
            if current_hash == snapshot.hash:
                logger.info("Page already matches snapshot, no rollback needed")
                return True
            
            # Restore HTML
            await page.set_content(snapshot.html)
            
            # Restore cookies
            if snapshot.cookies:
                await page.context.add_cookies(snapshot.cookies)
            
            # Restore localStorage
            if snapshot.local_storage:
                await page.evaluate("""(storage) => {
                    localStorage.clear();
                    for (const [key, value] of Object.entries(storage)) {
                        localStorage.setItem(key, value);
                    }
                }""", snapshot.local_storage)
            
            # Restore sessionStorage
            if snapshot.session_storage:
                await page.evaluate("""(storage) => {
                    sessionStorage.clear();
                    for (const [key, value] of Object.entries(storage)) {
                        sessionStorage.setItem(key, value);
                    }
                }""", snapshot.session_storage)
            
            # Restore scroll position
            await page.evaluate(f"""(x, y) => {{
                window.scrollTo({snapshot.scroll_position['x']}, {snapshot.scroll_position['y']});
            }}""")
            
            logger.info(f"Successfully rolled back to snapshot from {snapshot.timestamp}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rollback to snapshot: {e}")
            return False
    
    def get_snapshots(self, page_id: str = "default") -> List[DOMSnapshot]:
        """Get all snapshots for a page."""
        return self.snapshots.get(page_id, [])
    
    def get_latest_snapshot(self, page_id: str = "default") -> Optional[DOMSnapshot]:
        """Get the most recent snapshot for a page."""
        snapshots = self.get_snapshots(page_id)
        return snapshots[-1] if snapshots else None


class SandboxedExecutor:
    """Executes actions in a sandboxed environment for testing."""
    
    def __init__(self, config: SafetyConfig):
        self.config = config
        self.test_results: Dict[str, ValidationResult] = {}
    
    async def test_action(
        self,
        action: ActionContext,
        page,
        test_id: Optional[str] = None,
    ) -> ValidationResult:
        """Test an action in a sandboxed environment."""
        test_id = test_id or f"test_{int(time.time())}"
        
        try:
            # Create a new page for sandbox testing
            context = page.context
            sandbox_page = await context.new_page()
            
            # Copy current page state to sandbox
            current_html = await page.content()
            await sandbox_page.set_content(current_html)
            
            # Set a timeout for sandbox execution
            start_time = time.time()
            
            # Execute action in sandbox
            result = await self._execute_in_sandbox(
                action, sandbox_page, timeout=self.config.sandbox_timeout
            )
            
            execution_time = time.time() - start_time
            
            # Close sandbox page
            await sandbox_page.close()
            
            # Create validation result
            validation_result = ValidationResult(
                is_valid=result["success"],
                risk_level=action.risk_override or self._assess_risk(action),
                warnings=result.get("warnings", []),
                errors=result.get("errors", []),
                sandbox_test_passed=result["success"],
                execution_time_estimate=execution_time,
            )
            
            self.test_results[test_id] = validation_result
            return validation_result
            
        except asyncio.TimeoutError:
            logger.warning(f"Sandbox test timed out for action: {action.action_type}")
            return ValidationResult(
                is_valid=False,
                risk_level=RiskLevel.HIGH,
                errors=["Sandbox test timed out - action may be too slow or infinite"],
                sandbox_test_passed=False,
            )
        except Exception as e:
            logger.error(f"Sandbox test failed: {e}")
            return ValidationResult(
                is_valid=False,
                risk_level=RiskLevel.HIGH,
                errors=[f"Sandbox test error: {str(e)}"],
                sandbox_test_passed=False,
            )
    
    async def _execute_in_sandbox(
        self,
        action: ActionContext,
        sandbox_page,
        timeout: float,
    ) -> Dict[str, Any]:
        """Execute action in sandbox and collect results."""
        warnings = []
        errors = []
        
        try:
            # Add console listener for warnings/errors
            console_messages = []
            sandbox_page.on("console", lambda msg: console_messages.append(msg))
            
            # Add error listener
            page_errors = []
            sandbox_page.on("pageerror", lambda err: page_errors.append(str(err)))
            
            # Execute based on action type
            if action.action_type == ActionType.CLICK:
                if action.selector:
                    await sandbox_page.click(action.selector, timeout=timeout * 1000)
            
            elif action.action_type == ActionType.TYPE:
                if action.selector and action.value:
                    await sandbox_page.fill(action.selector, str(action.value))
            
            elif action.action_type == ActionType.NAVIGATE:
                if action.url:
                    await sandbox_page.goto(action.url, timeout=timeout * 1000)
            
            elif action.action_type == ActionType.EXECUTE_SCRIPT:
                if action.value:
                    await sandbox_page.evaluate(str(action.value))
            
            # Check for console warnings
            for msg in console_messages:
                if msg.type == "warning":
                    warnings.append(f"Console warning: {msg.text}")
                elif msg.type == "error":
                    errors.append(f"Console error: {msg.text}")
            
            # Check for page errors
            errors.extend(page_errors)
            
            return {
                "success": len(errors) == 0,
                "warnings": warnings,
                "errors": errors,
            }
            
        except Exception as e:
            return {
                "success": False,
                "warnings": warnings,
                "errors": errors + [str(e)],
            }
    
    def _assess_risk(self, action: ActionContext) -> RiskLevel:
        """Assess risk level of an action."""
        risk_level = RiskLevel.LOW
        
        # Check action type
        if action.action_type in [ ActionType.DELETE, ActionType.CLOSE]:
            risk_level = RiskLevel.CRITICAL
        elif action.action_type in [ActionType.NAVIGATE, ActionType.SUBMIT, ActionType.FORM_SUBMIT]:
            risk_level = RiskLevel.HIGH
        elif action.action_type in [ActionType.EXECUTE_SCRIPT, ActionType.UPLOAD_FILE]:
            risk_level = RiskLevel.MEDIUM
        
        # Check for destructive patterns
        if action.selector:
            for pattern in self.config.destructive_patterns:
                if pattern.lower() in action.selector.lower():
                    risk_level = RiskLevel.CRITICAL
                    break
        
        if action.value:
            value_str = str(action.value).lower()
            for pattern in self.config.destructive_patterns:
                if pattern.lower() in value_str:
                    risk_level = RiskLevel.CRITICAL
                    break
        
        return risk_level


class ActionValidator:
    """Main action validator with safety checks and validation pipeline."""
    
    def __init__(
        self,
        config: Optional[SafetyConfig] = None,
        snapshot_manager: Optional[DOMSnapshotManager] = None,
        sandbox_executor: Optional[SandboxedExecutor] = None,
    ):
        self.config = config or SafetyConfig()
        self.snapshot_manager = snapshot_manager or DOMSnapshotManager(self.config)
        self.sandbox_executor = sandbox_executor or SandboxedExecutor(self.config)
        
        # Validation rules
        self.validators: List[Callable[[ActionContext, Any], ValidationResult]] = [
            self._validate_selector_safety,
            self._validate_domain_safety,
            self._validate_action_type_safety,
            self._validate_value_safety,
            self._validate_timing,
        ]
        
        # Action history for pattern detection
        self.action_history: List[ActionContext] = []
        self.max_history = 100
        
        # Confirmation callback
        self._confirmation_callback: Optional[Callable[[ActionContext, RiskLevel], bool]] = None
    
    def set_confirmation_callback(self, callback: Callable[[ActionContext, RiskLevel], bool]):
        """Set callback for user confirmation of high-risk actions."""
        self._confirmation_callback = callback
    
    async def validate_action(
        self,
        action: ActionContext,
        page,
        take_snapshot: bool = True,
        test_in_sandbox: bool = False,
    ) -> ValidationResult:
        """
        Validate an action before execution.
        
        Args:
            action: The action to validate
            page: The Playwright page object
            take_snapshot: Whether to take a DOM snapshot before execution
            test_in_sandbox: Whether to test the action in a sandbox first
            
        Returns:
            ValidationResult with validation outcome
        """
        logger.info(f"Validating action: {action.action_type.value}")
        
        # Run all validators
        validation_results = []
        for validator in self.validators:
            result = validator(action, page)
            validation_results.append(result)
        
        # Combine results
        combined_result = self._combine_validation_results(validation_results)
        
        # Check if action requires confirmation
        if combined_result.risk_level >= self.config.require_confirmation_above:
            combined_result.requires_confirmation = True
            
            # If we have a confirmation callback, use it
            if self._confirmation_callback and not action.user_confirmed:
                confirmed = self._confirmation_callback(action, combined_result.risk_level)
                if not confirmed:
                    combined_result.is_valid = False
                    combined_result.errors.append("Action cancelled by user")
                    return combined_result
        
        # Take snapshot if enabled and action has medium or higher risk
        if (self.config.enable_rollback and take_snapshot and 
            combined_result.risk_level >= RiskLevel.MEDIUM):
            try:
                page_id = id(page)
                await self.snapshot_manager.take_snapshot(page, str(page_id))
                logger.debug(f"Snapshot taken for page {page_id}")
            except Exception as e:
                logger.warning(f"Failed to take snapshot: {e}")
                combined_result.warnings.append(f"Snapshot failed: {str(e)}")
        
        # Test in sandbox if enabled and requested
        if (self.config.enable_sandbox and test_in_sandbox and 
            combined_result.risk_level >= RiskLevel.MEDIUM):
            sandbox_result = await self.sandbox_executor.test_action(action, page)
            combined_result.sandbox_test_passed = sandbox_result.sandbox_test_passed
            
            if not sandbox_result.is_valid:
                combined_result.is_valid = False
                combined_result.errors.extend(sandbox_result.errors)
                combined_result.warnings.extend(sandbox_result.warnings)
        
        # Add to history
        self._add_to_history(action)
        
        # Check for suspicious patterns in history
        if self._detect_suspicious_pattern():
            combined_result.warnings.append("Suspicious action pattern detected")
            if combined_result.risk_level < RiskLevel.HIGH:
                combined_result.risk_level = RiskLevel.HIGH
        
        return combined_result
    
    def _validate_selector_safety(self, action: ActionContext, page) -> ValidationResult:
        """Validate selector for safety issues."""
        warnings = []
        errors = []
        
        if not action.selector:
            return ValidationResult(
                is_valid=True,
                risk_level=RiskLevel.LOW,
            )
        
        selector = action.selector.lower()
        
        # Check blocked selectors
        for blocked in self.config.blocked_selectors:
            if blocked.lower() in selector:
                errors.append(f"Selector matches blocked pattern: {blocked}")
                return ValidationResult(
                    is_valid=False,
                    risk_level=RiskLevel.CRITICAL,
                    errors=errors,
                )
        
        # Check for potentially dangerous selectors
        dangerous_patterns = [
            "script",
            "iframe",
            "object",
            "embed",
            "form[action*='delete']",
            "button[onclick*='delete']",
            "input[type='file']",
            "a[href*='javascript']",
        ]
        
        for pattern in dangerous_patterns:
            if pattern in selector:
                warnings.append(f"Selector contains potentially dangerous pattern: {pattern}")
        
        # Determine risk level based on selector
        risk_level = RiskLevel.LOW
        if any(word in selector for word in ["delete", "remove", "destroy", "drop"]):
            risk_level = RiskLevel.HIGH
        elif any(word in selector for word in ["submit", "save", "confirm", "ok"]):
            risk_level = RiskLevel.MEDIUM
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            risk_level=risk_level,
            warnings=warnings,
            errors=errors,
        )
    
    def _validate_domain_safety(self, action: ActionContext, page) -> ValidationResult:
        """Validate domain for safety issues."""
        warnings = []
        errors = []
        
        # Check current domain
        try:
            current_url = page.url
            from urllib.parse import urlparse
            parsed = urlparse(current_url)
            domain = parsed.netloc
            
            # Check blocked domains
            for blocked in self.config.blocked_domains:
                if blocked in domain:
                    errors.append(f"Domain {domain} is blocked")
                    return ValidationResult(
                        is_valid=False,
                        risk_level=RiskLevel.CRITICAL,
                        errors=errors,
                    )
            
            # Check for navigation to external domains
            if action.url:
                action_parsed = urlparse(action.url)
                action_domain = action_parsed.netloc
                
                if action_domain and action_domain != domain:
                    warnings.append(f"Action navigates to external domain: {action_domain}")
                    return ValidationResult(
                        is_valid=True,
                        risk_level=RiskLevel.HIGH,
                        warnings=warnings,
                    )
        
        except Exception as e:
            warnings.append(f"Could not validate domain: {str(e)}")
        
        return ValidationResult(
            is_valid=True,
            risk_level=RiskLevel.LOW,
            warnings=warnings,
            errors=errors,
        )
    
    def _validate_action_type_safety(self, action: ActionContext, page) -> ValidationResult:
        """Validate action type for safety."""
        risk_level = RiskLevel.LOW
        warnings = []
        
        # High-risk action types
        high_risk_actions = {
            ActionType.DELETE,
            ActionType.CLOSE,
            ActionType.NAVIGATE,
            ActionType.SUBMIT,
            ActionType.FORM_SUBMIT,
        }
        
        medium_risk_actions = {
            ActionType.EXECUTE_SCRIPT,
            ActionType.UPLOAD_FILE,
            ActionType.DOWNLOAD_FILE,
            ActionType.TYPE,
        }
        
        if action.action_type in high_risk_actions:
            risk_level = RiskLevel.HIGH
            warnings.append(f"Action type {action.action_type.value} is high risk")
        elif action.action_type in medium_risk_actions:
            risk_level = RiskLevel.MEDIUM
        
        return ValidationResult(
            is_valid=True,
            risk_level=risk_level,
            warnings=warnings,
        )
    
    def _validate_value_safety(self, action: ActionContext, page) -> ValidationResult:
        """Validate action value for safety issues."""
        warnings = []
        errors = []
        
        if action.value is None:
            return ValidationResult(
                is_valid=True,
                risk_level=RiskLevel.LOW,
            )
        
        value_str = str(action.value).lower()
        
        # Check for destructive patterns in value
        for pattern in self.config.destructive_patterns:
            if pattern.lower() in value_str:
                errors.append(f"Value contains destructive pattern: {pattern}")
                return ValidationResult(
                    is_valid=False,
                    risk_level=RiskLevel.CRITICAL,
                    errors=errors,
                )
        
        # Check for sensitive data patterns
        sensitive_patterns = [
            r"\b\d{3}[-.]?\d{2}[-.]?\d{4}\b",  # SSN
            r"\b\d{16}\b",  # Credit card
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b",  # IP address
        ]
        
        import re
        for pattern in sensitive_patterns:
            if re.search(pattern, value_str):
                warnings.append("Value may contain sensitive data")
                return ValidationResult(
                    is_valid=True,
                    risk_level=RiskLevel.HIGH,
                    warnings=warnings,
                )
        
        return ValidationResult(
            is_valid=True,
            risk_level=RiskLevel.LOW,
            warnings=warnings,
            errors=errors,
        )
    
    def _validate_timing(self, action: ActionContext, page) -> ValidationResult:
        """Validate action timing and rate limiting."""
        warnings = []
        
        # Check for rapid repeated actions
        if len(self.action_history) >= 3:
            recent_actions = self.action_history[-3:]
            time_diffs = [
                recent_actions[i+1].timestamp - recent_actions[i].timestamp
                for i in range(len(recent_actions) - 1)
            ]
            
            # If actions are happening too quickly (< 100ms apart)
            if all(diff < 0.1 for diff in time_diffs):
                warnings.append("Actions are occurring very rapidly - possible automation detected")
                return ValidationResult(
                    is_valid=True,
                    risk_level=RiskLevel.MEDIUM,
                    warnings=warnings,
                )
        
        return ValidationResult(
            is_valid=True,
            risk_level=RiskLevel.LOW,
        )
    
    def _combine_validation_results(self, results: List[ValidationResult]) -> ValidationResult:
        """Combine multiple validation results into one."""
        if not results:
            return ValidationResult(
                is_valid=True,
                risk_level=RiskLevel.LOW,
            )
        
        # Start with first result
        combined = results[0]
        
        for result in results[1:]:
            # Take highest risk level
            if result.risk_level.value > combined.risk_level.value:
                combined.risk_level = result.risk_level
            
            # Combine warnings and errors
            combined.warnings.extend(result.warnings)
            combined.errors.extend(result.errors)
            
            # If any result is invalid, combined is invalid
            if not result.is_valid:
                combined.is_valid = False
        
        return combined
    
    def _add_to_history(self, action: ActionContext):
        """Add action to history for pattern detection."""
        self.action_history.append(action)
        if len(self.action_history) > self.max_history:
            self.action_history.pop(0)
    
    def _detect_suspicious_pattern(self) -> bool:
        """Detect suspicious patterns in action history."""
        if len(self.action_history) < 5:
            return False
        
        # Check for repeated identical actions
        recent_actions = self.action_history[-5:]
        action_types = [a.action_type for a in recent_actions]
        
        # If all recent actions are the same type
        if len(set(action_types)) == 1:
            return True
        
        # Check for rapid navigation pattern
        nav_actions = [
            a for a in recent_actions
            if a.action_type in [ActionType.NAVIGATE, ActionType.GO_BACK, ActionType.GO_FORWARD]
        ]
        if len(nav_actions) >= 3:
            return True
        
        return False
    
    async def get_rollback_snapshot(self, page_id: str = "default") -> Optional[DOMSnapshot]:
        """Get the latest snapshot for rollback."""
        return self.snapshot_manager.get_latest_snapshot(page_id)
    
    async def rollback(self, page, snapshot: Optional[DOMSnapshot] = None) -> bool:
        """Rollback to a snapshot."""
        if not snapshot:
            page_id = str(id(page))
            snapshot = self.get_rollback_snapshot(page_id)
        
        if not snapshot:
            logger.error("No snapshot available for rollback")
            return False
        
        return await self.snapshot_manager.rollback_to_snapshot(page, snapshot)
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return {
            "total_validations": len(self.action_history),
            "risk_distribution": self._get_risk_distribution(),
            "common_warnings": self._get_common_warnings(),
            "sandbox_test_results": len(self.sandbox_executor.test_results),
        }
    
    def _get_risk_distribution(self) -> Dict[str, int]:
        """Get distribution of risk levels from history."""
        distribution = {level.name: 0 for level in RiskLevel}
        # Note: We don't store risk levels in history, so this is a placeholder
        return distribution
    
    def _get_common_warnings(self) -> List[str]:
        """Get common warnings from recent validations."""
        # This would analyze recent validation results
        return []


# Integration with existing nexus modules
class SafeActor:
    """Wrapper for browser actor with safety validation."""
    
    def __init__(self, actor, validator: Optional[ActionValidator] = None):
        self.actor = actor
        self.validator = validator or ActionValidator()
        self.page = getattr(actor, 'page', None)
    
    async def safe_click(self, selector: str, **kwargs) -> bool:
        """Safely click an element with validation."""
        action = ActionContext(
            action_type=ActionType.CLICK,
            selector=selector,
            metadata=kwargs,
        )
        
        result = await self.validator.validate_action(
            action, self.page, test_in_sandbox=True
        )
        
        if not result.is_valid:
            logger.warning(f"Click action rejected: {result.errors}")
            return False
        
        if result.requires_confirmation:
            # In a real implementation, this would wait for user confirmation
            logger.info("Click action requires user confirmation")
        
        # Execute the action
        try:
            await self.actor.click(selector, **kwargs)
            return True
        except Exception as e:
            logger.error(f"Click action failed: {e}")
            
            # Attempt rollback if enabled
            if self.validator.config.enable_rollback:
                await self.validator.rollback(self.page)
            
            return False
    
    async def safe_navigate(self, url: str, **kwargs) -> bool:
        """Safely navigate to a URL with validation."""
        action = ActionContext(
            action_type=ActionType.NAVIGATE,
            url=url,
            metadata=kwargs,
        )
        
        result = await self.validator.validate_action(
            action, self.page, test_in_sandbox=False
        )
        
        if not result.is_valid:
            logger.warning(f"Navigation rejected: {result.errors}")
            return False
        
        if result.risk_level >= RiskLevel.HIGH:
            logger.warning(f"High-risk navigation to: {url}")
        
        # Execute the action
        try:
            await self.actor.goto(url, **kwargs)
            return True
        except Exception as e:
            logger.error(f"Navigation failed: {e}")
            return False
    
    async def safe_type(self, selector: str, text: str, **kwargs) -> bool:
        """Safely type text into an element with validation."""
        action = ActionContext(
            action_type=ActionType.TYPE,
            selector=selector,
            value=text,
            metadata=kwargs,
        )
        
        result = await self.validator.validate_action(
            action, self.page, test_in_sandbox=True
        )
        
        if not result.is_valid:
            logger.warning(f"Type action rejected: {result.errors}")
            return False
        
        # Execute the action
        try:
            await self.actor.type(selector, text, **kwargs)
            return True
        except Exception as e:
            logger.error(f"Type action failed: {e}")
            
            # Attempt rollback if enabled
            if self.validator.config.enable_rollback:
                await self.validator.rollback(self.page)
            
            return False


# Factory function for easy integration
def create_action_validator(
    enable_sandbox: bool = True,
    enable_rollback: bool = True,
    risk_threshold: RiskLevel = RiskLevel.HIGH,
) -> ActionValidator:
    """Create an ActionValidator with common configuration."""
    config = SafetyConfig(
        enable_sandbox=enable_sandbox,
        enable_rollback=enable_rollback,
        require_confirmation_above=risk_threshold,
    )
    
    return ActionValidator(config=config)


# Export main classes
__all__ = [
    'RiskLevel',
    'ActionType',
    'DOMSnapshot',
    'ActionContext',
    'ValidationResult',
    'SafetyConfig',
    'DOMSnapshotManager',
    'SandboxedExecutor',
    'ActionValidator',
    'SafeActor',
    'create_action_validator',
]