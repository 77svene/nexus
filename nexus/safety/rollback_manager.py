"""
nexus/safety/rollback_manager.py

Action Validation & Safety System - Prevent destructive actions with pre-execution validation,
sandboxed testing environments, and rollback capabilities. Critical for production automation.
"""

import asyncio
import json
import time
import hashlib
import logging
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from pathlib import Path
import tempfile
import shutil
import pickle
from datetime import datetime

from playwright.async_api import Page, Browser, BrowserContext

from nexus.actor.element import Element
from nexus.actor.utils import get_element_by_selector
from nexus.agent.views import ActionResult


class RiskLevel(Enum):
    """Risk levels for browser actions"""
    LOW = 1      # Reading data, clicking safe elements
    MEDIUM = 2   # Form submissions, navigation
    HIGH = 3     # Data deletion, payment actions
    CRITICAL = 4 # Account changes, irreversible operations


class ActionType(Enum):
    """Types of browser actions for risk assessment"""
    NAVIGATE = "navigate"
    CLICK = "click"
    TYPE = "type"
    SELECT = "select"
    SUBMIT = "submit"
    DELETE = "delete"
    UPLOAD = "upload"
    DOWNLOAD = "download"
    EXECUTE_SCRIPT = "execute_script"
    CLOSE = "close"
    SCREENSHOT = "screenshot"
    EXTRACT = "extract"


@dataclass
class ActionRiskProfile:
    """Risk assessment for a browser action"""
    action_type: ActionType
    risk_level: RiskLevel
    risk_score: float  # 0.0 to 1.0
    potential_impact: str
    reversible: bool
    requires_confirmation: bool
    validation_rules: List[str]
    sandbox_recommended: bool


@dataclass
class DOMSnapshot:
    """Snapshot of DOM state for rollback"""
    timestamp: float
    url: str
    title: str
    html: str
    cookies: List[Dict]
    local_storage: Dict[str, str]
    session_storage: Dict[str, str]
    screenshot_path: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def snapshot_id(self) -> str:
        """Generate unique ID for this snapshot"""
        content = f"{self.timestamp}:{self.url}:{len(self.html)}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return asdict(self)


@dataclass
class ValidationResult:
    """Result of action validation"""
    valid: bool
    risk_profile: ActionRiskProfile
    warnings: List[str]
    errors: List[str]
    recommendations: List[str]
    sandbox_test_passed: Optional[bool] = None
    confirmation_required: bool = False


class RollbackManager:
    """
    Manages DOM snapshots and rollback capabilities for browser automation.
    Provides pre-execution validation and safety mechanisms.
    """

    def __init__(
        self,
        page: Page,
        max_snapshots: int = 50,
        snapshot_dir: Optional[Path] = None,
        enable_sandbox: bool = True,
        auto_confirm_threshold: RiskLevel = RiskLevel.MEDIUM,
        logger: Optional[logging.Logger] = None
    ):
        self.page = page
        self.max_snapshots = max_snapshots
        self.enable_sandbox = enable_sandbox
        self.auto_confirm_threshold = auto_confirm_threshold
        
        # Snapshot storage
        self._snapshots: List[DOMSnapshot] = []
        self._snapshot_dir = snapshot_dir or Path(tempfile.mkdtemp(prefix="browser_safety_"))
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Risk assessment cache
        self._risk_cache: Dict[str, ActionRiskProfile] = {}
        
        # Validation rules
        self._validation_rules: Dict[str, Callable] = self._initialize_validation_rules()
        
        # User confirmation callbacks
        self._confirmation_handlers: List[Callable] = []
        
        # Statistics
        self._stats = {
            "total_actions": 0,
            "blocked_actions": 0,
            "rollbacks_performed": 0,
            "snapshots_taken": 0,
            "high_risk_actions": 0,
            "sandbox_tests": 0
        }
        
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info(f"RollbackManager initialized. Snapshots dir: {self._snapshot_dir}")

    def _initialize_validation_rules(self) -> Dict[str, Callable]:
        """Initialize validation rule functions"""
        return {
            "no_password_fields": self._validate_no_passwords,
            "no_sensitive_urls": self._validate_sensitive_urls,
            "no_destructive_selectors": self._validate_destructive_selectors,
            "safe_navigation": self._validate_safe_navigation,
            "form_validation": self._validate_form_submission,
            "script_safety": self._validate_script_execution,
        }

    async def take_snapshot(self, label: str = "") -> DOMSnapshot:
        """
        Take a snapshot of current DOM state for potential rollback.
        
        Args:
            label: Optional label for the snapshot
            
        Returns:
            DOMSnapshot object
        """
        try:
            # Get current page state
            html = await self.page.content()
            url = self.page.url
            title = await self.page.title()
            
            # Get storage state
            cookies = await self.page.context.cookies()
            
            # Get local and session storage via JavaScript
            local_storage = await self.page.evaluate("""() => {
                const items = {};
                for (let i = 0; i < localStorage.length; i++) {
                    const key = localStorage.key(i);
                    items[key] = localStorage.getItem(key);
                }
                return items;
            }""")
            
            session_storage = await self.page.evaluate("""() => {
                const items = {};
                for (let i = 0; i < sessionStorage.length; i++) {
                    const key = sessionStorage.key(i);
                    items[key] = sessionStorage.getItem(key);
                }
                return items;
            }""")
            
            # Take screenshot if enabled
            screenshot_path = None
            try:
                screenshot_path = str(self._snapshot_dir / f"snapshot_{int(time.time())}.png")
                await self.page.screenshot(path=screenshot_path, full_page=False)
            except Exception as e:
                self.logger.warning(f"Failed to take screenshot: {e}")
            
            # Create snapshot
            snapshot = DOMSnapshot(
                timestamp=time.time(),
                url=url,
                title=title,
                html=html,
                cookies=cookies,
                local_storage=local_storage,
                session_storage=session_storage,
                screenshot_path=screenshot_path,
                metadata={
                    "label": label,
                    "viewport": await self.page.viewport_size,
                    "user_agent": await self.page.evaluate("navigator.userAgent")
                }
            )
            
            # Add to snapshots list
            self._snapshots.append(snapshot)
            self._stats["snapshots_taken"] += 1
            
            # Trim old snapshots if exceeding max
            if len(self._snapshots) > self.max_snapshots:
                removed = self._snapshots.pop(0)
                # Clean up old screenshot
                if removed.screenshot_path and Path(removed.screenshot_path).exists():
                    Path(removed.screenshot_path).unlink(missing_ok=True)
            
            self.logger.debug(f"Snapshot taken: {snapshot.snapshot_id} - {label}")
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Failed to take snapshot: {e}")
            raise

    async def rollback_to_snapshot(self, snapshot_id: str) -> bool:
        """
        Rollback to a specific snapshot.
        
        Args:
            snapshot_id: ID of the snapshot to rollback to
            
        Returns:
            True if rollback successful
        """
        snapshot = self._find_snapshot(snapshot_id)
        if not snapshot:
            self.logger.error(f"Snapshot {snapshot_id} not found")
            return False
        
        try:
            self.logger.info(f"Rolling back to snapshot {snapshot_id}")
            
            # Navigate to the snapshot URL if different
            if self.page.url != snapshot.url:
                await self.page.goto(snapshot.url, wait_until="domcontentloaded")
            
            # Restore DOM content
            await self.page.set_content(snapshot.html)
            
            # Restore cookies
            if snapshot.cookies:
                await self.page.context.add_cookies(snapshot.cookies)
            
            # Restore local storage
            if snapshot.local_storage:
                await self.page.evaluate("""(items) => {
                    localStorage.clear();
                    for (const [key, value] of Object.entries(items)) {
                        localStorage.setItem(key, value);
                    }
                }""", snapshot.local_storage)
            
            # Restore session storage
            if snapshot.session_storage:
                await self.page.evaluate("""(items) => {
                    sessionStorage.clear();
                    for (const [key, value] of Object.entries(items)) {
                        sessionStorage.setItem(key, value);
                    }
                }""", snapshot.session_storage)
            
            # Wait for page to stabilize
            await self.page.wait_for_load_state("networkidle")
            
            self._stats["rollbacks_performed"] += 1
            self.logger.info(f"Successfully rolled back to {snapshot_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return False

    async def rollback_last(self, steps: int = 1) -> bool:
        """
        Rollback to the nth previous snapshot.
        
        Args:
            steps: Number of snapshots to go back
            
        Returns:
            True if rollback successful
        """
        if len(self._snapshots) < steps:
            self.logger.warning(f"Not enough snapshots for {steps} step rollback")
            return False
        
        target_snapshot = self._snapshots[-steps]
        return await self.rollback_to_snapshot(target_snapshot.snapshot_id)

    async def validate_action(
        self,
        action_type: ActionType,
        selector: Optional[str] = None,
        value: Optional[str] = None,
        url: Optional[str] = None,
        script: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> ValidationResult:
        """
        Validate a browser action before execution.
        
        Args:
            action_type: Type of action to validate
            selector: CSS selector for element actions
            value: Value for input actions
            url: URL for navigation
            script: JavaScript for script execution
            metadata: Additional metadata
            
        Returns:
            ValidationResult with risk assessment
        """
        self._stats["total_actions"] += 1
        
        # Get risk profile for action
        risk_profile = self._assess_risk(
            action_type, selector, value, url, script, metadata
        )
        
        # Run validation rules
        warnings = []
        errors = []
        recommendations = []
        
        for rule_name, rule_func in self._validation_rules.items():
            try:
                rule_result = await rule_func(
                    action_type, selector, value, url, script, metadata
                )
                if rule_result.get("warning"):
                    warnings.append(rule_result["warning"])
                if rule_result.get("error"):
                    errors.append(rule_result["error"])
                if rule_result.get("recommendation"):
                    recommendations.append(rule_result["recommendation"])
            except Exception as e:
                self.logger.warning(f"Validation rule {rule_name} failed: {e}")
        
        # Determine if confirmation is required
        confirmation_required = (
            risk_profile.risk_level.value >= self.auto_confirm_threshold.value
            and risk_profile.requires_confirmation
        )
        
        # Sandbox test if enabled and recommended
        sandbox_test_passed = None
        if self.enable_sandbox and risk_profile.sandbox_recommended:
            sandbox_test_passed = await self._test_in_sandbox(
                action_type, selector, value, url, script
            )
            if not sandbox_test_passed:
                warnings.append("Action failed sandbox test")
        
        # Check if action should be blocked
        valid = len(errors) == 0
        if not valid:
            self._stats["blocked_actions"] += 1
        
        if risk_profile.risk_level == RiskLevel.HIGH:
            self._stats["high_risk_actions"] += 1
        
        return ValidationResult(
            valid=valid,
            risk_profile=risk_profile,
            warnings=warnings,
            errors=errors,
            recommendations=recommendations,
            sandbox_test_passed=sandbox_test_passed,
            confirmation_required=confirmation_required
        )

    async def safe_execute(
        self,
        action_func: Callable,
        action_type: ActionType,
        selector: Optional[str] = None,
        value: Optional[str] = None,
        url: Optional[str] = None,
        script: Optional[str] = None,
        metadata: Optional[Dict] = None,
        require_snapshot: bool = True,
        auto_rollback_on_failure: bool = True
    ) -> ActionResult:
        """
        Safely execute an action with validation and rollback capabilities.
        
        Args:
            action_func: Async function to execute the action
            action_type: Type of action
            selector: CSS selector for element actions
            value: Value for input actions
            url: URL for navigation
            script: JavaScript for script execution
            metadata: Additional metadata
            require_snapshot: Whether to take snapshot before execution
            auto_rollback_on_failure: Whether to auto-rollback on failure
            
        Returns:
            ActionResult with execution status
        """
        # Validate action first
        validation = await self.validate_action(
            action_type, selector, value, url, script, metadata
        )
        
        if not validation.valid:
            return ActionResult(
                success=False,
                error=f"Action validation failed: {', '.join(validation.errors)}",
                metadata={"validation": asdict(validation)}
            )
        
        # Request confirmation if needed
        if validation.confirmation_required:
            confirmed = await self._request_confirmation(validation)
            if not confirmed:
                return ActionResult(
                    success=False,
                    error="Action cancelled by user",
                    metadata={"validation": asdict(validation)}
                )
        
        # Take snapshot if required
        snapshot = None
        if require_snapshot:
            snapshot = await self.take_snapshot(
                label=f"pre_{action_type.value}_{selector or url or 'action'}"
            )
        
        try:
            # Execute the action
            result = await action_func()
            
            # Check if action succeeded
            if isinstance(result, ActionResult) and not result.success:
                if auto_rollback_on_failure and snapshot:
                    self.logger.warning(f"Action failed, rolling back to {snapshot.snapshot_id}")
                    await self.rollback_to_snapshot(snapshot.snapshot_id)
                return result
            
            return ActionResult(
                success=True,
                metadata={
                    "validation": asdict(validation),
                    "snapshot_id": snapshot.snapshot_id if snapshot else None
                }
            )
            
        except Exception as e:
            self.logger.error(f"Action execution failed: {e}")
            
            if auto_rollback_on_failure and snapshot:
                self.logger.warning(f"Rolling back due to exception")
                await self.rollback_to_snapshot(snapshot.snapshot_id)
            
            return ActionResult(
                success=False,
                error=str(e),
                metadata={
                    "validation": asdict(validation),
                    "snapshot_id": snapshot.snapshot_id if snapshot else None
                }
            )

    def _assess_risk(
        self,
        action_type: ActionType,
        selector: Optional[str],
        value: Optional[str],
        url: Optional[str],
        script: Optional[str],
        metadata: Optional[Dict]
    ) -> ActionRiskProfile:
        """Assess risk level for an action"""
        cache_key = f"{action_type.value}:{selector}:{url}"
        if cache_key in self._risk_cache:
            return self._risk_cache[cache_key]
        
        # Default risk assessment
        risk_level = RiskLevel.LOW
        risk_score = 0.1
        potential_impact = "Minimal impact"
        reversible = True
        requires_confirmation = False
        sandbox_recommended = False
        validation_rules = []
        
        # Action-specific risk assessment
        if action_type == ActionType.NAVIGATE:
            if url:
                if any(x in url.lower() for x in ["delete", "remove", "destroy", "drop"]):
                    risk_level = RiskLevel.HIGH
                    risk_score = 0.8
                    potential_impact = "Navigation to potentially destructive page"
                    requires_confirmation = True
                elif any(x in url.lower() for x in ["admin", "settings", "account"]):
                    risk_level = RiskLevel.MEDIUM
                    risk_score = 0.5
                    potential_impact = "Navigation to sensitive area"
                    validation_rules.append("safe_navigation")
        
        elif action_type == ActionType.CLICK:
            if selector:
                if any(x in selector.lower() for x in ["delete", "remove", "cancel", "close"]):
                    risk_level = RiskLevel.HIGH
                    risk_score = 0.7
                    potential_impact = "Clicking potentially destructive element"
                    requires_confirmation = True
                    validation_rules.append("no_destructive_selectors")
                elif any(x in selector.lower() for x in ["submit", "save", "confirm"]):
                    risk_level = RiskLevel.MEDIUM
                    risk_score = 0.4
                    potential_impact = "Form submission"
                    validation_rules.append("form_validation")
        
        elif action_type == ActionType.TYPE:
            if selector and "password" in selector.lower():
                risk_level = RiskLevel.HIGH
                risk_score = 0.9
                potential_impact = "Entering sensitive information"
                reversible = False
                requires_confirmation = True
                validation_rules.append("no_password_fields")
            elif value and len(value) > 1000:
                risk_level = RiskLevel.MEDIUM
                risk_score = 0.3
                potential_impact = "Large text input"
        
        elif action_type == ActionType.SUBMIT:
            risk_level = RiskLevel.MEDIUM
            risk_score = 0.5
            potential_impact = "Form submission with potential side effects"
            requires_confirmation = True
            validation_rules.append("form_validation")
        
        elif action_type == ActionType.DELETE:
            risk_level = RiskLevel.HIGH
            risk_score = 0.9
            potential_impact = "Data deletion"
            reversible = False
            requires_confirmation = True
            sandbox_recommended = True
        
        elif action_type == ActionType.EXECUTE_SCRIPT:
            risk_level = RiskLevel.HIGH
            risk_score = 0.8
            potential_impact = "Arbitrary code execution"
            requires_confirmation = True
            sandbox_recommended = True
            validation_rules.append("script_safety")
        
        elif action_type == ActionType.CLOSE:
            risk_level = RiskLevel.MEDIUM
            risk_score = 0.6
            potential_impact = "Closing browser/tab"
            requires_confirmation = True
        
        profile = ActionRiskProfile(
            action_type=action_type,
            risk_level=risk_level,
            risk_score=risk_score,
            potential_impact=potential_impact,
            reversible=reversible,
            requires_confirmation=requires_confirmation,
            validation_rules=validation_rules,
            sandbox_recommended=sandbox_recommended
        )
        
        self._risk_cache[cache_key] = profile
        return profile

    async def _test_in_sandbox(
        self,
        action_type: ActionType,
        selector: Optional[str],
        value: Optional[str],
        url: Optional[str],
        script: Optional[str]
    ) -> bool:
        """Test action in sandbox environment"""
        if not self.enable_sandbox:
            return True
        
        self._stats["sandbox_tests"] += 1
        
        try:
            # Create a new browser context for sandbox
            context = await self.page.context.browser.new_context()
            sandbox_page = await context.new_page()
            
            # Copy current page state to sandbox
            current_html = await self.page.content()
            await sandbox_page.set_content(current_html)
            
            # Test the action in sandbox
            test_passed = False
            
            if action_type == ActionType.CLICK and selector:
                try:
                    element = await sandbox_page.query_selector(selector)
                    if element:
                        await element.click(timeout=5000)
                        test_passed = True
                except:
                    test_passed = False
            
            elif action_type == ActionType.TYPE and selector and value:
                try:
                    element = await sandbox_page.query_selector(selector)
                    if element:
                        await element.fill(value)
                        test_passed = True
                except:
                    test_passed = False
            
            elif action_type == ActionType.NAVIGATE and url:
                try:
                    response = await sandbox_page.goto(url, timeout=10000)
                    test_passed = response and response.ok
                except:
                    test_passed = False
            
            elif action_type == ActionType.EXECUTE_SCRIPT and script:
                try:
                    await sandbox_page.evaluate(script)
                    test_passed = True
                except:
                    test_passed = False
            
            else:
                # For other actions, assume they pass sandbox
                test_passed = True
            
            # Clean up sandbox
            await context.close()
            
            return test_passed
            
        except Exception as e:
            self.logger.warning(f"Sandbox test failed: {e}")
            return False

    async def _request_confirmation(self, validation: ValidationResult) -> bool:
        """Request user confirmation for high-risk actions"""
        if not self._confirmation_handlers:
            # Default: log warning and require manual intervention
            self.logger.warning(
                f"High-risk action requires confirmation: {validation.risk_profile.potential_impact}"
            )
            return False
        
        # Call registered confirmation handlers
        for handler in self._confirmation_handlers:
            try:
                if await handler(validation):
                    return True
            except Exception as e:
                self.logger.error(f"Confirmation handler failed: {e}")
        
        return False

    def register_confirmation_handler(self, handler: Callable):
        """Register a confirmation handler callback"""
        self._confirmation_handlers.append(handler)

    # Validation rule implementations
    async def _validate_no_passwords(self, action_type, selector, value, url, script, metadata):
        """Check for password field interactions"""
        if action_type == ActionType.TYPE and selector:
            if any(x in selector.lower() for x in ["password", "passwd", "pwd", "secret"]):
                return {
                    "warning": "Interacting with password field",
                    "recommendation": "Consider using secure credential management"
                }
        return {}

    async def _validate_sensitive_urls(self, action_type, selector, value, url, script, metadata):
        """Check for sensitive URL patterns"""
        if action_type == ActionType.NAVIGATE and url:
            sensitive_patterns = [
                "delete", "remove", "drop", "destroy", "reset",
                "admin", "root", "sudo", "superuser",
                "payment", "checkout", "billing"
            ]
            url_lower = url.lower()
            for pattern in sensitive_patterns:
                if pattern in url_lower:
                    return {
                        "warning": f"URL contains sensitive pattern: {pattern}",
                        "recommendation": "Verify this action is intentional"
                    }
        return {}

    async def _validate_destructive_selectors(self, action_type, selector, value, url, script, metadata):
        """Check for destructive element selectors"""
        if action_type == ActionType.CLICK and selector:
            destructive_patterns = [
                "delete", "remove", "destroy", "drop",
                "cancel", "close", "exit", "quit",
                "reset", "clear", "wipe"
            ]
            selector_lower = selector.lower()
            for pattern in destructive_patterns:
                if pattern in selector_lower:
                    return {
                        "warning": f"Selector contains destructive pattern: {pattern}",
                        "recommendation": "Double-check element purpose"
                    }
        return {}

    async def _validate_safe_navigation(self, action_type, selector, value, url, script, metadata):
        """Validate navigation safety"""
        if action_type == ActionType.NAVIGATE and url:
            # Check if navigating away from current domain
            current_domain = self._extract_domain(self.page.url)
            target_domain = self._extract_domain(url)
            
            if current_domain and target_domain and current_domain != target_domain:
                return {
                    "warning": f"Cross-domain navigation: {current_domain} -> {target_domain}",
                    "recommendation": "Verify target domain is trusted"
                }
        return {}

    async def _validate_form_submission(self, action_type, selector, value, url, script, metadata):
        """Validate form submissions"""
        if action_type in [ActionType.SUBMIT, ActionType.CLICK]:
            if selector and any(x in selector.lower() for x in ["submit", "save", "confirm"]):
                # Check for empty required fields
                try:
                    empty_fields = await self.page.evaluate("""() => {
                        const required = document.querySelectorAll('[required]');
                        const empty = [];
                        required.forEach(field => {
                            if (!field.value) empty.push(field.name || field.id);
                        });
                        return empty;
                    }""")
                    
                    if empty_fields:
                        return {
                            "warning": f"Form has empty required fields: {', '.join(empty_fields)}",
                            "recommendation": "Fill required fields before submission"
                        }
                except:
                    pass
        return {}

    async def _validate_script_execution(self, action_type, selector, value, url, script, metadata):
        """Validate script execution safety"""
        if action_type == ActionType.EXECUTE_SCRIPT and script:
            dangerous_patterns = [
                "eval(", "Function(", "setTimeout(", "setInterval(",
                "document.write", "innerHTML", "outerHTML",
                "localStorage.clear", "sessionStorage.clear",
                "window.location", "document.cookie"
            ]
            
            script_lower = script.lower()
            for pattern in dangerous_patterns:
                if pattern in script_lower:
                    return {
                        "warning": f"Script contains potentially dangerous pattern: {pattern}",
                        "recommendation": "Review script for security issues"
                    }
        return {}

    def _extract_domain(self, url: str) -> Optional[str]:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc
        except:
            return None

    def _find_snapshot(self, snapshot_id: str) -> Optional[DOMSnapshot]:
        """Find snapshot by ID"""
        for snapshot in self._snapshots:
            if snapshot.snapshot_id == snapshot_id:
                return snapshot
        return None

    def get_snapshots(self, limit: int = 10) -> List[Dict]:
        """Get recent snapshots"""
        snapshots = self._snapshots[-limit:] if limit else self._snapshots
        return [
            {
                "id": s.snapshot_id,
                "timestamp": datetime.fromtimestamp(s.timestamp).isoformat(),
                "url": s.url,
                "title": s.title,
                "label": s.metadata.get("label", "")
            }
            for s in reversed(snapshots)
        ]

    def get_statistics(self) -> Dict:
        """Get safety statistics"""
        return {
            **self._stats,
            "current_snapshots": len(self._snapshots),
            "risk_cache_size": len(self._risk_cache),
            "snapshot_dir": str(self._snapshot_dir)
        }

    async def export_snapshot(self, snapshot_id: str, export_path: Path) -> bool:
        """Export a snapshot to file"""
        snapshot = self._find_snapshot(snapshot_id)
        if not snapshot:
            return False
        
        try:
            export_path.parent.mkdir(parents=True, exist_ok=True)
            with open(export_path, 'wb') as f:
                pickle.dump(snapshot, f)
            return True
        except Exception as e:
            self.logger.error(f"Failed to export snapshot: {e}")
            return False

    async def import_snapshot(self, import_path: Path) -> Optional[str]:
        """Import a snapshot from file"""
        try:
            with open(import_path, 'rb') as f:
                snapshot = pickle.load(f)
            
            if not isinstance(snapshot, DOMSnapshot):
                self.logger.error("Invalid snapshot format")
                return None
            
            self._snapshots.append(snapshot)
            return snapshot.snapshot_id
        except Exception as e:
            self.logger.error(f"Failed to import snapshot: {e}")
            return None

    async def cleanup(self):
        """Clean up resources"""
        # Remove old screenshots
        for snapshot in self._snapshots:
            if snapshot.screenshot_path and Path(snapshot.screenshot_path).exists():
                try:
                    Path(snapshot.screenshot_path).unlink(missing_ok=True)
                except:
                    pass
        
        # Clear snapshots
        self._snapshots.clear()
        self._risk_cache.clear()
        
        self.logger.info("RollbackManager cleaned up")


# Integration helpers for existing codebase
class SafetyWrapper:
    """
    Wrapper to add safety features to existing browser automation code.
    """
    
    def __init__(self, rollback_manager: RollbackManager):
        self.rm = rollback_manager
    
    async def safe_click(self, selector: str, **kwargs) -> ActionResult:
        """Safely click an element with validation"""
        from nexus.actor.mouse import click
        
        async def click_action():
            return await click(self.rm.page, selector, **kwargs)
        
        return await self.rm.safe_execute(
            click_action,
            ActionType.CLICK,
            selector=selector
        )
    
    async def safe_type(self, selector: str, text: str, **kwargs) -> ActionResult:
        """Safely type text with validation"""
        from nexus.actor.element import type_text
        
        async def type_action():
            return await type_text(self.rm.page, selector, text, **kwargs)
        
        return await self.rm.safe_execute(
            type_action,
            ActionType.TYPE,
            selector=selector,
            value=text
        )
    
    async def safe_navigate(self, url: str, **kwargs) -> ActionResult:
        """Safely navigate to URL with validation"""
        async def navigate_action():
            return await self.rm.page.goto(url, **kwargs)
        
        return await self.rm.safe_execute(
            navigate_action,
            ActionType.NAVIGATE,
            url=url
        )
    
    async def safe_extract(self, selector: str, attribute: str = "textContent") -> ActionResult:
        """Safely extract data from element"""
        async def extract_action():
            element = await self.rm.page.query_selector(selector)
            if element:
                value = await element.get_attribute(attribute)
                return ActionResult(success=True, data=value)
            return ActionResult(success=False, error="Element not found")
        
        return await self.rm.safe_execute(
            extract_action,
            ActionType.EXTRACT,
            selector=selector,
            require_snapshot=False  # Read-only, no snapshot needed
        )


# Factory function for easy integration
def create_safety_manager(
    page: Page,
    **kwargs
) -> Tuple[RollbackManager, SafetyWrapper]:
    """
    Create and return safety manager and wrapper instances.
    
    Args:
        page: Playwright page instance
        **kwargs: Additional arguments for RollbackManager
        
    Returns:
        Tuple of (RollbackManager, SafetyWrapper)
    """
    manager = RollbackManager(page, **kwargs)
    wrapper = SafetyWrapper(manager)
    return manager, wrapper


# Export main classes
__all__ = [
    "RollbackManager",
    "SafetyWrapper",
    "DOMSnapshot",
    "ValidationResult",
    "ActionRiskProfile",
    "RiskLevel",
    "ActionType",
    "create_safety_manager"
]