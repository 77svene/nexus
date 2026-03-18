"""
nexus/safety/sandbox.py

Action Validation & Safety System - Prevent destructive actions with pre-execution validation,
sandboxed testing environments, and rollback capabilities.
"""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Awaitable, Union
import logging
from datetime import datetime

from playwright.async_api import Page, BrowserContext, async_playwright

from nexus.actor.element import ElementActor
from nexus.actor.mouse import MouseActor
from nexus.actor.page import PageActor
from nexus.agent.views import Action, ActionType


logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk levels for browser actions"""
    LOW = 1      # Read-only operations, no side effects
    MEDIUM = 2   # Minor modifications, easily reversible
    HIGH = 3     # Significant changes, potentially irreversible
    CRITICAL = 4 # Destructive actions, data loss possible


class ActionTypeCategory(Enum):
    """Categories of browser actions for risk assessment"""
    READ_ONLY = "read_only"
    FORM_INPUT = "form_input"
    NAVIGATION = "navigation"
    DOM_MODIFICATION = "dom_modification"
    DATA_EXTRACTION = "data_extraction"
    FILE_OPERATION = "file_operation"
    AUTHENTICATION = "authentication"
    DESTRUCTIVE = "destructive"


@dataclass
class RiskAssessment:
    """Assessment of action risk"""
    risk_level: RiskLevel
    risk_score: float  # 0.0 to 1.0
    category: ActionTypeCategory
    warnings: List[str]
    requires_confirmation: bool
    reversible: bool
    estimated_impact: str


@dataclass
class DOMSnapshot:
    """Snapshot of DOM state for rollback"""
    timestamp: float
    page_url: str
    dom_hash: str
    dom_content: str
    scroll_position: Dict[str, int]
    viewport_size: Dict[str, int]
    cookies: List[Dict[str, Any]]
    local_storage: Dict[str, str]
    session_storage: Dict[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SandboxResult:
    """Result of sandboxed execution"""
    success: bool
    error: Optional[str]
    execution_time: float
    dom_changes: List[str]
    network_requests: List[str]
    console_logs: List[str]
    risk_assessment: RiskAssessment


class ActionValidator:
    """Validates actions before execution and assesses risk"""
    
    # Risk mapping for different action types
    RISK_MAPPING = {
        # Read-only actions (LOW risk)
        ActionType.GET_TEXT: (RiskLevel.LOW, ActionTypeCategory.READ_ONLY),
        ActionType.GET_ATTRIBUTE: (RiskLevel.LOW, ActionTypeCategory.READ_ONLY),
        ActionType.SCREENSHOT: (RiskLevel.LOW, ActionTypeCategory.READ_ONLY),
        ActionType.SCROLL: (RiskLevel.LOW, ActionTypeCategory.READ_ONLY),
        ActionType.HOVER: (RiskLevel.LOW, ActionTypeCategory.READ_ONLY),
        
        # Form input actions (MEDIUM risk)
        ActionType.CLICK: (RiskLevel.MEDIUM, ActionTypeCategory.FORM_INPUT),
        ActionType.TYPE: (RiskLevel.MEDIUM, ActionTypeCategory.FORM_INPUT),
        ActionType.SELECT_OPTION: (RiskLevel.MEDIUM, ActionTypeCategory.FORM_INPUT),
        ActionType.CHECK: (RiskLevel.MEDIUM, ActionTypeCategory.FORM_INPUT),
        ActionType.UNCHECK: (RiskLevel.MEDIUM, ActionTypeCategory.FORM_INPUT),
        
        # Navigation actions (HIGH risk)
        ActionType.NAVIGATE: (RiskLevel.HIGH, ActionTypeCategory.NAVIGATION),
        ActionType.GO_BACK: (RiskLevel.HIGH, ActionTypeCategory.NAVIGATION),
        ActionType.GO_FORWARD: (RiskLevel.HIGH, ActionTypeCategory.NAVIGATION),
        ActionType.REFRESH: (RiskLevel.HIGH, ActionTypeCategory.NAVIGATION),
        
        # DOM modification actions (HIGH risk)
        ActionType.EVALUATE: (RiskLevel.HIGH, ActionTypeCategory.DOM_MODIFICATION),
        ActionType.EXECUTE_SCRIPT: (RiskLevel.HIGH, ActionTypeCategory.DOM_MODIFICATION),
        
        # Potentially destructive actions (CRITICAL risk)
        ActionType.DELETE: (RiskLevel.CRITICAL, ActionTypeCategory.DESTRUCTIVE),
        ActionType.SUBMIT_FORM: (RiskLevel.CRITICAL, ActionTypeCategory.AUTHENTICATION),
        ActionType.CLOSE: (RiskLevel.CRITICAL, ActionTypeCategory.DESTRUCTIVE),
    }
    
    # Patterns that increase risk
    HIGH_RISK_PATTERNS = [
        r"delete", r"remove", r"destroy", r"drop", r"truncate",
        r"password", r"credit.?card", r"ssn", r"social.?security",
        r"admin", r"root", r"sudo", r"format", r"reset",
        r"submit", r"confirm", r"purchase", r"buy", r"pay"
    ]
    
    def __init__(self):
        self.custom_validators: Dict[ActionType, Callable[[Action], Awaitable[bool]]] = {}
    
    def register_validator(self, action_type: ActionType, 
                          validator: Callable[[Action], Awaitable[bool]]):
        """Register custom validator for specific action type"""
        self.custom_validators[action_type] = validator
    
    async def assess_risk(self, action: Action, context: Optional[Dict] = None) -> RiskAssessment:
        """Assess risk level of an action"""
        action_type = action.type
        
        # Get base risk from mapping
        if action_type in self.RISK_MAPPING:
            base_risk, category = self.RISK_MAPPING[action_type]
        else:
            base_risk, category = RiskLevel.MEDIUM, ActionTypeCategory.FORM_INPUT
        
        # Calculate risk score (0.0 to 1.0)
        risk_score = base_risk.value / 4.0
        
        warnings = []
        requires_confirmation = base_risk.value >= RiskLevel.HIGH.value
        reversible = base_risk.value <= RiskLevel.MEDIUM.value
        
        # Check for high-risk patterns in action parameters
        action_str = json.dumps(action.dict()).lower()
        for pattern in self.HIGH_RISK_PATTERNS:
            if pattern in action_str:
                risk_score = min(1.0, risk_score + 0.2)
                warnings.append(f"Action contains high-risk pattern: {pattern}")
                if base_risk.value < RiskLevel.HIGH.value:
                    requires_confirmation = True
        
        # Check custom validators
        if action_type in self.custom_validators:
            try:
                is_safe = await self.custom_validators[action_type](action)
                if not is_safe:
                    risk_score = min(1.0, risk_score + 0.3)
                    warnings.append("Custom validator flagged action as unsafe")
            except Exception as e:
                logger.warning(f"Custom validator failed: {e}")
        
        # Adjust risk based on context
        if context:
            # If we're on a sensitive page (login, payment, etc.)
            if context.get("page_category") in ["login", "payment", "admin"]:
                risk_score = min(1.0, risk_score + 0.2)
                warnings.append("Action on sensitive page")
            
            # If user is not logged in but action requires auth
            if context.get("requires_auth") and not context.get("is_authenticated"):
                risk_score = min(1.0, risk_score + 0.3)
                warnings.append("Action requires authentication but user not logged in")
        
        # Determine final risk level
        if risk_score >= 0.75:
            final_risk = RiskLevel.CRITICAL
        elif risk_score >= 0.5:
            final_risk = RiskLevel.HIGH
        elif risk_score >= 0.25:
            final_risk = RiskLevel.MEDIUM
        else:
            final_risk = RiskLevel.LOW
        
        # Estimate impact
        impact_map = {
            RiskLevel.LOW: "No side effects, read-only operation",
            RiskLevel.MEDIUM: "Minor page modifications, easily reversible",
            RiskLevel.HIGH: "Significant changes, may require manual intervention",
            RiskLevel.CRITICAL: "Destructive action, potential data loss"
        }
        
        return RiskAssessment(
            risk_level=final_risk,
            risk_score=risk_score,
            category=category,
            warnings=warnings,
            requires_confirmation=requires_confirmation or final_risk.value >= RiskLevel.HIGH.value,
            reversible=reversible,
            estimated_impact=impact_map[final_risk]
        )


class DOMSnapshotter:
    """Creates and manages DOM snapshots for rollback"""
    
    def __init__(self, max_snapshots: int = 10):
        self.max_snapshots = max_snapshots
        self.snapshots: List[DOMSnapshot] = []
    
    async def take_snapshot(self, page: Page) -> DOMSnapshot:
        """Take a snapshot of current DOM state"""
        try:
            # Get DOM content
            dom_content = await page.content()
            dom_hash = hashlib.md5(dom_content.encode()).hexdigest()
            
            # Get page state
            scroll_position = await page.evaluate("""() => {
                return {
                    x: window.scrollX,
                    y: window.scrollY
                }
            }""")
            
            viewport_size = page.viewport_size
            
            # Get storage (simplified - in production would need async storage API)
            cookies = await page.context.cookies()
            
            # Create snapshot
            snapshot = DOMSnapshot(
                timestamp=time.time(),
                page_url=page.url,
                dom_hash=dom_hash,
                dom_content=dom_content,
                scroll_position=scroll_position,
                viewport_size={"width": viewport_size["width"], "height": viewport_size["height"]} if viewport_size else {},
                cookies=cookies,
                local_storage={},  # Would need page.evaluate to get actual values
                session_storage={}  # Would need page.evaluate to get actual values
            )
            
            # Add to snapshots list
            self.snapshots.append(snapshot)
            
            # Trim old snapshots if exceeding max
            if len(self.snapshots) > self.max_snapshots:
                self.snapshots = self.snapshots[-self.max_snapshots:]
            
            logger.debug(f"DOM snapshot taken: {dom_hash[:8]} at {page.url}")
            return snapshot
            
        except Exception as e:
            logger.error(f"Failed to take DOM snapshot: {e}")
            raise
    
    async def restore_snapshot(self, page: Page, snapshot: DOMSnapshot) -> bool:
        """Restore DOM to a previous snapshot"""
        try:
            # Navigate to the snapshot URL if different
            if page.url != snapshot.page_url:
                await page.goto(snapshot.page_url)
            
            # Restore DOM content
            await page.set_content(snapshot.dom_content)
            
            # Restore scroll position
            await page.evaluate(f"""() => {{
                window.scrollTo({snapshot.scroll_position['x']}, {snapshot.scroll_position['y']});
            }}""")
            
            # Restore cookies
            if snapshot.cookies:
                await page.context.add_cookies(snapshot.cookies)
            
            # Note: localStorage and sessionStorage restoration would require
            # additional implementation with page.evaluate
            
            logger.info(f"DOM restored to snapshot: {snapshot.dom_hash[:8]}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore DOM snapshot: {e}")
            return False
    
    def get_snapshot_history(self) -> List[Dict[str, Any]]:
        """Get history of snapshots"""
        return [
            {
                "timestamp": s.timestamp,
                "url": s.page_url,
                "hash": s.dom_hash[:8],
                "datetime": datetime.fromtimestamp(s.timestamp).isoformat()
            }
            for s in self.snapshots
        ]


class SandboxExecutor:
    """Executes actions in a sandboxed environment for testing"""
    
    def __init__(self, browser_context: Optional[BrowserContext] = None):
        self.browser_context = browser_context
        self.sandbox_page: Optional[Page] = None
        self._monitoring_active = False
        self._captured_requests: List[str] = []
        self._captured_logs: List[str] = []
    
    async def initialize(self, browser_context: BrowserContext):
        """Initialize sandbox with browser context"""
        self.browser_context = browser_context
        self.sandbox_page = await self.browser_context.new_page()
        
        # Set up monitoring
        self._setup_monitoring()
    
    def _setup_monitoring(self):
        """Set up network and console monitoring"""
        if not self.sandbox_page:
            return
        
        # Monitor network requests
        self.sandbox_page.on("request", lambda req: self._captured_requests.append(req.url))
        
        # Monitor console logs
        self.sandbox_page.on("console", lambda msg: self._captured_logs.append(f"{msg.type}: {msg.text}"))
    
    async def execute_in_sandbox(self, action: Action, 
                                original_page: Page) -> SandboxResult:
        """Execute action in sandbox and return results"""
        if not self.sandbox_page:
            raise RuntimeError("Sandbox not initialized")
        
        start_time = time.time()
        self._captured_requests = []
        self._captured_logs = []
        
        try:
            # Clone current page state to sandbox
            await self._clone_page_state(original_page, self.sandbox_page)
            
            # Take initial DOM snapshot
            initial_dom = await self.sandbox_page.content()
            
            # Execute the action in sandbox
            # Note: This is simplified - actual implementation would need to
            # map Action to appropriate actor method
            success = await self._execute_action_in_sandbox(action)
            
            # Take final DOM snapshot
            final_dom = await self.sandbox_page.content()
            
            # Analyze DOM changes
            dom_changes = self._analyze_dom_changes(initial_dom, final_dom)
            
            execution_time = time.time() - start_time
            
            # Assess risk based on sandbox execution
            risk_assessment = await self._assess_sandbox_impact(
                dom_changes, 
                self._captured_requests,
                self._captured_logs
            )
            
            return SandboxResult(
                success=success,
                error=None if success else "Action failed in sandbox",
                execution_time=execution_time,
                dom_changes=dom_changes,
                network_requests=self._captured_requests.copy(),
                console_logs=self._captured_logs.copy(),
                risk_assessment=risk_assessment
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return SandboxResult(
                success=False,
                error=str(e),
                execution_time=execution_time,
                dom_changes=[],
                network_requests=self._captured_requests.copy(),
                console_logs=self._captured_logs.copy(),
                risk_assessment=RiskAssessment(
                    risk_level=RiskLevel.HIGH,
                    risk_score=0.8,
                    category=ActionTypeCategory.DESTRUCTIVE,
                    warnings=[f"Sandbox execution failed: {str(e)}"],
                    requires_confirmation=True,
                    reversible=False,
                    estimated_impact="Unknown - sandbox execution failed"
                )
            )
    
    async def _clone_page_state(self, source: Page, target: Page):
        """Clone state from source page to target page"""
        try:
            # Navigate to same URL
            await target.goto(source.url)
            
            # Copy cookies
            cookies = await source.context.cookies()
            if cookies:
                await target.context.add_cookies(cookies)
            
            # Note: In production, would also copy localStorage, sessionStorage,
            # and other page state
            
        except Exception as e:
            logger.warning(f"Failed to fully clone page state: {e}")
    
    async def _execute_action_in_sandbox(self, action: Action) -> bool:
        """Execute action in sandbox page"""
        # Simplified implementation - in production would use actual actors
        try:
            if action.type == ActionType.NAVIGATE:
                await self.sandbox_page.goto(action.parameters.get("url", ""))
            elif action.type == ActionType.CLICK:
                selector = action.parameters.get("selector", "")
                if selector:
                    await self.sandbox_page.click(selector)
            elif action.type == ActionType.TYPE:
                selector = action.parameters.get("selector", "")
                text = action.parameters.get("text", "")
                if selector and text:
                    await self.sandbox_page.fill(selector, text)
            # Add more action types as needed
            
            return True
        except Exception as e:
            logger.error(f"Sandbox action execution failed: {e}")
            return False
    
    def _analyze_dom_changes(self, before: str, after: str) -> List[str]:
        """Analyze changes between DOM states"""
        changes = []
        
        if before != after:
            changes.append("DOM structure modified")
            
            # Simple heuristic checks
            if len(after) < len(before) * 0.9:
                changes.append("Significant content removal detected")
            
            if "<form" in before.lower() and "<form" not in after.lower():
                changes.append("Form element removed")
            
            if "delete" in after.lower() or "remove" in after.lower():
                changes.append("Destructive language detected in DOM")
        
        return changes
    
    async def _assess_sandbox_impact(self, dom_changes: List[str],
                                   network_requests: List[str],
                                   console_logs: List[str]) -> RiskAssessment:
        """Assess risk based on sandbox execution results"""
        risk_score = 0.0
        warnings = []
        
        # DOM changes impact
        if dom_changes:
            risk_score += 0.3
            warnings.extend([f"DOM change: {change}" for change in dom_changes])
        
        # Network requests impact
        external_requests = [r for r in network_requests if not r.startswith("data:")]
        if external_requests:
            risk_score += 0.2
            warnings.append(f"{len(external_requests)} external network requests")
        
        # Console errors impact
        errors = [log for log in console_logs if "error" in log.lower()]
        if errors:
            risk_score += 0.2
            warnings.append(f"{len(errors)} console errors")
        
        # Determine risk level
        if risk_score >= 0.7:
            risk_level = RiskLevel.CRITICAL
        elif risk_score >= 0.5:
            risk_level = RiskLevel.HIGH
        elif risk_score >= 0.3:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        return RiskAssessment(
            risk_level=risk_level,
            risk_score=min(1.0, risk_score),
            category=ActionTypeCategory.DOM_MODIFICATION,
            warnings=warnings,
            requires_confirmation=risk_level.value >= RiskLevel.HIGH.value,
            reversible=risk_level.value <= RiskLevel.MEDIUM.value,
            estimated_impact=f"Sandbox execution: {len(dom_changes)} DOM changes, "
                           f"{len(network_requests)} network requests"
        )
    
    async def cleanup(self):
        """Clean up sandbox resources"""
        if self.sandbox_page:
            await self.sandbox_page.close()
            self.sandbox_page = None


class SafetyManager:
    """Main safety manager coordinating all safety features"""
    
    def __init__(self, browser_context: Optional[BrowserContext] = None,
                 user_confirmation_callback: Optional[Callable[[Action, RiskAssessment], Awaitable[bool]]] = None):
        self.validator = ActionValidator()
        self.snapshotter = DOMSnapshotter()
        self.sandbox_executor = SandboxExecutor()
        self.browser_context = browser_context
        self.user_confirmation_callback = user_confirmation_callback
        
        # Safety configuration
        self.config = {
            "enable_sandbox_testing": True,
            "sandbox_risk_threshold": RiskLevel.HIGH.value,
            "auto_snapshot_before_action": True,
            "require_confirmation_above": RiskLevel.MEDIUM.value,
            "max_rollback_snapshots": 10,
            "enable_risk_logging": True
        }
        
        # Statistics
        self.stats = {
            "total_actions": 0,
            "blocked_actions": 0,
            "confirmed_actions": 0,
            "sandbox_tests": 0,
            "rollbacks_performed": 0
        }
    
    async def initialize(self, browser_context: BrowserContext):
        """Initialize safety manager with browser context"""
        self.browser_context = browser_context
        await self.sandbox_executor.initialize(browser_context)
        logger.info("Safety manager initialized")
    
    async def validate_and_execute(self, action: Action, 
                                  execute_func: Callable[..., Awaitable[Any]],
                                  page: Page,
                                  context: Optional[Dict] = None) -> Any:
        """
        Validate action, optionally test in sandbox, and execute with safety measures
        
        Args:
            action: The action to validate and execute
            execute_func: Async function to execute the action
            page: Playwright page object
            context: Additional context for risk assessment
        
        Returns:
            Result from execute_func if action is approved
        """
        self.stats["total_actions"] += 1
        
        try:
            # Step 1: Initial risk assessment
            risk_assessment = await self.validator.assess_risk(action, context)
            
            if self.config["enable_risk_logging"]:
                logger.info(f"Action {action.type} assessed as {risk_assessment.risk_level.name} "
                          f"(score: {risk_assessment.risk_score:.2f})")
                for warning in risk_assessment.warnings:
                    logger.warning(f"  - {warning}")
            
            # Step 2: Check if sandbox testing is needed
            if (self.config["enable_sandbox_testing"] and 
                risk_assessment.risk_level.value >= self.config["sandbox_risk_threshold"]):
                
                logger.info(f"Testing action in sandbox due to high risk...")
                sandbox_result = await self.sandbox_executor.execute_in_sandbox(action, page)
                self.stats["sandbox_tests"] += 1
                
                if not sandbox_result.success:
                    self.stats["blocked_actions"] += 1
                    raise ActionBlockedError(
                        f"Action failed in sandbox: {sandbox_result.error}",
                        risk_assessment,
                        sandbox_result
                    )
                
                # Update risk assessment based on sandbox results
                risk_assessment = sandbox_result.risk_assessment
            
            # Step 3: Check if user confirmation is required
            if (risk_assessment.requires_confirmation or 
                risk_assessment.risk_level.value >= self.config["require_confirmation_above"]):
                
                if self.user_confirmation_callback:
                    confirmed = await self.user_confirmation_callback(action, risk_assessment)
                    if not confirmed:
                        self.stats["blocked_actions"] += 1
                        raise ActionBlockedError(
                            "User rejected action",
                            risk_assessment
                        )
                    self.stats["confirmed_actions"] += 1
                else:
                    # No confirmation callback, block high-risk actions
                    if risk_assessment.risk_level.value >= RiskLevel.HIGH.value:
                        self.stats["blocked_actions"] += 1
                        raise ActionBlockedError(
                            f"Action requires user confirmation (risk: {risk_assessment.risk_level.name})",
                            risk_assessment
                        )
            
            # Step 4: Take DOM snapshot if configured
            snapshot = None
            if self.config["auto_snapshot_before_action"]:
                try:
                    snapshot = await self.snapshotter.take_snapshot(page)
                except Exception as e:
                    logger.warning(f"Failed to take DOM snapshot: {e}")
            
            # Step 5: Execute the action
            try:
                result = await execute_func()
                return result
            except Exception as e:
                # Step 6: Rollback on failure if snapshot exists
                if snapshot:
                    logger.warning(f"Action failed, attempting rollback: {e}")
                    rollback_success = await self.snapshotter.restore_snapshot(page, snapshot)
                    if rollback_success:
                        self.stats["rollbacks_performed"] += 1
                        logger.info("Rollback successful")
                    else:
                        logger.error("Rollback failed")
                raise ActionExecutionError(f"Action execution failed: {e}", risk_assessment)
        
        except (ActionBlockedError, ActionExecutionError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error in safety manager: {e}")
            raise ActionExecutionError(f"Safety manager error: {e}", None)
    
    async def take_manual_snapshot(self, page: Page) -> str:
        """Take a manual snapshot and return its ID"""
        snapshot = await self.snapshotter.take_snapshot(page)
        return snapshot.dom_hash
    
    async def rollback_to_snapshot(self, page: Page, snapshot_hash: str) -> bool:
        """Rollback to a specific snapshot by hash"""
        for snapshot in self.snapshotter.snapshots:
            if snapshot.dom_hash == snapshot_hash:
                success = await self.snapshotter.restore_snapshot(page, snapshot)
                if success:
                    self.stats["rollbacks_performed"] += 1
                return success
        return False
    
    def get_safety_stats(self) -> Dict[str, Any]:
        """Get safety statistics"""
        return {
            **self.stats,
            "snapshot_count": len(self.snapshotter.snapshots),
            "config": self.config
        }
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update safety configuration"""
        self.config.update(new_config)
        logger.info(f"Safety config updated: {new_config}")
    
    async def cleanup(self):
        """Clean up safety manager resources"""
        await self.sandbox_executor.cleanup()
        logger.info("Safety manager cleaned up")


class ActionBlockedError(Exception):
    """Exception raised when an action is blocked by safety system"""
    def __init__(self, message: str, risk_assessment: RiskAssessment, 
                 sandbox_result: Optional[SandboxResult] = None):
        super().__init__(message)
        self.risk_assessment = risk_assessment
        self.sandbox_result = sandbox_result


class ActionExecutionError(Exception):
    """Exception raised when action execution fails"""
    def __init__(self, message: str, risk_assessment: Optional[RiskAssessment]):
        super().__init__(message)
        self.risk_assessment = risk_assessment


# Integration helpers for existing codebase
class SafeActorWrapper:
    """Wrapper for existing actors to add safety checks"""
    
    def __init__(self, actor: Union[ElementActor, MouseActor, PageActor],
                 safety_manager: SafetyManager, page: Page):
        self.actor = actor
        self.safety_manager = safety_manager
        self.page = page
    
    async def execute_action(self, action: Action, **kwargs) -> Any:
        """Execute action with safety checks"""
        async def execute():
            # Map action type to actor method
            if hasattr(self.actor, action.type.value):
                method = getattr(self.actor, action.type.value)
                return await method(**action.parameters, **kwargs)
            else:
                raise ValueError(f"Unsupported action type: {action.type}")
        
        return await self.safety_manager.validate_and_execute(
            action, execute, self.page
        )


# Factory function for easy integration
def create_safety_manager(browser_context: Optional[BrowserContext] = None,
                         user_confirmation_callback: Optional[Callable] = None) -> SafetyManager:
    """Factory function to create and configure safety manager"""
    manager = SafetyManager(browser_context, user_confirmation_callback)
    
    # Register custom validators for specific action types
    async def validate_navigation(action: Action) -> bool:
        """Custom validator for navigation actions"""
        url = action.parameters.get("url", "")
        # Block navigation to known dangerous domains
        dangerous_domains = ["malware.com", "phishing.net"]
        return not any(domain in url for domain in dangerous_domains)
    
    manager.validator.register_validator(ActionType.NAVIGATE, validate_navigation)
    
    return manager


# Example usage in agent code:
"""
# In agent/service.py or similar:
from nexus.safety.sandbox import create_safety_manager

class SafeAgent:
    def __init__(self, browser_context):
        self.safety_manager = create_safety_manager(
            browser_context,
            user_confirmation_callback=self.request_user_confirmation
        )
        await self.safety_manager.initialize(browser_context)
    
    async def request_user_confirmation(self, action, risk_assessment):
        # Show confirmation dialog to user
        print(f"High-risk action detected: {action.type}")
        print(f"Risk level: {risk_assessment.risk_level.name}")
        print(f"Warnings: {risk_assessment.warnings}")
        return input("Proceed? (y/n): ").lower() == 'y'
    
    async def execute_action(self, action, page):
        # Wrap the actual execution
        async def execute():
            # Your existing action execution logic here
            pass
        
        return await self.safety_manager.validate_and_execute(
            action, execute, page
        )
"""