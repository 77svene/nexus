"""testing/agent_tester.py"""

import asyncio
import json
import logging
import random
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from unittest.mock import AsyncMock, MagicMock, patch

# Import existing core modules
from core.distributed.executor import DistributedExecutor
from core.distributed.consensus import ConsensusManager
from core.distributed.state_manager import StateManager
from core.resilience.circuit_breaker import CircuitBreaker
from core.resilience.retry_policy import RetryPolicy
from core.resilience.fallback_manager import FallbackManager
from core.composition.capability_graph import CapabilityGraph
from core.composition.optimizer import Optimizer
from core.composition.planner import Planner
from monitoring.tracing import TracingManager
from monitoring.metrics_collector import MetricsCollector
from monitoring.cost_tracker import CostTracker

# Import plugin modules
from plugins.llm_application_dev.skills.prompt_engineering_patterns.scripts.optimize_prompt import PromptOptimizer

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class TestStatus(Enum):
    """Status of a test execution."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"

class TestType(Enum):
    """Type of test being executed."""
    UNIT = "unit"
    INTEGRATION = "integration"
    PROPERTY = "property"
    MUTATION = "mutation"
    PERFORMANCE = "performance"
    CHAOS = "chaos"

@dataclass
class TestCase:
    """Represents a single test case."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    test_type: TestType = TestType.UNIT
    agent_id: str = ""
    input_data: Dict[str, Any] = field(default_factory=dict)
    expected_output: Optional[Dict[str, Any]] = None
    expected_behavior: Optional[Callable] = None
    timeout: float = 30.0
    retries: int = 0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert test case to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "test_type": self.test_type.value,
            "agent_id": self.agent_id,
            "input_data": self.input_data,
            "expected_output": self.expected_output,
            "timeout": self.timeout,
            "retries": self.retries,
            "tags": self.tags,
            "metadata": self.metadata
        }

@dataclass
class TestResult:
    """Represents the result of a test execution."""
    test_case_id: str
    status: TestStatus
    actual_output: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    assertions_passed: int = 0
    assertions_failed: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None
    mutation_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert test result to dictionary."""
        return {
            "test_case_id": self.test_case_id,
            "status": self.status.value,
            "actual_output": self.actual_output,
            "error_message": self.error_message,
            "execution_time": self.execution_time,
            "assertions_passed": self.assertions_passed,
            "assertions_failed": self.assertions_failed,
            "metrics": self.metrics,
            "trace_id": self.trace_id,
            "mutation_score": self.mutation_score
        }

class MockLLMResponse:
    """Simulates LLM responses for testing without actual API calls."""
    
    def __init__(self, response_data: Dict[str, Any], latency: float = 0.1):
        self.response_data = response_data
        self.latency = latency
        self.call_count = 0
        self.call_history = []
    
    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Simulate LLM generation with configurable response."""
        self.call_count += 1
        call_info = {
            "prompt": prompt,
            "kwargs": kwargs,
            "timestamp": time.time()
        }
        self.call_history.append(call_info)
        
        # Simulate latency
        await asyncio.sleep(self.latency)
        
        # Return configured response or generate based on prompt
        if "response" in self.response_data:
            return self.response_data["response"]
        
        # Generate contextual response based on prompt
        return self._generate_contextual_response(prompt, kwargs)
    
    def _generate_contextual_response(self, prompt: str, kwargs: Dict) -> Dict[str, Any]:
        """Generate a contextual response based on the prompt."""
        # Simple pattern matching for common test scenarios
        if "error" in prompt.lower():
            return {"error": "Simulated error response", "status": "failed"}
        elif "success" in prompt.lower():
            return {"result": "Success", "data": {"processed": True}, "status": "success"}
        elif "complex" in prompt.lower():
            return {
                "thought_process": ["Step 1", "Step 2", "Step 3"],
                "final_answer": "Complex response generated",
                "confidence": 0.95
            }
        else:
            return {
                "response": f"Mock response to: {prompt[:100]}...",
                "metadata": {"model": "test-model", "tokens_used": len(prompt.split())}
            }

class TestAssertion:
    """Provides assertion methods for test validation."""
    
    @staticmethod
    def assert_equals(actual: Any, expected: Any, message: str = ""):
        """Assert that actual equals expected."""
        if actual != expected:
            raise AssertionError(f"{message} - Expected: {expected}, Got: {actual}")
    
    @staticmethod
    def assert_contains(container: Any, item: Any, message: str = ""):
        """Assert that container contains item."""
        if item not in container:
            raise AssertionError(f"{message} - {item} not found in {container}")
    
    @staticmethod
    def assert_greater_than(actual: float, threshold: float, message: str = ""):
        """Assert that actual is greater than threshold."""
        if actual <= threshold:
            raise AssertionError(f"{message} - Expected > {threshold}, Got: {actual}")
    
    @staticmethod
    def assert_less_than(actual: float, threshold: float, message: str = ""):
        """Assert that actual is less than threshold."""
        if actual >= threshold:
            raise AssertionError(f"{message} - Expected < {threshold}, Got: {actual}")
    
    @staticmethod
    def assert_response_structure(response: Dict, required_keys: List[str], message: str = ""):
        """Assert that response has required structure."""
        missing_keys = [key for key in required_keys if key not in response]
        if missing_keys:
            raise AssertionError(f"{message} - Missing keys: {missing_keys}")
    
    @staticmethod
    def assert_performance(execution_time: float, max_time: float, message: str = ""):
        """Assert that execution time is within acceptable limits."""
        if execution_time > max_time:
            raise AssertionError(f"{message} - Execution time {execution_time}s exceeds limit {max_time}s")

class PropertyGenerator:
    """Generates property-based test cases using various strategies."""
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.strategies = {
            "string": self._generate_string,
            "number": self._generate_number,
            "boolean": self._generate_boolean,
            "list": self._generate_list,
            "dict": self._generate_dict,
            "edge_case": self._generate_edge_case
        }
    
    def generate_test_cases(self, 
                           agent_id: str,
                           input_schema: Dict[str, Any],
                           count: int = 100,
                           strategy: str = "mixed") -> List[TestCase]:
        """Generate property-based test cases."""
        test_cases = []
        
        for i in range(count):
            test_case = TestCase(
                name=f"property_test_{agent_id}_{i}",
                description=f"Property-based test case {i} for {agent_id}",
                test_type=TestType.PROPERTY,
                agent_id=agent_id,
                input_data=self._generate_input(input_schema, strategy),
                tags=["property", "auto-generated"]
            )
            test_cases.append(test_case)
        
        return test_cases
    
    def _generate_input(self, schema: Dict[str, Any], strategy: str) -> Dict[str, Any]:
        """Generate input data based on schema."""
        generated = {}
        
        for field_name, field_spec in schema.items():
            field_type = field_spec.get("type", "string")
            
            if strategy == "edge_case" or (strategy == "mixed" and self.rng.random() < 0.2):
                generated[field_name] = self._generate_edge_case(field_type, field_spec)
            else:
                generator = self.strategies.get(field_type, self._generate_string)
                generated[field_name] = generator(field_spec)
        
        return generated
    
    def _generate_string(self, spec: Dict) -> str:
        """Generate random string."""
        length = spec.get("length", self.rng.randint(1, 100))
        chars = spec.get("chars", "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ")
        return ''.join(self.rng.choice(chars) for _ in range(length))
    
    def _generate_number(self, spec: Dict) -> Union[int, float]:
        """Generate random number."""
        min_val = spec.get("min", 0)
        max_val = spec.get("max", 1000)
        is_float = spec.get("float", False)
        
        if is_float:
            return self.rng.uniform(min_val, max_val)
        else:
            return self.rng.randint(min_val, max_val)
    
    def _generate_boolean(self, spec: Dict) -> bool:
        """Generate random boolean."""
        return self.rng.random() < 0.5
    
    def _generate_list(self, spec: Dict) -> List:
        """Generate random list."""
        min_items = spec.get("min_items", 0)
        max_items = spec.get("max_items", 10)
        item_schema = spec.get("items", {"type": "string"})
        
        count = self.rng.randint(min_items, max_items)
        return [self._generate_input({"item": item_schema}, "random")["item"] for _ in range(count)]
    
    def _generate_dict(self, spec: Dict) -> Dict:
        """Generate random dictionary."""
        return self._generate_input(spec.get("properties", {}), "random")
    
    def _generate_edge_case(self, field_type: str, spec: Dict) -> Any:
        """Generate edge case values."""
        edge_cases = {
            "string": ["", " ", "\n", "\t", "null", "undefined", "NaN", "a" * 10000, 
                      "!@#$%^&*()", "<script>alert('xss')</script>", "'; DROP TABLE users; --"],
            "number": [0, -0, 1, -1, 0.0001, 999999999, float('inf'), float('-inf'), float('nan')],
            "boolean": [True, False],
            "list": [[], [None], [""], list(range(1000))],
            "dict": [{}, {"": ""}, {"key": None}, {str(i): i for i in range(100)}]
        }
        
        cases = edge_cases.get(field_type, [""])
        return self.rng.choice(cases)

class MutationTester:
    """Performs mutation testing on agent code."""
    
    def __init__(self):
        self.mutation_operators = [
            self._mutate_arithmetic_operators,
            self._mutate_comparison_operators,
            self._mutate_logical_operators,
            self._mutate_boundary_values,
            self._mutate_return_values,
            self._mutate_exception_handling
        ]
    
    def generate_mutations(self, source_code: str, mutation_count: int = 50) -> List[str]:
        """Generate mutated versions of source code."""
        mutations = []
        
        for i in range(mutation_count):
            mutation_op = self.rng.choice(self.mutation_operators)
            mutated = mutation_op(source_code)
            if mutated != source_code:
                mutations.append(mutated)
        
        return mutations
    
    def _mutate_arithmetic_operators(self, code: str) -> str:
        """Mutate arithmetic operators."""
        replacements = {
            "+": "-",
            "-": "+",
            "*": "/",
            "/": "*",
            "%": "**"
        }
        
        mutated = code
        for old, new in replacements.items():
            if old in mutated:
                mutated = mutated.replace(old, new, 1)
                break
        
        return mutated
    
    def _mutate_comparison_operators(self, code: str) -> str:
        """Mutate comparison operators."""
        replacements = {
            "==": "!=",
            "!=": "==",
            ">": "<",
            "<": ">",
            ">=": "<=",
            "<=": ">="
        }
        
        mutated = code
        for old, new in replacements.items():
            if old in mutated:
                mutated = mutated.replace(old, new, 1)
                break
        
        return mutated
    
    def _mutate_logical_operators(self, code: str) -> str:
        """Mutate logical operators."""
        replacements = {
            "and": "or",
            "or": "and",
            "not ": ""
        }
        
        mutated = code
        for old, new in replacements.items():
            if old in mutated:
                mutated = mutated.replace(old, new, 1)
                break
        
        return mutated
    
    def _mutate_boundary_values(self, code: str) -> str:
        """Mutate boundary values."""
        # Simple boundary mutation - in production would use AST parsing
        import re
        
        # Find numbers and mutate them
        def mutate_number(match):
            num = int(match.group())
            if num == 0:
                return "1"
            elif num == 1:
                return "0"
            elif num > 0:
                return str(num - 1)
            else:
                return str(num + 1)
        
        return re.sub(r'\b\d+\b', mutate_number, code)
    
    def _mutate_return_values(self, code: str) -> str:
        """Mutate return values."""
        if "return True" in code:
            return code.replace("return True", "return False", 1)
        elif "return False" in code:
            return code.replace("return False", "return True", 1)
        elif "return None" in code:
            return code.replace("return None", "return {}", 1)
        return code
    
    def _mutate_exception_handling(self, code: str) -> str:
        """Mutate exception handling."""
        if "except Exception:" in code:
            return code.replace("except Exception:", "except:", 1)
        elif "raise" in code:
            return code.replace("raise", "pass", 1)
        return code

class AgentTestSuite:
    """Manages a collection of test cases and their execution."""
    
    def __init__(self, name: str, description: str = ""):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.test_cases: List[TestCase] = []
        self.results: List[TestResult] = []
        self.created_at = time.time()
        self.config = {
            "parallel_execution": True,
            "max_workers": 4,
            "stop_on_failure": False,
            "timeout": 300.0,
            "retry_failed": True,
            "max_retries": 3
        }
    
    def add_test_case(self, test_case: TestCase):
        """Add a test case to the suite."""
        self.test_cases.append(test_case)
    
    def add_test_cases(self, test_cases: List[TestCase]):
        """Add multiple test cases to the suite."""
        self.test_cases.extend(test_cases)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of test suite execution."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in self.results if r.status == TestStatus.FAILED)
        errors = sum(1 for r in self.results if r.status == TestStatus.ERROR)
        skipped = sum(1 for r in self.results if r.status == TestStatus.SKIPPED)
        
        return {
            "suite_id": self.id,
            "suite_name": self.name,
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "skipped": skipped,
            "pass_rate": passed / total if total > 0 else 0,
            "execution_time": sum(r.execution_time for r in self.results),
            "created_at": self.created_at
        }

class AgentTester:
    """Main testing framework for nexus."""
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 enable_monitoring: bool = True,
                 enable_tracing: bool = True):
        self.config = self._load_config(config_path)
        self.mock_llm = MockLLMResponse(self.config.get("mock_llm_responses", {}))
        self.property_generator = PropertyGenerator(seed=self.config.get("random_seed"))
        self.mutation_tester = MutationTester()
        self.assertions = TestAssertion()
        
        # Initialize monitoring and tracing if enabled
        self.enable_monitoring = enable_monitoring
        self.enable_tracing = enable_tracing
        
        if enable_monitoring:
            self.metrics_collector = MetricsCollector()
            self.cost_tracker = CostTracker()
        
        if enable_tracing:
            self.tracing_manager = TracingManager()
        
        # Test suites registry
        self.test_suites: Dict[str, AgentTestSuite] = {}
        
        # Agent registry (for mocking)
        self.agent_registry: Dict[str, Any] = {}
        
        # Integration with existing modules
        self.distributed_executor = DistributedExecutor()
        self.consensus_manager = ConsensusManager()
        self.state_manager = StateManager()
        self.circuit_breaker = CircuitBreaker()
        self.retry_policy = RetryPolicy()
        self.fallback_manager = FallbackManager()
        self.capability_graph = CapabilityGraph()
        self.optimizer = Optimizer()
        self.planner = Planner()
        self.prompt_optimizer = PromptOptimizer()
        
        logger.info("AgentTester initialized with configuration: %s", self.config)
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            "mock_llm_responses": {
                "default_latency": 0.1,
                "error_rate": 0.05,
                "timeout_rate": 0.02
            },
            "test_generation": {
                "property_test_count": 100,
                "mutation_test_count": 50,
                "edge_case_probability": 0.2
            },
            "execution": {
                "parallel": True,
                "max_workers": 4,
                "default_timeout": 30.0
            },
            "reporting": {
                "output_format": "json",
                "include_traces": True,
                "include_metrics": True
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                # Merge with defaults
                for key, value in user_config.items():
                    if key in default_config and isinstance(default_config[key], dict):
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
        
        return default_config
    
    def register_agent(self, agent_id: str, agent_instance: Any):
        """Register an agent for testing."""
        self.agent_registry[agent_id] = agent_instance
        logger.info("Registered agent: %s", agent_id)
    
    def create_test_suite(self, name: str, description: str = "") -> AgentTestSuite:
        """Create a new test suite."""
        suite = AgentTestSuite(name, description)
        self.test_suites[suite.id] = suite
        return suite
    
    def generate_unit_tests(self, 
                           agent_id: str, 
                           input_schema: Dict[str, Any],
                           count: int = 50) -> List[TestCase]:
        """Generate unit tests for an agent."""
        test_cases = []
        
        for i in range(count):
            test_case = TestCase(
                name=f"unit_test_{agent_id}_{i}",
                description=f"Unit test {i} for agent {agent_id}",
                test_type=TestType.UNIT,
                agent_id=agent_id,
                input_data=self.property_generator._generate_input(input_schema, "mixed"),
                tags=["unit", "auto-generated"]
            )
            test_cases.append(test_case)
        
        return test_cases
    
    def generate_integration_tests(self,
                                  workflow_id: str,
                                  agent_sequence: List[str],
                                  test_count: int = 20) -> List[TestCase]:
        """Generate integration tests for a workflow."""
        test_cases = []
        
        for i in range(test_count):
            # Generate input for first agent in sequence
            first_agent = agent_sequence[0]
            input_schema = self._get_agent_input_schema(first_agent)
            
            test_case = TestCase(
                name=f"integration_test_{workflow_id}_{i}",
                description=f"Integration test {i} for workflow {workflow_id}",
                test_type=TestType.INTEGRATION,
                agent_id=workflow_id,
                input_data={
                    "workflow_id": workflow_id,
                    "agent_sequence": agent_sequence,
                    "initial_input": self.property_generator._generate_input(input_schema, "mixed")
                },
                tags=["integration", "workflow", "auto-generated"],
                metadata={"agent_sequence": agent_sequence}
            )
            test_cases.append(test_case)
        
        return test_cases
    
    def generate_property_tests(self,
                               agent_id: str,
                               input_schema: Dict[str, Any],
                               property_checks: List[Callable]) -> List[TestCase]:
        """Generate property-based tests with custom property checks."""
        base_tests = self.property_generator.generate_test_cases(
            agent_id, 
            input_schema, 
            count=self.config["test_generation"]["property_test_count"]
        )
        
        # Add property checks to metadata
        for test_case in base_tests:
            test_case.metadata["property_checks"] = property_checks
        
        return base_tests
    
    async def run_test_case(self, test_case: TestCase) -> TestResult:
        """Execute a single test case."""
        start_time = time.time()
        result = TestResult(
            test_case_id=test_case.id,
            status=TestStatus.RUNNING
        )
        
        try:
            # Start tracing if enabled
            if self.enable_tracing:
                result.trace_id = self.tracing_manager.start_trace(
                    f"test_{test_case.id}",
                    {"test_type": test_case.test_type.value, "agent_id": test_case.agent_id}
                )
            
            # Execute based on test type
            if test_case.test_type == TestType.UNIT:
                result = await self._run_unit_test(test_case, result)
            elif test_case.test_type == TestType.INTEGRATION:
                result = await self._run_integration_test(test_case, result)
            elif test_case.test_type == TestType.PROPERTY:
                result = await self._run_property_test(test_case, result)
            elif test_case.test_type == TestType.MUTATION:
                result = await self._run_mutation_test(test_case, result)
            elif test_case.test_type == TestType.PERFORMANCE:
                result = await self._run_performance_test(test_case, result)
            elif test_case.test_type == TestType.CHAOS:
                result = await self._run_chaos_test(test_case, result)
            else:
                result.status = TestStatus.SKIPPED
                result.error_message = f"Unknown test type: {test_case.test_type}"
        
        except asyncio.TimeoutError:
            result.status = TestStatus.TIMEOUT
            result.error_message = f"Test timed out after {test_case.timeout} seconds"
        
        except Exception as e:
            result.status = TestStatus.ERROR
            result.error_message = str(e)
            logger.exception("Error executing test case %s", test_case.id)
        
        finally:
            # Record execution time
            result.execution_time = time.time() - start_time
            
            # Stop tracing
            if self.enable_tracing and result.trace_id:
                self.tracing_manager.end_trace(result.trace_id, {
                    "status": result.status.value,
                    "execution_time": result.execution_time
                })
            
            # Record metrics
            if self.enable_monitoring:
                self.metrics_collector.record_test_execution(
                    test_case.test_type.value,
                    result.status.value,
                    result.execution_time
                )
        
        return result
    
    async def _run_unit_test(self, test_case: TestCase, result: TestResult) -> TestResult:
        """Execute a unit test."""
        agent = self.agent_registry.get(test_case.agent_id)
        
        if not agent:
            result.status = TestStatus.ERROR
            result.error_message = f"Agent {test_case.agent_id} not registered"
            return result
        
        try:
            # Mock LLM calls if agent uses LLM
            with patch.object(agent, 'llm_client', self.mock_llm):
                # Execute agent with test input
                if hasattr(agent, 'process'):
                    output = await agent.process(test_case.input_data)
                elif hasattr(agent, 'run'):
                    output = await agent.run(test_case.input_data)
                elif callable(agent):
                    output = await agent(test_case.input_data)
                else:
                    raise ValueError(f"Agent {test_case.agent_id} has no executable method")
                
                result.actual_output = output
                
                # Validate output
                if test_case.expected_output:
                    self._validate_output(output, test_case.expected_output, result)
                elif test_case.expected_behavior:
                    test_case.expected_behavior(output, result)
                else:
                    # Basic validation - ensure output is not None
                    if output is None:
                        result.status = TestStatus.FAILED
                        result.error_message = "Agent returned None"
                    else:
                        result.status = TestStatus.PASSED
                        result.assertions_passed += 1
        
        except AssertionError as e:
            result.status = TestStatus.FAILED
            result.error_message = str(e)
        
        except Exception as e:
            result.status = TestStatus.ERROR
            result.error_message = f"Agent execution failed: {str(e)}"
        
        return result
    
    async def _run_integration_test(self, test_case: TestCase, result: TestResult) -> TestResult:
        """Execute an integration test."""
        workflow_data = test_case.input_data
        agent_sequence = workflow_data.get("agent_sequence", [])
        current_input = workflow_data.get("initial_input", {})
        
        try:
            # Execute nexus in sequence
            for i, agent_id in enumerate(agent_sequence):
                agent = self.agent_registry.get(agent_id)
                
                if not agent:
                    result.status = TestStatus.ERROR
                    result.error_message = f"Agent {agent_id} not found in sequence"
                    return result
                
                # Execute current agent
                with patch.object(agent, 'llm_client', self.mock_llm):
                    if hasattr(agent, 'process'):
                        output = await agent.process(current_input)
                    elif hasattr(agent, 'run'):
                        output = await agent.run(current_input)
                    else:
                        output = await agent(current_input)
                
                # Use output as input for next agent
                current_input = output
                
                # Record intermediate results
                result.metrics[f"agent_{i}_output"] = output
            
            # Final output is the result of last agent
            result.actual_output = current_input
            
            # Validate workflow completion
            if test_case.expected_output:
                self._validate_output(current_input, test_case.expected_output, result)
            else:
                result.status = TestStatus.PASSED
                result.assertions_passed = len(agent_sequence)
        
        except Exception as e:
            result.status = TestStatus.ERROR
            result.error_message = f"Workflow execution failed: {str(e)}"
        
        return result
    
    async def _run_property_test(self, test_case: TestCase, result: TestResult) -> TestResult:
        """Execute a property-based test."""
        # First run the basic test
        result = await self._run_unit_test(test_case, result)
        
        # If passed, run property checks
        if result.status == TestStatus.PASSED and "property_checks" in test_case.metadata:
            property_checks = test_case.metadata["property_checks"]
            
            for check in property_checks:
                try:
                    check(result.actual_output, test_case.input_data)
                    result.assertions_passed += 1
                except AssertionError as e:
                    result.status = TestStatus.FAILED
                    result.error_message = f"Property check failed: {str(e)}"
                    result.assertions_failed += 1
                    break
        
        return result
    
    async def _run_mutation_test(self, test_case: TestCase, result: TestResult) -> TestResult:
        """Execute a mutation test."""
        agent = self.agent_registry.get(test_case.agent_id)
        
        if not agent:
            result.status = TestStatus.ERROR
            result.error_message = f"Agent {test_case.agent_id} not registered"
            return result
        
        try:
            # Get agent source code (simplified - in production would use inspect)
            import inspect
            source_code = inspect.getsource(agent.__class__)
            
            # Generate mutations
            mutations = self.mutation_tester.generate_mutations(
                source_code,
                count=self.config["test_generation"]["mutation_test_count"]
            )
            
            # Test each mutation
            killed_mutations = 0
            total_mutations = len(mutations)
            
            for i, mutated_code in enumerate(mutations):
                # Create mutated agent (simplified - in production would use exec or importlib)
                try:
                    # This is a simplified approach - in production would be more sophisticated
                    local_vars = {}
                    exec(mutated_code, globals(), local_vars)
                    
                    # Find the mutated class
                    mutated_class = None
                    for var in local_vars.values():
                        if isinstance(var, type) and hasattr(var, '__name__'):
                            if var.__name__ == agent.__class__.__name__:
                                mutated_class = var
                                break
                    
                    if mutated_class:
                        mutated_agent = mutated_class()
                        
                        # Run test with mutated agent
                        with patch.object(mutated_agent, 'llm_client', self.mock_llm):
                            if hasattr(mutated_agent, 'process'):
                                mutated_output = await mutated_agent.process(test_case.input_data)
                            else:
                                mutated_output = await mutated_agent(test_case.input_data)
                        
                        # Compare with original output
                        if mutated_output != result.actual_output:
                            killed_mutations += 1
                
                except Exception:
                    # Mutation caused error - counts as killed
                    killed_mutations += 1
            
            # Calculate mutation score
            mutation_score = killed_mutations / total_mutations if total_mutations > 0 else 0
            result.mutation_score = mutation_score
            
            # Determine test status based on mutation score
            if mutation_score >= 0.8:  # 80% mutation score threshold
                result.status = TestStatus.PASSED
                result.assertions_passed += 1
            else:
                result.status = TestStatus.FAILED
                result.error_message = f"Mutation score {mutation_score:.2f} below threshold"
                result.assertions_failed += 1
        
        except Exception as e:
            result.status = TestStatus.ERROR
            result.error_message = f"Mutation testing failed: {str(e)}"
        
        return result
    
    async def _run_performance_test(self, test_case: TestCase, result: TestResult) -> TestResult:
        """Execute a performance test."""
        # Run test multiple times to measure performance
        iterations = test_case.metadata.get("iterations", 10)
        execution_times = []
        
        for i in range(iterations):
            iter_start = time.time()
            
            # Run the test
            iter_result = await self._run_unit_test(test_case, TestResult(test_case.id, TestStatus.RUNNING))
            
            iter_time = time.time() - iter_start
            execution_times.append(iter_time)
            
            # Check for failures
            if iter_result.status != TestStatus.PASSED:
                result.status = TestStatus.FAILED
                result.error_message = f"Performance test failed on iteration {i}: {iter_result.error_message}"
                return result
        
        # Calculate performance metrics
        avg_time = sum(execution_times) / len(execution_times)
        max_time = max(execution_times)
        min_time = min(execution_times)
        
        result.metrics.update({
            "iterations": iterations,
            "average_time": avg_time,
            "max_time": max_time,
            "min_time": min_time,
            "std_dev": (sum((t - avg_time) ** 2 for t in execution_times) / len(execution_times)) ** 0.5
        })
        
        # Check against performance requirements
        max_allowed_time = test_case.metadata.get("max_allowed_time", 1.0)
        
        if avg_time <= max_allowed_time:
            result.status = TestStatus.PASSED
            result.assertions_passed += 1
        else:
            result.status = TestStatus.FAILED
            result.error_message = f"Average execution time {avg_time:.3f}s exceeds limit {max_allowed_time}s"
            result.assertions_failed += 1
        
        result.actual_output = {"performance_metrics": result.metrics}
        return result
    
    async def _run_chaos_test(self, test_case: TestCase, result: TestResult) -> TestResult:
        """Execute a chaos test with random failures and delays."""
        chaos_config = test_case.metadata.get("chaos_config", {})
        
        # Configure chaos parameters
        failure_rate = chaos_config.get("failure_rate", 0.1)
        delay_rate = chaos_config.get("delay_rate", 0.2)
        max_delay = chaos_config.get("max_delay", 2.0)
        
        # Create chaos-aware mock LLM
        chaos_llm = ChaosMockLLM(
            base_llm=self.mock_llm,
            failure_rate=failure_rate,
            delay_rate=delay_rate,
            max_delay=max_delay
        )
        
        agent = self.agent_registry.get(test_case.agent_id)
        
        if not agent:
            result.status = TestStatus.ERROR
            result.error_message = f"Agent {test_case.agent_id} not registered"
            return result
        
        try:
            # Run test with chaos
            with patch.object(agent, 'llm_client', chaos_llm):
                if hasattr(agent, 'process'):
                    output = await agent.process(test_case.input_data)
                elif hasattr(agent, 'run'):
                    output = await agent.run(test_case.input_data)
                else:
                    output = await agent(test_case.input_data)
                
                result.actual_output = output
                
                # Check if agent handled chaos gracefully
                if output is not None and not isinstance(output, dict) or "error" not in output:
                    result.status = TestStatus.PASSED
                    result.assertions_passed += 1
                    result.metrics["chaos_events"] = chaos_llm.chaos_events
                else:
                    result.status = TestStatus.FAILED
                    result.error_message = "Agent did not handle chaos gracefully"
                    result.assertions_failed += 1
        
        except Exception as e:
            result.status = TestStatus.FAILED
            result.error_message = f"Agent failed under chaos conditions: {str(e)}"
            result.assertions_failed += 1
        
        return result
    
    def _validate_output(self, actual: Any, expected: Any, result: TestResult):
        """Validate actual output against expected output."""
        if isinstance(expected, dict) and isinstance(actual, dict):
            # Deep comparison for dictionaries
            for key, expected_value in expected.items():
                if key not in actual:
                    result.status = TestStatus.FAILED
                    result.error_message = f"Missing key in output: {key}"
                    result.assertions_failed += 1
                    return
                
                if actual[key] != expected_value:
                    result.status = TestStatus.FAILED
                    result.error_message = f"Value mismatch for key {key}: expected {expected_value}, got {actual[key]}"
                    result.assertions_failed += 1
                    return
            
            result.status = TestStatus.PASSED
            result.assertions_passed += len(expected)
        
        elif actual == expected:
            result.status = TestStatus.PASSED
            result.assertions_passed += 1
        
        else:
            result.status = TestStatus.FAILED
            result.error_message = f"Output mismatch: expected {expected}, got {actual}"
            result.assertions_failed += 1
    
    def _get_agent_input_schema(self, agent_id: str) -> Dict[str, Any]:
        """Get input schema for an agent (simplified)."""
        # In production, this would query agent metadata or configuration
        return {
            "query": {"type": "string", "length": 50},
            "context": {"type": "dict", "properties": {
                "user_id": {"type": "string"},
                "session_id": {"type": "string"}
            }},
            "parameters": {"type": "dict"}
        }
    
    async def run_test_suite(self, suite_id: str) -> Dict[str, Any]:
        """Execute all tests in a test suite."""
        suite = self.test_suites.get(suite_id)
        
        if not suite:
            raise ValueError(f"Test suite {suite_id} not found")
        
        logger.info("Starting test suite: %s (%d tests)", suite.name, len(suite.test_cases))
        
        # Execute tests
        if suite.config["parallel_execution"] and len(suite.test_cases) > 1:
            # Parallel execution
            semaphore = asyncio.Semaphore(suite.config["max_workers"])
            
            async def run_with_semaphore(test_case):
                async with semaphore:
                    return await self.run_test_case(test_case)
            
            tasks = [run_with_semaphore(tc) for tc in suite.test_cases]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    error_result = TestResult(
                        test_case_id=suite.test_cases[i].id,
                        status=TestStatus.ERROR,
                        error_message=str(result)
                    )
                    suite.results.append(error_result)
                else:
                    suite.results.append(result)
        
        else:
            # Sequential execution
            for test_case in suite.test_cases:
                result = await self.run_test_case(test_case)
                suite.results.append(result)
                
                # Stop on failure if configured
                if suite.config["stop_on_failure"] and result.status in [TestStatus.FAILED, TestStatus.ERROR]:
                    logger.warning("Stopping suite execution due to failure in test: %s", test_case.name)
                    break
        
        # Generate summary
        summary = suite.get_summary()
        logger.info("Test suite completed: %s - Pass rate: %.2f%%", 
                   suite.name, summary["pass_rate"] * 100)
        
        return summary
    
    def generate_test_report(self, suite_id: str, format: str = "json") -> str:
        """Generate a test report in the specified format."""
        suite = self.test_suites.get(suite_id)
        
        if not suite:
            raise ValueError(f"Test suite {suite_id} not found")
        
        report_data = {
            "summary": suite.get_summary(),
            "test_cases": [tc.to_dict() for tc in suite.test_cases],
            "results": [r.to_dict() for r in suite.results],
            "generated_at": time.time()
        }
        
        if format == "json":
            return json.dumps(report_data, indent=2)
        elif format == "html":
            return self._generate_html_report(report_data)
        elif format == "markdown":
            return self._generate_markdown_report(report_data)
        else:
            raise ValueError(f"Unsupported report format: {format}")
    
    def _generate_html_report(self, report_data: Dict) -> str:
        """Generate HTML test report."""
        # Simplified HTML report generation
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Report - {report_data['summary']['suite_name']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
                .passed {{ color: green; }}
                .failed {{ color: red; }}
                .error {{ color: orange; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Test Report: {report_data['summary']['suite_name']}</h1>
            <div class="summary">
                <h2>Summary</h2>
                <p>Total Tests: {report_data['summary']['total_tests']}</p>
                <p class="passed">Passed: {report_data['summary']['passed']}</p>
                <p class="failed">Failed: {report_data['summary']['failed']}</p>
                <p class="error">Errors: {report_data['summary']['errors']}</p>
                <p>Pass Rate: {report_data['summary']['pass_rate'] * 100:.1f}%</p>
            </div>
            
            <h2>Test Results</h2>
            <table>
                <tr>
                    <th>Test Name</th>
                    <th>Type</th>
                    <th>Status</th>
                    <th>Execution Time</th>
                    <th>Error</th>
                </tr>
        """
        
        for result in report_data['results']:
            test_case = next((tc for tc in report_data['test_cases'] 
                            if tc['id'] == result['test_case_id']), {})
            
            status_class = result['status']
            html += f"""
                <tr>
                    <td>{test_case.get('name', 'Unknown')}</td>
                    <td>{test_case.get('test_type', 'Unknown')}</td>
                    <td class="{status_class}">{result['status']}</td>
                    <td>{result['execution_time']:.3f}s</td>
                    <td>{result.get('error_message', '')}</td>
                </tr>
            """
        
        html += """
            </table>
        </body>
        </html>
        """
        
        return html
    
    def _generate_markdown_report(self, report_data: Dict) -> str:
        """Generate Markdown test report."""
        summary = report_data['summary']
        
        md = f"""# Test Report: {summary['suite_name']}

## Summary
- **Total Tests:** {summary['total_tests']}
- **Passed:** {summary['passed']} ✅
- **Failed:** {summary['failed']} ❌
- **Errors:** {summary['errors']} ⚠️
- **Pass Rate:** {summary['pass_rate'] * 100:.1f}%
- **Execution Time:** {summary['execution_time']:.2f}s

## Test Results

| Test Name | Type | Status | Time | Error |
|-----------|------|--------|------|-------|
"""
        
        for result in report_data['results']:
            test_case = next((tc for tc in report_data['test_cases'] 
                            if tc['id'] == result['test_case_id']), {})
            
            status_emoji = {
                "passed": "✅",
                "failed": "❌",
                "error": "⚠️",
                "skipped": "⏭️",
                "timeout": "⏰"
            }.get(result['status'], "❓")
            
            md += f"| {test_case.get('name', 'Unknown')} | {test_case.get('test_type', 'Unknown')} | {status_emoji} {result['status']} | {result['execution_time']:.3f}s | {result.get('error_message', '')} |\n"
        
        return md

class ChaosMockLLM(MockLLMResponse):
    """Mock LLM that introduces chaos (failures, delays) for testing resilience."""
    
    def __init__(self, base_llm: MockLLMResponse, failure_rate: float = 0.1, 
                 delay_rate: float = 0.2, max_delay: float = 2.0):
        super().__init__(base_llm.response_data, base_llm.latency)
        self.base_llm = base_llm
        self.failure_rate = failure_rate
        self.delay_rate = delay_rate
        self.max_delay = max_delay
        self.chaos_events = []
        self.rng = random.Random()
    
    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response with chaos injection."""
        event_type = None
        
        # Random failure
        if self.rng.random() < self.failure_rate:
            event_type = "failure"
            self.chaos_events.append({"type": "failure", "timestamp": time.time()})
            raise Exception("Simulated LLM failure")
        
        # Random delay
        if self.rng.random() < self.delay_rate:
            event_type = "delay"
            delay = self.rng.uniform(0.1, self.max_delay)
            self.chaos_events.append({"type": "delay", "duration": delay, "timestamp": time.time()})
            await asyncio.sleep(delay)
        
        # Normal response
        if not event_type:
            event_type = "normal"
        
        self.chaos_events.append({"type": event_type, "timestamp": time.time()})
        return await self.base_llm.generate(prompt, **kwargs)

# Example usage and factory functions
def create_agent_tester(config_path: Optional[str] = None) -> AgentTester:
    """Factory function to create an AgentTester instance."""
    return AgentTester(config_path=config_path)

def create_mock_agent(agent_id: str, behavior: Callable) -> Any:
    """Create a mock agent for testing."""
    class MockAgent:
        def __init__(self, agent_id: str, behavior: Callable):
            self.agent_id = agent_id
            self.behavior = behavior
            self.llm_client = None
        
        async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
            return await self.behavior(input_data)
    
    return MockAgent(agent_id, behavior)

# Export main classes and functions
__all__ = [
    "AgentTester",
    "AgentTestSuite",
    "TestCase",
    "TestResult",
    "TestStatus",
    "TestType",
    "MockLLMResponse",
    "PropertyGenerator",
    "MutationTester",
    "TestAssertion",
    "ChaosMockLLM",
    "create_agent_tester",
    "create_mock_agent"
]