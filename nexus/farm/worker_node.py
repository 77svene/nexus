import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Awaitable
import aiohttp
from aiohttp import web
import aioredis
from contextlib import asynccontextmanager

from nexus.actor.playground import Playground
from nexus.agent.service import AgentService
from nexus.agent.views import AgentConfig

logger = logging.getLogger(__name__)


class WorkerStatus(Enum):
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"
    ERROR = "error"


class TaskPriority(Enum):
    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


@dataclass
class Task:
    id: str
    type: str
    payload: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    session_id: Optional[str] = None
    timeout: int = 300  # seconds
    max_retries: int = 3
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def to_dict(self):
        data = asdict(self)
        data['priority'] = self.priority.value
        data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict):
        data['priority'] = TaskPriority(data['priority'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


@dataclass
class TaskResult:
    task_id: str
    worker_id: str
    status: str  # "success", "error", "timeout"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    completed_at: datetime = None
    
    def __post_init__(self):
        if self.completed_at is None:
            self.completed_at = datetime.utcnow()


@dataclass
class WorkerNodeConfig:
    worker_id: str = None
    master_url: str = "http://localhost:8000"
    redis_url: str = "redis://localhost:6379"
    max_concurrent_tasks: int = 5
    health_check_interval: int = 30  # seconds
    task_timeout: int = 300  # seconds
    browser_pool_size: int = 3
    enable_session_persistence: bool = True
    cloud_provider: Optional[str] = None  # "aws", "gcp", or None for local
    cloud_config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.worker_id is None:
            self.worker_id = f"worker-{uuid.uuid4().hex[:8]}"


class BrowserPool:
    """Manages a pool of browser instances for task execution"""
    
    def __init__(self, pool_size: int = 3):
        self.pool_size = pool_size
        self.available_browsers: List[Playground] = []
        self.in_use_browsers: Dict[str, Playground] = {}  # task_id -> browser
        self.sessions: Dict[str, Dict[str, Any]] = {}  # session_id -> session_data
        self._lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize the browser pool"""
        for i in range(self.pool_size):
            try:
                browser = Playground()
                await browser.__aenter__()
                self.available_browsers.append(browser)
                logger.info(f"Initialized browser instance {i+1}/{self.pool_size}")
            except Exception as e:
                logger.error(f"Failed to initialize browser instance: {e}")
                raise
    
    async def acquire_browser(self, task_id: str, session_id: Optional[str] = None) -> Playground:
        """Acquire a browser instance from the pool"""
        async with self._lock:
            if not self.available_browsers:
                raise Exception("No available browser instances")
            
            browser = self.available_browsers.pop()
            self.in_use_browsers[task_id] = browser
            
            # Restore session if available
            if session_id and session_id in self.sessions:
                session_data = self.sessions[session_id]
                await self._restore_session(browser, session_data)
            
            return browser
    
    async def release_browser(self, task_id: str, session_id: Optional[str] = None, 
                            save_session: bool = False):
        """Release a browser instance back to the pool"""
        async with self._lock:
            if task_id not in self.in_use_browsers:
                return
            
            browser = self.in_use_browsers.pop(task_id)
            
            # Save session if requested
            if session_id and save_session:
                session_data = await self._capture_session(browser)
                self.sessions[session_id] = session_data
            
            # Clear browser state for next use
            try:
                await browser.clear_state()
                self.available_browsers.append(browser)
            except Exception as e:
                logger.error(f"Error clearing browser state: {e}")
                # Create a new browser instance to replace the problematic one
                try:
                    new_browser = Playground()
                    await new_browser.__aenter__()
                    self.available_browsers.append(new_browser)
                except Exception as e2:
                    logger.error(f"Failed to create replacement browser: {e2}")
    
    async def _capture_session(self, browser: Playground) -> Dict[str, Any]:
        """Capture current browser session state"""
        try:
            # Capture cookies, local storage, and other session data
            cookies = await browser.get_cookies()
            local_storage = await browser.evaluate("() => JSON.stringify(localStorage)")
            session_storage = await browser.evaluate("() => JSON.stringify(sessionStorage)")
            
            return {
                "cookies": cookies,
                "local_storage": json.loads(local_storage) if local_storage else {},
                "session_storage": json.loads(session_storage) if session_storage else {},
                "url": await browser.get_current_url(),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.warning(f"Failed to capture session: {e}")
            return {}
    
    async def _restore_session(self, browser: Playground, session_data: Dict[str, Any]):
        """Restore browser session from saved data"""
        try:
            # Navigate to the saved URL first
            if "url" in session_data:
                await browser.goto(session_data["url"])
            
            # Restore cookies
            if "cookies" in session_data and session_data["cookies"]:
                await browser.set_cookies(session_data["cookies"])
            
            # Restore local storage
            if "local_storage" in session_data:
                for key, value in session_data["local_storage"].items():
                    await browser.evaluate(f"() => localStorage.setItem('{key}', '{value}')")
            
            # Restore session storage
            if "session_storage" in session_data:
                for key, value in session_data["session_storage"].items():
                    await browser.evaluate(f"() => sessionStorage.setItem('{key}', '{value}')")
            
            logger.debug(f"Restored session from {session_data.get('timestamp', 'unknown')}")
        except Exception as e:
            logger.warning(f"Failed to restore session: {e}")
    
    async def cleanup(self):
        """Clean up all browser instances"""
        async with self._lock:
            # Close all in-use browsers
            for task_id, browser in self.in_use_browsers.items():
                try:
                    await browser.__aexit__(None, None, None)
                except Exception as e:
                    logger.error(f"Error closing browser for task {task_id}: {e}")
            
            # Close all available browsers
            for browser in self.available_browsers:
                try:
                    await browser.__aexit__(None, None, None)
                except Exception as e:
                    logger.error(f"Error closing browser: {e}")
            
            self.available_browsers.clear()
            self.in_use_browsers.clear()


class TaskExecutor:
    """Executes tasks using browser instances"""
    
    def __init__(self, browser_pool: BrowserPool):
        self.browser_pool = browser_pool
        self.agent_service = AgentService()
    
    async def execute_task(self, task: Task) -> TaskResult:
        """Execute a single task"""
        start_time = datetime.utcnow()
        task_id = task.id
        browser = None
        
        try:
            # Acquire browser instance
            browser = await self.browser_pool.acquire_browser(task_id, task.session_id)
            
            # Execute based on task type
            if task.type == "agent_task":
                result = await self._execute_agent_task(browser, task)
            elif task.type == "browser_script":
                result = await self._execute_browser_script(browser, task)
            elif task.type == "playground_action":
                result = await self._execute_playground_action(browser, task)
            else:
                raise ValueError(f"Unknown task type: {task.type}")
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            return TaskResult(
                task_id=task_id,
                worker_id="",  # Will be set by worker
                status="success",
                result=result,
                execution_time=execution_time
            )
            
        except asyncio.TimeoutError:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            return TaskResult(
                task_id=task_id,
                worker_id="",
                status="timeout",
                error=f"Task timed out after {task.timeout} seconds",
                execution_time=execution_time
            )
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            logger.exception(f"Error executing task {task_id}")
            return TaskResult(
                task_id=task_id,
                worker_id="",
                status="error",
                error=str(e),
                execution_time=execution_time
            )
        finally:
            # Release browser instance
            if browser:
                save_session = task.session_id is not None
                await self.browser_pool.release_browser(task_id, task.session_id, save_session)
    
    async def _execute_agent_task(self, browser: Playground, task: Task) -> Dict[str, Any]:
        """Execute an agent-based task"""
        config = AgentConfig(**task.payload.get("config", {}))
        agent = self.agent_service.create_agent(config, browser)
        
        # Set timeout for the agent execution
        try:
            result = await asyncio.wait_for(
                agent.run(task.payload.get("goal", "")),
                timeout=task.timeout
            )
            return {"agent_result": result}
        except asyncio.TimeoutError:
            agent.cancel()
            raise
    
    async def _execute_browser_script(self, browser: Playground, task: Task) -> Dict[str, Any]:
        """Execute a browser script"""
        script = task.payload.get("script", "")
        script_type = task.payload.get("script_type", "javascript")
        
        if script_type == "javascript":
            result = await browser.evaluate(script)
        elif script_type == "python":
            # Execute Python code in a controlled environment
            result = await self._execute_python_script(browser, script)
        else:
            raise ValueError(f"Unsupported script type: {script_type}")
        
        return {"script_result": result}
    
    async def _execute_python_script(self, browser: Playground, script: str) -> Any:
        """Execute Python script with browser context"""
        # Create a restricted globals dictionary
        globals_dict = {
            "__builtins__": __builtins__,
            "browser": browser,
            "asyncio": asyncio,
            "json": json,
        }
        
        # Execute the script
        try:
            exec_result = exec(script, globals_dict)
            if asyncio.iscoroutine(exec_result):
                return await exec_result
            return exec_result
        except Exception as e:
            raise Exception(f"Python script execution failed: {str(e)}")
    
    async def _execute_playground_action(self, browser: Playground, task: Task) -> Dict[str, Any]:
        """Execute a playground action"""
        action = task.payload.get("action")
        params = task.payload.get("params", {})
        
        # Map action to browser method
        action_map = {
            "goto": browser.goto,
            "click": browser.click,
            "type": browser.type,
            "screenshot": browser.screenshot,
            "get_text": browser.get_text,
            "get_attribute": browser.get_attribute,
        }
        
        if action not in action_map:
            raise ValueError(f"Unknown playground action: {action}")
        
        method = action_map[action]
        result = await method(**params)
        return {"action_result": result}


class WorkerNode:
    """Main worker node that processes tasks from the master"""
    
    def __init__(self, config: WorkerNodeConfig):
        self.config = config
        self.worker_id = config.worker_id
        self.status = WorkerStatus.IDLE
        self.current_tasks: Dict[str, Task] = {}
        self.browser_pool = BrowserPool(config.browser_pool_size)
        self.task_executor = TaskExecutor(self.browser_pool)
        self.redis: Optional[aioredis.Redis] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._health_check_task: Optional[asyncio.Task] = None
        self._task_processing_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the worker node"""
        logger.info(f"Starting worker node {self.worker_id}")
        
        try:
            # Initialize browser pool
            await self.browser_pool.initialize()
            
            # Connect to Redis
            self.redis = await aioredis.from_url(
                self.config.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            
            # Create HTTP session
            self.session = aiohttp.ClientSession()
            
            # Register with master
            await self._register_with_master()
            
            # Start background tasks
            self._running = True
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            self._task_processing_task = asyncio.create_task(self._task_processing_loop())
            
            logger.info(f"Worker node {self.worker_id} started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start worker node: {e}")
            await self.cleanup()
            raise
    
    async def stop(self):
        """Stop the worker node"""
        logger.info(f"Stopping worker node {self.worker_id}")
        self._running = False
        
        # Cancel background tasks
        if self._health_check_task:
            self._health_check_task.cancel()
        if self._task_processing_task:
            self._task_processing_task.cancel()
        
        # Wait for current tasks to complete
        await self._wait_for_current_tasks()
        
        # Cleanup resources
        await self.cleanup()
        
        # Unregister from master
        await self._unregister_from_master()
        
        logger.info(f"Worker node {self.worker_id} stopped")
    
    async def _register_with_master(self):
        """Register this worker with the master node"""
        registration_data = {
            "worker_id": self.worker_id,
            "capabilities": {
                "max_concurrent_tasks": self.config.max_concurrent_tasks,
                "browser_pool_size": self.config.browser_pool_size,
                "supported_task_types": ["agent_task", "browser_script", "playground_action"],
                "cloud_provider": self.config.cloud_provider,
            },
            "status": self.status.value,
            "registered_at": datetime.utcnow().isoformat()
        }
        
        try:
            async with self.session.post(
                f"{self.config.master_url}/api/workers/register",
                json=registration_data
            ) as response:
                if response.status == 200:
                    logger.info("Successfully registered with master")
                else:
                    error = await response.text()
                    logger.error(f"Failed to register with master: {error}")
        except Exception as e:
            logger.error(f"Error registering with master: {e}")
    
    async def _unregister_from_master(self):
        """Unregister this worker from the master node"""
        try:
            async with self.session.post(
                f"{self.config.master_url}/api/workers/{self.worker_id}/unregister"
            ) as response:
                if response.status == 200:
                    logger.info("Successfully unregistered from master")
                else:
                    logger.warning("Failed to unregister from master")
        except Exception as e:
            logger.error(f"Error unregistering from master: {e}")
    
    async def _health_check_loop(self):
        """Periodically send health checks to master"""
        while self._running:
            try:
                await self._send_health_check()
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _send_health_check(self):
        """Send health check to master"""
        health_data = {
            "worker_id": self.worker_id,
            "status": self.status.value,
            "current_tasks": len(self.current_tasks),
            "available_browsers": len(self.browser_pool.available_browsers),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            async with self.session.post(
                f"{self.config.master_url}/api/workers/{self.worker_id}/health",
                json=health_data
            ) as response:
                if response.status != 200:
                    logger.warning(f"Health check failed: {response.status}")
        except Exception as e:
            logger.error(f"Error sending health check: {e}")
    
    async def _task_processing_loop(self):
        """Main loop for processing tasks"""
        while self._running:
            try:
                # Check if we can accept more tasks
                if len(self.current_tasks) >= self.config.max_concurrent_tasks:
                    await asyncio.sleep(1)
                    continue
                
                # Fetch next task from Redis queue
                task_data = await self._fetch_next_task()
                if task_data:
                    task = Task.from_dict(task_data)
                    await self._process_task(task)
                else:
                    # No tasks available, wait a bit
                    await asyncio.sleep(0.5)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in task processing loop: {e}")
                await asyncio.sleep(1)
    
    async def _fetch_next_task(self) -> Optional[Dict]:
        """Fetch next task from Redis queue"""
        try:
            # Use priority queue (sorted set)
            task_json = await self.redis.zpopmin(f"worker:{self.worker_id}:tasks", count=1)
            if task_json:
                return json.loads(task_json[0][0])
            return None
        except Exception as e:
            logger.error(f"Error fetching task: {e}")
            return None
    
    async def _process_task(self, task: Task):
        """Process a single task"""
        task_id = task.id
        logger.info(f"Processing task {task_id} of type {task.type}")
        
        # Add to current tasks
        self.current_tasks[task_id] = task
        self.status = WorkerStatus.BUSY
        
        # Update master about task start
        await self._update_task_status(task_id, "processing")
        
        try:
            # Execute task with timeout
            result = await asyncio.wait_for(
                self.task_executor.execute_task(task),
                timeout=task.timeout
            )
            
            # Set worker ID in result
            result.worker_id = self.worker_id
            
            # Send result to master
            await self._send_task_result(result)
            
            logger.info(f"Task {task_id} completed with status: {result.status}")
            
        except asyncio.TimeoutError:
            logger.warning(f"Task {task_id} timed out")
            result = TaskResult(
                task_id=task_id,
                worker_id=self.worker_id,
                status="timeout",
                error=f"Task execution timed out after {task.timeout} seconds"
            )
            await self._send_task_result(result)
            
        except Exception as e:
            logger.exception(f"Error processing task {task_id}")
            result = TaskResult(
                task_id=task_id,
                worker_id=self.worker_id,
                status="error",
                error=str(e)
            )
            await self._send_task_result(result)
            
        finally:
            # Remove from current tasks
            if task_id in self.current_tasks:
                del self.current_tasks[task_id]
            
            # Update status
            if not self.current_tasks:
                self.status = WorkerStatus.IDLE
    
    async def _update_task_status(self, task_id: str, status: str):
        """Update task status in master"""
        try:
            async with self.session.post(
                f"{self.config.master_url}/api/tasks/{task_id}/status",
                json={"status": status, "worker_id": self.worker_id}
            ) as response:
                if response.status != 200:
                    logger.warning(f"Failed to update task status: {response.status}")
        except Exception as e:
            logger.error(f"Error updating task status: {e}")
    
    async def _send_task_result(self, result: TaskResult):
        """Send task result to master"""
        try:
            result_data = asdict(result)
            result_data['completed_at'] = result.completed_at.isoformat()
            
            async with self.session.post(
                f"{self.config.master_url}/api/tasks/{result.task_id}/result",
                json=result_data
            ) as response:
                if response.status == 200:
                    logger.debug(f"Sent result for task {result.task_id}")
                else:
                    logger.warning(f"Failed to send task result: {response.status}")
        except Exception as e:
            logger.error(f"Error sending task result: {e}")
    
    async def _wait_for_current_tasks(self, timeout: int = 30):
        """Wait for current tasks to complete"""
        if not self.current_tasks:
            return
        
        logger.info(f"Waiting for {len(self.current_tasks)} tasks to complete...")
        
        start_time = datetime.utcnow()
        while self.current_tasks and (datetime.utcnow() - start_time).total_seconds() < timeout:
            await asyncio.sleep(1)
        
        if self.current_tasks:
            logger.warning(f"Timeout waiting for tasks: {list(self.current_tasks.keys())}")
    
    async def cleanup(self):
        """Cleanup all resources"""
        # Cleanup browser pool
        await self.browser_pool.cleanup()
        
        # Close Redis connection
        if self.redis:
            await self.redis.close()
        
        # Close HTTP session
        if self.session:
            await self.session.close()


class WorkerNodeManager:
    """Manages multiple worker nodes (for local scaling)"""
    
    def __init__(self, base_config: WorkerNodeConfig, num_workers: int = 1):
        self.base_config = base_config
        self.num_workers = num_workers
        self.workers: List[WorkerNode] = []
        self._running = False
    
    async def start(self):
        """Start all worker nodes"""
        logger.info(f"Starting {self.num_workers} worker nodes")
        self._running = True
        
        for i in range(self.num_workers):
            # Create unique worker ID for each worker
            worker_config = WorkerNodeConfig(
                worker_id=f"{self.base_config.worker_id}-{i}",
                master_url=self.base_config.master_url,
                redis_url=self.base_config.redis_url,
                max_concurrent_tasks=self.base_config.max_concurrent_tasks,
                health_check_interval=self.base_config.health_check_interval,
                task_timeout=self.base_config.task_timeout,
                browser_pool_size=self.base_config.browser_pool_size,
                enable_session_persistence=self.base_config.enable_session_persistence,
                cloud_provider=self.base_config.cloud_provider,
                cloud_config=self.base_config.cloud_config
            )
            
            worker = WorkerNode(worker_config)
            await worker.start()
            self.workers.append(worker)
        
        logger.info(f"All {self.num_workers} worker nodes started")
    
    async def stop(self):
        """Stop all worker nodes"""
        logger.info("Stopping all worker nodes")
        self._running = False
        
        # Stop all workers concurrently
        stop_tasks = [worker.stop() for worker in self.workers]
        await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        self.workers.clear()
        logger.info("All worker nodes stopped")
    
    async def scale(self, target_workers: int):
        """Scale the number of worker nodes"""
        current_workers = len(self.workers)
        
        if target_workers > current_workers:
            # Scale up
            for i in range(current_workers, target_workers):
                worker_config = WorkerNodeConfig(
                    worker_id=f"{self.base_config.worker_id}-{i}",
                    master_url=self.base_config.master_url,
                    redis_url=self.base_config.redis_url,
                    max_concurrent_tasks=self.base_config.max_concurrent_tasks,
                    health_check_interval=self.base_config.health_check_interval,
                    task_timeout=self.base_config.task_timeout,
                    browser_pool_size=self.base_config.browser_pool_size,
                    enable_session_persistence=self.base_config.enable_session_persistence,
                    cloud_provider=self.base_config.cloud_provider,
                    cloud_config=self.base_config.cloud_config
                )
                
                worker = WorkerNode(worker_config)
                await worker.start()
                self.workers.append(worker)
                
        elif target_workers < current_workers:
            # Scale down
            workers_to_remove = self.workers[target_workers:]
            self.workers = self.workers[:target_workers]
            
            # Stop removed workers
            for worker in workers_to_remove:
                await worker.stop()
        
        logger.info(f"Scaled from {current_workers} to {target_workers} workers")


# Cloud provider integration helpers
class CloudWorkerDeployer:
    """Deploys worker nodes to cloud providers"""
    
    @staticmethod
    async def deploy_aws(config: WorkerNodeConfig, num_instances: int = 1):
        """Deploy worker nodes to AWS"""
        # Implementation would use boto3 to launch EC2 instances
        # with the worker node code pre-installed
        logger.info(f"Deploying {num_instances} worker nodes to AWS")
        # This is a placeholder - actual implementation would:
        # 1. Launch EC2 instances
        # 2. Install dependencies
        # 3. Start worker nodes
        # 4. Register with master
        pass
    
    @staticmethod
    async def deploy_gcp(config: WorkerNodeConfig, num_instances: int = 1):
        """Deploy worker nodes to GCP"""
        # Implementation would use google-cloud-compute
        logger.info(f"Deploying {num_instances} worker nodes to GCP")
        pass
    
    @staticmethod
    async def deploy_docker(config: WorkerNodeConfig, num_containers: int = 1):
        """Deploy worker nodes as Docker containers"""
        # Implementation would use Docker API
        logger.info(f"Deploying {num_containers} worker containers")
        pass


# CLI interface for running worker nodes
async def run_worker_node_cli():
    """Command-line interface for running a worker node"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run a browser farm worker node")
    parser.add_argument("--worker-id", help="Worker node ID")
    parser.add_argument("--master-url", default="http://localhost:8000", help="Master node URL")
    parser.add_argument("--redis-url", default="redis://localhost:6379", help="Redis URL")
    parser.add_argument("--max-tasks", type=int, default=5, help="Maximum concurrent tasks")
    parser.add_argument("--browser-pool", type=int, default=3, help="Browser pool size")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of worker nodes to run")
    
    args = parser.parse_args()
    
    config = WorkerNodeConfig(
        worker_id=args.worker_id,
        master_url=args.master_url,
        redis_url=args.redis_url,
        max_concurrent_tasks=args.max_tasks,
        browser_pool_size=args.browser_pool
    )
    
    if args.num_workers == 1:
        worker = WorkerNode(config)
        try:
            await worker.start()
            # Keep running until interrupted
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await worker.stop()
    else:
        manager = WorkerNodeManager(config, args.num_workers)
        try:
            await manager.start()
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await manager.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_worker_node_cli())