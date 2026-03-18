"""
Dependency Manager for Hot-Reloadable Extensions with Dependency Isolation
Enables live reloading of extensions without restarting the entire UI.
Each extension runs in isolated namespace with controlled dependency injection.
"""

import sys
import os
import importlib
import importlib.util
import inspect
import hashlib
import json
import time
import threading
import traceback
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import ast
import weakref

# Global references to core modules we'll need
modules = sys.modules


class ReloadStatus(Enum):
    """Status of extension reload operation"""
    PENDING = "pending"
    RELOADING = "reloading"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class ModuleInfo:
    """Information about a module in the dependency graph"""
    name: str
    file_path: str
    module: Any = None
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    checksum: Optional[str] = None
    last_reload: float = 0.0
    reload_count: int = 0


@dataclass
class ExtensionState:
    """State information for an extension"""
    name: str
    path: str
    main_module: str
    modules: Dict[str, ModuleInfo] = field(default_factory=dict)
    namespace: Dict[str, Any] = field(default_factory=dict)
    status: ReloadStatus = ReloadStatus.PENDING
    last_error: Optional[str] = None
    health_check: Optional[Callable] = None
    rollback_snapshot: Optional[Dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class DependencyGraph:
    """Manages module dependencies and reload order"""
    
    def __init__(self):
        self.graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_graph: Dict[str, Set[str]] = defaultdict(set)
        self.modules: Dict[str, ModuleInfo] = {}
        
    def add_module(self, module_info: ModuleInfo):
        """Add a module to the dependency graph"""
        self.modules[module_info.name] = module_info
        self.graph[module_info.name] = module_info.dependencies.copy()
        
        # Update reverse dependencies
        for dep in module_info.dependencies:
            self.reverse_graph[dep].add(module_info.name)
    
    def remove_module(self, module_name: str):
        """Remove a module from the dependency graph"""
        if module_name in self.modules:
            # Remove from reverse dependencies
            for dep in self.modules[module_name].dependencies:
                if dep in self.reverse_graph:
                    self.reverse_graph[dep].discard(module_name)
            
            # Remove from forward dependencies
            del self.graph[module_name]
            del self.modules[module_name]
    
    def get_reload_order(self, module_names: Set[str]) -> List[str]:
        """Get topological order for reloading affected modules"""
        # Build subgraph of affected modules and their dependents
        affected = set()
        queue = deque(module_names)
        
        while queue:
            module = queue.popleft()
            if module in affected:
                continue
            affected.add(module)
            
            # Add all dependents
            for dependent in self.reverse_graph.get(module, set()):
                if dependent not in affected:
                    queue.append(dependent)
        
        # Topological sort on affected modules
        in_degree = {m: 0 for m in affected}
        for module in affected:
            for dep in self.graph.get(module, set()):
                if dep in affected:
                    in_degree[module] += 1
        
        queue = deque([m for m in affected if in_degree[m] == 0])
        order = []
        
        while queue:
            module = queue.popleft()
            order.append(module)
            
            for dependent in self.reverse_graph.get(module, set()):
                if dependent in affected:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)
        
        if len(order) != len(affected):
            # Circular dependency detected
            raise ValueError(f"Circular dependency detected among modules: {affected}")
        
        return order
    
    def get_dependencies(self, module_name: str, transitive: bool = False) -> Set[str]:
        """Get dependencies of a module"""
        if not transitive:
            return self.graph.get(module_name, set()).copy()
        
        # Get all transitive dependencies
        visited = set()
        queue = deque([module_name])
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            
            for dep in self.graph.get(current, set()):
                if dep not in visited:
                    queue.append(dep)
        
        visited.discard(module_name)
        return visited


class NamespaceIsolator:
    """Provides isolated namespace for extension modules"""
    
    def __init__(self, extension_name: str, core_modules: Dict[str, Any]):
        self.extension_name = extension_name
        self.core_modules = core_modules
        self.isolated_modules: Dict[str, Any] = {}
        
    def create_isolated_module(self, module_name: str, module: Any) -> Any:
        """Create an isolated version of a module"""
        # Create a new module object
        isolated = type(sys)(module_name)
        isolated.__file__ = getattr(module, '__file__', '')
        isolated.__loader__ = getattr(module, '__loader__', None)
        isolated.__package__ = getattr(module, '__package__', '')
        
        # Copy module attributes but filter out private ones
        for key, value in module.__dict__.items():
            if not key.startswith('_') or key in ('__file__', '__path__', '__package__'):
                setattr(isolated, key, value)
        
        # Inject controlled dependencies
        self._inject_dependencies(isolated)
        
        return isolated
    
    def _inject_dependencies(self, module: Any):
        """Inject controlled dependencies into isolated module"""
        # Provide access to core modules through controlled interface
        for core_name, core_module in self.core_modules.items():
            if hasattr(module, core_name):
                continue  # Don't override existing attributes
            
            # Create a proxy for the core module
            proxy = self._create_module_proxy(core_name, core_module)
            setattr(module, core_name, proxy)
    
    def _create_module_proxy(self, name: str, module: Any) -> Any:
        """Create a proxy object for a module with controlled access"""
        class ModuleProxy:
            def __init__(self, name, module):
                self._name = name
                self._module = module
                self._accessed_attrs = set()
            
            def __getattr__(self, attr):
                self._accessed_attrs.add(attr)
                return getattr(self._module, attr)
            
            def __setattr__(self, attr, value):
                if attr.startswith('_'):
                    super().__setattr__(attr, value)
                else:
                    setattr(self._module, attr, value)
        
        return ModuleProxy(name, module)


class ExtensionHealthChecker:
    """Performs health checks on extensions"""
    
    @staticmethod
    def check_extension(extension_state: ExtensionState) -> Tuple[bool, str]:
        """Check if extension is healthy"""
        try:
            # Check if main module can be imported
            if extension_state.main_module not in sys.modules:
                return False, f"Main module {extension_state.main_module} not loaded"
            
            # Check if health check function exists and passes
            if extension_state.health_check:
                result = extension_state.health_check()
                if not result:
                    return False, "Health check function returned False"
            
            # Check for common error patterns
            main_module = sys.modules.get(extension_state.main_module)
            if main_module:
                # Check if required attributes exist
                required_attrs = ['script', 'callbacks']
                for attr in required_attrs:
                    if hasattr(main_module, attr):
                        attr_value = getattr(main_module, attr)
                        if attr_value is None:
                            return False, f"Required attribute {attr} is None"
            
            return True, "Extension is healthy"
            
        except Exception as e:
            return False, f"Health check failed: {str(e)}"
    
    @staticmethod
    def create_health_check(extension_path: str) -> Optional[Callable]:
        """Create a health check function for an extension"""
        health_check_path = os.path.join(extension_path, 'health_check.py')
        if os.path.exists(health_check_path):
            try:
                spec = importlib.util.spec_from_file_location(
                    f"health_check_{os.path.basename(extension_path)}",
                    health_check_path
                )
                health_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(health_module)
                
                if hasattr(health_module, 'check'):
                    return health_module.check
            except:
                pass
        return None


class DependencyManager:
    """Main dependency manager for hot-reloading extensions"""
    
    def __init__(self, extensions_dir: str = "extensions-builtin"):
        self.extensions_dir = Path(extensions_dir)
        self.extensions: Dict[str, ExtensionState] = {}
        self.dependency_graph = DependencyGraph()
        self.namespace_isolator = None
        self.health_checker = ExtensionHealthChecker()
        self.watcher_thread: Optional[threading.Thread] = None
        self.watcher_running = False
        self.file_checksums: Dict[str, str] = {}
        self.reload_callbacks: List[Callable] = []
        self.lock = threading.RLock()
        
        # Core modules that extensions can depend on
        self.core_modules = self._get_core_modules()
        
        # Initialize namespace isolator
        self.namespace_isolator = NamespaceIsolator(
            "dependency_manager",
            self.core_modules
        )
        
        # Initialize built-in extensions
        self._initialize_builtin_extensions()
    
    def _get_core_modules(self) -> Dict[str, Any]:
        """Get core modules that extensions can depend on"""
        core_module_names = [
            'modules.processing',
            'modules.shared',
            'modules.sd_models',
            'modules.devices',
            'modules.ui',
            'modules.script_callbacks',
            'modules.extensions',
            'modules.paths',
            'modules.images',
            'modules.sd_samplers',
            'modules.sd_vae',
            'modules.hypernetworks',
            'modules.textual_inversion',
            'modules.extra_networks',
            'modules.ui_common',
            'modules.ui_postprocessing',
            'modules.ui_extensions',
            'modules.generation_parameters_copypaste',
            'modules.sd_hijack',
            'modules.sd_hijack_optimizations',
            'modules.sd_unet',
            'modules.sd_disable_initialization',
            'modules.sd_models_config',
            'modules.sd_vae_taesd',
            'modules.lowvram',
            'modules.modelloader',
            'modules.errors',
            'modules.progress',
            'modules.prompt_parser',
            'modules.styles',
            'modules.generation_parameters',
        ]
        
        core_modules = {}
        for module_name in core_module_names:
            if module_name in sys.modules:
                core_modules[module_name.split('.')[-1]] = sys.modules[module_name]
        
        return core_modules
    
    def _initialize_builtin_extensions(self):
        """Initialize built-in extensions from extensions-builtin directory"""
        if not self.extensions_dir.exists():
            return
        
        for extension_dir in self.extensions_dir.iterdir():
            if extension_dir.is_dir() and not extension_dir.name.startswith('.'):
                self.register_extension(extension_dir.name, str(extension_dir))
    
    def register_extension(self, extension_name: str, extension_path: str) -> bool:
        """Register an extension with the dependency manager"""
        with self.lock:
            if extension_name in self.extensions:
                return False
            
            # Find main module (usually scripts/*.py or install.py)
            main_module = self._find_main_module(extension_path)
            if not main_module:
                return False
            
            # Create extension state
            extension_state = ExtensionState(
                name=extension_name,
                path=extension_path,
                main_module=main_module,
                health_check=self.health_checker.create_health_check(extension_path)
            )
            
            # Scan and register all modules in the extension
            self._scan_extension_modules(extension_state)
            
            # Create isolated namespace
            extension_state.namespace = self._create_extension_namespace(extension_state)
            
            self.extensions[extension_name] = extension_state
            return True
    
    def _find_main_module(self, extension_path: str) -> Optional[str]:
        """Find the main module of an extension"""
        # Check for scripts directory
        scripts_dir = os.path.join(extension_path, 'scripts')
        if os.path.exists(scripts_dir):
            for file in os.listdir(scripts_dir):
                if file.endswith('.py') and not file.startswith('_'):
                    module_name = f"extensions.{os.path.basename(extension_path)}.scripts.{file[:-3]}"
                    return module_name
        
        # Check for install.py
        install_path = os.path.join(extension_path, 'install.py')
        if os.path.exists(install_path):
            module_name = f"extensions.{os.path.basename(extension_path)}.install"
            return module_name
        
        # Check for any .py file in root
        for file in os.listdir(extension_path):
            if file.endswith('.py') and not file.startswith('_'):
                module_name = f"extensions.{os.path.basename(extension_path)}.{file[:-3]}"
                return module_name
        
        return None
    
    def _scan_extension_modules(self, extension_state: ExtensionState):
        """Scan and register all modules in an extension"""
        extension_path = Path(extension_state.path)
        extension_name = extension_state.name
        
        for py_file in extension_path.rglob("*.py"):
            if py_file.name.startswith('_'):
                continue
            
            # Create module name
            rel_path = py_file.relative_to(extension_path)
            module_parts = list(rel_path.with_suffix('').parts)
            module_name = f"extensions.{extension_name}.{'.'.join(module_parts)}"
            
            # Calculate file checksum
            checksum = self._calculate_file_checksum(str(py_file))
            
            # Create module info
            module_info = ModuleInfo(
                name=module_name,
                file_path=str(py_file),
                checksum=checksum,
                last_reload=time.time()
            )
            
            # Analyze dependencies
            module_info.dependencies = self._analyze_module_dependencies(
                str(py_file), extension_name
            )
            
            # Add to extension and graph
            extension_state.modules[module_name] = module_info
            self.dependency_graph.add_module(module_info)
    
    def _analyze_module_dependencies(self, file_path: str, extension_name: str) -> Set[str]:
        """Analyze dependencies of a module using AST"""
        dependencies = set()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=file_path)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        dependencies.add(alias.name.split('.')[0])
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_parts = node.module.split('.')
                        
                        # Check if it's an import from within the extension
                        if module_parts[0] == 'extensions' and len(module_parts) > 1:
                            if module_parts[1] == extension_name:
                                # Internal dependency
                                dep_module = f"extensions.{extension_name}.{'.'.join(module_parts[2:])}"
                                dependencies.add(dep_module)
                        else:
                            # External dependency
                            dependencies.add(module_parts[0])
            
            # Also check for common dependencies
            common_deps = {'modules', 'torch', 'numpy', 'PIL', 'gradio', 'fastapi'}
            for dep in common_deps:
                if dep in content:
                    dependencies.add(dep)
        
        except Exception as e:
            print(f"Warning: Could not analyze dependencies for {file_path}: {e}")
        
        return dependencies
    
    def _calculate_file_checksum(self, file_path: str) -> str:
        """Calculate MD5 checksum of a file"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            return hashlib.md5(content).hexdigest()
        except:
            return ""
    
    def _create_extension_namespace(self, extension_state: ExtensionState) -> Dict[str, Any]:
        """Create isolated namespace for an extension"""
        namespace = {
            '__name__': f"extensions.{extension_state.name}",
            '__package__': f"extensions.{extension_state.name}",
            '__path__': [extension_state.path],
            '__file__': os.path.join(extension_state.path, '__init__.py'),
            'extension_name': extension_state.name,
            'extension_path': extension_state.path,
        }
        
        # Add core modules to namespace
        for name, module in self.core_modules.items():
            namespace[name] = module
        
        return namespace
    
    def reload_extension(self, extension_name: str, force: bool = False) -> bool:
        """Reload an extension with dependency isolation"""
        with self.lock:
            if extension_name not in self.extensions:
                print(f"Extension {extension_name} not found")
                return False
            
            extension_state = self.extensions[extension_name]
            
            # Check if reload is needed
            if not force and not self._check_reload_needed(extension_state):
                print(f"Extension {extension_name} doesn't need reload")
                return True
            
            print(f"Reloading extension {extension_name}...")
            extension_state.status = ReloadStatus.RELOADING
            extension_state.updated_at = time.time()
            
            try:
                # Create snapshot for rollback
                self._create_rollback_snapshot(extension_state)
                
                # Get modules that need reloading
                modules_to_reload = self._get_modules_to_reload(extension_state)
                
                # Get reload order
                reload_order = self.dependency_graph.get_reload_order(modules_to_reload)
                
                # Reload modules in order
                for module_name in reload_order:
                    if not self._reload_module(module_name, extension_state):
                        raise Exception(f"Failed to reload module {module_name}")
                
                # Perform health check
                is_healthy, message = self.health_checker.check_extension(extension_state)
                if not is_healthy:
                    raise Exception(f"Health check failed: {message}")
                
                # Update status
                extension_state.status = ReloadStatus.SUCCESS
                extension_state.last_error = None
                
                # Notify callbacks
                self._notify_reload_callbacks(extension_name, True)
                
                print(f"Successfully reloaded extension {extension_name}")
                return True
                
            except Exception as e:
                # Rollback on failure
                print(f"Failed to reload extension {extension_name}: {e}")
                traceback.print_exc()
                
                extension_state.status = ReloadStatus.FAILED
                extension_state.last_error = str(e)
                
                # Perform rollback
                self._rollback_extension(extension_state)
                
                # Notify callbacks
                self._notify_reload_callbacks(extension_name, False)
                
                return False
    
    def _check_reload_needed(self, extension_state: ExtensionState) -> bool:
        """Check if any modules in the extension have changed"""
        for module_info in extension_state.modules.values():
            current_checksum = self._calculate_file_checksum(module_info.file_path)
            if current_checksum != module_info.checksum:
                return True
        return False
    
    def _get_modules_to_reload(self, extension_state: ExtensionState) -> Set[str]:
        """Get set of modules that need reloading"""
        modules_to_reload = set()
        
        for module_name, module_info in extension_state.modules.items():
            current_checksum = self._calculate_file_checksum(module_info.file_path)
            if current_checksum != module_info.checksum:
                modules_to_reload.add(module_name)
                # Update checksum
                module_info.checksum = current_checksum
        
        return modules_to_reload
    
    def _reload_module(self, module_name: str, extension_state: ExtensionState) -> bool:
        """Reload a single module"""
        try:
            module_info = extension_state.modules.get(module_name)
            if not module_info:
                return False
            
            # Remove from sys.modules if present
            if module_name in sys.modules:
                del sys.modules[module_name]
            
            # Load module with isolated namespace
            spec = importlib.util.spec_from_file_location(
                module_name,
                module_info.file_path,
                submodule_search_locations=[]
            )
            
            if not spec or not spec.loader:
                return False
            
            # Create module in isolated namespace
            module = importlib.util.module_from_spec(spec)
            module.__dict__.update(extension_state.namespace)
            
            # Execute module
            spec.loader.exec_module(module)
            
            # Store module
            module_info.module = module
            module_info.last_reload = time.time()
            module_info.reload_count += 1
            
            # Add to sys.modules for import resolution
            sys.modules[module_name] = module
            
            return True
            
        except Exception as e:
            print(f"Error reloading module {module_name}: {e}")
            traceback.print_exc()
            return False
    
    def _create_rollback_snapshot(self, extension_state: ExtensionState):
        """Create a snapshot for rollback"""
        snapshot = {
            'modules': {},
            'namespace': extension_state.namespace.copy(),
            'timestamp': time.time()
        }
        
        for module_name, module_info in extension_state.modules.items():
            if module_info.module:
                snapshot['modules'][module_name] = {
                    'module': module_info.module,
                    'checksum': module_info.checksum,
                    'reload_count': module_info.reload_count
                }
        
        extension_state.rollback_snapshot = snapshot
    
    def _rollback_extension(self, extension_state: ExtensionState):
        """Rollback extension to previous state"""
        if not extension_state.rollback_snapshot:
            return
        
        snapshot = extension_state.rollback_snapshot
        
        try:
            # Restore modules
            for module_name, module_data in snapshot['modules'].items():
                if module_name in extension_state.modules:
                    module_info = extension_state.modules[module_name]
                    module_info.module = module_data['module']
                    module_info.checksum = module_data['checksum']
                    module_info.reload_count = module_data['reload_count']
                    
                    # Restore in sys.modules
                    if module_info.module:
                        sys.modules[module_name] = module_info.module
            
            # Restore namespace
            extension_state.namespace = snapshot['namespace']
            
            extension_state.status = ReloadStatus.ROLLED_BACK
            print(f"Rolled back extension {extension_state.name}")
            
        except Exception as e:
            print(f"Error during rollback: {e}")
            extension_state.status = ReloadStatus.FAILED
    
    def reload_module(self, module_name: str) -> bool:
        """Reload a specific module and its dependents"""
        with self.lock:
            # Find which extension owns this module
            extension_name = None
            for ext_name, ext_state in self.extensions.items():
                if module_name in ext_state.modules:
                    extension_name = ext_name
                    break
            
            if not extension_name:
                print(f"Module {module_name} not found in any extension")
                return False
            
            # Get all modules that depend on this one
            dependents = self.dependency_graph.get_dependencies(module_name, transitive=True)
            modules_to_reload = {module_name}.union(dependents)
            
            # Get reload order
            try:
                reload_order = self.dependency_graph.get_reload_order(modules_to_reload)
            except ValueError as e:
                print(f"Cannot reload: {e}")
                return False
            
            # Reload modules
            extension_state = self.extensions[extension_name]
            for mod_name in reload_order:
                if mod_name in extension_state.modules:
                    if not self._reload_module(mod_name, extension_state):
                        print(f"Failed to reload module {mod_name}")
                        return False
            
            return True
    
    def add_reload_callback(self, callback: Callable):
        """Add a callback to be called when an extension is reloaded"""
        self.reload_callbacks.append(callback)
    
    def _notify_reload_callbacks(self, extension_name: str, success: bool):
        """Notify all registered callbacks about a reload"""
        for callback in self.reload_callbacks:
            try:
                callback(extension_name, success)
            except Exception as e:
                print(f"Error in reload callback: {e}")
    
    def start_file_watcher(self, interval: float = 2.0):
        """Start watching for file changes"""
        if self.watcher_running:
            return
        
        self.watcher_running = True
        self.watcher_thread = threading.Thread(
            target=self._file_watcher_loop,
            args=(interval,),
            daemon=True
        )
        self.watcher_thread.start()
    
    def stop_file_watcher(self):
        """Stop watching for file changes"""
        self.watcher_running = False
        if self.watcher_thread:
            self.watcher_thread.join(timeout=5.0)
    
    def _file_watcher_loop(self, interval: float):
        """Main loop for file watcher"""
        while self.watcher_running:
            try:
                self._check_for_changes()
                time.sleep(interval)
            except Exception as e:
                print(f"Error in file watcher: {e}")
                time.sleep(interval)
    
    def _check_for_changes(self):
        """Check for file changes and trigger reloads"""
        with self.lock:
            for extension_name, extension_state in self.extensions.items():
                if extension_state.status == ReloadStatus.RELOADING:
                    continue
                
                # Check if any module has changed
                needs_reload = False
                for module_info in extension_state.modules.values():
                    current_checksum = self._calculate_file_checksum(module_info.file_path)
                    if current_checksum != module_info.checksum:
                        needs_reload = True
                        break
                
                if needs_reload:
                    print(f"Detected changes in extension {extension_name}, scheduling reload...")
                    # Schedule reload in a separate thread to avoid blocking
                    threading.Thread(
                        target=self.reload_extension,
                        args=(extension_name,),
                        daemon=True
                    ).start()
    
    def get_extension_info(self, extension_name: str) -> Optional[Dict[str, Any]]:
        """Get information about an extension"""
        if extension_name not in self.extensions:
            return None
        
        ext = self.extensions[extension_name]
        
        return {
            'name': ext.name,
            'path': ext.path,
            'main_module': ext.main_module,
            'status': ext.status.value,
            'last_error': ext.last_error,
            'modules': {
                name: {
                    'file_path': info.file_path,
                    'last_reload': info.last_reload,
                    'reload_count': info.reload_count,
                    'dependencies': list(info.dependencies)
                }
                for name, info in ext.modules.items()
            },
            'created_at': ext.created_at,
            'updated_at': ext.updated_at
        }
    
    def get_all_extensions(self) -> List[Dict[str, Any]]:
        """Get information about all registered extensions"""
        return [
            self.get_extension_info(name)
            for name in self.extensions.keys()
        ]
    
    def clear_extension_cache(self, extension_name: str):
        """Clear cached modules for an extension"""
        with self.lock:
            if extension_name not in self.extensions:
                return
            
            extension_state = self.extensions[extension_name]
            
            # Remove modules from sys.modules
            for module_name in extension_state.modules.keys():
                if module_name in sys.modules:
                    del sys.modules[module_name]
            
            # Clear module references
            for module_info in extension_state.modules.values():
                module_info.module = None
    
    def export_dependency_graph(self, output_path: str):
        """Export dependency graph to JSON for visualization"""
        graph_data = {
            'modules': {},
            'extensions': {}
        }
        
        # Export module information
        for module_name, module_info in self.dependency_graph.modules.items():
            graph_data['modules'][module_name] = {
                'file_path': module_info.file_path,
                'dependencies': list(module_info.dependencies),
                'dependents': list(module_info.dependents),
                'reload_count': module_info.reload_count
            }
        
        # Export extension information
        for ext_name, ext_state in self.extensions.items():
            graph_data['extensions'][ext_name] = {
                'modules': list(ext_state.modules.keys()),
                'status': ext_state.status.value
            }
        
        with open(output_path, 'w') as f:
            json.dump(graph_data, f, indent=2)


# Global instance
dependency_manager = DependencyManager()


def reload_extension(extension_name: str, force: bool = False) -> bool:
    """Reload an extension (convenience function)"""
    return dependency_manager.reload_extension(extension_name, force)


def reload_module(module_name: str) -> bool:
    """Reload a specific module (convenience function)"""
    return dependency_manager.reload_module(module_name)


def start_auto_reload(interval: float = 2.0):
    """Start automatic reloading on file changes"""
    dependency_manager.start_file_watcher(interval)


def stop_auto_reload():
    """Stop automatic reloading"""
    dependency_manager.stop_file_watcher()


def get_extension_status(extension_name: str) -> Optional[Dict[str, Any]]:
    """Get status of an extension"""
    return dependency_manager.get_extension_info(extension_name)


# Integration with existing extension system
def patch_extension_loader():
    """Patch the existing extension loader to use our dependency manager"""
    try:
        import modules.extensions as extensions_module
        
        # Store original load_extensions function
        original_load_extensions = extensions_module.load_extensions
        
        def patched_load_extensions():
            # Call original function
            original_load_extensions()
            
            # Register loaded extensions with dependency manager
            for ext in extensions_module.extensions:
                if hasattr(ext, 'path') and hasattr(ext, 'name'):
                    dependency_manager.register_extension(ext.name, ext.path)
        
        # Replace the function
        extensions_module.load_extensions = patched_load_extensions
        
        print("Dependency manager integrated with extension loader")
        
    except Exception as e:
        print(f"Warning: Could not patch extension loader: {e}")


# Auto-patch on import
try:
    patch_extension_loader()
except:
    pass


# Example health check for an extension
def example_health_check():
    """Example health check function for extensions"""
    # Check if required modules are loaded
    required_modules = ['torch', 'numpy']
    for module in required_modules:
        if module not in sys.modules:
            return False
    
    # Check if GPU is available if needed
    try:
        import torch
        if not torch.cuda.is_available():
            print("Warning: CUDA not available")
    except:
        pass
    
    return True


if __name__ == "__main__":
    # Example usage
    print("Dependency Manager for Stable Diffusion WebUI")
    print("=" * 50)
    
    # List registered extensions
    extensions = dependency_manager.get_all_extensions()
    print(f"Registered {len(extensions)} extensions")
    
    for ext in extensions[:3]:  # Show first 3
        print(f"\nExtension: {ext['name']}")
        print(f"  Status: {ext['status']}")
        print(f"  Modules: {len(ext['modules'])}")
    
    # Start auto-reload (for development)
    # start_auto_reload()
    
    print("\nDependency manager ready. Use reload_extension('extension_name') to reload extensions.")