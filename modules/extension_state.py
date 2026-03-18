"""
Atomic Extension State Management for nexus
Transaction-based extension state with rollback capability.
Extensions can safely experiment without risking UI stability.
"""

import os
import json
import copy
import time
import hashlib
import threading
import traceback
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
from collections import defaultdict

import modules.scripts as scripts
import modules.extensions as extensions
from modules import shared, paths, errors


class ExtensionState(Enum):
    """Extension state enumeration."""
    DISABLED = "disabled"
    ENABLED = "enabled"
    FAILED = "failed"
    SUSPENDED = "suspended"
    EXPERIMENTAL = "experimental"


class TransactionType(Enum):
    """Transaction type enumeration."""
    ENABLE = "enable"
    DISABLE = "disable"
    CONFIG_UPDATE = "config_update"
    ROLLBACK = "rollback"
    CHECKPOINT = "checkpoint"


@dataclass
class ExtensionCheckpoint:
    """Checkpoint for extension state rollback."""
    extension_name: str
    timestamp: float
    state: ExtensionState
    config_hash: str
    config_snapshot: Dict[str, Any]
    enabled_scripts: List[str]
    transaction_id: str


@dataclass
class ExtensionTransaction:
    """Transaction for atomic extension operations."""
    transaction_id: str
    timestamp: float
    transaction_type: TransactionType
    extension_name: str
    old_state: ExtensionState
    new_state: ExtensionState
    changes: Dict[str, Any] = field(default_factory=dict)
    completed: bool = False
    rolled_back: bool = False
    error_message: Optional[str] = None


@dataclass
class ExtensionStability:
    """Stability metrics for an extension."""
    extension_name: str
    total_loads: int = 0
    successful_loads: int = 0
    failed_loads: int = 0
    crash_count: int = 0
    last_crash_time: float = 0.0
    average_load_time: float = 0.0
    stability_score: float = 100.0
    stability_rating: str = "A"
    last_updated: float = field(default_factory=time.time)


class ExtensionStateManager:
    """
    Manages extension state with transaction-based operations and rollback capability.
    Implements copy-on-write state management for safe experimentation.
    """
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        with self._lock:
            self._initialized = True
            
            # Core state storage
            self._extension_states: Dict[str, ExtensionState] = {}
            self._extension_configs: Dict[str, Dict[str, Any]] = {}
            self._extension_scripts: Dict[str, List[str]] = {}
            
            # Transaction and checkpoint management
            self._transactions: Dict[str, ExtensionTransaction] = {}
            self._checkpoints: Dict[str, ExtensionCheckpoint] = {}
            self._active_transactions: Set[str] = set()
            
            # Stability tracking
            self._stability_metrics: Dict[str, ExtensionStability] = {}
            
            # Copy-on-write state
            self._state_copies: Dict[str, Dict[str, Any]] = {}
            self._dirty_extensions: Set[str] = set()
            
            # Configuration
            self._config_path = Path(paths.data_path) / "extension_states.json"
            self._stability_path = Path(paths.data_path) / "extension_stability.json"
            self._checkpoints_path = Path(paths.data_path) / "extension_checkpoints"
            
            # Create checkpoints directory
            self._checkpoints_path.mkdir(exist_ok=True)
            
            # Load persisted state
            self._load_state()
            self._load_stability_metrics()
            
            # Register shutdown handler
            import atexit
            atexit.register(self._save_state)
    
    def _generate_transaction_id(self) -> str:
        """Generate a unique transaction ID."""
        timestamp = str(time.time())
        random_data = os.urandom(16).hex()
        return hashlib.sha256(f"{timestamp}{random_data}".encode()).hexdigest()[:16]
    
    def _load_state(self) -> None:
        """Load extension state from disk."""
        try:
            if self._config_path.exists():
                with open(self._config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Restore extension states
                    for ext_name, state_str in data.get('states', {}).items():
                        try:
                            self._extension_states[ext_name] = ExtensionState(state_str)
                        except ValueError:
                            self._extension_states[ext_name] = ExtensionState.DISABLED
                    
                    # Restore extension configs
                    self._extension_configs = data.get('configs', {})
                    
                    # Restore extension scripts
                    self._extension_scripts = data.get('scripts', {})
                    
                    print(f"[ExtensionStateManager] Loaded state for {len(self._extension_states)} extensions")
        except Exception as e:
            print(f"[ExtensionStateManager] Error loading state: {e}")
            errors.report(f"Failed to load extension state: {e}", exc_info=True)
    
    def _save_state(self) -> None:
        """Save extension state to disk."""
        try:
            with self._lock:
                data = {
                    'states': {name: state.value for name, state in self._extension_states.items()},
                    'configs': self._extension_configs,
                    'scripts': self._extension_scripts,
                    'last_saved': time.time()
                }
                
                # Atomic write using temporary file
                temp_path = self._config_path.with_suffix('.tmp')
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                # Replace original file
                temp_path.replace(self._config_path)
                
                print(f"[ExtensionStateManager] Saved state for {len(self._extension_states)} extensions")
        except Exception as e:
            print(f"[ExtensionStateManager] Error saving state: {e}")
            errors.report(f"Failed to save extension state: {e}", exc_info=True)
    
    def _load_stability_metrics(self) -> None:
        """Load stability metrics from disk."""
        try:
            if self._stability_path.exists():
                with open(self._stability_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    for ext_name, metrics in data.get('metrics', {}).items():
                        self._stability_metrics[ext_name] = ExtensionStability(
                            extension_name=ext_name,
                            **metrics
                        )
                    
                    print(f"[ExtensionStateManager] Loaded stability metrics for {len(self._stability_metrics)} extensions")
        except Exception as e:
            print(f"[ExtensionStateManager] Error loading stability metrics: {e}")
    
    def _save_stability_metrics(self) -> None:
        """Save stability metrics to disk."""
        try:
            with self._lock:
                data = {
                    'metrics': {
                        name: asdict(metrics) for name, metrics in self._stability_metrics.items()
                    },
                    'last_saved': time.time()
                }
                
                # Atomic write
                temp_path = self._stability_path.with_suffix('.tmp')
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                temp_path.replace(self._stability_path)
        except Exception as e:
            print(f"[ExtensionStateManager] Error saving stability metrics: {e}")
    
    def _calculate_stability_rating(self, metrics: ExtensionStability) -> str:
        """Calculate stability rating based on crash frequency and success rate."""
        if metrics.total_loads == 0:
            return "A"
        
        success_rate = (metrics.successful_loads / metrics.total_loads) * 100
        crash_rate = (metrics.crash_count / max(metrics.total_loads, 1)) * 100
        
        # Calculate score (0-100)
        score = success_rate - (crash_rate * 2)
        metrics.stability_score = max(0, min(100, score))
        
        # Assign rating
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def _update_stability_metrics(self, extension_name: str, success: bool, 
                                 load_time: float = 0.0, crashed: bool = False) -> None:
        """Update stability metrics for an extension."""
        with self._lock:
            if extension_name not in self._stability_metrics:
                self._stability_metrics[extension_name] = ExtensionStability(
                    extension_name=extension_name
                )
            
            metrics = self._stability_metrics[extension_name]
            metrics.total_loads += 1
            metrics.last_updated = time.time()
            
            if success:
                metrics.successful_loads += 1
                # Update average load time
                if load_time > 0:
                    total_time = metrics.average_load_time * (metrics.successful_loads - 1)
                    metrics.average_load_time = (total_time + load_time) / metrics.successful_loads
            else:
                metrics.failed_loads += 1
            
            if crashed:
                metrics.crash_count += 1
                metrics.last_crash_time = time.time()
            
            # Update stability rating
            metrics.stability_rating = self._calculate_stability_rating(metrics)
            
            # Auto-save periodically
            if metrics.total_loads % 10 == 0:
                self._save_stability_metrics()
    
    def _create_state_copy(self, extension_name: str) -> Dict[str, Any]:
        """Create a copy-on-write state for an extension."""
        with self._lock:
            if extension_name in self._state_copies:
                return self._state_copies[extension_name]
            
            # Create deep copy of current state
            state_copy = {
                'state': self._extension_states.get(extension_name, ExtensionState.DISABLED),
                'config': copy.deepcopy(self._extension_configs.get(extension_name, {})),
                'scripts': copy.deepcopy(self._extension_scripts.get(extension_name, []))
            }
            
            self._state_copies[extension_name] = state_copy
            return state_copy
    
    def _apply_state_copy(self, extension_name: str) -> None:
        """Apply a state copy to the actual state."""
        with self._lock:
            if extension_name not in self._state_copies:
                return
            
            state_copy = self._state_copies[extension_name]
            
            # Apply the copy
            self._extension_states[extension_name] = state_copy['state']
            self._extension_configs[extension_name] = state_copy['config']
            self._extension_scripts[extension_name] = state_copy['scripts']
            
            # Mark as dirty for persistence
            self._dirty_extensions.add(extension_name)
            
            # Clean up the copy
            del self._state_copies[extension_name]
    
    def _discard_state_copy(self, extension_name: str) -> None:
        """Discard a state copy without applying."""
        with self._lock:
            if extension_name in self._state_copies:
                del self._state_copies[extension_name]
    
    def begin_transaction(self, extension_name: str, transaction_type: TransactionType,
                         changes: Optional[Dict[str, Any]] = None) -> str:
        """
        Begin a new transaction for an extension.
        
        Args:
            extension_name: Name of the extension
            transaction_type: Type of transaction
            changes: Dictionary of changes to apply
            
        Returns:
            Transaction ID
        """
        with self._lock:
            transaction_id = self._generate_transaction_id()
            
            # Create state copy for copy-on-write
            state_copy = self._create_state_copy(extension_name)
            
            # Get current state
            current_state = self._extension_states.get(extension_name, ExtensionState.DISABLED)
            
            # Create transaction
            transaction = ExtensionTransaction(
                transaction_id=transaction_id,
                timestamp=time.time(),
                transaction_type=transaction_type,
                extension_name=extension_name,
                old_state=current_state,
                new_state=current_state,  # Will be updated in commit
                changes=changes or {}
            )
            
            self._transactions[transaction_id] = transaction
            self._active_transactions.add(transaction_id)
            
            print(f"[ExtensionStateManager] Started transaction {transaction_id} for {extension_name}")
            return transaction_id
    
    def commit_transaction(self, transaction_id: str, new_state: Optional[ExtensionState] = None,
                          success: bool = True, error_message: Optional[str] = None) -> bool:
        """
        Commit a transaction.
        
        Args:
            transaction_id: ID of the transaction to commit
            new_state: New state to set (if applicable)
            success: Whether the operation was successful
            error_message: Error message if operation failed
            
        Returns:
            True if commit was successful
        """
        with self._lock:
            if transaction_id not in self._transactions:
                print(f"[ExtensionStateManager] Transaction {transaction_id} not found")
                return False
            
            transaction = self._transactions[transaction_id]
            extension_name = transaction.extension_name
            
            try:
                if success:
                    # Update new state if provided
                    if new_state is not None:
                        transaction.new_state = new_state
                    
                    # Apply state copy
                    if extension_name in self._state_copies:
                        state_copy = self._state_copies[extension_name]
                        state_copy['state'] = transaction.new_state
                        
                        # Apply changes from transaction
                        for key, value in transaction.changes.items():
                            if key == 'config':
                                state_copy['config'].update(value)
                            elif key == 'scripts':
                                state_copy['scripts'] = value
                        
                        self._apply_state_copy(extension_name)
                    
                    # Mark transaction as completed
                    transaction.completed = True
                    
                    # Update stability metrics
                    if transaction.transaction_type in [TransactionType.ENABLE, TransactionType.CONFIG_UPDATE]:
                        self._update_stability_metrics(extension_name, success=True)
                    
                    print(f"[ExtensionStateManager] Committed transaction {transaction_id} for {extension_name}")
                    
                    # Auto-save after successful commit
                    if len(self._dirty_extensions) >= 5:
                        self._save_state()
                        self._dirty_extensions.clear()
                    
                    return True
                else:
                    # Operation failed - rollback
                    return self.rollback_transaction(transaction_id, error_message)
                    
            except Exception as e:
                print(f"[ExtensionStateManager] Error committing transaction {transaction_id}: {e}")
                errors.report(f"Failed to commit extension transaction: {e}", exc_info=True)
                return self.rollback_transaction(transaction_id, str(e))
            finally:
                # Clean up active transaction
                if transaction_id in self._active_transactions:
                    self._active_transactions.remove(transaction_id)
    
    def rollback_transaction(self, transaction_id: str, error_message: Optional[str] = None) -> bool:
        """
        Rollback a transaction.
        
        Args:
            transaction_id: ID of the transaction to rollback
            error_message: Error message for the rollback
            
        Returns:
            True if rollback was successful
        """
        with self._lock:
            if transaction_id not in self._transactions:
                print(f"[ExtensionStateManager] Transaction {transaction_id} not found for rollback")
                return False
            
            transaction = self._transactions[transaction_id]
            extension_name = transaction.extension_name
            
            try:
                # Discard state copy (reverts to original state)
                self._discard_state_copy(extension_name)
                
                # Mark transaction as rolled back
                transaction.rolled_back = True
                transaction.completed = True
                transaction.error_message = error_message or "Transaction rolled back"
                
                # Update stability metrics for failure
                self._update_stability_metrics(extension_name, success=False, crashed=True)
                
                # If this was an enable transaction that failed, mark extension as failed
                if transaction.transaction_type == TransactionType.ENABLE:
                    self._extension_states[extension_name] = ExtensionState.FAILED
                    self._dirty_extensions.add(extension_name)
                
                print(f"[ExtensionStateManager] Rolled back transaction {transaction_id} for {extension_name}")
                
                # Notify user about the failure
                self._notify_extension_failure(extension_name, transaction.error_message)
                
                return True
                
            except Exception as e:
                print(f"[ExtensionStateManager] Error rolling back transaction {transaction_id}: {e}")
                errors.report(f"Failed to rollback extension transaction: {e}", exc_info=True)
                return False
            finally:
                # Clean up active transaction
                if transaction_id in self._active_transactions:
                    self._active_transactions.remove(transaction_id)
    
    def create_checkpoint(self, extension_name: str, operation_name: str = "") -> str:
        """
        Create a checkpoint for an extension before risky operations.
        
        Args:
            extension_name: Name of the extension
            operation_name: Description of the operation
            
        Returns:
            Checkpoint ID (transaction ID)
        """
        with self._lock:
            # Create a transaction for the checkpoint
            transaction_id = self.begin_transaction(
                extension_name,
                TransactionType.CHECKPOINT,
                {'operation': operation_name}
            )
            
            # Create checkpoint data
            checkpoint = ExtensionCheckpoint(
                extension_name=extension_name,
                timestamp=time.time(),
                state=self._extension_states.get(extension_name, ExtensionState.DISABLED),
                config_hash=self._calculate_config_hash(extension_name),
                config_snapshot=copy.deepcopy(self._extension_configs.get(extension_name, {})),
                enabled_scripts=copy.deepcopy(self._extension_scripts.get(extension_name, [])),
                transaction_id=transaction_id
            )
            
            # Save checkpoint to disk
            checkpoint_path = self._checkpoints_path / f"{extension_name}_{transaction_id}.json"
            try:
                with open(checkpoint_path, 'w', encoding='utf-8') as f:
                    json.dump(asdict(checkpoint), f, indent=2, ensure_ascii=False)
                
                self._checkpoints[transaction_id] = checkpoint
                print(f"[ExtensionStateManager] Created checkpoint for {extension_name}: {operation_name}")
                
            except Exception as e:
                print(f"[ExtensionStateManager] Error saving checkpoint: {e}")
                # Still return transaction ID for tracking
            
            return transaction_id
    
    def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Restore an extension to a previous checkpoint.
        
        Args:
            checkpoint_id: ID of the checkpoint to restore
            
        Returns:
            True if restore was successful
        """
        with self._lock:
            if checkpoint_id not in self._checkpoints:
                # Try to load from disk
                checkpoint_files = list(self._checkpoints_path.glob(f"*_{checkpoint_id}.json"))
                if not checkpoint_files:
                    print(f"[ExtensionStateManager] Checkpoint {checkpoint_id} not found")
                    return False
                
                try:
                    with open(checkpoint_files[0], 'r', encoding='utf-8') as f:
                        checkpoint_data = json.load(f)
                    
                    checkpoint = ExtensionCheckpoint(**checkpoint_data)
                    self._checkpoints[checkpoint_id] = checkpoint
                except Exception as e:
                    print(f"[ExtensionStateManager] Error loading checkpoint: {e}")
                    return False
            
            checkpoint = self._checkpoints[checkpoint_id]
            extension_name = checkpoint.extension_name
            
            try:
                # Begin a restore transaction
                restore_transaction_id = self.begin_transaction(
                    extension_name,
                    TransactionType.ROLLBACK,
                    {'checkpoint_id': checkpoint_id}
                )
                
                # Restore state from checkpoint
                self._extension_states[extension_name] = checkpoint.state
                self._extension_configs[extension_name] = copy.deepcopy(checkpoint.config_snapshot)
                self._extension_scripts[extension_name] = copy.deepcopy(checkpoint.enabled_scripts)
                
                # Apply the restored state
                self._apply_state_copy(extension_name)
                
                # Commit the restore transaction
                self.commit_transaction(restore_transaction_id, checkpoint.state, success=True)
                
                print(f"[ExtensionStateManager] Restored checkpoint for {extension_name}")
                return True
                
            except Exception as e:
                print(f"[ExtensionStateManager] Error restoring checkpoint: {e}")
                errors.report(f"Failed to restore extension checkpoint: {e}", exc_info=True)
                return False
    
    def _calculate_config_hash(self, extension_name: str) -> str:
        """Calculate hash of extension configuration."""
        config = self._extension_configs.get(extension_name, {})
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def _notify_extension_failure(self, extension_name: str, error_message: str) -> None:
        """Notify user about extension failure."""
        try:
            # Try to use the shared notification system
            if hasattr(shared, 'state') and hasattr(shared.state, 'job'):
                # Add to job messages if available
                msg = f"Extension '{extension_name}' failed and was disabled: {error_message}"
                if hasattr(shared.state, 'job_messages'):
                    shared.state.job_messages.append(msg)
            
            # Print to console
            print(f"[ExtensionStateManager] Extension failure: {extension_name} - {error_message}")
            
            # Try to show UI notification if available
            try:
                from modules import ui
                if hasattr(ui, 'notification'):
                    ui.notification.error(f"Extension disabled: {error_message}")
            except:
                pass
                
        except Exception as e:
            print(f"[ExtensionStateManager] Error sending notification: {e}")
    
    # Public API Methods
    
    def get_extension_state(self, extension_name: str) -> ExtensionState:
        """Get the current state of an extension."""
        with self._lock:
            return self._extension_states.get(extension_name, ExtensionState.DISABLED)
    
    def set_extension_state(self, extension_name: str, state: ExtensionState,
                           config_updates: Optional[Dict[str, Any]] = None) -> bool:
        """
        Set the state of an extension with transaction support.
        
        Args:
            extension_name: Name of the extension
            state: New state to set
            config_updates: Optional configuration updates
            
        Returns:
            True if operation was successful
        """
        with self._lock:
            # Begin transaction
            transaction_id = self.begin_transaction(
                extension_name,
                TransactionType.ENABLE if state == ExtensionState.ENABLED else TransactionType.DISABLE,
                {'config': config_updates} if config_updates else {}
            )
            
            try:
                # Update state in the copy
                if extension_name in self._state_copies:
                    self._state_copies[extension_name]['state'] = state
                
                # Commit transaction
                return self.commit_transaction(transaction_id, state, success=True)
                
            except Exception as e:
                print(f"[ExtensionStateManager] Error setting extension state: {e}")
                return self.rollback_transaction(transaction_id, str(e))
    
    def enable_extension(self, extension_name: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """Enable an extension."""
        return self.set_extension_state(extension_name, ExtensionState.ENABLED, config)
    
    def disable_extension(self, extension_name: str) -> bool:
        """Disable an extension."""
        return self.set_extension_state(extension_name, ExtensionState.DISABLED)
    
    def record_extension_crash(self, extension_name: str, error_message: str) -> None:
        """Record an extension crash and disable it."""
        with self._lock:
            # Update stability metrics
            self._update_stability_metrics(extension_name, success=False, crashed=True)
            
            # Set state to FAILED
            self._extension_states[extension_name] = ExtensionState.FAILED
            self._dirty_extensions.add(extension_name)
            
            # Notify user
            self._notify_extension_failure(extension_name, error_message)
            
            # Auto-save
            self._save_state()
    
    def get_extension_stability(self, extension_name: str) -> Optional[ExtensionStability]:
        """Get stability metrics for an extension."""
        with self._lock:
            return self._stability_metrics.get(extension_name)
    
    def get_all_extension_states(self) -> Dict[str, ExtensionState]:
        """Get states of all extensions."""
        with self._lock:
            return self._extension_states.copy()
    
    def get_marketplace_data(self) -> List[Dict[str, Any]]:
        """
        Get data for extension marketplace with stability ratings.
        
        Returns:
            List of extension data with stability information
        """
        with self._lock:
            marketplace_data = []
            
            # Get all available extensions
            try:
                available_extensions = extensions.list_extensions()
            except:
                available_extensions = []
            
            for ext in available_extensions:
                ext_name = ext.name if hasattr(ext, 'name') else str(ext)
                
                # Get stability metrics
                stability = self._stability_metrics.get(ext_name)
                
                # Get current state
                state = self._extension_states.get(ext_name, ExtensionState.DISABLED)
                
                # Get config
                config = self._extension_configs.get(ext_name, {})
                
                # Build marketplace entry
                entry = {
                    'name': ext_name,
                    'state': state.value,
                    'enabled': state == ExtensionState.ENABLED,
                    'stability_rating': stability.stability_rating if stability else "A",
                    'stability_score': stability.stability_score if stability else 100.0,
                    'crash_count': stability.crash_count if stability else 0,
                    'total_loads': stability.total_loads if stability else 0,
                    'success_rate': (stability.successful_loads / max(stability.total_loads, 1)) * 100 if stability else 100.0,
                    'last_crash': stability.last_crash_time if stability else 0,
                    'config': config,
                    'is_builtin': ext_name.startswith('extensions-builtin/') if hasattr(ext, 'path') else False
                }
                
                marketplace_data.append(entry)
            
            # Sort by stability score (best first)
            marketplace_data.sort(key=lambda x: x['stability_score'], reverse=True)
            
            return marketplace_data
    
    def cleanup_old_checkpoints(self, max_age_days: int = 7) -> int:
        """
        Clean up old checkpoints.
        
        Args:
            max_age_days: Maximum age of checkpoints in days
            
        Returns:
            Number of checkpoints cleaned up
        """
        with self._lock:
            cleaned_count = 0
            current_time = time.time()
            max_age_seconds = max_age_days * 24 * 60 * 60
            
            # Clean up in-memory checkpoints
            checkpoints_to_remove = []
            for checkpoint_id, checkpoint in self._checkpoints.items():
                if current_time - checkpoint.timestamp > max_age_seconds:
                    checkpoints_to_remove.append(checkpoint_id)
            
            for checkpoint_id in checkpoints_to_remove:
                del self._checkpoints[checkpoint_id]
                cleaned_count += 1
            
            # Clean up checkpoint files
            try:
                for checkpoint_file in self._checkpoints_path.glob("*.json"):
                    try:
                        # Extract timestamp from filename or file content
                        file_age = current_time - checkpoint_file.stat().st_mtime
                        if file_age > max_age_seconds:
                            checkpoint_file.unlink()
                            cleaned_count += 1
                    except:
                        continue
            except Exception as e:
                print(f"[ExtensionStateManager] Error cleaning checkpoint files: {e}")
            
            if cleaned_count > 0:
                print(f"[ExtensionStateManager] Cleaned up {cleaned_count} old checkpoints")
            
            return cleaned_count
    
    def force_save(self) -> None:
        """Force save all state to disk."""
        with self._lock:
            self._save_state()
            self._save_stability_metrics()
            self._dirty_extensions.clear()
    
    def get_transaction_history(self, extension_name: Optional[str] = None,
                               limit: int = 50) -> List[ExtensionTransaction]:
        """
        Get transaction history.
        
        Args:
            extension_name: Filter by extension name (optional)
            limit: Maximum number of transactions to return
            
        Returns:
            List of transactions
        """
        with self._lock:
            transactions = list(self._transactions.values())
            
            # Filter by extension name if provided
            if extension_name:
                transactions = [t for t in transactions if t.extension_name == extension_name]
            
            # Sort by timestamp (newest first)
            transactions.sort(key=lambda t: t.timestamp, reverse=True)
            
            return transactions[:limit]
    
    def get_active_transactions(self) -> List[str]:
        """Get list of active transaction IDs."""
        with self._lock:
            return list(self._active_transactions)


# Global instance
extension_state_manager = ExtensionStateManager()


# Integration functions for existing codebase

def initialize_extension_state_management() -> None:
    """Initialize extension state management system."""
    print("[ExtensionStateManager] Initializing extension state management...")
    
    # Clean up old checkpoints on startup
    extension_state_manager.cleanup_old_checkpoints()
    
    # Load all extension states
    all_states = extension_state_manager.get_all_extension_states()
    print(f"[ExtensionStateManager] Loaded states for {len(all_states)} extensions")


def wrap_extension_load(extension_name: str, load_func, *args, **kwargs):
    """
    Wrapper for extension loading with state management.
    
    Args:
        extension_name: Name of the extension
        load_func: Original load function
        *args, **kwargs: Arguments for load function
        
    Returns:
        Result of load function or None if failed
    """
    start_time = time.time()
    transaction_id = None
    
    try:
        # Check if extension is enabled
        state = extension_state_manager.get_extension_state(extension_name)
        if state == ExtensionState.DISABLED:
            print(f"[ExtensionStateManager] Extension {extension_name} is disabled, skipping load")
            return None
        
        if state == ExtensionState.FAILED:
            print(f"[ExtensionStateManager] Extension {extension_name} previously failed, skipping load")
            return None
        
        # Create checkpoint before loading
        transaction_id = extension_state_manager.create_checkpoint(
            extension_name, 
            f"Loading extension {extension_name}"
        )
        
        # Load the extension
        result = load_func(*args, **kwargs)
        
        # Calculate load time
        load_time = time.time() - start_time
        
        # Update stability metrics on success
        extension_state_manager._update_stability_metrics(
            extension_name, 
            success=True, 
            load_time=load_time
        )
        
        # Commit the transaction
        extension_state_manager.commit_transaction(
            transaction_id, 
            ExtensionState.ENABLED, 
            success=True
        )
        
        return result
        
    except Exception as e:
        print(f"[ExtensionStateManager] Error loading extension {extension_name}: {e}")
        traceback.print_exc()
        
        # Record crash
        extension_state_manager.record_extension_crash(
            extension_name, 
            f"Failed to load: {str(e)}"
        )
        
        # Rollback if we have a transaction
        if transaction_id:
            extension_state_manager.rollback_transaction(
                transaction_id, 
                f"Load failed: {str(e)}"
            )
        
        return None


def get_extension_stability_rating(extension_name: str) -> str:
    """Get stability rating for an extension."""
    stability = extension_state_manager.get_extension_stability(extension_name)
    if stability:
        return stability.stability_rating
    return "A"


def is_extension_safe_to_enable(extension_name: str) -> bool:
    """Check if an extension is safe to enable based on stability."""
    stability = extension_state_manager.get_extension_stability(extension_name)
    if not stability:
        return True  # No history, assume safe
    
    # Check if extension has crashed recently (within last hour)
    if stability.crash_count > 0:
        time_since_crash = time.time() - stability.last_crash_time
        if time_since_crash < 3600:  # 1 hour
            return False
    
    # Check stability rating
    return stability.stability_rating in ["A", "B", "C"]


# Monkey-patch extension loading if extensions module is available
try:
    import modules.extensions as ext_module
    
    # Store original functions
    _original_load_extensions = getattr(ext_module, 'load_extensions', None)
    _original_load_extension = getattr(ext_module, 'load_extension', None)
    
    def patched_load_extension(extension_path, *args, **kwargs):
        """Patched load_extension with state management."""
        extension_name = os.path.basename(extension_path)
        return wrap_extension_load(
            extension_name,
            _original_load_extension,
            extension_path,
            *args,
            **kwargs
        )
    
    def patched_load_extensions(*args, **kwargs):
        """Patched load_extensions with state management."""
        # Get list of extensions first
        extensions_list = ext_module.list_extensions()
        
        # Load each extension with state management
        for ext in extensions_list:
            ext_name = ext.name if hasattr(ext, 'name') else str(ext)
            ext_path = ext.path if hasattr(ext, 'path') else str(ext)
            
            # Skip if disabled
            state = extension_state_manager.get_extension_state(ext_name)
            if state == ExtensionState.DISABLED:
                print(f"[ExtensionStateManager] Skipping disabled extension: {ext_name}")
                continue
            
            # Load with state management
            wrap_extension_load(ext_name, _original_load_extension, ext_path)
    
    # Apply patches if functions exist
    if _original_load_extension:
        ext_module.load_extension = patched_load_extension
    if _original_load_extensions:
        ext_module.load_extensions = patched_load_extensions
    
    print("[ExtensionStateManager] Successfully patched extension loading functions")
    
except Exception as e:
    print(f"[ExtensionStateManager] Could not patch extension loading: {e}")


# Auto-initialize when module is imported
initialize_extension_state_management()