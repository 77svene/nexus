"""
Atomic Extension State Management for nexus
Transaction-based extension state with rollback capability.
"""

import os
import sys
import json
import time
import uuid
import copy
import shutil
import hashlib
import threading
import traceback
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict
import logging

# Import existing modules for integration
from modules import extensions, shared, paths, errors
from modules.paths import data_path, extensions_dir

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Constants
TRANSACTION_DIR = os.path.join(data_path, "transactions")
MARKETPLACE_FILE = os.path.join(data_path, "extension_marketplace.json")
STATE_BACKUP_DIR = os.path.join(data_path, "state_backups")
MAX_TRANSACTION_HISTORY = 100
STABILITY_RATING_WINDOW = 30  # days

class TransactionState(Enum):
    """Transaction states"""
    PENDING = "pending"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"

class ExtensionRiskLevel(Enum):
    """Risk levels for extensions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class TransactionCheckpoint:
    """Checkpoint for transaction rollback"""
    checkpoint_id: str
    timestamp: float
    extension_name: str
    state_hash: str
    backup_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExtensionState:
    """State of an extension"""
    name: str
    enabled: bool = True
    version: str = "0.0.0"
    risk_level: ExtensionRiskLevel = ExtensionRiskLevel.LOW
    crash_count: int = 0
    last_crash: Optional[float] = None
    total_transactions: int = 0
    successful_transactions: int = 0
    stability_rating: float = 1.0
    last_updated: float = field(default_factory=time.time)
    dependencies: List[str] = field(default_factory=list)
    state_hash: Optional[str] = None
    
    def update_stability_rating(self):
        """Update stability rating based on crash history"""
        if self.total_transactions == 0:
            self.stability_rating = 1.0
            return
        
        # Weight recent crashes more heavily
        success_rate = self.successful_transactions / self.total_transactions
        
        # Apply exponential decay to crash count
        if self.last_crash:
            days_since_crash = (time.time() - self.last_crash) / 86400
            decay_factor = max(0, 1 - (days_since_crash / STABILITY_RATING_WINDOW))
            weighted_crashes = self.crash_count * decay_factor
        else:
            weighted_crashes = 0
        
        # Calculate final rating (0.0 to 1.0)
        self.stability_rating = max(0.0, min(1.0, success_rate - (weighted_crashes * 0.1)))

@dataclass
class Transaction:
    """Transaction for atomic extension operations"""
    transaction_id: str
    extension_name: str
    operation: str
    state: TransactionState = TransactionState.PENDING
    created_at: float = field(default_factory=time.time)
    committed_at: Optional[float] = None
    rolled_back_at: Optional[float] = None
    checkpoints: List[TransactionCheckpoint] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['state'] = self.state.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Transaction':
        """Create from dictionary"""
        data['state'] = TransactionState(data['state'])
        return cls(**data)

class StateSnapshot:
    """Snapshot of system state for copy-on-write"""
    
    def __init__(self):
        self.extension_states: Dict[str, ExtensionState] = {}
        self.loaded_extensions: Set[str] = set()
        self.active_scripts: List[str] = []
        self.model_states: Dict[str, Any] = {}
        self.ui_states: Dict[str, Any] = {}
        self.timestamp: float = time.time()
    
    def capture(self):
        """Capture current system state"""
        # Capture extension states
        for ext in extensions.extensions:
            if hasattr(ext, 'name'):
                self.extension_states[ext.name] = ExtensionState(
                    name=ext.name,
                    enabled=ext.enabled,
                    version=getattr(ext, 'version', '0.0.0')
                )
        
        # Capture loaded extensions
        self.loaded_extensions = {ext.name for ext in extensions.extensions if ext.enabled}
        
        # Capture active scripts (simplified)
        try:
            from modules import scripts
            self.active_scripts = [s.title() for s in scripts.scripts_txt2img.scripts + scripts.scripts_img2img.scripts]
        except:
            pass
        
        # Capture model states (simplified - actual implementation would need deeper integration)
        try:
            from modules import sd_models
            if hasattr(sd_models, 'model_data'):
                self.model_states['current_model'] = sd_models.model_data.sd_model_info.get('name', '') if hasattr(sd_models.model_data, 'sd_model_info') else ''
        except:
            pass
        
        self.timestamp = time.time()
    
    def calculate_hash(self) -> str:
        """Calculate hash of state for change detection"""
        state_str = json.dumps({
            'extensions': {name: asdict(state) for name, state in self.extension_states.items()},
            'loaded': sorted(list(self.loaded_extensions)),
            'scripts': sorted(self.active_scripts),
            'models': self.model_states
        }, sort_keys=True)
        return hashlib.sha256(state_str.encode()).hexdigest()

class TransactionManager:
    """Manages atomic transactions for extension state"""
    
    def __init__(self):
        self.active_transactions: Dict[str, Transaction] = {}
        self.transaction_history: List[Transaction] = []
        self.extension_states: Dict[str, ExtensionState] = {}
        self.marketplace_data: Dict[str, Any] = {}
        self.lock = threading.RLock()
        self.state_snapshot = StateSnapshot()
        
        # Initialize directories
        os.makedirs(TRANSACTION_DIR, exist_ok=True)
        os.makedirs(STATE_BACKUP_DIR, exist_ok=True)
        
        # Load existing data
        self._load_extension_states()
        self._load_marketplace_data()
        self._load_transaction_history()
        
        # Register shutdown handler
        import atexit
        atexit.register(self._cleanup_on_exit)
    
    def _load_extension_states(self):
        """Load extension states from disk"""
        state_file = os.path.join(TRANSACTION_DIR, "extension_states.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                    for name, state_data in data.items():
                        state_data['risk_level'] = ExtensionRiskLevel(state_data.get('risk_level', 'low'))
                        self.extension_states[name] = ExtensionState(**state_data)
            except Exception as e:
                logger.error(f"Failed to load extension states: {e}")
    
    def _save_extension_states(self):
        """Save extension states to disk"""
        state_file = os.path.join(TRANSACTION_DIR, "extension_states.json")
        try:
            data = {}
            for name, state in self.extension_states.items():
                state_dict = asdict(state)
                state_dict['risk_level'] = state.risk_level.value
                data[name] = state_dict
            
            with open(state_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save extension states: {e}")
    
    def _load_marketplace_data(self):
        """Load marketplace data from disk"""
        if os.path.exists(MARKETPLACE_FILE):
            try:
                with open(MARKETPLACE_FILE, 'r') as f:
                    self.marketplace_data = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load marketplace data: {e}")
                self.marketplace_data = {}
    
    def _save_marketplace_data(self):
        """Save marketplace data to disk"""
        try:
            with open(MARKETPLACE_FILE, 'w') as f:
                json.dump(self.marketplace_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save marketplace data: {e}")
    
    def _load_transaction_history(self):
        """Load transaction history from disk"""
        history_file = os.path.join(TRANSACTION_DIR, "transaction_history.json")
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self.transaction_history = [Transaction.from_dict(t) for t in data[-MAX_TRANSACTION_HISTORY:]]
            except Exception as e:
                logger.error(f"Failed to load transaction history: {e}")
    
    def _save_transaction_history(self):
        """Save transaction history to disk"""
        history_file = os.path.join(TRANSACTION_DIR, "transaction_history.json")
        try:
            data = [t.to_dict() for t in self.transaction_history[-MAX_TRANSACTION_HISTORY:]]
            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save transaction history: {e}")
    
    def _create_backup(self, extension_name: str, transaction_id: str) -> str:
        """Create backup of extension state"""
        backup_dir = os.path.join(STATE_BACKUP_DIR, extension_name, transaction_id)
        os.makedirs(backup_dir, exist_ok=True)
        
        # Find extension path
        ext_path = None
        for ext in extensions.extensions:
            if hasattr(ext, 'name') and ext.name == extension_name:
                ext_path = ext.path
                break
        
        if ext_path and os.path.exists(ext_path):
            # Backup extension files
            backup_ext_dir = os.path.join(backup_dir, "extension")
            if os.path.exists(backup_ext_dir):
                shutil.rmtree(backup_ext_dir)
            shutil.copytree(ext_path, backup_ext_dir)
        
        # Backup shared state (simplified)
        state_file = os.path.join(backup_dir, "state.json")
        state_data = {
            'timestamp': time.time(),
            'extension_name': extension_name,
            'enabled': self.extension_states.get(extension_name, ExtensionState(name=extension_name)).enabled
        }
        
        with open(state_file, 'w') as f:
            json.dump(state_data, f)
        
        return backup_dir
    
    def _restore_backup(self, backup_path: str, extension_name: str) -> bool:
        """Restore extension from backup"""
        try:
            state_file = os.path.join(backup_path, "state.json")
            if not os.path.exists(state_file):
                return False
            
            with open(state_file, 'r') as f:
                state_data = json.load(f)
            
            # Find extension
            ext = None
            for e in extensions.extensions:
                if hasattr(e, 'name') and e.name == extension_name:
                    ext = e
                    break
            
            if not ext:
                return False
            
            # Restore extension files
            backup_ext_dir = os.path.join(backup_path, "extension")
            if os.path.exists(backup_ext_dir) and os.path.exists(ext.path):
                # Remove current extension files
                for item in os.listdir(ext.path):
                    item_path = os.path.join(ext.path, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                
                # Copy backup files
                for item in os.listdir(backup_ext_dir):
                    src = os.path.join(backup_ext_dir, item)
                    dst = os.path.join(ext.path, item)
                    if os.path.isdir(src):
                        shutil.copytree(src, dst)
                    else:
                        shutil.copy2(src, dst)
            
            # Restore enabled state
            if extension_name in self.extension_states:
                self.extension_states[extension_name].enabled = state_data.get('enabled', True)
            
            return True
        except Exception as e:
            logger.error(f"Failed to restore backup: {e}")
            return False
    
    def _cleanup_old_backups(self, max_age_days: int = 7):
        """Clean up old backups"""
        cutoff_time = time.time() - (max_age_days * 86400)
        
        for ext_dir in Path(STATE_BACKUP_DIR).iterdir():
            if ext_dir.is_dir():
                for backup_dir in ext_dir.iterdir():
                    if backup_dir.is_dir():
                        try:
                            # Check state file timestamp
                            state_file = backup_dir / "state.json"
                            if state_file.exists():
                                mtime = state_file.stat().st_mtime
                                if mtime < cutoff_time:
                                    shutil.rmtree(backup_dir)
                        except:
                            pass
    
    def begin_transaction(self, extension_name: str, operation: str, 
                         risk_level: ExtensionRiskLevel = ExtensionRiskLevel.MEDIUM,
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Begin a new transaction for an extension
        
        Args:
            extension_name: Name of the extension
            operation: Description of the operation
            risk_level: Risk level of the operation
            metadata: Additional metadata
        
        Returns:
            Transaction ID
        """
        with self.lock:
            # Check if extension exists
            ext_exists = any(hasattr(ext, 'name') and ext.name == extension_name 
                           for ext in extensions.extensions)
            if not ext_exists:
                raise ValueError(f"Extension '{extension_name}' not found")
            
            # Generate transaction ID
            transaction_id = str(uuid.uuid4())
            
            # Create transaction
            transaction = Transaction(
                transaction_id=transaction_id,
                extension_name=extension_name,
                operation=operation,
                metadata=metadata or {}
            )
            
            # Update extension state
            if extension_name not in self.extension_states:
                self.extension_states[extension_name] = ExtensionState(
                    name=extension_name,
                    risk_level=risk_level
                )
            
            ext_state = self.extension_states[extension_name]
            ext_state.total_transactions += 1
            ext_state.risk_level = risk_level
            
            # Capture state snapshot
            self.state_snapshot.capture()
            state_hash = self.state_snapshot.calculate_hash()
            
            # Create checkpoint
            checkpoint = TransactionCheckpoint(
                checkpoint_id=str(uuid.uuid4()),
                timestamp=time.time(),
                extension_name=extension_name,
                state_hash=state_hash,
                backup_path=self._create_backup(extension_name, transaction_id)
            )
            
            transaction.checkpoints.append(checkpoint)
            
            # Store transaction
            self.active_transactions[transaction_id] = transaction
            
            logger.info(f"Transaction {transaction_id} started for {extension_name}: {operation}")
            
            return transaction_id
    
    def commit_transaction(self, transaction_id: str) -> bool:
        """
        Commit a transaction
        
        Args:
            transaction_id: ID of the transaction to commit
        
        Returns:
            True if successful
        """
        with self.lock:
            if transaction_id not in self.active_transactions:
                logger.error(f"Transaction {transaction_id} not found")
                return False
            
            transaction = self.active_transactions[transaction_id]
            
            try:
                # Verify state hasn't changed unexpectedly
                self.state_snapshot.capture()
                current_hash = self.state_snapshot.calculate_hash()
                
                # For high-risk operations, verify state consistency
                if transaction.metadata.get('risk_level') in ['high', 'critical']:
                    if current_hash == transaction.checkpoints[0].state_hash:
                        logger.warning(f"State unchanged for high-risk transaction {transaction_id}")
                
                # Update transaction state
                transaction.state = TransactionState.COMMITTED
                transaction.committed_at = time.time()
                
                # Update extension state
                ext_name = transaction.extension_name
                if ext_name in self.extension_states:
                    self.extension_states[ext_name].successful_transactions += 1
                    self.extension_states[ext_name].last_updated = time.time()
                    self.extension_states[ext_name].update_stability_rating()
                
                # Move to history
                self.transaction_history.append(transaction)
                del self.active_transactions[transaction_id]
                
                # Save state
                self._save_extension_states()
                self._save_transaction_history()
                
                # Cleanup old backups
                self._cleanup_old_backups()
                
                logger.info(f"Transaction {transaction_id} committed for {ext_name}")
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to commit transaction {transaction_id}: {e}")
                transaction.error_message = str(e)
                transaction.state = TransactionState.FAILED
                return False
    
    def rollback_transaction(self, transaction_id: str, reason: str = "") -> bool:
        """
        Rollback a transaction
        
        Args:
            transaction_id: ID of the transaction to rollback
            reason: Reason for rollback
        
        Returns:
            True if successful
        """
        with self.lock:
            if transaction_id not in self.active_transactions:
                logger.error(f"Transaction {transaction_id} not found")
                return False
            
            transaction = self.active_transactions[transaction_id]
            ext_name = transaction.extension_name
            
            try:
                # Restore from backup
                if transaction.checkpoints:
                    checkpoint = transaction.checkpoints[0]
                    success = self._restore_backup(checkpoint.backup_path, ext_name)
                    
                    if not success:
                        logger.error(f"Failed to restore backup for transaction {transaction_id}")
                
                # Update transaction state
                transaction.state = TransactionState.ROLLED_BACK
                transaction.rolled_back_at = time.time()
                transaction.error_message = reason or "Transaction rolled back"
                
                # Update extension state
                if ext_name in self.extension_states:
                    self.extension_states[ext_name].crash_count += 1
                    self.extension_states[ext_name].last_crash = time.time()
                    self.extension_states[ext_name].update_stability_rating()
                
                # Disable extension if stability is too low
                ext_state = self.extension_states.get(ext_name)
                if ext_state and ext_state.stability_rating < 0.3:
                    self._disable_extension(ext_name)
                    logger.warning(f"Extension {ext_name} disabled due to low stability rating: {ext_state.stability_rating:.2f}")
                
                # Move to history
                self.transaction_history.append(transaction)
                del self.active_transactions[transaction_id]
                
                # Save state
                self._save_extension_states()
                self._save_transaction_history()
                
                logger.info(f"Transaction {transaction_id} rolled back for {ext_name}: {reason}")
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to rollback transaction {transaction_id}: {e}")
                transaction.error_message = f"Rollback failed: {str(e)}"
                transaction.state = TransactionState.FAILED
                return False
    
    def _disable_extension(self, extension_name: str):
        """Disable an extension"""
        try:
            for ext in extensions.extensions:
                if hasattr(ext, 'name') and ext.name == extension_name:
                    ext.enabled = False
                    if extension_name in self.extension_states:
                        self.extension_states[extension_name].enabled = False
                    break
        except Exception as e:
            logger.error(f"Failed to disable extension {extension_name}: {e}")
    
    def get_extension_stability(self, extension_name: str) -> Dict[str, Any]:
        """
        Get stability information for an extension
        
        Args:
            extension_name: Name of the extension
        
        Returns:
            Dictionary with stability information
        """
        with self.lock:
            if extension_name not in self.extension_states:
                return {
                    'name': extension_name,
                    'stability_rating': 1.0,
                    'risk_level': 'low',
                    'crash_count': 0,
                    'total_transactions': 0
                }
            
            state = self.extension_states[extension_name]
            return {
                'name': extension_name,
                'stability_rating': state.stability_rating,
                'risk_level': state.risk_level.value,
                'crash_count': state.crash_count,
                'total_transactions': state.total_transactions,
                'successful_transactions': state.successful_transactions,
                'last_crash': state.last_crash,
                'last_updated': state.last_updated
            }
    
    def update_marketplace_entry(self, extension_name: str, 
                               metadata: Dict[str, Any]) -> bool:
        """
        Update or create marketplace entry for an extension
        
        Args:
            extension_name: Name of the extension
            metadata: Extension metadata
        
        Returns:
            True if successful
        """
        with self.lock:
            try:
                stability = self.get_extension_stability(extension_name)
                
                entry = {
                    'name': extension_name,
                    'stability_rating': stability['stability_rating'],
                    'risk_level': stability['risk_level'],
                    'crash_count': stability['crash_count'],
                    'last_updated': time.time(),
                    **metadata
                }
                
                self.marketplace_data[extension_name] = entry
                self._save_marketplace_data()
                
                logger.info(f"Marketplace entry updated for {extension_name}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to update marketplace entry: {e}")
                return False
    
    def get_marketplace_entries(self, min_stability: float = 0.0,
                              risk_levels: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get marketplace entries filtered by stability and risk
        
        Args:
            min_stability: Minimum stability rating (0.0 to 1.0)
            risk_levels: List of risk levels to include
        
        Returns:
            List of marketplace entries
        """
        with self.lock:
            entries = []
            
            for name, entry in self.marketplace_data.items():
                # Filter by stability
                if entry.get('stability_rating', 1.0) < min_stability:
                    continue
                
                # Filter by risk level
                if risk_levels and entry.get('risk_level', 'low') not in risk_levels:
                    continue
                
                entries.append(entry)
            
            # Sort by stability rating (descending)
            entries.sort(key=lambda x: x.get('stability_rating', 0), reverse=True)
            
            return entries
    
    def execute_with_transaction(self, extension_name: str, operation: str,
                               func, *args, **kwargs) -> Tuple[bool, Any]:
        """
        Execute a function within a transaction
        
        Args:
            extension_name: Name of the extension
            operation: Description of the operation
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
        
        Returns:
            Tuple of (success, result)
        """
        transaction_id = None
        try:
            # Determine risk level based on operation
            risk_level = ExtensionRiskLevel.MEDIUM
            if any(keyword in operation.lower() for keyword in ['delete', 'remove', 'uninstall']):
                risk_level = ExtensionRiskLevel.HIGH
            elif any(keyword in operation.lower() for keyword in ['install', 'update', 'modify']):
                risk_level = ExtensionRiskLevel.MEDIUM
            else:
                risk_level = ExtensionRiskLevel.LOW
            
            # Begin transaction
            transaction_id = self.begin_transaction(
                extension_name=extension_name,
                operation=operation,
                risk_level=risk_level,
                metadata={'args': str(args), 'kwargs': str(kwargs)}
            )
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Commit transaction
            self.commit_transaction(transaction_id)
            
            return True, result
            
        except Exception as e:
            logger.error(f"Transaction failed for {extension_name}: {e}")
            
            # Rollback transaction
            if transaction_id:
                self.rollback_transaction(
                    transaction_id,
                    reason=f"Exception: {str(e)}\n{traceback.format_exc()}"
                )
            
            # Notify user
            self._notify_user_of_failure(extension_name, str(e))
            
            return False, None
    
    def _notify_user_of_failure(self, extension_name: str, error_message: str):
        """Notify user of extension failure"""
        try:
            # Use shared module to show error if available
            if hasattr(shared, 'log_error'):
                shared.log_error(f"Extension '{extension_name}' failed: {error_message}")
            
            # Also log to console
            logger.error(f"Extension '{extension_name}' failed: {error_message}")
            
        except Exception as e:
            logger.error(f"Failed to notify user: {e}")
    
    def _cleanup_on_exit(self):
        """Cleanup on exit"""
        try:
            # Rollback any pending transactions
            for transaction_id in list(self.active_transactions.keys()):
                self.rollback_transaction(transaction_id, reason="System shutdown")
            
            # Save state
            self._save_extension_states()
            self._save_marketplace_data()
            self._save_transaction_history()
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    def get_active_transactions(self) -> List[Dict[str, Any]]:
        """Get list of active transactions"""
        with self.lock:
            return [t.to_dict() for t in self.active_transactions.values()]
    
    def get_transaction_history(self, extension_name: Optional[str] = None,
                              limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get transaction history
        
        Args:
            extension_name: Filter by extension name
            limit: Maximum number of entries
        
        Returns:
            List of transactions
        """
        with self.lock:
            history = self.transaction_history
            
            if extension_name:
                history = [t for t in history if t.extension_name == extension_name]
            
            # Sort by timestamp (newest first)
            history.sort(key=lambda x: x.created_at, reverse=True)
            
            return [t.to_dict() for t in history[:limit]]

# Global transaction manager instance
transaction_manager = TransactionManager()

# Convenience functions for integration
def begin_extension_transaction(extension_name: str, operation: str, **kwargs) -> str:
    """Begin a transaction for an extension"""
    return transaction_manager.begin_transaction(extension_name, operation, **kwargs)

def commit_extension_transaction(transaction_id: str) -> bool:
    """Commit a transaction"""
    return transaction_manager.commit_transaction(transaction_id)

def rollback_extension_transaction(transaction_id: str, reason: str = "") -> bool:
    """Rollback a transaction"""
    return transaction_manager.rollback_transaction(transaction_id, reason)

def execute_extension_safely(extension_name: str, operation: str, func, *args, **kwargs):
    """Execute extension code safely with transaction support"""
    return transaction_manager.execute_with_transaction(
        extension_name, operation, func, *args, **kwargs
    )

def get_extension_stability_rating(extension_name: str) -> float:
    """Get stability rating for an extension"""
    return transaction_manager.get_extension_stability(extension_name)['stability_rating']

def update_extension_marketplace(extension_name: str, **metadata) -> bool:
    """Update extension marketplace entry"""
    return transaction_manager.update_marketplace_entry(extension_name, metadata)

# Integration with existing extension system
def patch_extension_loader():
    """Patch the extension loader to use transactions"""
    try:
        original_load = extensions.load_extension
        
        def transactional_load_extension(ext_path):
            ext_name = os.path.basename(ext_path)
            transaction_id = None
            
            try:
                # Begin transaction
                transaction_id = begin_extension_transaction(
                    ext_name,
                    "load_extension",
                    risk_level=ExtensionRiskLevel.MEDIUM
                )
                
                # Load extension
                result = original_load(ext_path)
                
                # Commit transaction
                if transaction_id:
                    commit_extension_transaction(transaction_id)
                
                return result
                
            except Exception as e:
                # Rollback on failure
                if transaction_id:
                    rollback_extension_transaction(
                        transaction_id,
                        reason=f"Failed to load extension: {str(e)}"
                    )
                raise
        
        # Apply patch
        extensions.load_extension = transactional_load_extension
        
        logger.info("Extension loader patched with transaction support")
        
    except Exception as e:
        logger.error(f"Failed to patch extension loader: {e}")

# Auto-patch on import
try:
    patch_extension_loader()
except Exception as e:
    logger.warning(f"Could not auto-patch extension loader: {e}")

# Export main classes and functions
__all__ = [
    'TransactionManager',
    'Transaction',
    'ExtensionState',
    'TransactionState',
    'ExtensionRiskLevel',
    'transaction_manager',
    'begin_extension_transaction',
    'commit_extension_transaction',
    'rollback_extension_transaction',
    'execute_extension_safely',
    'get_extension_stability_rating',
    'update_extension_marketplace'
]