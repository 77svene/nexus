# utils/hpo.py
import os
import yaml
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from copy import deepcopy
import json

# Optional imports with fallback
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

from .general import increment_path, colorstr, check_yaml, check_file
from .torch_utils import torch_distributed_zero_first
from .callbacks import Callbacks
from .metrics import fitness
from .loss import ComputeLoss
from .. import train

LOGGER = logging.getLogger(__name__)

# Default hyperparameter search space for YOLOv5
DEFAULT_SEARCH_SPACE = {
    # Optimization
    'lr0': (1e-5, 1e-1, 'log'),  # Initial learning rate
    'lrf': (0.01, 1.0, 'log'),  # Final learning rate factor
    'momentum': (0.6, 0.98),  # SGD momentum
    'weight_decay': (0.0, 0.001),  # Optimizer weight decay
    'warmup_epochs': (0.0, 5.0),  # Warmup epochs
    'warmup_momentum': (0.0, 0.95),  # Warmup initial momentum
    'warmup_bias_lr': (0.0, 0.2),  # Warmup initial bias lr
    
    # Loss weights
    'box': (0.02, 0.2),  # Box loss weight
    'cls': (0.2, 4.0),  # Class loss weight
    'cls_pw': (0.5, 2.0),  # Class positive weight
    'obj': (0.2, 4.0),  # Object loss weight
    'obj_pw': (0.5, 2.0),  # Object positive weight
    'iou_t': (0.1, 0.7),  # IoU training threshold
    'anchor_t': (2.0, 8.0),  # Anchor-multiple threshold
    'fl_gamma': (0.0, 2.0),  # Focal loss gamma
    
    # Augmentation
    'hsv_h': (0.0, 0.1),  # HSV-Hue augmentation
    'hsv_s': (0.0, 0.9),  # HSV-Saturation augmentation
    'hsv_v': (0.0, 0.9),  # HSV-Value augmentation
    'degrees': (0.0, 45.0),  # Rotation degrees
    'translate': (0.0, 0.9),  # Translation
    'scale': (0.0, 0.9),  # Scaling
    'shear': (0.0, 10.0),  # Shear degrees
    'perspective': (0.0, 0.001),  # Perspective
    'flipud': (0.0, 1.0),  # Flip up-down
    'fliplr': (0.0, 1.0),  # Flip left-right
    'mosaic': (0.0, 1.0),  # Mosaic augmentation
    'mixup': (0.0, 1.0),  # Mixup augmentation
    'copy_paste': (0.0, 1.0),  # Copy-paste augmentation
}


class HPOSearchSpace:
    """Hyperparameter search space definition and sampling."""
    
    def __init__(self, search_space: Optional[Dict] = None):
        self.search_space = search_space or DEFAULT_SEARCH_SPACE
        
    def sample(self, trial: Any) -> Dict:
        """Sample hyperparameters from search space using Optuna trial."""
        params = {}
        for name, spec in self.search_space.items():
            if len(spec) == 3:
                low, high, scale = spec
                if scale == 'log':
                    params[name] = trial.suggest_float(name, low, high, log=True)
                else:
                    params[name] = trial.suggest_float(name, low, high)
            elif len(spec) == 2:
                low, high = spec
                if isinstance(low, int) and isinstance(high, int):
                    params[name] = trial.suggest_int(name, low, high)
                else:
                    params[name] = trial.suggest_float(name, low, high)
            elif isinstance(spec, list):
                params[name] = trial.suggest_categorical(name, spec)
        return params
    
    def to_yaml(self, path: str):
        """Save search space to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.search_space, f, default_flow_style=False)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'HPOSearchSpace':
        """Load search space from YAML file."""
        search_space = check_yaml(path)
        with open(search_space) as f:
            return cls(yaml.safe_load(f))


class HPOResults:
    """Store and analyze HPO results."""
    
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        self.best_trial = None
        self.best_fitness = -np.inf
        
        # Setup TensorBoard writer if available
        self.tb_writer = None
        if TENSORBOARD_AVAILABLE:
            tb_dir = self.save_dir / 'tensorboard'
            tb_dir.mkdir(exist_ok=True)
            self.tb_writer = SummaryWriter(tb_dir)
    
    def add_trial(self, trial_id: int, params: Dict, metrics: Dict, fitness_value: float):
        """Add trial results."""
        result = {
            'trial_id': trial_id,
            'params': params,
            'metrics': metrics,
            'fitness': fitness_value
        }
        self.results.append(result)
        
        # Update best trial
        if fitness_value > self.best_fitness:
            self.best_fitness = fitness_value
            self.best_trial = result
        
        # Log to TensorBoard
        if self.tb_writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(f'hpo/{key}', value, trial_id)
            self.tb_writer.add_scalar('hpo/fitness', fitness_value, trial_id)
            
            # Log hyperparameters
            for key, value in params.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(f'hpo/params/{key}', value, trial_id)
    
    def save(self):
        """Save results to CSV and JSON."""
        # Save detailed results to JSON
        with open(self.save_dir / 'hpo_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save summary to CSV
        summary_data = []
        for result in self.results:
            row = {'trial_id': result['trial_id'], 'fitness': result['fitness']}
            row.update(result['params'])
            row.update(result['metrics'])
            summary_data.append(row)
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            df.to_csv(self.save_dir / 'hpo_summary.csv', index=False)
        
        # Save best configuration
        if self.best_trial:
            best_config = self.best_trial['params'].copy()
            best_config['best_fitness'] = self.best_fitness
            with open(self.save_dir / 'best_hyperparameters.yaml', 'w') as f:
                yaml.dump(best_config, f, default_flow_style=False)
        
        LOGGER.info(f"Results saved to {self.save_dir}")
    
    def close(self):
        """Close TensorBoard writer."""
        if self.tb_writer:
            self.tb_writer.close()


class YOLOv5HPO:
    """Hyperparameter Optimization for YOLOv5."""
    
    def __init__(self, 
                 opt: Any,
                 search_space: Optional[Dict] = None,
                 n_trials: int = 100,
                 timeout: Optional[int] = None,
                 n_jobs: int = 1,
                 study_name: str = 'nexus_hpo',
                 direction: str = 'maximize',
                 pruner: Optional[Any] = None,
                 sampler: Optional[Any] = None):
        
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for HPO. Install with: pip install optuna"
            )
        
        self.opt = opt
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.study_name = study_name
        self.direction = direction
        
        # Setup search space
        self.search_space = HPOSearchSpace(search_space)
        
        # Setup results tracking
        self.save_dir = increment_path(Path(opt.project) / 'hpo', exist_ok=opt.exist_ok)
        self.results = HPOResults(self.save_dir)
        
        # Setup Optuna study
        self.sampler = sampler or TPESampler(seed=opt.seed)
        self.pruner = pruner or MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        
        # Create study with storage if specified
        storage = getattr(opt, 'hpo_storage', None)
        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=self.sampler,
            pruner=self.pruner,
            storage=storage,
            load_if_exists=True
        )
    
    def objective(self, trial: Any) -> float:
        """Objective function for Optuna optimization."""
        # Sample hyperparameters
        params = self.search_space.sample(trial)
        
        # Create a copy of opt with new hyperparameters
        trial_opt = deepcopy(self.opt)
        
        # Update hyperparameters
        for key, value in params.items():
            if hasattr(trial_opt, key):
                setattr(trial_opt, key, value)
        
        # Set trial-specific save directory
        trial_dir = self.save_dir / f'trial_{trial.number}'
        trial_dir.mkdir(exist_ok=True)
        trial_opt.project = str(trial_dir)
        trial_opt.name = 'train'
        
        # Disable some outputs for cleaner HPO logs
        trial_opt.verbose = False
        trial_opt.save_period = -1  # Don't save periodic checkpoints
        
        try:
            # Run training
            results = train.run(**vars(trial_opt))
            
            # Extract metrics
            if results is None:
                return -np.inf
            
            metrics = results.results_dict if hasattr(results, 'results_dict') else {}
            
            # Calculate fitness
            fitness_value = fitness(np.array([[
                metrics.get('metrics/mAP_0.5', 0),
                metrics.get('metrics/mAP_0.5:0.95', 0),
                metrics.get('metrics/precision', 0),
                metrics.get('metrics/recall', 0)
            ]]))
            
            # Log results
            self.results.add_trial(
                trial_id=trial.number,
                params=params,
                metrics=metrics,
                fitness_value=float(fitness_value)
            )
            
            # Report intermediate values for pruning
            for epoch, val in enumerate(metrics.get('val/loss', [])):
                trial.report(val, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return float(fitness_value)
            
        except Exception as e:
            LOGGER.warning(f"Trial {trial.number} failed: {e}")
            return -np.inf
    
    def run(self) -> Dict:
        """Run hyperparameter optimization."""
        LOGGER.info(f"Starting Hyperparameter Optimization with {self.n_trials} trials")
        LOGGER.info(f"Search space: {list(self.search_space.search_space.keys())}")
        
        # Save search space
        self.search_space.to_yaml(self.save_dir / 'search_space.yaml')
        
        # Run optimization
        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=True
        )
        
        # Save results
        self.results.save()
        
        # Get best trial
        best_trial = self.study.best_trial
        LOGGER.info(f"Best trial: {best_trial.number}")
        LOGGER.info(f"Best fitness: {best_trial.value}")
        LOGGER.info(f"Best parameters: {best_trial.params}")
        
        return {
            'best_trial': best_trial.number,
            'best_fitness': best_trial.value,
            'best_params': best_trial.params,
            'all_results': self.results.results
        }
    
    def suggest_from_previous(self, results_file: str) -> Dict:
        """Suggest hyperparameters based on previous optimization results."""
        if not Path(results_file).exists():
            LOGGER.warning(f"Results file not found: {results_file}")
            return {}
        
        with open(results_file) as f:
            results = json.load(f)
        
        if not results:
            return {}
        
        # Sort by fitness
        sorted_results = sorted(results, key=lambda x: x['fitness'], reverse=True)
        
        # Return top 5 configurations
        suggestions = []
        for i, result in enumerate(sorted_results[:5]):
            suggestions.append({
                'rank': i + 1,
                'fitness': result['fitness'],
                'params': result['params']
            })
        
        return suggestions


def run_hpo(opt, 
            search_space: Optional[Dict] = None,
            n_trials: int = 100,
            timeout: Optional[int] = None,
            n_jobs: int = 1) -> Dict:
    """
    Main function to run hyperparameter optimization.
    
    Args:
        opt: Parsed command line arguments from train.py
        search_space: Custom search space dictionary
        n_trials: Number of optimization trials
        timeout: Timeout in seconds
        n_jobs: Number of parallel jobs
    
    Returns:
        Dictionary with optimization results
    """
    hpo = YOLOv5HPO(
        opt=opt,
        search_space=search_space,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs
    )
    
    return hpo.run()


def create_search_space_template(path: str = 'search_space.yaml'):
    """Create a template search space YAML file."""
    search_space = HPOSearchSpace()
    search_space.to_yaml(path)
    LOGGER.info(f"Created search space template at {path}")
    return path


def load_best_hyperparameters(hpo_dir: str) -> Dict:
    """Load best hyperparameters from HPO results."""
    best_config_path = Path(hpo_dir) / 'best_hyperparameters.yaml'
    if not best_config_path.exists():
        raise FileNotFoundError(f"Best hyperparameters not found at {best_config_path}")
    
    with open(best_config_path) as f:
        return yaml.safe_load(f)


def apply_best_hyperparameters(opt, hpo_dir: str) -> Any:
    """Apply best hyperparameters from HPO to training options."""
    best_params = load_best_hyperparameters(hpo_dir)
    
    # Remove non-hyperparameter keys
    best_params.pop('best_fitness', None)
    
    # Update options
    for key, value in best_params.items():
        if hasattr(opt, key):
            setattr(opt, key, value)
            LOGGER.info(f"Applied hyperparameter: {key} = {value}")
    
    return opt


# Integration with existing training
def train_with_hpo(opt, n_trials: int = 50):
    """Run training with hyperparameter optimization."""
    from .. import train as train_module
    
    # First run HPO
    LOGGER.info("Starting Hyperparameter Optimization...")
    hpo_results = run_hpo(opt, n_trials=n_trials)
    
    # Apply best hyperparameters
    opt = apply_best_hyperparameters(opt, hpo_results['best_params'])
    
    # Run final training with best hyperparameters
    LOGGER.info("Starting final training with best hyperparameters...")
    opt.project = str(Path(opt.project) / 'final_training')
    opt.name = 'best_hyperparameters'
    
    # Import and run training
    train_module.run(**vars(opt))
    
    return hpo_results


if __name__ == '__main__':
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--hpo-trials', type=int, default=50, help='Number of HPO trials')
    parser.add_argument('--hpo-timeout', type=int, default=None, help='HPO timeout in seconds')
    parser.add_argument('--create-search-space', action='store_true', help='Create search space template')
    opt, _ = parser.parse_known_args()
    
    if opt.create_search_space:
        create_search_space_template()
    else:
        # This would be called from train.py with proper arguments
        print("Import and use run_hpo() or YOLOv5HPO class in your training script")