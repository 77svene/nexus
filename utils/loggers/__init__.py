# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Logging utils."""

import csv
import json
import os
import warnings
from pathlib import Path

import torch
from packaging.version import parse

from utils.general import LOGGER, colorstr, cv2
from utils.loggers.clearml.clearml_utils import ClearmlLogger
from utils.loggers.wandb.wandb_utils import WandbLogger
from utils.plots import plot_images, plot_labels, plot_results
from utils.torch_utils import de_parallel

LOGGERS = ("csv", "tb", "wandb", "clearml", "comet")  # *.csv, TensorBoard, Weights & Biases, ClearML
RANK = int(os.getenv("RANK", -1))

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:

    def SummaryWriter(*args):
        """Fall back to SummaryWriter returning None if TensorBoard is not installed."""
        return None  # None = SummaryWriter(str)


try:
    import wandb

    assert hasattr(wandb, "__version__")  # verify package import not local dir
    if parse(wandb.__version__) >= parse("0.12.2") and RANK in {0, -1}:
        try:
            wandb_login_success = wandb.login(timeout=30)
        except wandb.errors.UsageError:  # known non-TTY terminal issue
            wandb_login_success = False
        if not wandb_login_success:
            wandb = None
except (ImportError, AssertionError):
    wandb = None

try:
    import clearml

    assert hasattr(clearml, "__version__")  # verify package import not local dir
except (ImportError, AssertionError):
    clearml = None

try:
    if RANK in {0, -1}:
        import comet_ml

        assert hasattr(comet_ml, "__version__")  # verify package import not local dir
        from utils.loggers.comet import CometLogger

    else:
        comet_ml = None
except (ImportError, AssertionError):
    comet_ml = None


def _json_default(value):
    """Format `value` for JSON serialization (e.g. unwrap tensors).

    Fall back to strings.
    """
    if isinstance(value, torch.Tensor):
        try:
            value = value.item()
        except ValueError:  # "only one element tensors can be converted to Python scalars"
            pass
    return value if isinstance(value, float) else str(value)


class Loggers:
    """Initializes and manages various logging utilities for tracking YOLOv5 training and validation metrics."""

    def __init__(self, save_dir=None, weights=None, opt=None, hyp=None, logger=None, include=LOGGERS):
        """Initializes loggers for YOLOv5 training and validation metrics, paths, and options."""
        self.save_dir = save_dir
        self.weights = weights
        self.opt = opt
        self.hyp = hyp
        self.plots = not opt.noplots  # plot results
        self.logger = logger  # for printing results to console
        self.include = include
        self.keys = [
            "train/box_loss",
            "train/obj_loss",
            "train/cls_loss",  # train loss
            "metrics/precision",
            "metrics/recall",
            "metrics/mAP_0.5",
            "metrics/mAP_0.5:0.95",  # metrics
            "val/box_loss",
            "val/obj_loss",
            "val/cls_loss",  # val loss
            "x/lr0",
            "x/lr1",
            "x/lr2",
        ]  # params
        self.best_keys = ["best/epoch", "best/precision", "best/recall", "best/mAP_0.5", "best/mAP_0.5:0.95"]
        for k in LOGGERS:
            setattr(self, k, None)  # init empty logger dictionary
        self.csv = True  # always log to csv
        self.ndjson_console = "ndjson_console" in self.include  # log ndjson to console
        self.ndjson_file = "ndjson_file" in self.include  # log ndjson to file
        
        # Hyperparameter optimization logging
        self.hyp_opt_csv = None
        self.hyp_opt_tb = None
        self.hyp_opt_best = None
        self.hyp_opt_trials = []

        # Messages
        if not comet_ml:
            prefix = colorstr("Comet: ")
            s = f"{prefix}run 'pip install comet_ml' to automatically track and visualize YOLOv5 🚀 runs in Comet"
            self.logger.info(s)
        # TensorBoard
        s = self.save_dir
        if "tb" in self.include and not self.opt.evolve:
            prefix = colorstr("TensorBoard: ")
            self.logger.info(f"{prefix}Start with 'tensorboard --logdir {s.parent}', view at http://localhost:6006/")
            self.tb = SummaryWriter(str(s))

        # W&B
        if wandb and "wandb" in self.include:
            self.opt.hyp = self.hyp  # add hyperparameters
            self.wandb = WandbLogger(self.opt)
        else:
            self.wandb = None

        # ClearML
        if clearml and "clearml" in self.include:
            try:
                self.clearml = ClearmlLogger(self.opt, self.hyp)
            except Exception:
                self.clearml = None
                prefix = colorstr("ClearML: ")
                LOGGER.warning(
                    f"{prefix}WARNING ⚠️ ClearML is installed but not configured, skipping ClearML logging."
                    f" See https://docs.ultralytics.com/nexus/tutorials/clearml_logging_integration#readme"
                )

        else:
            self.clearml = None

        # Comet
        if comet_ml and "comet" in self.include:
            if isinstance(self.opt.resume, str) and self.opt.resume.startswith("comet://"):
                run_id = self.opt.resume.split("/")[-1]
                self.comet_logger = CometLogger(self.opt, self.hyp, run_id=run_id)

            else:
                self.comet_logger = CometLogger(self.opt, self.hyp)

        else:
            self.comet_logger = None

    @property
    def remote_dataset(self):
        """Fetches dataset dictionary from remote logging services like ClearML, Weights & Biases, or Comet ML."""
        data_dict = None
        if self.clearml:
            data_dict = self.clearml.data_dict
        if self.wandb:
            data_dict = self.wandb.data_dict
        if self.comet_logger:
            data_dict = self.comet_logger.data_dict

        return data_dict

    def on_train_start(self):
        """Initializes the training process for Comet ML logger if it's configured."""
        if self.comet_logger:
            self.comet_logger.on_train_start()

    def on_pretrain_routine_start(self):
        """Invokes pre-training routine start hook for Comet ML logger if available."""
        if self.comet_logger:
            self.comet_logger.on_pretrain_routine_start()

    def on_pretrain_routine_end(self, labels, names):
        """Callback that runs at the end of pre-training routine, logging label plots if enabled."""
        if self.plots:
            plot_labels(labels, names, self.save_dir)
            paths = self.save_dir.glob("*labels*.jpg")  # training labels
            if self.wandb:
                self.wandb.log({"Labels": [wandb.Image(str(x), caption=x.name) for x in paths]})
            if self.comet_logger:
                self.comet_logger.on_pretrain_routine_end(paths)
            if self.clearml:
                for path in paths:
                    self.clearml.log_plot(title=path.stem, plot_path=path)

    def on_train_batch_end(self, model, ni, imgs, targets, paths, vals):
        """Logs training batch end events, plots images, and updates external loggers with batch-end data."""
        log_dict = dict(zip(self.keys[:3], vals))
        # Callback runs on train batch end
        # ni: number integrated batches (since train start)
        if self.plots:
            if ni < 3:
                f = self.save_dir / f"train_batch{ni}.jpg"  # filename
                plot_images(imgs, targets, paths, f)
                if ni == 0 and self.tb and not self.opt.sync_bn:
                    log_tensorboard_graph(self.tb, model, imgsz=(self.opt.imgsz, self.opt.imgsz))
            if ni == 10 and (self.wandb or self.clearml):
                files = sorted(self.save_dir.glob("train*.jpg"))
                if self.wandb:
                    self.wandb.log({"Mosaics": [wandb.Image(str(f), caption=f.name) for f in files if f.exists()]})
                if self.clearml:
                    self.clearml.log_debug_samples(files, title="Mosaics")

        if self.comet_logger:
            self.comet_logger.on_train_batch_end(log_dict)

    def on_train_epoch_end(self, epoch, vals):
        """Logs training epoch end events, updates CSV, TensorBoard, and other loggers with epoch metrics."""
        if self.csv:
            self.write_csv(epoch, vals)
        if self.tb:
            for k, v in zip(self.keys, vals):
                self.tb.add_scalar(k, v, epoch)
        if self.wandb:
            self.wandb.log(dict(zip(self.keys, vals)))
        if self.clearml:
            self.clearml.log_metrics(dict(zip(self.keys, vals)), epoch)
        if self.comet_logger:
            self.comet_logger.on_train_epoch_end(dict(zip(self.keys, vals)), epoch)

    def on_val_end(self, vals):
        """Logs validation end events, updates CSV, TensorBoard, and other loggers with validation metrics."""
        if self.csv:
            self.write_csv(self.epoch, vals)
        if self.tb:
            for k, v in zip(self.keys[3:7], vals):
                self.tb.add_scalar(k, v, self.epoch)
        if self.wandb:
            self.wandb.log(dict(zip(self.keys[3:7], vals)))
        if self.clearml:
            self.clearml.log_metrics(dict(zip(self.keys[3:7], vals)), self.epoch)
        if self.comet_logger:
            self.comet_logger.on_val_end(dict(zip(self.keys[3:7], vals)))

    def on_fit_epoch_end(self, vals, epoch, best_fitness, fi):
        """Logs end of fit epoch, updates CSV, TensorBoard, and other loggers with fitness and best metrics."""
        if self.csv:
            self.write_csv(epoch, vals)
        if self.tb:
            for k, v in zip(self.keys, vals):
                self.tb.add_scalar(k, v, epoch)
            if best_fitness == fi:
                for k, v in zip(self.best_keys, [epoch] + vals[3:7]):
                    self.tb.add_scalar(k, v, epoch)
        if self.wandb:
            self.wandb.log(dict(zip(self.keys, vals)))
            if best_fitness == fi:
                self.wandb.log(dict(zip(self.best_keys, [epoch] + vals[3:7])))
        if self.clearml:
            self.clearml.log_metrics(dict(zip(self.keys, vals)), epoch)
            if best_fitness == fi:
                self.clearml.log_metrics(dict(zip(self.best_keys, [epoch] + vals[3:7])), epoch)
        if self.comet_logger:
            self.comet_logger.on_fit_epoch_end(dict(zip(self.keys, vals)), epoch)

    def on_model_save(self, last, epoch, final_epoch, best_fitness, fi):
        """Logs model save events, updates W&B and ClearML loggers if it's the final epoch."""
        if self.wandb:
            if (epoch + 1) % self.opt.save_period == 0 and not final_epoch:
                self.wandb.log_model(last.parent, self.opt, epoch, fi, is_best=best_fitness == fi)
        if self.clearml:
            if (epoch + 1) % self.opt.save_period == 0 and not final_epoch:
                self.clearml.log_model(last.parent, epoch, is_best=best_fitness == fi)

    def on_train_end(self, last, best, vals):
        """Logs end of training, saves model artifacts, and updates W&B, ClearML, and Comet loggers."""
        if self.csv:
            self.write_csv(self.epoch, vals)
        if self.tb:
            for k, v in zip(self.keys, vals):
                self.tb.add_scalar(k, v, self.epoch)
            for k, v in zip(self.best_keys, [self.epoch] + vals[3:7]):
                self.tb.add_scalar(k, v, self.epoch)
        if self.wandb:
            self.wandb.log(dict(zip(self.keys, vals)))
            self.wandb.log(dict(zip(self.best_keys, [self.epoch] + vals[3:7])))
            self.wandb.log_artifact(str(best if best.exists() else last), name="model", type="model")
        if self.clearml:
            self.clearml.log_metrics(dict(zip(self.keys, vals)), self.epoch)
            self.clearml.log_metrics(dict(zip(self.best_keys, [self.epoch] + vals[3:7])), self.epoch)
            self.clearml.log_model(best if best.exists() else last, epoch=self.epoch, is_best=True)
        if self.comet_logger:
            self.comet_logger.on_train_end(dict(zip(self.keys, vals)), self.epoch)

    def on_train_end_cleanup(self):
        """Cleans up resources and closes loggers at the end of training."""
        if self.wandb:
            self.wandb.finish_run()

    def write_csv(self, epoch, vals):
        """Writes training metrics to a CSV file for the given epoch."""
        if not hasattr(self, "csv_writer"):
            self.csv_writer = csv.writer(open(str(self.save_dir / "results.csv"), "a", newline=""))
            self.csv_writer.writerow(
                ["epoch", "train/box_loss", "train/obj_loss", "train/cls_loss", "metrics/precision",
                 "metrics/recall", "metrics/mAP_0.5", "metrics/mAP_0.5:0.95", "val/box_loss", "val/obj_loss",
                 "val/cls_loss", "x/lr0", "x/lr1", "x/lr2"]
            )
        self.csv_writer.writerow([epoch] + vals)

    def log_hyperparameters(self, hyp):
        """Logs hyperparameters to TensorBoard and other loggers."""
        if self.tb:
            for k, v in hyp.items():
                self.tb.add_scalar(f"hyperparameters/{k}", v, 0)
        if self.wandb:
            self.wandb.log({"hyperparameters": hyp})
        if self.clearml:
            self.clearml.log_parameters(hyp)

    def init_hyp_optimization(self, hyp_search_space):
        """Initialize hyperparameter optimization logging.
        
        Args:
            hyp_search_space (dict): Dictionary defining hyperparameter search space
        """
        # Create CSV for hyperparameter optimization results
        hyp_opt_dir = self.save_dir / "hyp_optimization"
        hyp_opt_dir.mkdir(exist_ok=True)
        
        self.hyp_opt_csv = hyp_opt_dir / "hyp_opt_results.csv"
        self.hyp_opt_best = {"best_fitness": -1, "best_hyp": None, "best_metrics": None}
        
        # Write CSV header if file doesn't exist
        if not self.hyp_opt_csv.exists():
            with open(self.hyp_opt_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                # Header: trial, hyperparameters (JSON), fitness, precision, recall, mAP_0.5, mAP_0.5:0.95
                writer.writerow(["trial", "hyperparameters", "fitness", "precision", "recall", 
                               "mAP_0.5", "mAP_0.5:0.95", "box_loss", "obj_loss", "cls_loss"])
        
        # Initialize TensorBoard writer for hyperparameter optimization
        if self.tb:
            self.hyp_opt_tb = SummaryWriter(str(hyp_opt_dir))
        
        # Log search space to TensorBoard
        if self.hyp_opt_tb:
            self.hyp_opt_tb.add_text("hyp_search_space", json.dumps(hyp_search_space, indent=2), 0)
        
        # Store search space
        self.hyp_search_space = hyp_search_space
        
        LOGGER.info(f"Hyperparameter optimization initialized. Results will be saved to: {hyp_opt_dir}")

    def log_hyp_opt_trial(self, trial_num, hyp, metrics, fitness):
        """Log results from a hyperparameter optimization trial.
        
        Args:
            trial_num (int): Trial number
            hyp (dict): Hyperparameters used in this trial
            metrics (dict): Validation metrics from the trial
            fitness (float): Fitness score (weighted combination of metrics)
        """
        if self.hyp_opt_csv is None:
            LOGGER.warning("Hyperparameter optimization not initialized. Call init_hyp_optimization first.")
            return
        
        # Extract metrics with defaults
        precision = metrics.get("metrics/precision", 0)
        recall = metrics.get("metrics/recall", 0)
        mAP_05 = metrics.get("metrics/mAP_0.5", 0)
        mAP_05_095 = metrics.get("metrics/mAP_0.5:0.95", 0)
        box_loss = metrics.get("val/box_loss", 0)
        obj_loss = metrics.get("val/obj_loss", 0)
        cls_loss = metrics.get("val/cls_loss", 0)
        
        # Log to CSV
        with open(self.hyp_opt_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                trial_num,
                json.dumps(hyp, default=_json_default),
                fitness,
                precision,
                recall,
                mAP_05,
                mAP_05_095,
                box_loss,
                obj_loss,
                cls_loss
            ])
        
        # Log to TensorBoard
        if self.hyp_opt_tb:
            # Log scalars for this trial
            self.hyp_opt_tb.add_scalar("hyp_optimization/fitness", fitness, trial_num)
            self.hyp_opt_tb.add_scalar("hyp_optimization/precision", precision, trial_num)
            self.hyp_opt_tb.add_scalar("hyp_optimization/recall", recall, trial_num)
            self.hyp_opt_tb.add_scalar("hyp_optimization/mAP_0.5", mAP_05, trial_num)
            self.hyp_opt_tb.add_scalar("hyp_optimization/mAP_0.5:0.95", mAP_05_095, trial_num)
            
            # Log hyperparameters as hparams
            try:
                # Convert all values to float for TensorBoard hparams
                hparams = {}
                for k, v in hyp.items():
                    if isinstance(v, (int, float)):
                        hparams[k] = float(v)
                    else:
                        hparams[k] = str(v)
                
                # Log hparams with metrics
                self.hyp_opt_tb.add_hparams(
                    hparams,
                    {
                        "hparam/fitness": fitness,
                        "hparam/precision": precision,
                        "hparam/recall": recall,
                        "hparam/mAP_0.5": mAP_05,
                        "hparam/mAP_0.5:0.95": mAP_05_095
                    },
                    run_name=f"trial_{trial_num}"
                )
            except Exception as e:
                LOGGER.warning(f"Failed to log hyperparameters to TensorBoard: {e}")
        
        # Track best trial
        if fitness > self.hyp_opt_best["best_fitness"]:
            self.hyp_opt_best = {
                "best_fitness": fitness,
                "best_hyp": hyp.copy(),
                "best_metrics": metrics.copy(),
                "trial_num": trial_num
            }
            
            # Save best hyperparameters to JSON
            best_hyp_file = self.save_dir / "hyp_optimization" / "best_hyperparameters.json"
            with open(best_hyp_file, 'w') as f:
                json.dump({
                    "trial": trial_num,
                    "fitness": fitness,
                    "hyperparameters": hyp,
                    "metrics": metrics
                }, f, indent=2, default=_json_default)
            
            LOGGER.info(f"New best hyperparameters found in trial {trial_num} with fitness: {fitness:.4f}")
        
        # Store trial for later analysis
        self.hyp_opt_trials.append({
            "trial": trial_num,
            "hyp": hyp.copy(),
            "metrics": metrics.copy(),
            "fitness": fitness
        })

    def finalize_hyp_optimization(self):
        """Finalize hyperparameter optimization and log summary."""
        if not self.hyp_opt_trials:
            LOGGER.warning("No hyperparameter optimization trials to finalize.")
            return
        
        # Log summary to TensorBoard
        if self.hyp_opt_tb:
            # Create summary table
            summary_text = "Hyperparameter Optimization Summary\n"
            summary_text += "=" * 50 + "\n"
            summary_text += f"Total trials: {len(self.hyp_opt_trials)}\n"
            summary_text += f"Best fitness: {self.hyp_opt_best['best_fitness']:.4f}\n"
            summary_text += f"Best trial: {self.hyp_opt_best.get('trial_num', 'N/A')}\n\n"
            
            summary_text += "Best Hyperparameters:\n"
            for k, v in self.hyp_opt_best['best_hyp'].items():
                summary_text += f"  {k}: {v}\n"
            
            summary_text += "\nBest Metrics:\n"
            for k, v in self.hyp_opt_best['best_metrics'].items():
                summary_text += f"  {k}: {v:.4f}\n"
            
            self.hyp_opt_tb.add_text("hyp_optimization/summary", summary_text, 0)
            
            # Log fitness progression
            trials = [t["trial"] for t in self.hyp_opt_trials]
            fitnesses = [t["fitness"] for t in self.hyp_opt_trials]
            for trial, fitness in zip(trials, fitnesses):
                self.hyp_opt_tb.add_scalar("hyp_optimization/progression", fitness, trial)
            
            # Close TensorBoard writer
            self.hyp_opt_tb.close()
        
        # Save complete results to JSON
        results_file = self.save_dir / "hyp_optimization" / "all_trials.json"
        with open(results_file, 'w') as f:
            json.dump(self.hyp_opt_trials, f, indent=2, default=_json_default)
        
        LOGGER.info(f"Hyperparameter optimization completed. {len(self.hyp_opt_trials)} trials run.")
        LOGGER.info(f"Best fitness: {self.hyp_opt_best['best_fitness']:.4f}")
        LOGGER.info(f"Best hyperparameters saved to: {self.save_dir / 'hyp_optimization' / 'best_hyperparameters.json'}")
        LOGGER.info(f"All trial results saved to: {results_file}")

    def suggest_hyperparameters(self, method="bayesian"):
        """Suggest next hyperparameters based on optimization history.
        
        Args:
            method (str): Optimization method ('bayesian', 'random', 'grid')
        
        Returns:
            dict: Suggested hyperparameters for next trial
        """
        if not hasattr(self, 'hyp_search_space'):
            LOGGER.warning("Hyperparameter search space not defined. Call init_hyp_optimization first.")
            return {}
        
        if not self.hyp_opt_trials:
            # First trial: sample from search space
            return self._sample_from_search_space()
        
        if method == "bayesian":
            return self._bayesian_suggestion()
        elif method == "random":
            return self._sample_from_search_space()
        elif method == "grid":
            return self._grid_suggestion()
        else:
            LOGGER.warning(f"Unknown optimization method: {method}. Using random sampling.")
            return self._sample_from_search_space()

    def _sample_from_search_space(self):
        """Sample hyperparameters from the defined search space."""
        import random
        import numpy as np
        
        suggested_hyp = {}
        for param, config in self.hyp_search_space.items():
            if config["type"] == "float":
                if config.get("log", False):
                    # Log-uniform sampling
                    log_low = np.log(config["low"])
                    log_high = np.log(config["high"])
                    suggested_hyp[param] = np.exp(random.uniform(log_low, log_high))
                else:
                    suggested_hyp[param] = random.uniform(config["low"], config["high"])
            elif config["type"] == "int":
                suggested_hyp[param] = random.randint(config["low"], config["high"])
            elif config["type"] == "categorical":
                suggested_hyp[param] = random.choice(config["choices"])
            else:
                LOGGER.warning(f"Unknown parameter type for {param}: {config['type']}")
        
        return suggested_hyp

    def _bayesian_suggestion(self):
        """Suggest hyperparameters using Bayesian optimization (simplified)."""
        # This is a simplified implementation. In production, you'd use a library like Optuna
        # or scikit-optimize for proper Bayesian optimization
        
        if len(self.hyp_opt_trials) < 3:
            # Not enough data for Bayesian optimization, use random sampling
            return self._sample_from_search_space()
        
        # Simple implementation: find best trial and perturb its hyperparameters
        best_trial = max(self.hyp_opt_trials, key=lambda x: x["fitness"])
        best_hyp = best_trial["hyp"]
        
        suggested_hyp = {}
        for param, config in self.hyp_search_space.items():
            if param in best_hyp:
                current_value = best_hyp[param]
                if config["type"] == "float":
                    # Perturb by ±10% of range
                    range_size = config["high"] - config["low"]
                    perturbation = random.uniform(-0.1 * range_size, 0.1 * range_size)
                    suggested_value = current_value + perturbation
                    # Clip to bounds
                    suggested_hyp[param] = max(config["low"], min(config["high"], suggested_value))
                elif config["type"] == "int":
                    # Perturb by ±1
                    perturbation = random.randint(-1, 1)
                    suggested_hyp[param] = max(config["low"], min(config["high"], current_value + perturbation))
                elif config["type"] == "categorical":
                    # Occasionally explore other options
                    if random.random() < 0.2:  # 20% chance to explore
                        suggested_hyp[param] = random.choice(config["choices"])
                    else:
                        suggested_hyp[param] = current_value
            else:
                # Parameter not in best trial, sample from scratch
                suggested_hyp[param] = self._sample_from_search_space()[param]
        
        return suggested_hyp

    def _grid_suggestion(self):
        """Suggest next hyperparameters in a grid search pattern."""
        # This would require storing the grid state, which is complex
        # For simplicity, we'll use random sampling
        LOGGER.info("Grid search not fully implemented. Using random sampling.")
        return self._sample_from_search_space()


def log_tensorboard_graph(tb, model, imgsz=(640, 640)):
    """Logs the model graph to TensorBoard for visualization."""
    try:
        p = next(model.parameters())  # for device, type
        x = torch.zeros((1, 3, *imgsz), device=p.device, dtype=p.dtype)  # input
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress jit trace warning
            tb.add_graph(torch.jit.trace(de_parallel(model), x, strict=False), [])
    except Exception as e:
        LOGGER.warning(f"WARNING ⚠️ TensorBoard graph visualization failure {e}")