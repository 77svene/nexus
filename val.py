# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Validate a trained YOLOv5 detection model on a detection dataset.

Usage:
    $ python val.py --weights nexuss.pt --data coco128.yaml --img 640

Usage - formats:
    $ python val.py --weights nexuss.pt                 # PyTorch
                              nexuss.torchscript        # TorchScript
                              nexuss.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                              nexuss_openvino_model     # OpenVINO
                              nexuss.engine             # TensorRT
                              nexuss.mlpackage          # CoreML (macOS-only)
                              nexuss_saved_model        # TensorFlow SavedModel
                              nexuss.pb                 # TensorFlow GraphDef
                              nexuss.tflite             # TensorFlow Lite
                              nexuss_edgetpu.tflite     # TensorFlow Edge TPU
                              nexuss_paddle_model       # PaddlePaddle
"""

import argparse
import json
import os
import subprocess
import sys
import time
import threading
import psutil
import logging
from pathlib import Path
from functools import wraps
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    Profile,
    check_dataset,
    check_img_size,
    check_requirements,
    check_yaml,
    coco80_to_coco91_class,
    colorstr,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    xywh2xyxy,
    xyxy2xywh,
)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, smart_inference_mode


class EdgeResilienceMonitor:
    """Monitor system health and manage model resilience for edge deployments."""
    
    def __init__(self, 
                 temp_threshold: float = 80.0,
                 memory_threshold: float = 85.0,
                 watchdog_timeout: int = 30,
                 fallback_model_path: Optional[str] = None,
                 degradation_enabled: bool = True):
        self.temp_threshold = temp_threshold
        self.memory_threshold = memory_threshold
        self.watchdog_timeout = watchdog_timeout
        self.fallback_model_path = fallback_model_path
        self.degradation_enabled = degradation_enabled
        
        self.watchdog_timer = None
        self.last_health_check = time.time()
        self.health_status = {
            'temperature_ok': True,
            'memory_ok': True,
            'inference_ok': True,
            'consecutive_failures': 0
        }
        
        self.original_imgsz = None
        self.original_model = None
        self.current_model = None
        self.is_degraded = False
        
        # Setup logging for resilience monitoring
        self.logger = logging.getLogger('EdgeResilience')
        self.logger.setLevel(logging.INFO)
        
    def get_system_temperature(self) -> float:
        """Get current system temperature (platform-specific)."""
        try:
            # Linux thermal zone reading
            if os.path.exists('/sys/class/thermal/thermal_zone0/temp'):
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    temp = float(f.read().strip()) / 1000.0
                    return temp
            # macOS using powermetrics (requires sudo)
            elif sys.platform == 'darwin':
                try:
                    result = subprocess.run(
                        ['sudo', 'powermetrics', '--samplers', 'smc', '-i', '100', '-n', '1'],
                        capture_output=True, text=True, timeout=5
                    )
                    for line in result.stdout.split('\n'):
                        if 'CPU die temperature' in line:
                            temp = float(line.split(':')[-1].strip().replace('°C', ''))
                            return temp
                except:
                    pass
            # Windows using wmi (requires wmi module)
            elif sys.platform == 'win32':
                try:
                    import wmi
                    w = wmi.WMI(namespace="root\\OpenHardwareMonitor")
                    temperature_infos = w.Sensor()
                    for sensor in temperature_infos:
                        if sensor.SensorType == 'Temperature':
                            return float(sensor.Value)
                except ImportError:
                    pass
            return 0.0  # Default if cannot read
        except Exception as e:
            self.logger.warning(f"Failed to read temperature: {e}")
            return 0.0
    
    def get_memory_usage(self) -> float:
        """Get current memory usage percentage."""
        try:
            memory = psutil.virtual_memory()
            return memory.percent
        except Exception as e:
            self.logger.warning(f"Failed to read memory usage: {e}")
            return 0.0
    
    def check_system_health(self) -> Dict[str, Any]:
        """Perform comprehensive system health check."""
        temp = self.get_system_temperature()
        memory = self.get_memory_usage()
        
        health = {
            'temperature': temp,
            'memory_percent': memory,
            'temperature_ok': temp < self.temp_threshold,
            'memory_ok': memory < self.memory_threshold,
            'timestamp': time.time()
        }
        
        self.health_status.update(health)
        self.last_health_check = time.time()
        
        if not health['temperature_ok']:
            self.logger.warning(f"High temperature detected: {temp:.1f}°C (threshold: {self.temp_threshold}°C)")
        
        if not health['memory_ok']:
            self.logger.warning(f"High memory usage detected: {memory:.1f}% (threshold: {self.memory_threshold}%)")
        
        return health
    
    def start_watchdog(self, timeout: Optional[int] = None):
        """Start watchdog timer for inference monitoring."""
        if timeout is None:
            timeout = self.watchdog_timeout
            
        def watchdog_timeout_handler():
            self.logger.error("Watchdog timeout! Inference appears to be hung.")
            self.health_status['inference_ok'] = False
            
        self.watchdog_timer = threading.Timer(timeout, watchdog_timeout_handler)
        self.watchdog_timer.daemon = True
        self.watchdog_timer.start()
    
    def reset_watchdog(self):
        """Reset the watchdog timer (call during successful inference)."""
        if self.watchdog_timer:
            self.watchdog_timer.cancel()
            self.start_watchdog()
    
    def stop_watchdog(self):
        """Stop the watchdog timer."""
        if self.watchdog_timer:
            self.watchdog_timer.cancel()
            self.watchdog_timer = None
    
    def should_degrade(self) -> bool:
        """Check if system conditions warrant performance degradation."""
        if not self.degradation_enabled:
            return False
            
        health = self.check_system_health()
        
        # Check for consecutive failures
        if not health['temperature_ok'] or not health['memory_ok']:
            self.health_status['consecutive_failures'] += 1
        else:
            self.health_status['consecutive_failures'] = 0
            
        # Degrade if 3 consecutive failures or critical conditions
        return (self.health_status['consecutive_failures'] >= 3 or 
                health['temperature'] > self.temp_threshold * 1.1 or
                health['memory_percent'] > self.memory_threshold * 1.1)
    
    def apply_degradation(self, model, imgsz: int) -> Tuple[Any, int]:
        """Apply graceful degradation strategies."""
        if not self.should_degrade():
            return model, imgsz
            
        self.logger.info("Applying graceful degradation strategies...")
        self.is_degraded = True
        
        # Strategy 1: Reduce input size
        if imgsz > 320:
            new_imgsz = max(320, imgsz // 2)
            self.logger.info(f"Degrading: Reducing input size from {imgsz} to {new_imgsz}")
            return model, new_imgsz
        
        # Strategy 2: Switch to fallback model if available
        if self.fallback_model_path and os.path.exists(self.fallback_model_path):
            self.logger.info(f"Degrading: Switching to fallback model: {self.fallback_model_path}")
            try:
                device = next(model.parameters()).device
                fallback_model = DetectMultiBackend(self.fallback_model_path, device=device)
                return fallback_model, imgsz
            except Exception as e:
                self.logger.error(f"Failed to load fallback model: {e}")
        
        # Strategy 3: Reduce precision (if using CUDA)
        if hasattr(model, 'half') and next(model.parameters()).is_cuda:
            self.logger.info("Degrading: Converting model to FP16")
            model = model.half()
            
        return model, imgsz
    
    def reset_degradation(self):
        """Reset degradation state."""
        self.is_degraded = False
        self.health_status['consecutive_failures'] = 0


def health_check_decorator(monitor: EdgeResilienceMonitor):
    """Decorator to wrap inference functions with health monitoring."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Pre-inference health check
            health = monitor.check_system_health()
            if not health['temperature_ok'] or not health['memory_ok']:
                monitor.logger.warning("System health check failed before inference")
            
            # Start watchdog
            monitor.start_watchdog()
            
            try:
                # Execute inference
                result = func(*args, **kwargs)
                
                # Reset watchdog on success
                monitor.reset_watchdog()
                monitor.health_status['inference_ok'] = True
                monitor.health_status['consecutive_failures'] = 0
                
                return result
                
            except Exception as e:
                # Log inference failure
                monitor.logger.error(f"Inference failed: {e}")
                monitor.health_status['inference_ok'] = False
                monitor.health_status['consecutive_failures'] += 1
                raise
            finally:
                # Always stop watchdog
                monitor.stop_watchdog()
                
        return wrapper
    return decorator


def save_one_txt(predn, save_conf, shape, file):
    """Saves one detection result to a txt file in normalized xywh format, optionally including confidence.

    Args:
        predn (torch.Tensor): Predicted bounding boxes and associated confidence scores and classes in xyxy format,
            tensor of shape (N, 6) where N is the number of detections.
        save_conf (bool): If True, saves the confidence scores along with the bounding box coordinates.
        shape (tuple): Shape of the original image as (height, width).
        file (str | Path): File path where the result will be saved.

    Returns:
        None

    Examples:
        ```python
        predn = torch.tensor([[10, 20, 30, 40, 0.9, 1]])  # example prediction
        save_one_txt(predn, save_conf=True, shape=(640, 480), file="output.txt")
        ```

    Notes:
        The xyxy bounding box format represents the coordinates (xmin, ymin, xmax, ymax).
        The xywh format represents the coordinates (center_x, center_y, width, height) and is normalized by the width and
        height of the image.
    """
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, "a") as f:
            f.write(("%g " * len(line)).rstrip() % line + "\n")


def save_one_json(predn, jdict, path, class_map):
    """Saves a single JSON detection result, including image ID, category ID, bounding box, and confidence score.

    Args:
        predn (torch.Tensor): Predicted detections in xyxy format with shape (n, 6) where n is the number of detections.
            The tensor should contain [x_min, y_min, x_max, y_max, confidence, class_id] for each detection.
        jdict (list[dict]): List to collect JSON formatted detection results.
        path (pathlib.Path): Path object of the image file, used to extract image_id.
        class_map (dict[int, int]): Mapping from model class indices to dataset-specific category IDs.

    Returns:
        None: Appends detection results as dictionaries to `jdict` list in-place.

    Examples:
        ```python
        predn = torch.tensor([[100, 50, 200, 150, 0.9, 0], [50, 30, 100, 80, 0.8, 1]])
        jdict = []
        path = Path("42.jpg")
        class_map = {0: 18, 1: 19}
        save_one_json(predn, jdict, path, class_map)
        ```
        This will append to `jdict`:
        ```
        [
            {'image_id': 42, 'category_id': 18, 'bbox': [125.0, 75.0, 100.0, 100.0], 'score': 0.9},
            {'image_id': 42, 'category_id': 19, 'bbox': [75.0, 55.0, 50.0, 50.0], 'score': 0.8}
        ]
        ```

    Notes:
        The `bbox` values are formatted as [x, y, width, height], where x and y represent the top-left corner of the box.
    """
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append(
            {
                "image_id": image_id,
                "category_id": class_map[int(p[5])],
                "bbox": [round(x, 3) for x in b],
                "score": round(p[4], 5),
            }
        )


def process_batch(detections, labels, iouv):
    """Return a correct prediction matrix given detections and labels at various IoU thresholds.

    Args:
        detections (np.ndarray): Array of shape (N, 6) where each row corresponds to a detection with format [x1, y1,
            x2, y2, conf, class].
        labels (np.ndarray): Array of shape (M, 5) where each row corresponds to a ground truth label with format
            [class, x1, y1, x2, y2].
        iouv (np.ndarray): Array of IoU thresholds to evaluate at.

    Returns:
        correct (np.ndarray): A binary array of shape (N, len(iouv)) indicating whether each detection is a true
            positive for each IoU threshold. There are 10 IoU levels used in the evaluation.

    Examples:
        ```python
        detections = np.array([[50, 50, 200, 200, 0.9, 1], [30, 30, 150, 150, 0.7, 0]])
        labels = np.array([[1, 50, 50, 200, 200]])
        iouv = np.linspace(0.5, 0.95, 10)
        correct = process_batch(detections, labels, iouv)
        ```

    Notes:
        - This function is used as part of the evaluation pipeline for object detection models.
        - IoU (Intersection over Union) is a common evaluation metric for object detection performance.
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


@smart_inference_mode()
def run(
    data,
    weights=None,  # model.pt path(s)
    batch_size=32,  # batch size
    imgsz=640,  # inference size (pixels)
    conf_thres=0.001,  # confidence threshold
    iou_thres=0.6,  # NMS IoU threshold
    max_det=300,  # maximum detections per image
    task="val",  # train, val, test, speed or study
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    workers=8,  # dataloader workers (per RANK in DDP mode)
    single_cls=False,  # treat as single-class dataset
    augment=False,  # augmented inference
    verbose=False,  # verbose output
    save_txt=False,  # save results to *.txt
    save_hybrid=False,  # save label+prediction hybrid results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_json=False,  # save a COCO-JSON results file
    project=ROOT / "runs/val",  # save to project/name
    name="exp",  # save to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    half=True,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    dataloader=None,
    save_dir=Path(""),
    plots=True,
    callbacks=Callbacks(),
    compute_loss=None,
    # New resilience parameters
    resilience_enabled=True,
    temp_threshold=80.0,
    memory_threshold=85.0,
    watchdog_timeout=30,
    fallback_weights=None,
    degradation_enabled=True,
):
    # Initialize resilience monitor
    monitor = None
    if resilience_enabled:
        monitor = EdgeResilienceMonitor(
            temp_threshold=temp_threshold,
            memory_threshold=memory_threshold,
            watchdog_timeout=watchdog_timeout,
            fallback_model_path=fallback_weights,
            degradation_enabled=degradation_enabled
        )
        LOGGER.info(f"Edge resilience monitoring enabled (temp: {temp_threshold}°C, memory: {memory_threshold}%)")
    
    # Load model with resilience wrapper
    model = None
    original_imgsz = imgsz
    
    def load_model_with_resilience(weights_path, current_imgsz):
        nonlocal model
        model = DetectMultiBackend(weights_path, device=device, dnn=dnn, data=data, fp16=half)
        stride, pt = model.stride, model.pt
        imgsz = check_img_size(current_imgsz, s=stride)  # check image size
        
        # Apply initial degradation if needed
        if monitor and monitor.should_degrade():
            model, imgsz = monitor.apply_degradation(model, imgsz)
            
        return model, imgsz, stride, pt
    
    # Load initial model
    model, imgsz, stride, pt = load_model_with_resilience(weights, imgsz)
    
    # Define inference function with health monitoring
    @health_check_decorator(monitor) if monitor else lambda f: f
    def model_inference(im):
        """Run model inference with health monitoring."""
        return model(im, augment=augment, visualize=False)
    
    # Half precision
    half &= pt and device.type != "cpu"  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()
    
    # Configure
    model.eval()
    cuda = device.type != "cpu"
    
    # Data
    data = check_dataset(data)  # check
    
    # Dataloader
    if pt and not single_cls:
        model.names = data["names"]
    
    # Initialize
    nc = int(data["nc"])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    
    # Dataloader
    if not training:
        if pt and not single_cls:
            model.names = data["names"]
        pad = 0.5 if task == "speed" else 0.0
        task = task if task in ("train", "val", "test") else "val"  # path to train/val/test images
        dataloader = create_dataloader(
            data[task],
            imgsz,
            batch_size,
            stride,
            single_cls,
            pad=pad,
            rect=pt,
            workers=workers,
            prefix=colorstr(f"{task}: "),
        )[0]
    
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = model.names if hasattr(model, "names") else model.module.names  # get class names
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    class_map = coco80_to_coco91_class() if data.get("is_coco") else list(range(1000))
    s = ("%22s" + "%11s" * 6) % ("Images", "Instances", "P", "R", "mAP50", "mAP50-95", "Status")
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dt = Profile(), Profile(), Profile()  # profiling times
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    callbacks.run("on_val_start")
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar
    
    resilience_status = "Normal"
    consecutive_failures = 0
    max_consecutive_failures = 3
    
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        # System health check before each batch
        if monitor:
            health = monitor.check_system_health()
            if not health['temperature_ok'] or not health['memory_ok']:
                resilience_status = "Warning"
                consecutive_failures += 1
                LOGGER.warning(f"System health warning - Temp: {health['temperature']:.1f}°C, Memory: {health['memory_percent']:.1f}%")
                
                # Apply degradation if needed
                if consecutive_failures >= max_consecutive_failures:
                    model, imgsz = monitor.apply_degradation(model, imgsz)
                    resilience_status = "Degraded"
                    consecutive_failures = 0
            else:
                consecutive_failures = 0
                if monitor.is_degraded:
                    resilience_status = "Degraded"
                else:
                    resilience_status = "Normal"
        
        callbacks.run("on_val_batch_start")
        t1 = time.time()
        with dt[0]:
            if cuda:
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            nb, _, height, width = im.shape  # batch size, channels, height, width
        
        # Inference with resilience monitoring
        with dt[1]:
            try:
                preds = model_inference(im)  # Use monitored inference
                consecutive_failures = 0  # Reset on success
            except Exception as e:
                LOGGER.error(f"Inference failed: {e}")
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures and fallback_weights:
                    LOGGER.warning(f"Switching to fallback model: {fallback_weights}")
                    model, imgsz, stride, pt = load_model_with_resilience(fallback_weights, original_imgsz)
                    resilience_status = "Fallback"
                    consecutive_failures = 0
                    # Retry inference with fallback model
                    preds = model_inference(im)
                else:
                    raise
        
        # Loss
        with dt[2]:
            if compute_loss:
                loss += compute_loss(preds, targets)[1]  # box, obj, cls
            
        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        preds = non_max_suppression(
            preds, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls, max_det=max_det
        )
        
        # Metrics
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1
            
            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue
            
            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred
            
            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)
            
            # Save
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / "labels" / f"{path.stem}.txt")
            if save_json:
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            callbacks.run("on_val_image_end", pred, predn, path, names, im[si])
        
        # Plot
        if plots and batch_i < 3:
            plot_images(im, targets, paths, save_dir / f"val_batch{batch_i}_labels.jpg", names)  # labels
            plot_images(im, output_to_target(preds), paths, save_dir / f"val_batch{batch_i}_pred.jpg", names)  # pred
        
        callbacks.run("on_val_batch_end")
        
        # Update progress bar with resilience status
        pbar.set_description(f"{s} [{resilience_status}]")
    
    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class
    
    # Print results
    pf = "%22s" + "%11i" * 2 + "%11.3g" * 4  # print format
    LOGGER.info(pf % ("all", seen, nt.sum(), mp, mr, map50, map))
    
    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
    
    # Print speeds
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}" % t)
    
    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run("on_val_end", nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix)
    
    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ""  # weights
        anno_json = str(Path(data.get("path", "../coco")) / "annotations/instances_val2017.json")  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        LOGGER.info(f"\nEvaluating pycocotools mAP... saving {pred_json}...")
        with open(pred_json, "w") as f:
            json.dump(jdict, f)
        
        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements("pycocotools>=2.0.6")
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            
            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, "bbox")
            if data.get("is_coco"):
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            LOGGER.info(f"pycocotools unable to run: {e}")
    
    # Return results
    model.float()  # for half validation
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    
    # Final resilience report
    if monitor:
        LOGGER.info(f"\nResilience Report:")
        LOGGER.info(f"  Final Status: {resilience_status}")
        LOGGER.info(f"  Temperature: {monitor.get_system_temperature():.1f}°C")
        LOGGER.info(f"  Memory Usage: {monitor.get_memory_usage():.1f}%")
        if monitor.is_degraded:
            LOGGER.info("  Model ran in degraded mode")
    
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "nexuss.pt", help="model path(s)")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="maximum detections per image")
    parser.add_argument("--task", default="val", help="train, val, test, speed or study")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--single-cls", action="store_true", help="treat as single-class dataset")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--verbose", action="store_true", help="report mAP by class")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-hybrid", action="store_true", help="save label+prediction hybrid results to *.txt")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-json", action="store_true", help="save a COCO-JSON results file")
    parser.add_argument("--project", default=ROOT / "runs/val", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    # New resilience arguments
    parser.add_argument("--resilience", action="store_true", default=True, help="enable edge resilience monitoring")
    parser.add_argument("--no-resilience", action="store_true", help="disable edge resilience monitoring")
    parser.add_argument("--temp-threshold", type=float, default=80.0, help="temperature threshold in °C")
    parser.add_argument("--memory-threshold", type=float, default=85.0, help="memory usage threshold in %")
    parser.add_argument("--watchdog-timeout", type=int, default=30, help="watchdog timeout in seconds")
    parser.add_argument("--fallback-weights", type=str, default=None, help="fallback model weights path")
    parser.add_argument("--no-degradation", action="store_true", help="disable graceful degradation")
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith("coco.yaml")
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    
    # Handle resilience flags
    if opt.no_resilience:
        opt.resilience = False
    
    if opt.no_degradation:
        opt.degradation_enabled = False
    else:
        opt.degradation_enabled = True
    
    if opt.task in ("train", "val", "test"):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/nexus/issues/1466
            LOGGER.info(f"WARNING: confidence threshold {opt.conf_thres} > 0.001 produces incorrect results")
        run(**vars(opt))
    
    elif opt.task == "speed":
        # Add speed test with resilience monitoring
        LOGGER.info("Running speed test with resilience monitoring...")
        for w in opt.weights:
            run(
                data=opt.data,
                weights=w,
                batch_size=opt.batch_size,
                imgsz=opt.imgsz,
                conf_thres=opt.conf_thres,
                iou_thres=opt.iou_thres,
                max_det=opt.max_det,
                task="speed",
                device=opt.device,
                workers=opt.workers,
                single_cls=opt.single_cls,
                augment=opt.augment,
                verbose=False,
                save_txt=False,
                save_hybrid=False,
                save_conf=False,
                save_json=False,
                project=opt.project,
                name=opt.name,
                exist_ok=opt.exist_ok,
                half=opt.half,
                dnn=opt.dnn,
                resilience_enabled=opt.resilience,
                temp_threshold=opt.temp_threshold,
                memory_threshold=opt.memory_threshold,
                watchdog_timeout=opt.watchdog_timeout,
                fallback_weights=opt.fallback_weights,
                degradation_enabled=opt.degradation_enabled,
            )
    
    elif opt.task == "study":
        # Add study with resilience monitoring
        LOGGER.info("Running study with resilience monitoring...")
        x = list(range(320, 800, 64))  # x axis (image sizes)
        for w in opt.weights:
            f = f"study_{Path(opt.data).stem}_{Path(w).stem}.txt"  # filename to save to
            y = []  # y axis
            for i in x:
                LOGGER.info(f"\nRunning {f} point {i}...")
                r, _, t = run(
                    data=opt.data,
                    weights=w,
                    batch_size=max(opt.batch_size, 64),
                    imgsz=i,
                    conf_thres=opt.conf_thres,
                    iou_thres=opt.iou_thres,
                    max_det=opt.max_det,
                    task="val",
                    device=opt.device,
                    workers=opt.workers,
                    single_cls=opt.single_cls,
                    augment=opt.augment,
                    verbose=False,
                    save_txt=False,
                    save_hybrid=False,
                    save_conf=False,
                    save_json=False,
                    project=opt.project,
                    name=opt.name,
                    exist_ok=opt.exist_ok,
                    half=opt.half,
                    dnn=opt.dnn,
                    resilience_enabled=opt.resilience,
                    temp_threshold=opt.temp_threshold,
                    memory_threshold=opt.memory_threshold,
                    watchdog_timeout=opt.watchdog_timeout,
                    fallback_weights=opt.fallback_weights,
                    degradation_enabled=opt.degradation_enabled,
                )
                y.append(r + t)  # results and times
            # Save study results
            np.savetxt(f, y, fmt="%10.4g")  # save
        os.system("zip -r study.zip study_*.txt")
        plot_val_study(x=x)  # plot


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)