# utils/scaling.py
# Advanced Data Augmentation & Scaling for YOLOv5

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import yaml

from utils.augmentations import letterbox, random_perspective, augment_hsv
from utils.general import colorstr, check_version
from models.yolo import Model
from models.common import Conv, BottleneckCSP, SPP, SPPF, Focus, Concat


class Mosaic9:
    """Mosaic9 augmentation: combines 9 images into a single training sample.
    
    Creates a 3x3 grid of images with random scaling and positioning,
    providing richer context and object relationships for training.
    """
    
    def __init__(self, img_size=640, center_ratio=0.5):
        self.img_size = img_size
        self.center_ratio = center_ratio
        
    def __call__(self, images: List[np.ndarray], labels: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Mosaic9 augmentation.
        
        Args:
            images: List of 9 images (HWC, BGR format)
            labels: List of corresponding labels [N, 5] (class, x_center, y_center, width, height)
            
        Returns:
            mosaic_img: Mosaic-augmented image
            mosaic_labels: Combined and transformed labels
        """
        assert len(images) == 9 and len(labels) == 9, "Mosaic9 requires exactly 9 images"
        
        # Create output canvas
        mosaic_img = np.full((self.img_size * 3, self.img_size * 3, 3), 114, dtype=np.uint8)
        
        # Random center point for the mosaic
        center_x = int(self.img_size * (0.5 + random.uniform(-self.center_ratio, self.center_ratio)))
        center_y = int(self.img_size * (0.5 + random.uniform(-self.center_ratio, self.center_ratio)))
        
        mosaic_labels = []
        
        # Grid positions for 3x3 mosaic
        positions = [
            (-1, -1), (0, -1), (1, -1),  # Top row
            (-1, 0), (0, 0), (1, 0),     # Middle row
            (-1, 1), (0, 1), (1, 1)      # Bottom row
        ]
        
        for idx, (img, labels_i) in enumerate(zip(images, labels)):
            h, w = img.shape[:2]
            
            # Random scale factor (0.5 to 1.5)
            scale = random.uniform(0.5, 1.5)
            
            # Calculate position in the 3x3 grid
            grid_x, grid_y = positions[idx]
            
            # Place image in the mosaic
            if grid_x == -1:  # Left column
                x_offset = center_x - int(w * scale)
            elif grid_x == 0:  # Middle column
                x_offset = center_x - int(w * scale / 2)
            else:  # Right column
                x_offset = center_x
                
            if grid_y == -1:  # Top row
                y_offset = center_y - int(h * scale)
            elif grid_y == 0:  # Middle row
                y_offset = center_y - int(h * scale / 2)
            else:  # Bottom row
                y_offset = center_y
            
            # Resize image
            img_resized = cv2.resize(img, (int(w * scale), int(h * scale)))
            
            # Calculate valid region to paste
            img_h, img_w = img_resized.shape[:2]
            
            # Clip coordinates to canvas boundaries
            x1 = max(0, x_offset)
            y1 = max(0, y_offset)
            x2 = min(mosaic_img.shape[1], x_offset + img_w)
            y2 = min(mosaic_img.shape[0], y_offset + img_h)
            
            # Calculate corresponding region in resized image
            img_x1 = max(0, -x_offset)
            img_y1 = max(0, -y_offset)
            img_x2 = img_x1 + (x2 - x1)
            img_y2 = img_y1 + (y2 - y1)
            
            # Paste image
            if (x2 - x1) > 0 and (y2 - y1) > 0:
                mosaic_img[y1:y2, x1:x2] = img_resized[img_y1:img_y2, img_x1:img_x2]
                
                # Transform labels
                if len(labels_i) > 0:
                    labels_transformed = labels_i.copy()
                    
                    # Scale labels
                    labels_transformed[:, 1] = labels_transformed[:, 1] * scale * w
                    labels_transformed[:, 2] = labels_transformed[:, 2] * scale * h
                    labels_transformed[:, 3] = labels_transformed[:, 3] * scale * w
                    labels_transformed[:, 4] = labels_transformed[:, 4] * scale * h
                    
                    # Adjust for position offset
                    labels_transformed[:, 1] += x_offset
                    labels_transformed[:, 2] += y_offset
                    
                    # Convert to absolute coordinates and clip to canvas
                    labels_transformed[:, 1] = np.clip(labels_transformed[:, 1], 0, mosaic_img.shape[1])
                    labels_transformed[:, 2] = np.clip(labels_transformed[:, 2], 0, mosaic_img.shape[0])
                    
                    # Filter labels that are within canvas
                    mask = (labels_transformed[:, 1] > 0) & (labels_transformed[:, 2] > 0) & \
                           (labels_transformed[:, 1] < mosaic_img.shape[1]) & \
                           (labels_transformed[:, 2] < mosaic_img.shape[0])
                    labels_transformed = labels_transformed[mask]
                    
                    # Normalize to [0, 1]
                    labels_transformed[:, 1] /= mosaic_img.shape[1]
                    labels_transformed[:, 2] /= mosaic_img.shape[0]
                    labels_transformed[:, 3] /= mosaic_img.shape[1]
                    labels_transformed[:, 4] /= mosaic_img.shape[0]
                    
                    mosaic_labels.append(labels_transformed)
        
        # Resize to target size
        mosaic_img = cv2.resize(mosaic_img, (self.img_size, self.img_size))
        
        # Combine all labels
        if mosaic_labels:
            mosaic_labels = np.concatenate(mosaic_labels, axis=0)
        else:
            mosaic_labels = np.zeros((0, 5), dtype=np.float32)
            
        return mosaic_img, mosaic_labels


class MixUp:
    """MixUp augmentation: blends two images and their labels.
    
    Creates a weighted combination of two training samples,
    encouraging the model to learn more robust features.
    """
    
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        
    def __call__(self, img1: np.ndarray, labels1: np.ndarray, 
                 img2: np.ndarray, labels2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply MixUp augmentation.
        
        Args:
            img1, img2: Input images (HWC, BGR format)
            labels1, labels2: Corresponding labels [N, 5] (class, x_center, y_center, width, height)
            
        Returns:
            mixed_img: Blended image
            mixed_labels: Combined labels with mixing weights
        """
        # Sample mixing coefficient from Beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
            
        # Ensure images are same size
        assert img1.shape == img2.shape, "MixUp requires same-sized images"
        
        # Mix images
        mixed_img = (lam * img1 + (1 - lam) * img2).astype(np.uint8)
        
        # Combine labels with mixing weight
        if len(labels1) > 0:
            labels1_with_weight = np.hstack([labels1, np.full((len(labels1), 1), lam)])
        else:
            labels1_with_weight = np.zeros((0, 6), dtype=np.float32)
            
        if len(labels2) > 0:
            labels2_with_weight = np.hstack([labels2, np.full((len(labels2), 1), 1 - lam)])
        else:
            labels2_with_weight = np.zeros((0, 6), dtype=np.float32)
            
        mixed_labels = np.vstack([labels1_with_weight, labels2_with_weight])
        
        return mixed_img, mixed_labels


class CopyPaste:
    """Copy-Paste augmentation: copies objects from one image and pastes them onto another.
    
    Particularly effective for instance segmentation and detection tasks,
    creating new training samples with novel object configurations.
    """
    
    def __init__(self, max_objects=5, iou_threshold=0.3):
        self.max_objects = max_objects
        self.iou_threshold = iou_threshold
        
    def __call__(self, img_src: np.ndarray, labels_src: np.ndarray,
                 img_dst: np.ndarray, labels_dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Copy-Paste augmentation.
        
        Args:
            img_src, img_dst: Source and destination images
            labels_src, labels_dst: Corresponding labels [N, 5] (class, x_center, y_center, width, height)
            
        Returns:
            img_pasted: Image with pasted objects
            labels_pasted: Combined labels
        """
        if len(labels_src) == 0:
            return img_dst, labels_dst
            
        # Select random objects to copy
        num_objects = min(self.max_objects, len(labels_src))
        selected_indices = np.random.choice(len(labels_src), num_objects, replace=False)
        
        labels_to_copy = labels_src[selected_indices]
        img_pasted = img_dst.copy()
        h_dst, w_dst = img_dst.shape[:2]
        
        new_labels = []
        
        for label in labels_to_copy:
            class_id, x_center, y_center, width, height = label
            
            # Convert normalized coordinates to absolute
            x_center_abs = int(x_center * w_dst)
            y_center_abs = int(y_center * h_dst)
            width_abs = int(width * w_dst)
            height_abs = int(height * h_dst)
            
            # Calculate bounding box
            x1 = max(0, x_center_abs - width_abs // 2)
            y1 = max(0, y_center_abs - height_abs // 2)
            x2 = min(w_dst, x_center_abs + width_abs // 2)
            y2 = min(h_dst, y_center_abs + height_abs // 2)
            
            # Skip if too small
            if (x2 - x1) < 10 or (y2 - y1) < 10:
                continue
                
            # Check for overlap with existing objects (simplified IoU check)
            overlap = False
            for existing_label in labels_dst:
                ex_class, ex_x, ex_y, ex_w, ex_h = existing_label
                ex_x1 = max(0, int((ex_x - ex_w/2) * w_dst))
                ex_y1 = max(0, int((ex_y - ex_h/2) * h_dst))
                ex_x2 = min(w_dst, int((ex_x + ex_w/2) * w_dst))
                ex_y2 = min(h_dst, int((ex_y + ex_h/2) * h_dst))
                
                # Calculate IoU
                inter_x1 = max(x1, ex_x1)
                inter_y1 = max(y1, ex_y1)
                inter_x2 = min(x2, ex_x2)
                inter_y2 = min(y2, ex_y2)
                
                if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    area1 = (x2 - x1) * (y2 - y1)
                    area2 = (ex_x2 - ex_x1) * (ex_y2 - ex_y1)
                    iou = inter_area / (area1 + area2 - inter_area)
                    
                    if iou > self.iou_threshold:
                        overlap = True
                        break
            
            if not overlap:
                # Copy object region from source image
                # (In practice, you'd use instance masks for better results)
                # Here we use bounding box region for simplicity
                src_h, src_w = img_src.shape[:2]
                src_x1 = max(0, int((x_center - width/2) * src_w))
                src_y1 = max(0, int((y_center - height/2) * src_h))
                src_x2 = min(src_w, int((x_center + width/2) * src_w))
                src_y2 = min(src_h, int((y_center + height/2) * src_h))
                
                if (src_x2 - src_x1) > 0 and (src_y2 - src_y1) > 0:
                    object_region = img_src[src_y1:src_y2, src_x1:src_x2]
                    
                    # Resize to match destination size
                    object_resized = cv2.resize(object_region, (x2-x1, y2-y1))
                    
                    # Simple blending (could be improved with Poisson blending)
                    img_pasted[y1:y2, x1:x2] = cv2.addWeighted(
                        img_pasted[y1:y2, x1:x2], 0.5,
                        object_resized, 0.5, 0
                    )
                    
                    # Add label
                    new_label = np.array([class_id, 
                                         (x1 + x2) / (2 * w_dst),
                                         (y1 + y2) / (2 * h_dst),
                                         (x2 - x1) / w_dst,
                                         (y2 - y1) / h_dst])
                    new_labels.append(new_label)
        
        # Combine with existing labels
        if new_labels:
            new_labels = np.array(new_labels)
            labels_pasted = np.vstack([labels_dst, new_labels]) if len(labels_dst) > 0 else new_labels
        else:
            labels_pasted = labels_dst
            
        return img_pasted, labels_pasted


class RandAugment:
    """RandAugment: Automated augmentation policy with random transformations.
    
    Applies a random selection of transformations with random magnitudes,
    reducing the need for manual augmentation policy design.
    """
    
    def __init__(self, n_transforms=2, magnitude=9, magnitude_std=0.5):
        self.n_transforms = n_transforms
        self.magnitude = magnitude
        self.magnitude_std = magnitude_std
        
        # Define available transformations
        self.transforms = [
            self.auto_contrast,
            self.equalize,
            self.rotate,
            self.solarize,
            self.color,
            self.posterize,
            self.contrast,
            self.brightness,
            self.sharpness,
            self.shear_x,
            self.shear_y,
            self.translate_x,
            self.translate_y,
        ]
        
    def __call__(self, img: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply RandAugment.
        
        Args:
            img: Input image (HWC, BGR format)
            labels: Corresponding labels [N, 5] (class, x_center, y_center, width, height)
            
        Returns:
            img_aug: Augmented image
            labels: Unchanged labels (transformations preserve bounding boxes)
        """
        # Select random transformations
        ops = random.choices(self.transforms, k=self.n_transforms)
        
        # Apply transformations with random magnitude
        img_aug = img.copy()
        for op in ops:
            magnitude = max(0, min(10, self.magnitude + random.gauss(0, self.magnitude_std)))
            img_aug = op(img_aug, magnitude)
            
        return img_aug, labels
    
    def auto_contrast(self, img: np.ndarray, magnitude: float) -> np.ndarray:
        """Auto contrast adjustment."""
        return cv2.convertScaleAbs(img, alpha=255.0 / (np.max(img) - np.min(img) + 1e-6))
    
    def equalize(self, img: np.ndarray, magnitude: float) -> np.ndarray:
        """Histogram equalization."""
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    def rotate(self, img: np.ndarray, magnitude: float) -> np.ndarray:
        """Random rotation."""
        angle = (magnitude / 10) * 30  # Max 30 degrees
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderValue=(114, 114, 114))
    
    def solarize(self, img: np.ndarray, magnitude: float) -> np.ndarray:
        """Solarize (invert pixels above threshold)."""
        threshold = int((magnitude / 10) * 256)
        return np.where(img < threshold, img, 255 - img).astype(np.uint8)
    
    def color(self, img: np.ndarray, magnitude: float) -> np.ndarray:
        """Color saturation adjustment."""
        factor = 1 + (magnitude / 10) * 1.5  # Max 2.5x saturation
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        img_hsv[:,:,1] = np.clip(img_hsv[:,:,1] * factor, 0, 255)
        return cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def posterize(self, img: np.ndarray, magnitude: float) -> np.ndarray:
        """Posterize (reduce number of bits)."""
        bits = max(1, int(8 - (magnitude / 10) * 7))  # 1-8 bits
        shift = 8 - bits
        return ((img >> shift) << shift).astype(np.uint8)
    
    def contrast(self, img: np.ndarray, magnitude: float) -> np.ndarray:
        """Contrast adjustment."""
        factor = 1 + (magnitude / 10) * 1.5  # Max 2.5x contrast
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        return np.clip((img - mean) * factor + mean, 0, 255).astype(np.uint8)
    
    def brightness(self, img: np.ndarray, magnitude: float) -> np.ndarray:
        """Brightness adjustment."""
        delta = int((magnitude / 10) * 100)  # Max 100 brightness increase
        return np.clip(img.astype(np.int32) + delta, 0, 255).astype(np.uint8)
    
    def sharpness(self, img: np.ndarray, magnitude: float) -> np.ndarray:
        """Sharpness adjustment."""
        kernel = np.array([[-1, -1, -1],
                          [-1, 9 + magnitude/10, -1],
                          [-1, -1, -1]])
        return cv2.filter2D(img, -1, kernel)
    
    def shear_x(self, img: np.ndarray, magnitude: float) -> np.ndarray:
        """Shear in x direction."""
        shear = (magnitude / 10) * 0.3  # Max 0.3 shear
        h, w = img.shape[:2]
        M = np.float32([[1, shear, 0], [0, 1, 0]])
        return cv2.warpAffine(img, M, (w, h), borderValue=(114, 114, 114))
    
    def shear_y(self, img: np.ndarray, magnitude: float) -> np.ndarray:
        """Shear in y direction."""
        shear = (magnitude / 10) * 0.3  # Max 0.3 shear
        h, w = img.shape[:2]
        M = np.float32([[1, 0, 0], [shear, 1, 0]])
        return cv2.warpAffine(img, M, (w, h), borderValue=(114, 114, 114))
    
    def translate_x(self, img: np.ndarray, magnitude: float) -> np.ndarray:
        """Translate in x direction."""
        pixels = int((magnitude / 10) * img.shape[1] * 0.45)  # Max 45% translation
        h, w = img.shape[:2]
        M = np.float32([[1, 0, pixels], [0, 1, 0]])
        return cv2.warpAffine(img, M, (w, h), borderValue=(114, 114, 114))
    
    def translate_y(self, img: np.ndarray, magnitude: float) -> np.ndarray:
        """Translate in y direction."""
        pixels = int((magnitude / 10) * img.shape[0] * 0.45)  # Max 45% translation
        h, w = img.shape[:2]
        M = np.float32([[1, 0, 0], [0, 1, pixels]])
        return cv2.warpAffine(img, M, (w, h), borderValue=(114, 114, 114))


class ModelScaler:
    """Automated model scaling using compound scaling method.
    
    Scales model depth, width, and resolution uniformly to create
    custom-sized models (e.g., YOLOv5-nano, YOLOv5-xxxl) optimized
    for target latency or accuracy.
    """
    
    # Predefined scaling coefficients for different model sizes
    SCALING_PRESETS = {
        'nano': {'depth': 0.33, 'width': 0.25, 'resolution': 0.5},
        'tiny': {'depth': 0.33, 'width': 0.375, 'resolution': 0.5},
        'small': {'depth': 0.33, 'width': 0.50, 'resolution': 0.5},
        'medium': {'depth': 0.67, 'width': 0.75, 'resolution': 0.5},
        'large': {'depth': 1.0, 'width': 1.0, 'resolution': 1.0},
        'xlarge': {'depth': 1.33, 'width': 1.25, 'resolution': 1.0},
        'huge': {'depth': 1.67, 'width': 1.5, 'resolution': 1.0},
        'giant': {'depth': 2.0, 'width': 2.0, 'resolution': 1.0},
        'xxxl': {'depth': 3.0, 'width': 3.0, 'resolution': 1.5},
    }
    
    @staticmethod
    def scale_model(model_config: Dict, depth_scale: float = 1.0, 
                   width_scale: float = 1.0, resolution_scale: float = 1.0) -> Dict:
        """Scale model configuration using compound scaling.
        
        Args:
            model_config: Original model configuration dictionary
            depth_scale: Scaling factor for model depth (number of layers)
            width_scale: Scaling factor for model width (number of channels)
            resolution_scale: Scaling factor for input resolution
            
        Returns:
            Scaled model configuration dictionary
        """
        scaled_config = model_config.copy()
        
        # Scale depth (number of repeats in CSP layers)
        if 'depth_multiple' in scaled_config:
            scaled_config['depth_multiple'] = max(0.1, scaled_config['depth_multiple'] * depth_scale)
        
        # Scale width (channel multiplier)
        if 'width_multiple' in scaled_config:
            scaled_config['width_multiple'] = max(0.1, scaled_config['width_multiple'] * width_scale)
        
        # Scale anchors based on resolution
        if 'anchors' in scaled_config and resolution_scale != 1.0:
            scaled_anchors = []
            for anchor_set in scaled_config['anchors']:
                scaled_anchor = [int(a * resolution_scale) for a in anchor_set]
                scaled_anchors.append(scaled_anchor)
            scaled_config['anchors'] = scaled_anchors
        
        return scaled_config
    
    @staticmethod
    def create_scaled_model(model_name: str = 'nexuss', preset: str = None,
                           depth_scale: float = None, width_scale: float = None,
                           resolution_scale: float = None, nc: int = 80,
                           anchors: List = None) -> Tuple[Model, int]:
        """Create a scaled YOLOv5 model.
        
        Args:
            model_name: Base model name ('nexusn', 'nexuss', 'nexusm', 'nexusl', 'nexusx')
            preset: Predefined scaling preset (overrides individual scale factors)
            depth_scale: Custom depth scaling factor
            width_scale: Custom width scaling factor
            resolution_scale: Custom resolution scaling factor
            nc: Number of classes
            anchors: Custom anchors (if None, uses default)
            
        Returns:
            model: Scaled YOLOv5 model
            img_size: Scaled input image size
        """
        # Load base configuration
        cfg_path = Path(__file__).parent.parent / 'models' / f'{model_name}.yaml'
        if not cfg_path.exists():
            # Default to nexuss if specific config not found
            cfg_path = Path(__file__).parent.parent / 'models' / 'nexuss.yaml'
        
        with open(cfg_path) as f:
            model_config = yaml.safe_load(f)
        
        # Apply preset if specified
        if preset and preset in ModelScaler.SCALING_PRESETS:
            preset_config = ModelScaler.SCALING_PRESETS[preset]
            depth_scale = preset_config['depth']
            width_scale = preset_config['width']
            resolution_scale = preset_config['resolution']
        
        # Use default scales if not specified
        depth_scale = depth_scale or 1.0
        width_scale = width_scale or 1.0
        resolution_scale = resolution_scale or 1.0
        
        # Scale configuration
        scaled_config = ModelScaler.scale_model(model_config, depth_scale, width_scale, resolution_scale)
        
        # Update number of classes if specified
        if nc != 80:
            scaled_config['nc'] = nc
        
        # Update anchors if specified
        if anchors:
            scaled_config['anchors'] = anchors
        
        # Calculate scaled image size (multiple of 32)
        base_img_size = model_config.get('img_size', 640)
        scaled_img_size = int(base_img_size * resolution_scale)
        scaled_img_size = max(32, scaled_img_size - (scaled_img_size % 32))  # Ensure divisible by 32
        
        # Create model
        model = Model(cfg=scaled_config, ch=3, nc=nc, anchors=scaled_config.get('anchors'))
        
        return model, scaled_img_size
    
    @staticmethod
    def optimize_for_target(model_name: str = 'nexuss', target_type: str = 'latency',
                           target_value: float = 10.0, device: str = 'cpu') -> Tuple[Model, int]:
        """Optimize model scaling for target latency or accuracy.
        
        Args:
            model_name: Base model name
            target_type: 'latency' (ms) or 'accuracy' (mAP)
            target_value: Target value in ms (latency) or percentage (accuracy)
            device: Device for latency measurement ('cpu' or 'cuda')
            
        Returns:
            model: Optimized model
            img_size: Optimized input size
        """
        # This is a simplified optimization - in practice, you'd use a more sophisticated
        # approach with actual latency/accuracy measurements
        
        if target_type == 'latency':
            # Scale model based on target latency
            # Lower latency = smaller model
            if target_value < 5:  # Ultra-fast
                preset = 'nano'
            elif target_value < 10:  # Fast
                preset = 'tiny'
            elif target_value < 20:  # Balanced
                preset = 'small'
            elif target_value < 50:  # Accurate
                preset = 'medium'
            else:  # High accuracy
                preset = 'large'
                
        elif target_type == 'accuracy':
            # Scale model based on target accuracy
            # Higher accuracy = larger model
            if target_value < 30:  # Low accuracy
                preset = 'nano'
            elif target_value < 40:  # Moderate
                preset = 'tiny'
            elif target_value < 50:  # Good
                preset = 'small'
            elif target_value < 60:  # Better
                preset = 'medium'
            else:  # Best
                preset = 'large'
        else:
            raise ValueError(f"Unknown target_type: {target_type}")
        
        return ModelScaler.create_scaled_model(model_name=model_name, preset=preset)


class AdvancedAugmentationPipeline:
    """Complete augmentation pipeline combining all techniques.
    
    Integrates Mosaic9, MixUp, Copy-Paste, and RandAugment with
    configurable probabilities and parameters.
    """
    
    def __init__(self, img_size=640, augment=True, hyp=None):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp or {}
        
        # Initialize augmentations
        self.mosaic9 = Mosaic9(img_size=img_size)
        self.mixup = MixUp(alpha=self.hyp.get('mixup', 0.5))
        self.copy_paste = CopyPaste(max_objects=5, iou_threshold=0.3)
        self.randaugment = RandAugment(n_transforms=2, magnitude=9)
        
        # Augmentation probabilities
        self.p_mosaic9 = self.hyp.get('mosaic9', 0.5)
        self.p_mixup = self.hyp.get('mixup_prob', 0.3)
        self.p_copy_paste = self.hyp.get('copy_paste', 0.2)
        self.p_randaugment = self.hyp.get('randaugment', 0.5)
        
    def __call__(self, images: List[np.ndarray], labels: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Apply augmentation pipeline.
        
        Args:
            images: List of images for mosaic (9 images for Mosaic9, 2 for others)
            labels: List of corresponding labels
            
        Returns:
            img: Augmented image
            labels: Augmented labels
        """
        if not self.augment:
            # No augmentation, just resize first image
            img = letterbox(images[0], self.img_size, auto=False)[0]
            return img, labels[0]
        
        # Apply Mosaic9 with probability
        if len(images) >= 9 and random.random() < self.p_mosaic9:
            img, lbl = self.mosaic9(images[:9], labels[:9])
        else:
            # Use first image with letterbox
            img = letterbox(images[0], self.img_size, auto=False)[0]
            lbl = labels[0]
        
        # Apply Copy-Paste with probability
        if len(images) >= 2 and random.random() < self.p_copy_paste:
            img, lbl = self.copy_paste(images[1], labels[1], img, lbl)
        
        # Apply MixUp with probability
        if len(images) >= 2 and random.random() < self.p_mixup:
            img2 = letterbox(images[1], self.img_size, auto=False)[0]
            lbl2 = labels[1]
            img, lbl = self.mixup(img, lbl, img2, lbl2)
        
        # Apply RandAugment with probability
        if random.random() < self.p_randaugment:
            img, lbl = self.randaugment(img, lbl)
        
        # Apply standard augmentations (HSV, perspective, etc.)
        img, lbl = random_perspective(img, lbl,
                                     degrees=self.hyp.get('degrees', 0.0),
                                     translate=self.hyp.get('translate', 0.1),
                                     scale=self.hyp.get('scale', 0.5),
                                     shear=self.hyp.get('shear', 0.0),
                                     perspective=self.hyp.get('perspective', 0.0)
                                     )
        
        # Apply HSV augmentation
        augment_hsv(img, 
                   hgain=self.hyp.get('hsv_h', 0.015),
                   sgain=self.hyp.get('hsv_s', 0.7),
                   vgain=self.hyp.get('hsv_v', 0.4))
        
        return img, lbl


def apply_scaling_to_model(model: Model, scaling_config: Dict) -> Model:
    """Apply scaling configuration to an existing model.
    
    Args:
        model: YOLOv5 model instance
        scaling_config: Dictionary with scaling parameters
        
    Returns:
        model: Model with updated scaling
    """
    # Update model attributes based on scaling config
    if hasattr(model, 'yaml'):
        model.yaml = {**model.yaml, **scaling_config}
    
    # Update depth and width multiples
    if 'depth_multiple' in scaling_config:
        model.depth_multiple = scaling_config['depth_multiple']
    if 'width_multiple' in scaling_config:
        model.width_multiple = scaling_config['width_multiple']
    
    # Reinitialize model with new configuration
    # Note: In practice, you might need to rebuild the model layers
    # This is a simplified version
    
    return model


def get_model_complexity(model: Model, img_size: int = 640) -> Dict:
    """Calculate model complexity metrics.
    
    Args:
        model: YOLOv5 model
        img_size: Input image size
        
    Returns:
        Dictionary with complexity metrics
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate FLOPs (simplified)
    # In practice, use thop or similar library for accurate calculation
    input_tensor = torch.randn(1, 3, img_size, img_size)
    
    # Simple forward pass to estimate
    try:
        from thop import profile
        flops, _ = profile(model, inputs=(input_tensor,), verbose=False)
    except ImportError:
        # Rough estimate: 2 * params * spatial_ops
        flops = 2 * total_params * (img_size // 32) ** 2  # Very rough estimate
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'flops': flops,
        'params_millions': total_params / 1e6,
        'flops_billions': flops / 1e9,
    }


# Example usage and integration points
if __name__ == "__main__":
    # Example 1: Create scaled models
    print("Creating scaled YOLOv5 models...")
    
    # Create nano model
    nano_model, nano_size = ModelScaler.create_scaled_model(preset='nano')
    nano_metrics = get_model_complexity(nano_model, nano_size)
    print(f"YOLOv5-nano: {nano_metrics['params_millions']:.2f}M params, "
          f"{nano_metrics['flops_billions']:.2f}B FLOPs, {nano_size}px")
    
    # Create xxxl model
    xxxl_model, xxxl_size = ModelScaler.create_scaled_model(preset='xxxl')
    xxxl_metrics = get_model_complexity(xxxl_model, xxxl_size)
    print(f"YOLOv5-xxxl: {xxxl_metrics['params_millions']:.2f}M params, "
          f"{xxxl_metrics['flops_billions']:.2f}B FLOPs, {xxxl_size}px")
    
    # Example 2: Optimize for target latency
    fast_model, fast_size = ModelScaler.optimize_for_target(
        model_name='nexuss',
        target_type='latency',
        target_value=10.0,
        device='cuda'
    )
    print(f"Optimized for 10ms latency: {get_model_complexity(fast_model, fast_size)}")
    
    # Example 3: Test augmentations
    print("\nTesting augmentations...")
    # Create dummy data
    dummy_images = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8) for _ in range(9)]
    dummy_labels = [np.random.rand(5, 5).astype(np.float32) for _ in range(9)]
    
    # Test Mosaic9
    mosaic9 = Mosaic9(img_size=640)
    mosaic_img, mosaic_labels = mosaic9(dummy_images, dummy_labels)
    print(f"Mosaic9 output shape: {mosaic_img.shape}, labels: {mosaic_labels.shape}")
    
    # Test complete pipeline
    pipeline = AdvancedAugmentationPipeline(img_size=640)
    aug_img, aug_labels = pipeline(dummy_images, dummy_labels)
    print(f"Pipeline output shape: {aug_img.shape}, labels: {aug_labels.shape}")