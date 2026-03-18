# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Image augmentation functions."""

import math
import random

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from utils.general import LOGGER, check_version, colorstr, resample_segments, segment2box, xywhn2xyxy
from utils.metrics import bbox_ioa

IMAGENET_MEAN = 0.485, 0.456, 0.406  # RGB mean
IMAGENET_STD = 0.229, 0.224, 0.225  # RGB standard deviation


class Albumentations:
    """Provides optional data augmentation for YOLOv5 using Albumentations library if installed."""

    def __init__(self, size=640):
        """Initializes Albumentations class for optional data augmentation in YOLOv5 with specified input size."""
        self.transform = None
        prefix = colorstr("albumentations: ")
        try:
            import albumentations as A

            check_version(A.__version__, "1.0.3", hard=True)  # version requirement

            T = [
                A.RandomResizedCrop(height=size, width=size, scale=(0.8, 1.0), ratio=(0.9, 1.11), p=0.0),
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.0),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_lower=75, p=0.0),
            ]  # transforms
            self.transform = A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

            LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            LOGGER.info(f"{prefix}{e}")

    def __call__(self, im, labels, p=1.0):
        """Applies transformations to an image and labels with probability `p`, returning updated image and labels."""
        if self.transform and random.random() < p:
            new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])  # transformed
            im, labels = new["image"], np.array([[c, *b] for c, b in zip(new["class_labels"], new["bboxes"])])
        return im, labels


class RandAugment:
    """RandAugment data augmentation for YOLOv5."""
    
    def __init__(self, n=2, m=9):
        """Initialize RandAugment with n transformations and magnitude m."""
        self.n = n
        self.m = m
        self.augmentations = [
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
    
    def __call__(self, im, labels=None, p=0.5):
        """Apply RandAugment to image and optionally labels with probability p."""
        if random.random() > p:
            return im, labels
        
        ops = random.choices(self.augmentations, k=self.n)
        for op in ops:
            im, labels = op(im, labels, magnitude=self.m)
        return im, labels
    
    def auto_contrast(self, im, labels, magnitude):
        """Apply auto contrast adjustment."""
        if random.random() < 0.5:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im = cv2.equalizeHist(im)
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        return im, labels
    
    def equalize(self, im, labels, magnitude):
        """Apply histogram equalization."""
        im = hist_equalize(im, clahe=True, bgr=True)
        return im, labels
    
    def rotate(self, im, labels, magnitude):
        """Apply random rotation."""
        angle = random.uniform(-magnitude * 3, magnitude * 3)
        h, w = im.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        im = cv2.warpAffine(im, M, (w, h), borderValue=(114, 114, 114))
        
        if labels is not None and len(labels):
            # Transform bounding boxes
            boxes = labels[:, 1:].copy()
            boxes = boxes.reshape(-1, 2, 2)
            boxes = np.concatenate([boxes, np.ones_like(boxes[:, :, :1])], axis=-1)
            transformed = np.dot(boxes, M.T)
            labels[:, 1:] = transformed.reshape(-1, 4)
        return im, labels
    
    def solarize(self, im, labels, magnitude):
        """Apply solarize effect."""
        threshold = 256 - magnitude * 25
        im = np.where(im < threshold, im, 255 - im)
        return im.astype(np.uint8), labels
    
    def color(self, im, labels, magnitude):
        """Apply color jitter."""
        factor = 1 + (magnitude - 5) * 0.1
        im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        im = im.astype(np.float32)
        im[:, :, 1] *= factor
        im = np.clip(im, 0, 255).astype(np.uint8)
        im = cv2.cvtColor(im, cv2.COLOR_HSV2BGR)
        return im, labels
    
    def posterize(self, im, labels, magnitude):
        """Apply posterize effect."""
        bits = max(1, 8 - magnitude)
        im = np.bitwise_and(im, 255 << (8 - bits))
        return im, labels
    
    def contrast(self, im, labels, magnitude):
        """Apply contrast adjustment."""
        factor = 1 + (magnitude - 5) * 0.1
        mean = np.mean(im, axis=(0, 1), keepdims=True)
        im = np.clip((im - mean) * factor + mean, 0, 255).astype(np.uint8)
        return im, labels
    
    def brightness(self, im, labels, magnitude):
        """Apply brightness adjustment."""
        factor = 1 + (magnitude - 5) * 0.1
        im = np.clip(im * factor, 0, 255).astype(np.uint8)
        return im, labels
    
    def sharpness(self, im, labels, magnitude):
        """Apply sharpness adjustment."""
        factor = 1 + (magnitude - 5) * 0.1
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) * factor
        kernel[1, 1] = kernel[1, 1] - factor + 1
        im = cv2.filter2D(im, -1, kernel)
        return im, labels
    
    def shear_x(self, im, labels, magnitude):
        """Apply horizontal shear."""
        shear = random.uniform(-magnitude * 0.03, magnitude * 0.03)
        h, w = im.shape[:2]
        M = np.float32([[1, shear, 0], [0, 1, 0]])
        im = cv2.warpAffine(im, M, (w, h), borderValue=(114, 114, 114))
        
        if labels is not None and len(labels):
            boxes = labels[:, 1:].copy()
            boxes = boxes.reshape(-1, 2, 2)
            boxes = np.concatenate([boxes, np.ones_like(boxes[:, :, :1])], axis=-1)
            transformed = np.dot(boxes, M.T)
            labels[:, 1:] = transformed.reshape(-1, 4)
        return im, labels
    
    def shear_y(self, im, labels, magnitude):
        """Apply vertical shear."""
        shear = random.uniform(-magnitude * 0.03, magnitude * 0.03)
        h, w = im.shape[:2]
        M = np.float32([[1, 0, 0], [shear, 1, 0]])
        im = cv2.warpAffine(im, M, (w, h), borderValue=(114, 114, 114))
        
        if labels is not None and len(labels):
            boxes = labels[:, 1:].copy()
            boxes = boxes.reshape(-1, 2, 2)
            boxes = np.concatenate([boxes, np.ones_like(boxes[:, :, :1])], axis=-1)
            transformed = np.dot(boxes, M.T)
            labels[:, 1:] = transformed.reshape(-1, 4)
        return im, labels
    
    def translate_x(self, im, labels, magnitude):
        """Apply horizontal translation."""
        translate = random.uniform(-magnitude * 0.1, magnitude * 0.1)
        h, w = im.shape[:2]
        M = np.float32([[1, 0, translate * w], [0, 1, 0]])
        im = cv2.warpAffine(im, M, (w, h), borderValue=(114, 114, 114))
        
        if labels is not None and len(labels):
            boxes = labels[:, 1:].copy()
            boxes = boxes.reshape(-1, 2, 2)
            boxes = np.concatenate([boxes, np.ones_like(boxes[:, :, :1])], axis=-1)
            transformed = np.dot(boxes, M.T)
            labels[:, 1:] = transformed.reshape(-1, 4)
        return im, labels
    
    def translate_y(self, im, labels, magnitude):
        """Apply vertical translation."""
        translate = random.uniform(-magnitude * 0.1, magnitude * 0.1)
        h, w = im.shape[:2]
        M = np.float32([[1, 0, 0], [0, 1, translate * h]])
        im = cv2.warpAffine(im, M, (w, h), borderValue=(114, 114, 114))
        
        if labels is not None and len(labels):
            boxes = labels[:, 1:].copy()
            boxes = boxes.reshape(-1, 2, 2)
            boxes = np.concatenate([boxes, np.ones_like(boxes[:, :, :1])], axis=-1)
            transformed = np.dot(boxes, M.T)
            labels[:, 1:] = transformed.reshape(-1, 4)
        return im, labels


def mosaic9(images, labels, size=640):
    """Create a 9-image mosaic augmentation.
    
    Args:
        images: List of 9 images (numpy arrays)
        labels: List of 9 label arrays (each [cls, x1, y1, x2, y2])
        size: Output size (square)
    
    Returns:
        mosaic_img: Mosaic augmented image
        mosaic_labels: Combined labels for mosaic image
    """
    assert len(images) == 9, f"Expected 9 images for Mosaic9, got {len(images)}"
    
    # Create empty mosaic
    mosaic_img = np.full((size * 3, size * 3, 3), 114, dtype=np.uint8)
    
    # Random center point
    yc, xc = [int(random.uniform(size * 0.5, size * 1.5)) for _ in range(2)]
    
    mosaic_labels = []
    
    for i in range(9):
        img = images[i]
        h, w = img.shape[:2]
        
        # Place img in mosaic
        if i == 0:  # top left
            img_resized = cv2.resize(img, (size, size))
            mosaic_img[0:size, 0:size] = img_resized
            # Adjust labels
            if len(labels[i]):
                label = labels[i].copy()
                label[:, 1] = label[:, 1] * size / w
                label[:, 2] = label[:, 2] * size / h
                label[:, 3] = label[:, 3] * size / w
                label[:, 4] = label[:, 4] * size / h
                mosaic_labels.append(label)
                
        elif i == 1:  # top center
            img_resized = cv2.resize(img, (size, size))
            mosaic_img[0:size, size:size*2] = img_resized
            if len(labels[i]):
                label = labels[i].copy()
                label[:, 1] = label[:, 1] * size / w + size
                label[:, 2] = label[:, 2] * size / h
                label[:, 3] = label[:, 3] * size / w + size
                label[:, 4] = label[:, 4] * size / h
                mosaic_labels.append(label)
                
        elif i == 2:  # top right
            img_resized = cv2.resize(img, (size, size))
            mosaic_img[0:size, size*2:size*3] = img_resized
            if len(labels[i]):
                label = labels[i].copy()
                label[:, 1] = label[:, 1] * size / w + size * 2
                label[:, 2] = label[:, 2] * size / h
                label[:, 3] = label[:, 3] * size / w + size * 2
                label[:, 4] = label[:, 4] * size / h
                mosaic_labels.append(label)
                
        elif i == 3:  # center left
            img_resized = cv2.resize(img, (size, size))
            mosaic_img[size:size*2, 0:size] = img_resized
            if len(labels[i]):
                label = labels[i].copy()
                label[:, 1] = label[:, 1] * size / w
                label[:, 2] = label[:, 2] * size / h + size
                label[:, 3] = label[:, 3] * size / w
                label[:, 4] = label[:, 4] * size / h + size
                mosaic_labels.append(label)
                
        elif i == 4:  # center (main)
            # Place center image with random offset
            img_resized = cv2.resize(img, (size, size))
            y1 = max(0, yc - size // 2)
            y2 = min(size * 3, yc + size // 2)
            x1 = max(0, xc - size // 2)
            x2 = min(size * 3, xc + size // 2)
            
            # Crop if needed
            y1c = max(0, - (yc - size // 2))
            y2c = size - max(0, (yc + size // 2) - size * 3)
            x1c = max(0, - (xc - size // 2))
            x2c = size - max(0, (xc + size // 2) - size * 3)
            
            mosaic_img[y1:y2, x1:x2] = img_resized[y1c:y2c, x1c:x2c]
            
            if len(labels[i]):
                label = labels[i].copy()
                label[:, 1] = label[:, 1] * size / w + x1 - x1c
                label[:, 2] = label[:, 2] * size / h + y1 - y1c
                label[:, 3] = label[:, 3] * size / w + x1 - x1c
                label[:, 4] = label[:, 4] * size / h + y1 - y1c
                mosaic_labels.append(label)
                
        elif i == 5:  # center right
            img_resized = cv2.resize(img, (size, size))
            mosaic_img[size:size*2, size*2:size*3] = img_resized
            if len(labels[i]):
                label = labels[i].copy()
                label[:, 1] = label[:, 1] * size / w + size * 2
                label[:, 2] = label[:, 2] * size / h + size
                label[:, 3] = label[:, 3] * size / w + size * 2
                label[:, 4] = label[:, 4] * size / h + size
                mosaic_labels.append(label)
                
        elif i == 6:  # bottom left
            img_resized = cv2.resize(img, (size, size))
            mosaic_img[size*2:size*3, 0:size] = img_resized
            if len(labels[i]):
                label = labels[i].copy()
                label[:, 1] = label[:, 1] * size / w
                label[:, 2] = label[:, 2] * size / h + size * 2
                label[:, 3] = label[:, 3] * size / w
                label[:, 4] = label[:, 4] * size / h + size * 2
                mosaic_labels.append(label)
                
        elif i == 7:  # bottom center
            img_resized = cv2.resize(img, (size, size))
            mosaic_img[size*2:size*3, size:size*2] = img_resized
            if len(labels[i]):
                label = labels[i].copy()
                label[:, 1] = label[:, 1] * size / w + size
                label[:, 2] = label[:, 2] * size / h + size * 2
                label[:, 3] = label[:, 3] * size / w + size
                label[:, 4] = label[:, 4] * size / h + size * 2
                mosaic_labels.append(label)
                
        elif i == 8:  # bottom right
            img_resized = cv2.resize(img, (size, size))
            mosaic_img[size*2:size*3, size*2:size*3] = img_resized
            if len(labels[i]):
                label = labels[i].copy()
                label[:, 1] = label[:, 1] * size / w + size * 2
                label[:, 2] = label[:, 2] * size / h + size * 2
                label[:, 3] = label[:, 3] * size / w + size * 2
                label[:, 4] = label[:, 4] * size / h + size * 2
                mosaic_labels.append(label)
    
    # Combine all labels
    if mosaic_labels:
        mosaic_labels = np.concatenate(mosaic_labels, axis=0)
        # Clip to image boundaries
        mosaic_labels[:, 1:] = np.clip(mosaic_labels[:, 1:], 0, size * 3)
    else:
        mosaic_labels = np.zeros((0, 5))
    
    return mosaic_img, mosaic_labels


def mixup(images, labels, alpha=8.0):
    """MixUp augmentation for two images.
    
    Args:
        images: List of 2 images (numpy arrays)
        labels: List of 2 label arrays (each [cls, x1, y1, x2, y2])
        alpha: Beta distribution parameter
    
    Returns:
        mixed_img: Mixed image
        mixed_labels: Combined labels
    """
    assert len(images) == 2, f"MixUp requires exactly 2 images, got {len(images)}"
    
    # Sample lambda from Beta distribution
    lam = np.random.beta(alpha, alpha)
    
    img1, img2 = images
    labels1, labels2 = labels
    
    # Ensure images are same size
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    if h1 != h2 or w1 != w2:
        img2 = cv2.resize(img2, (w1, h1))
    
    # Mix images
    mixed_img = (img1 * lam + img2 * (1 - lam)).astype(np.uint8)
    
    # Mix labels (concatenate both sets)
    mixed_labels = []
    if len(labels1):
        mixed_labels.append(labels1)
    if len(labels2):
        mixed_labels.append(labels2)
    
    if mixed_labels:
        mixed_labels = np.concatenate(mixed_labels, axis=0)
    else:
        mixed_labels = np.zeros((0, 5))
    
    return mixed_img, mixed_labels


def copy_paste(images, labels, p=0.5):
    """Copy-Paste augmentation.
    
    Args:
        images: List of images
        labels: List of label arrays
        p: Probability of applying copy-paste
    
    Returns:
        Augmented images and labels
    """
    if len(images) < 2 or random.random() > p:
        return images, labels
    
    # Select random source and target
    src_idx = random.randint(0, len(images) - 1)
    tgt_idx = random.randint(0, len(images) - 1)
    while src_idx == tgt_idx:
        tgt_idx = random.randint(0, len(images) - 1)
    
    src_img = images[src_idx].copy()
    tgt_img = images[tgt_idx].copy()
    src_labels = labels[src_idx].copy() if len(labels[src_idx]) else np.zeros((0, 5))
    tgt_labels = labels[tgt_idx].copy() if len(labels[tgt_idx]) else np.zeros((0, 5))
    
    if len(src_labels) == 0:
        return images, labels
    
    # Randomly select objects to copy
    n_copy = random.randint(1, min(3, len(src_labels)))
    copy_indices = random.sample(range(len(src_labels)), n_copy)
    
    for idx in copy_indices:
        cls, x1, y1, x2, y2 = src_labels[idx]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Ensure coordinates are valid
        h, w = src_img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            continue
        
        # Copy object
        obj = src_img[y1:y2, x1:x2]
        
        # Random position in target
        th, tw = tgt_img.shape[:2]
        oh, ow = y2 - y1, x2 - x1
        
        # Ensure object fits in target
        if oh > th or ow > tw:
            continue
        
        ty = random.randint(0, th - oh)
        tx = random.randint(0, tw - ow)
        
        # Paste object
        tgt_img[ty:ty+oh, tx:tx+ow] = obj
        
        # Add new label
        new_label = np.array([[cls, tx, ty, tx + ow, ty + oh]])
        tgt_labels = np.vstack([tgt_labels, new_label]) if len(tgt_labels) else new_label
    
    # Update images and labels
    images[tgt_idx] = tgt_img
    labels[tgt_idx] = tgt_labels
    
    return images, labels


def scale_model(model_config, depth_multiple=1.0, width_multiple=1.0):
    """Scale model configuration using compound scaling.
    
    Args:
        model_config: Model configuration dictionary
        depth_multiple: Depth scaling factor
        width_multiple: Width scaling factor
    
    Returns:
        Scaled model configuration
    """
    scaled_config = model_config.copy()
    
    # Scale depth (number of layers)
    if 'depth_multiple' in scaled_config:
        scaled_config['depth_multiple'] *= depth_multiple
    
    # Scale width (number of channels)
    if 'width_multiple' in scaled_config:
        scaled_config['width_multiple'] *= width_multiple
    
    # Scale backbone and head
    for section in ['backbone', 'head']:
        if section in scaled_config:
            for i, layer in enumerate(scaled_config[section]):
                if len(layer) >= 3:  # [from, number, module, args]
                    # Scale number of repeats
                    if isinstance(layer[1], int):
                        scaled_config[section][i][1] = max(1, int(layer[1] * depth_multiple))
                    
                    # Scale channels in args
                    if len(layer) >= 4 and isinstance(layer[3], (list, tuple)):
                        for j, arg in enumerate(layer[3]):
                            if isinstance(arg, int) and arg > 64:  # Likely channel dimension
                                scaled_config[section][i][3][j] = max(8, int(arg * width_multiple / 8) * 8)
    
    return scaled_config


def get_model_scales(target_latency=None, target_accuracy=None):
    """Get model scaling factors for different model sizes.
    
    Args:
        target_latency: Target inference latency in ms
        target_accuracy: Target mAP accuracy
    
    Returns:
        Dictionary of scaling factors for different model sizes
    """
    # Predefined scaling configurations for different model sizes
    scales = {
        'nano': {'depth_multiple': 0.33, 'width_multiple': 0.25},
        'tiny': {'depth_multiple': 0.33, 'width_multiple': 0.375},
        'small': {'depth_multiple': 0.33, 'width_multiple': 0.50},
        'medium': {'depth_multiple': 0.67, 'width_multiple': 0.75},
        'large': {'depth_multiple': 1.0, 'width_multiple': 1.0},
        'xlarge': {'depth_multiple': 1.33, 'width_multiple': 1.25},
        'huge': {'depth_multiple': 1.67, 'width_multiple': 1.50},
        'giant': {'depth_multiple': 2.0, 'width_multiple': 1.75},
        'xxxl': {'depth_multiple': 2.5, 'width_multiple': 2.0},
    }
    
    if target_latency is not None:
        # Estimate scaling based on target latency
        # This is a simplified model - in practice, you'd use profiling data
        if target_latency < 5:  # Very fast
            return scales['nano']
        elif target_latency < 10:  # Fast
            return scales['tiny']
        elif target_latency < 20:  # Balanced
            return scales['small']
        elif target_latency < 40:  # Accurate
            return scales['medium']
        else:  # Very accurate
            return scales['large']
    
    if target_accuracy is not None:
        # Estimate scaling based on target accuracy
        # This is a simplified model - in practice, you'd use benchmark data
        if target_accuracy < 30:  # Low accuracy
            return scales['nano']
        elif target_accuracy < 35:  # Medium accuracy
            return scales['small']
        elif target_accuracy < 40:  # Good accuracy
            return scales['medium']
        else:  # High accuracy
            return scales['large']
    
    return scales


def normalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD, inplace=False):
    """Applies ImageNet normalization to RGB images in BCHW format, modifying them in-place if specified.

    Example: y = (x - mean) / std
    """
    return TF.normalize(x, mean, std, inplace=inplace)


def denormalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Reverses ImageNet normalization for BCHW format RGB images by applying `x = x * std + mean`."""
    for i in range(3):
        x[:, i] = x[:, i] * std[i] + mean[i]
    return x


def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    """Applies HSV color-space augmentation to an image with random gains for hue, saturation, and value."""
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed


def hist_equalize(im, clahe=True, bgr=False):
    """Equalizes image histogram, with optional CLAHE, for BGR or RGB image with shape (n,m,3) and range 0-255."""
    yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)  # convert YUV image to RGB


def replicate(im, labels):
    """Replicates half of the smallest object labels in an image for data augmentation.

    Returns augmented image and labels.
    """
    h, w = im.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    for i in s.argsort()[: round(s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        im[y1a:y2a, x1a:x2a] = im[y1b:y2b, x1b:x2b]  # im4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return im, labels


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """Resizes and pads image to new_shape with stride-multiple constraints, returns resized image, ratio, padding."""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = round(shape[1] * r), round(shape[0] * r)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def random_perspective(
    im, targets=(), segments=(), degrees=10, translate=0.1, scale=0.1, shear=10, perspective=0.0, border=(0, 0)
):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]
    """Applies random perspective transformation to an image, modifying the image and corresponding labels."""
    height = im.shape[0] + border[0] * 2  # shape(h,w,c)
    width = im.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)