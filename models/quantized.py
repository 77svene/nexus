"""Quantization-Aware Training (QAT) Pipeline for YOLOv5.

This module implements a seamless QAT pipeline for INT8 quantization, enabling 2-4x inference
speedup on edge devices (TensorRT, OpenVINO, TFLite) with minimal accuracy drop. Integrates
with existing YOLOv5 training and export workflows.
"""

import copy
import logging
import torch
import torch.nn as nn
from torch.quantization import (
    DeQuantStub,
    QuantStub,
    convert,
    get_default_qat_qconfig,
    prepare_qat,
    quantize_dynamic,
)
from torch.quantization.quantize_fx import prepare_qat_fx, convert_fx

from models.common import Conv, Bottleneck, C3, SPP, SPPF, Focus, Concat, Detect
from models.yolo import Model
from utils.torch_utils import copy_attr, initialize_weights

LOGGER = logging.getLogger(__name__)


class QuantizedConv(nn.Module):
    """Quantized version of Conv module with fake quantization nodes."""
    
    def __init__(self, conv, qconfig):
        super().__init__()
        self.conv = conv.conv
        self.bn = conv.bn
        self.act = conv.act
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.qconfig = qconfig
        
    def forward(self, x):
        x = self.quant(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.dequant(x)
        return x


class QuantizedBottleneck(nn.Module):
    """Quantized version of Bottleneck module."""
    
    def __init__(self, bottleneck, qconfig):
        super().__init__()
        self.cv1 = QuantizedConv(bottleneck.cv1, qconfig)
        self.cv2 = QuantizedConv(bottleneck.cv2, qconfig)
        self.add = bottleneck.add
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
    def forward(self, x):
        x = self.quant(x)
        out = self.cv2(self.cv1(x))
        if self.add:
            out = out + x
        out = self.dequant(out)
        return out


class QuantizedC3(nn.Module):
    """Quantized version of C3 module."""
    
    def __init__(self, c3, qconfig):
        super().__init__()
        self.cv1 = QuantizedConv(c3.cv1, qconfig)
        self.cv2 = QuantizedConv(c3.cv2, qconfig)
        self.cv3 = QuantizedConv(c3.cv3, qconfig)
        self.m = nn.Sequential(*[QuantizedBottleneck(b, qconfig) for b in c3.m])
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
    def forward(self, x):
        x = self.quant(x)
        out = self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
        out = self.dequant(out)
        return out


class QuantizedSPPF(nn.Module):
    """Quantized version of SPPF module."""
    
    def __init__(self, sppf, qconfig):
        super().__init__()
        self.cv1 = QuantizedConv(sppf.cv1, qconfig)
        self.cv2 = QuantizedConv(sppf.cv2, qconfig)
        self.m = sppf.m
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
    def forward(self, x):
        x = self.quant(x)
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        out = self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))
        out = self.dequant(out)
        return out


class QuantizedDetect(nn.Module):
    """Quantized version of Detect module."""
    
    def __init__(self, detect, qconfig):
        super().__init__()
        self.detect = detect
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
    def forward(self, x):
        x = [self.quant(xi) for xi in x]
        out = self.detect(x)
        if isinstance(out, (list, tuple)):
            out = [self.dequant(o) for o in out]
        else:
            out = self.dequant(out)
        return out


class QATWrapper(nn.Module):
    """Wrapper for YOLOv5 model with QAT support.
    
    Automatically replaces standard modules with quantization-aware versions
    and manages the quantization process.
    """
    
    def __init__(self, model, qconfig=None, quantize_backbone=True, quantize_head=True):
        super().__init__()
        self.model = model
        self.qconfig = qconfig or get_default_qat_qconfig('fbgemm')
        self.quantize_backbone = quantize_backbone
        self.quantize_head = quantize_head
        
        # Quantization stubs for model input/output
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
        # Replace modules with quantized versions
        self._quantize_modules()
        
    def _quantize_modules(self):
        """Replace standard modules with quantized versions."""
        for name, module in self.model.named_modules():
            if isinstance(module, Conv):
                if self._should_quantize_module(name):
                    setattr(self.model, name, QuantizedConv(module, self.qconfig))
            elif isinstance(module, Bottleneck):
                if self._should_quantize_module(name):
                    setattr(self.model, name, QuantizedBottleneck(module, self.qconfig))
            elif isinstance(module, C3):
                if self._should_quantize_module(name):
                    setattr(self.model, name, QuantizedC3(module, self.qconfig))
            elif isinstance(module, SPPF):
                if self._should_quantize_module(name):
                    setattr(self.model, name, QuantizedSPPF(module, self.qconfig))
            elif isinstance(module, Detect):
                if self.quantize_head:
                    setattr(self.model, name, QuantizedDetect(module, self.qconfig))
    
    def _should_quantize_module(self, module_name):
        """Determine if a module should be quantized based on configuration."""
        if 'model.24' in module_name:  # Detect layer
            return self.quantize_head
        else:  # Backbone
            return self.quantize_backbone
    
    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        if isinstance(x, (list, tuple)):
            x = [self.dequant(xi) for xi in x]
        else:
            x = self.dequant(x)
        return x
    
    def fuse_model(self):
        """Fuse Conv-BN layers for better quantization."""
        for module in self.model.modules():
            if isinstance(module, QuantizedConv):
                torch.quantization.fuse_modules(
                    module, 
                    ['conv', 'bn', 'act'], 
                    inplace=True
                )
    
    def prepare_qat(self):
        """Prepare model for quantization-aware training."""
        self.model.train()
        self.fuse_model()
        prepare_qat(self, self.qconfig, inplace=True)
        return self
    
    def convert_to_quantized(self):
        """Convert trained QAT model to fully quantized model."""
        self.model.eval()
        return convert(self, inplace=False)
    
    def export_quantized_onnx(self, file, img_size=(640, 640), batch_size=1, 
                              opset_version=12, simplify=True):
        """Export quantized model to ONNX format."""
        import onnx
        from onnxsim import simplify as onnx_simplify
        
        self.eval()
        img = torch.zeros(batch_size, 3, *img_size).to(next(self.parameters()).device)
        
        # Export to ONNX
        torch.onnx.export(
            self,
            img,
            file,
            opset_version=opset_version,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes={
                'images': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            } if batch_size > 1 else None
        )
        
        # Simplify ONNX model
        if simplify:
            try:
                onnx_model = onnx.load(file)
                onnx_model, check = onnx_simplify(onnx_model)
                assert check, "Simplified ONNX model could not be validated"
                onnx.save(onnx_model, file)
            except Exception as e:
                LOGGER.warning(f'ONNX simplification failed: {e}')
        
        LOGGER.info(f'Quantized ONNX model exported to {file}')
        return file


class QATTrainer:
    """Handles QAT training process with calibration and fine-tuning."""
    
    def __init__(self, model, device, qconfig=None):
        self.model = model
        self.device = device
        self.qconfig = qconfig or get_default_qat_qconfig('fbgemm')
        
    def prepare_for_qat(self, quantize_backbone=True, quantize_head=True):
        """Prepare model for QAT training."""
        # Create quantization-aware model
        qat_model = QATWrapper(
            self.model,
            qconfig=self.qconfig,
            quantize_backbone=quantize_backbone,
            quantize_head=quantize_head
        )
        
        # Prepare for QAT
        qat_model.prepare_qat()
        qat_model.to(self.device)
        
        LOGGER.info("Model prepared for Quantization-Aware Training")
        return qat_model
    
    def calibrate(self, calibration_loader, num_batches=100):
        """Run calibration using representative dataset."""
        self.model.eval()
        LOGGER.info("Starting calibration...")
        
        with torch.no_grad():
            for i, (images, _) in enumerate(calibration_loader):
                if i >= num_batches:
                    break
                images = images.to(self.device)
                self.model(images)
                if i % 10 == 0:
                    LOGGER.info(f"Calibration batch {i}/{num_batches}")
        
        LOGGER.info("Calibration completed")
    
    def train_step(self, images, targets, optimizer, scaler=None):
        """Single training step with QAT."""
        self.model.train()
        images = images.to(self.device)
        targets = targets.to(self.device)
        
        # Forward pass with automatic mixed precision
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = self.model(images)
            loss, loss_items = self.compute_loss(outputs, targets)
        
        # Backward pass
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        optimizer.zero_grad()
        return loss, loss_items
    
    def compute_loss(self, predictions, targets):
        """Compute loss for QAT training."""
        # Use the same loss computation as standard training
        from utils.loss import ComputeLoss
        compute_loss = ComputeLoss(self.model.model)
        return compute_loss(predictions, targets)


class QATConfig:
    """Configuration for QAT training."""
    
    def __init__(self, 
                 quantize_backbone=True,
                 quantize_head=True,
                 qconfig_name='fbgemm',
                 calibration_batches=100,
                 fine_tune_epochs=10,
                 learning_rate=1e-5,
                 per_channel=True,
                 symmetric_weights=True,
                 symmetric_activations=False):
        
        self.quantize_backbone = quantize_backbone
        self.quantize_head = quantize_head
        self.qconfig_name = qconfig_name
        self.calibration_batches = calibration_batches
        self.fine_tune_epochs = fine_tune_epochs
        self.learning_rate = learning_rate
        self.per_channel = per_channel
        self.symmetric_weights = symmetric_weights
        self.symmetric_activations = symmetric_activations
        
        # Get quantization config
        self.qconfig = self._get_qconfig()
    
    def _get_qconfig(self):
        """Get quantization configuration based on settings."""
        from torch.quantization import get_default_qconfig, QConfig
        from torch.quantization.observer import (
            MinMaxObserver,
            PerChannelMinMaxObserver,
            HistogramObserver,
        )
        
        if self.qconfig_name == 'custom':
            # Custom QConfig with specified parameters
            weight_observer = PerChannelMinMaxObserver if self.per_channel else MinMaxObserver
            activation_observer = HistogramObserver
            
            return QConfig(
                activation=activation_observer.with_args(
                    dtype=torch.quint8,
                    qscheme=torch.per_tensor_symmetric if self.symmetric_activations 
                           else torch.per_tensor_affine
                ),
                weight=weight_observer.with_args(
                    dtype=torch.qint8,
                    qscheme=torch.per_channel_symmetric if self.per_channel 
                           else torch.per_tensor_symmetric
                )
            )
        else:
            # Use PyTorch's default config
            return get_default_qconfig(self.qconfig_name)
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {
            'quantize_backbone': self.quantize_backbone,
            'quantize_head': self.quantize_head,
            'qconfig_name': self.qconfig_name,
            'calibration_batches': self.calibration_batches,
            'fine_tune_epochs': self.fine_tune_epochs,
            'learning_rate': self.learning_rate,
            'per_channel': self.per_channel,
            'symmetric_weights': self.symmetric_weights,
            'symmetric_activations': self.symmetric_activations
        }


def create_qat_model(model, config=None, device='cpu'):
    """Create QAT-ready model from YOLOv5 model.
    
    Args:
        model: YOLOv5 model instance
        config: QATConfig instance or None for defaults
        device: Target device
        
    Returns:
        QATWrapper instance ready for training
    """
    if config is None:
        config = QATConfig()
    
    trainer = QATTrainer(model, device, config.qconfig)
    qat_model = trainer.prepare_for_qat(
        quantize_backbone=config.quantize_backbone,
        quantize_head=config.quantize_head
    )
    
    return qat_model


def export_quantized_model(model, file, img_size=(640, 640), format='onnx', **kwargs):
    """Export quantized model to specified format.
    
    Args:
        model: Quantized model (QATWrapper or converted)
        file: Output file path
        img_size: Input image size
        format: Export format ('onnx', 'tflite', 'torchscript')
        **kwargs: Additional export arguments
        
    Returns:
        Path to exported model
    """
    if isinstance(model, QATWrapper):
        # Convert to fully quantized model first
        model = model.convert_to_quantized()
    
    if format == 'onnx':
        return model.export_quantized_onnx(file, img_size, **kwargs)
    elif format == 'tflite':
        return _export_quantized_tflite(model, file, img_size, **kwargs)
    elif format == 'torchscript':
        return _export_quantized_torchscript(model, file, img_size, **kwargs)
    else:
        raise ValueError(f"Unsupported export format: {format}")


def _export_quantized_tflite(model, file, img_size, **kwargs):
    """Export quantized model to TFLite format."""
    import tensorflow as tf
    import onnx
    from onnx_tf.backend import prepare
    
    # First export to ONNX
    onnx_file = file.replace('.tflite', '.onnx')
    model.export_quantized_onnx(onnx_file, img_size)
    
    # Convert ONNX to TensorFlow
    onnx_model = onnx.load(onnx_file)
    tf_rep = prepare(onnx_model)
    tf_model_path = file.replace('.tflite', '_tf')
    tf_rep.export_graph(tf_model_path)
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.int8]
    
    # Representative dataset for quantization
    def representative_dataset():
        for _ in range(100):
            data = torch.randn(1, 3, *img_size)
            yield [data.numpy()]
    
    converter.representative_dataset = representative_dataset
    tflite_model = converter.convert()
    
    with open(file, 'wb') as f:
        f.write(tflite_model)
    
    LOGGER.info(f'Quantized TFLite model exported to {file}')
    return file


def _export_quantized_torchscript(model, file, img_size, **kwargs):
    """Export quantized model to TorchScript format."""
    model.eval()
    img = torch.zeros(1, 3, *img_size)
    
    # Trace the model
    traced = torch.jit.trace(model, img)
    
    # Optimize for mobile
    from torch.utils.mobile_optimizer import optimize_for_mobile
    optimized = optimize_for_mobile(traced)
    
    optimized.save(file)
    LOGGER.info(f'Quantized TorchScript model exported to {file}')
    return file


def benchmark_quantized_model(model, img_size=(640, 640), num_runs=100, warmup=10):
    """Benchmark quantized model performance."""
    import time
    import numpy as np
    
    model.eval()
    device = next(model.parameters()).device
    img = torch.randn(1, 3, *img_size).to(device)
    
    # Warmup
    for _ in range(warmup):
        _ = model(img)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.time()
        _ = model(img)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        times.append(time.time() - start)
    
    avg_time = np.mean(times) * 1000  # Convert to ms
    std_time = np.std(times) * 1000
    
    LOGGER.info(f"Quantized model benchmark: {avg_time:.2f} ± {std_time:.2f} ms")
    return avg_time, std_time


# Integration with existing training pipeline
def integrate_qat_training(opt, device, tb_writer=None):
    """Integrate QAT into existing YOLOv5 training pipeline.
    
    Args:
        opt: Training options from train.py
        device: Torch device
        tb_writer: TensorBoard writer
        
    Returns:
        QAT model and trainer
    """
    from models.yolo import Model
    from utils.torch_utils import de_parallel
    
    # Load model
    model = Model(opt.cfg, ch=3, nc=opt.data['nc']).to(device)
    
    # Create QAT configuration
    qat_config = QATConfig(
        quantize_backbone=opt.quantize_backbone if hasattr(opt, 'quantize_backbone') else True,
        quantize_head=opt.quantize_head if hasattr(opt, 'quantize_head') else True,
        qconfig_name=opt.qconfig if hasattr(opt, 'qconfig') else 'fbgemm',
        calibration_batches=opt.calibration_batches if hasattr(opt, 'calibration_batches') else 100,
        fine_tune_epochs=opt.qat_epochs if hasattr(opt, 'qat_epochs') else 10
    )
    
    # Create QAT model
    qat_model = create_qat_model(model, qat_config, device)
    
    # Create trainer
    qat_trainer = QATTrainer(de_parallel(qat_model), device, qat_config.qconfig)
    
    LOGGER.info(f"QAT integration complete. Config: {qat_config.to_dict()}")
    
    return qat_model, qat_trainer, qat_config


# Utility functions for compatibility with existing code
def is_quantized_module(module):
    """Check if module is quantized."""
    return isinstance(module, (QuantizedConv, QuantizedBottleneck, 
                              QuantizedC3, QuantizedSPPF, QuantizedDetect))


def get_quantizable_modules(model):
    """Get list of modules that can be quantized."""
    quantizable = []
    for name, module in model.named_modules():
        if isinstance(module, (Conv, Bottleneck, C3, SPPF, SPP, Focus, Detect)):
            quantizable.append((name, module))
    return quantizable


def print_quantization_summary(model):
    """Print summary of quantization status."""
    total_params = 0
    quantized_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
    
    for module in model.modules():
        if is_quantized_module(module):
            for param in module.parameters():
                quantized_params += param.numel()
    
    quant_ratio = quantized_params / total_params * 100 if total_params > 0 else 0
    
    LOGGER.info(f"Quantization Summary:")
    LOGGER.info(f"  Total parameters: {total_params:,}")
    LOGGER.info(f"  Quantized parameters: {quantized_params:,}")
    LOGGER.info(f"  Quantization ratio: {quant_ratio:.1f}%")
    
    return {
        'total_params': total_params,
        'quantized_params': quantized_params,
        'quant_ratio': quant_ratio
    }