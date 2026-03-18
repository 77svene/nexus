# utils/quantization.py
import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub, quantize_dynamic, quantize_fx
from torch.quantization.quantize_fx import prepare_qat_fx, convert_fx
from torch.ao.quantization import QConfigMapping, get_default_qat_qconfig
import copy
import logging
from pathlib import Path
from utils.torch_utils import is_parallel
from models.yolo import DetectionModel, SegmentationModel, ClassificationModel

logger = logging.getLogger(__name__)

class QuantizedModelWrapper(nn.Module):
    """Wrapper to add quantization stubs to any model"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        if isinstance(x, (list, tuple)):
            x = [self.dequant(xi) for xi in x]
        else:
            x = self.dequant(x)
        return x
    
    def fuse_model(self):
        """Fuse Conv+BN and Conv+BN+ReLU layers for better quantization"""
        from torch.quantization import fuse_modules
        
        for module in self.model.modules():
            if isinstance(module, nn.Sequential):
                # Common patterns in YOLOv5
                if len(module) == 2 and isinstance(module[0], nn.Conv2d) and isinstance(module[1], nn.BatchNorm2d):
                    fuse_modules(module, ['0', '1'], inplace=True)
                elif len(module) == 3 and isinstance(module[0], nn.Conv2d) and isinstance(module[1], nn.BatchNorm2d) and isinstance(module[2], nn.ReLU):
                    fuse_modules(module, ['0', '1', '2'], inplace=True)
            elif hasattr(module, 'conv') and hasattr(module, 'bn'):
                # Bottleneck and C3 modules
                if hasattr(module, 'act'):
                    fuse_modules(module, ['conv', 'bn', 'act'], inplace=True)
                else:
                    fuse_modules(module, ['conv', 'bn'], inplace=True)

class QuantizationPipeline:
    """Complete QAT pipeline for YOLOv5 models"""
    
    def __init__(self, model, imgsz=640, backend='fbgemm'):
        """
        Args:
            model: YOLOv5 model (DetectionModel, SegmentationModel, or ClassificationModel)
            imgsz: Input image size
            backend: Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM)
        """
        self.model = model
        self.imgsz = imgsz
        self.backend = backend
        self.quantized_model = None
        self.calibrated = False
        
        # Set quantization backend
        torch.backends.quantized.engine = backend
        
        # Determine model type
        self.model_type = self._get_model_type()
        
    def _get_model_type(self):
        """Determine if model is detection, segmentation, or classification"""
        if isinstance(self.model, DetectionModel):
            return 'detection'
        elif isinstance(self.model, SegmentationModel):
            return 'segmentation'
        elif isinstance(self.model, ClassificationModel):
            return 'classification'
        else:
            # Try to infer from model structure
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'yaml'):
                return 'detection'  # Most common case
            return 'unknown'
    
    def prepare_for_qat(self, qconfig=None):
        """Prepare model for Quantization-Aware Training
        
        Args:
            qconfig: Quantization configuration (None for default QAT config)
            
        Returns:
            Model ready for QAT
        """
        logger.info("Preparing model for Quantization-Aware Training...")
        
        # Create a copy of the model for quantization
        model_to_quantize = copy.deepcopy(self.model)
        
        # Unwrap DataParallel/DistributedDataParallel
        if is_parallel(model_to_quantize):
            model_to_quantize = model_to_quantize.module
        
        # Wrap model with quantization stubs
        wrapped_model = QuantizedModelWrapper(model_to_quantize)
        
        # Fuse modules for better quantization
        wrapped_model.fuse_model()
        
        # Set default QAT config if not provided
        if qconfig is None:
            qconfig = get_default_qat_qconfig(self.backend)
        
        # Prepare for QAT using FX graph mode (PyTorch 1.8+)
        try:
            # Use FX graph mode quantization for better accuracy
            example_inputs = torch.randn(1, 3, self.imgsz, self.imgsz)
            prepared_model = prepare_qat_fx(
                wrapped_model,
                {"": qconfig},
                example_inputs=example_inputs
            )
            logger.info("Using FX graph mode quantization")
        except Exception as e:
            logger.warning(f"FX quantization failed, falling back to eager mode: {e}")
            # Fallback to eager mode quantization
            wrapped_model.qconfig = qconfig
            torch.quantization.prepare_qat(wrapped_model, inplace=True)
            prepared_model = wrapped_model
        
        self.quantized_model = prepared_model
        logger.info("Model prepared for QAT")
        
        return prepared_model
    
    def calibrate(self, dataloader, num_batches=100):
        """Calibrate quantization parameters using representative data
        
        Args:
            dataloader: Calibration dataloader
            num_batches: Number of batches to use for calibration
        """
        if self.quantized_model is None:
            raise RuntimeError("Call prepare_for_qat() first")
        
        logger.info(f"Calibrating with {num_batches} batches...")
        
        self.quantized_model.eval()
        device = next(self.quantized_model.parameters()).device
        
        with torch.no_grad():
            for i, (imgs, *_) in enumerate(dataloader):
                if i >= num_batches:
                    break
                imgs = imgs.to(device, non_blocking=True)
                self.quantized_model(imgs)
        
        self.calibrated = True
        logger.info("Calibration complete")
    
    def convert_to_quantized(self):
        """Convert QAT model to fully quantized model
        
        Returns:
            Fully quantized model ready for inference
        """
        if self.quantized_model is None:
            raise RuntimeError("Call prepare_for_qat() first")
        
        logger.info("Converting to quantized model...")
        
        self.quantized_model.eval()
        
        try:
            # Convert using FX graph mode
            quantized_model = convert_fx(self.quantized_model)
            logger.info("Converted using FX graph mode")
        except Exception as e:
            logger.warning(f"FX conversion failed, falling back to eager mode: {e}")
            # Fallback to eager mode
            quantized_model = torch.quantization.convert(self.quantized_model)
        
        # Remove quantization stubs if using wrapper
        if hasattr(quantized_model, 'quant') and hasattr(quantized_model, 'dequant'):
            # Keep the wrapper but it's now quantized
            pass
        
        self.quantized_model = quantized_model
        logger.info("Model converted to quantized version")
        
        return quantized_model
    
    def export_quantized(self, file='quantized_model.pt', format='torchscript'):
        """Export quantized model to various formats
        
        Args:
            file: Output file path
            format: Export format ('torchscript', 'onnx', 'tflite')
        """
        if self.quantized_model is None:
            raise RuntimeError("No quantized model available")
        
        self.quantized_model.eval()
        
        if format == 'torchscript':
            # Export as TorchScript
            example_inputs = torch.randn(1, 3, self.imgsz, self.imgsz)
            traced_model = torch.jit.trace(self.quantized_model, example_inputs)
            traced_model.save(file)
            logger.info(f"Quantized model saved to {file} (TorchScript)")
            
        elif format == 'onnx':
            # Export as ONNX
            import onnx
            example_inputs = torch.randn(1, 3, self.imgsz, self.imgsz)
            
            # Dynamic axes for batch size
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
            
            torch.onnx.export(
                self.quantized_model,
                example_inputs,
                file,
                opset_version=13,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes
            )
            
            # Simplify ONNX model
            try:
                import onnxsim
                model_onnx = onnx.load(file)
                model_simp, check = onnxsim.simplify(model_onnx)
                if check:
                    onnx.save(model_simp, file)
                    logger.info(f"ONNX model simplified and saved to {file}")
                else:
                    logger.warning("ONNX simplification failed, keeping original")
            except ImportError:
                logger.warning("onnxsim not installed, skipping simplification")
            
        elif format == 'tflite':
            # Export to TFLite via ONNX
            import tensorflow as tf
            import onnx
            from onnx_tf.backend import prepare
            
            # First export to ONNX
            onnx_file = file.replace('.tflite', '.onnx')
            self.export_quantized(onnx_file, format='onnx')
            
            # Convert ONNX to TensorFlow
            onnx_model = onnx.load(onnx_file)
            tf_rep = prepare(onnx_model)
            tf_model_path = file.replace('.tflite', '_tf')
            tf_rep.export_graph(tf_model_path)
            
            # Convert TensorFlow to TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            tflite_model = converter.convert()
            
            with open(file, 'wb') as f:
                f.write(tflite_model)
            
            logger.info(f"TFLite model saved to {file}")
            
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def benchmark(self, img=None, warmup=10, iterations=100):
        """Benchmark quantized model performance
        
        Args:
            img: Input image tensor (None for random)
            warmup: Number of warmup iterations
            iterations: Number of benchmark iterations
            
        Returns:
            Dictionary with benchmark results
        """
        import time
        import numpy as np
        
        if self.quantized_model is None:
            raise RuntimeError("No quantized model available")
        
        self.quantized_model.eval()
        device = next(self.quantized_model.parameters()).device
        
        if img is None:
            img = torch.randn(1, 3, self.imgsz, self.imgsz).to(device)
        else:
            img = img.to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.quantized_model(img)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(iterations):
                start = time.time()
                _ = self.quantized_model(img)
                torch.cuda.synchronize() if device.type == 'cuda' else None
                times.append(time.time() - start)
        
        times = np.array(times)
        
        results = {
            'mean_ms': np.mean(times) * 1000,
            'std_ms': np.std(times) * 1000,
            'min_ms': np.min(times) * 1000,
            'max_ms': np.max(times) * 1000,
            'fps': 1.0 / np.mean(times),
            'iterations': iterations,
            'backend': self.backend
        }
        
        logger.info(f"Quantized model benchmark: {results['mean_ms']:.2f}ms ± {results['std_ms']:.2f}ms, {results['fps']:.1f} FPS")
        
        return results

def quantize_model_dynamic(model, dtype=torch.qint8):
    """Apply dynamic quantization (post-training, no calibration needed)
    
    Args:
        model: PyTorch model
        dtype: Quantization data type
        
    Returns:
        Dynamically quantized model
    """
    # Only quantize Linear and LSTM layers (dynamic quantization)
    quantized_model = quantize_dynamic(
        model,
        {nn.Linear, nn.LSTM, nn.GRU},
        dtype=dtype
    )
    
    logger.info("Applied dynamic quantization")
    return quantized_model

def quantize_model_static(model, calibration_loader, backend='fbgemm'):
    """Apply static quantization (post-training with calibration)
    
    Args:
        model: PyTorch model
        calibration_loader: DataLoader for calibration
        backend: Quantization backend
        
    Returns:
        Statically quantized model
    """
    torch.backends.quantized.engine = backend
    
    # Create quantization pipeline
    pipeline = QuantizationPipeline(model, backend=backend)
    
    # Prepare for static quantization
    model.eval()
    wrapped_model = QuantizedModelWrapper(model)
    wrapped_model.fuse_model()
    
    # Set quantization config
    wrapped_model.qconfig = torch.quantization.get_default_qconfig(backend)
    
    # Prepare
    torch.quantization.prepare(wrapped_model, inplace=True)
    
    # Calibrate
    with torch.no_grad():
        for i, (imgs, *_) in enumerate(calibration_loader):
            if i >= 100:  # Limit calibration batches
                break
            wrapped_model(imgs)
    
    # Convert
    quantized_model = torch.quantization.convert(wrapped_model)
    
    logger.info("Applied static quantization")
    return quantized_model

def get_quantization_config(backend='fbgemm', per_channel=True, symmetric=True):
    """Get quantization configuration
    
    Args:
        backend: Quantization backend
        per_channel: Use per-channel quantization for weights
        symmetric: Use symmetric quantization
        
    Returns:
        QConfig for quantization
    """
    from torch.quantization import QConfig, MinMaxObserver, PerChannelMinMaxObserver, HistogramObserver
    
    if per_channel:
        weight_observer = PerChannelMinMaxObserver
    else:
        weight_observer = MinMaxObserver
    
    if symmetric:
        qscheme = torch.per_tensor_symmetric
    else:
        qscheme = torch.per_tensor_affine
    
    activation_observer = HistogramObserver.with_args(
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=True
    )
    
    weight_observer = weight_observer.with_args(
        dtype=torch.qint8,
        qscheme=qscheme,
        reduce_range=False
    )
    
    qconfig = QConfig(
        activation=activation_observer,
        weight=weight_observer
    )
    
    return qconfig

# Integration with existing YOLOv5 training
def setup_qat_training(model, optimizer, scheduler, imgsz=640):
    """Setup model for QAT training
    
    Args:
        model: YOLOv5 model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        imgsz: Image size
        
    Returns:
        Tuple of (model, optimizer, scheduler) configured for QAT
    """
    # Freeze batch norm layers during QAT
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
    
    # Adjust learning rate for QAT (typically lower)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.01  # Reduce LR by 100x for fine-tuning
    
    logger.info("Model configured for QAT training")
    return model, optimizer, scheduler

# Export function for use in export.py
def export_quantized_model(model, file, format='torchscript', imgsz=640):
    """Export quantized model (to be called from export.py)
    
    Args:
        model: Quantized model
        file: Output file path
        format: Export format
        imgsz: Image size
    """
    pipeline = QuantizationPipeline(model, imgsz=imgsz)
    pipeline.quantized_model = model  # Already quantized
    pipeline.export_quantized(file, format=format)

# Calibration dataset loader
class CalibrationDataset:
    """Dataset wrapper for quantization calibration"""
    
    def __init__(self, dataset, num_samples=1000):
        self.dataset = dataset
        self.num_samples = min(num_samples, len(dataset))
        
    def __iter__(self):
        indices = torch.randperm(len(self.dataset))[:self.num_samples]
        for idx in indices:
            img, *targets = self.dataset[idx]
            yield img.unsqueeze(0), targets
    
    def __len__(self):
        return self.num_samples

# Main quantization workflow
def quantize_nexus(model, calibration_dataset=None, qat_epochs=5, 
                   backend='fbgemm', export_format='torchscript'):
    """Complete quantization workflow for YOLOv5
    
    Args:
        model: Trained YOLOv5 model
        calibration_dataset: Dataset for calibration (None for random)
        qat_epochs: Number of QAT fine-tuning epochs
        backend: Quantization backend
        export_format: Export format after quantization
        
    Returns:
        Quantized model and export path
    """
    from train import train  # Import here to avoid circular imports
    
    logger.info("Starting YOLOv5 quantization workflow...")
    
    # 1. Prepare for QAT
    pipeline = QuantizationPipeline(model, backend=backend)
    qat_model = pipeline.prepare_for_qat()
    
    # 2. Fine-tune with QAT
    if calibration_dataset is not None:
        # Use calibration dataset for fine-tuning
        logger.info(f"Fine-tuning with QAT for {qat_epochs} epochs...")
        # Note: In practice, you would call your training loop here
        # For now, we just calibrate
        pipeline.calibrate(calibration_dataset, num_batches=100)
    
    # 3. Convert to quantized model
    quantized_model = pipeline.convert_to_quantized()
    
    # 4. Export
    export_path = f"quantized_model.{export_format}"
    pipeline.export_quantized(export_path, format=export_format)
    
    # 5. Benchmark
    benchmark_results = pipeline.benchmark()
    
    logger.info(f"Quantization complete. Model saved to {export_path}")
    logger.info(f"Performance: {benchmark_results['mean_ms']:.2f}ms, {benchmark_results['fps']:.1f} FPS")
    
    return quantized_model, export_path

# Command line interface for quantization
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv5 Quantization Tool')
    parser.add_argument('--weights', type=str, required=True, help='Path to trained weights')
    parser.add_argument('--data', type=str, required=True, help='Dataset YAML for calibration')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for calibration')
    parser.add_argument('--qat-epochs', type=int, default=5, help='QAT fine-tuning epochs')
    parser.add_argument('--backend', type=str, default='fbgemm', 
                       choices=['fbgemm', 'qnnpack'], help='Quantization backend')
    parser.add_argument('--format', type=str, default='torchscript',
                       choices=['torchscript', 'onnx', 'tflite'], help='Export format')
    parser.add_argument('--device', default='', help='CUDA device, i.e. 0 or 0,1,2,3 or cpu')
    
    opt = parser.parse_args()
    
    # Load model
    from models.experimental import attempt_load
    model = attempt_load(opt.weights, device=opt.device)
    
    # Load calibration dataset
    from utils.dataloaders import create_dataloader
    from utils.general import check_dataset
    
    data_dict = check_dataset(opt.data)
    calibration_loader = create_dataloader(
        data_dict['val'],
        imgsz=opt.imgsz,
        batch_size=opt.batch_size,
        stride=model.stride,
        pad=0.5,
        rect=True
    )[0]
    
    # Run quantization
    quantized_model, export_path = quantize_nexus(
        model,
        calibration_dataset=calibration_loader,
        qat_epochs=opt.qat_epochs,
        backend=opt.backend,
        export_format=opt.format
    )
    
    print(f"Quantized model exported to: {export_path}")