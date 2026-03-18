# utils/dynamic_shapes.py
# Dynamic Shape Inference & ONNX Export Utilities for YOLOv5
# Enables fully dynamic input shapes for ONNX and TensorRT exports

import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
from typing import Dict, List, Optional, Tuple, Union
import warnings
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DynamicShapeHandler:
    """
    Handler for dynamic shape operations in YOLOv5 models.
    Enables variable batch sizes, resolutions, and video streams without re-exporting.
    """
    
    @staticmethod
    def get_dynamic_axes(
        batch_dynamic: bool = True,
        height_dynamic: bool = True,
        width_dynamic: bool = True,
        input_names: List[str] = None,
        output_names: List[str] = None
    ) -> Dict[str, Dict[int, str]]:
        """
        Generate dynamic axes dictionary for ONNX export.
        
        Args:
            batch_dynamic: Allow dynamic batch size
            height_dynamic: Allow dynamic height dimension
            width_dynamic: Allow dynamic width dimension
            input_names: List of input tensor names
            output_names: List of output tensor names
            
        Returns:
            Dictionary mapping tensor names to dynamic axes specifications
        """
        if input_names is None:
            input_names = ['images']
        if output_names is None:
            output_names = ['output']
        
        dynamic_axes = {}
        
        # Input dynamic axes
        for name in input_names:
            axes = {}
            if batch_dynamic:
                axes[0] = 'batch_size'
            if height_dynamic:
                axes[2] = 'height'
            if width_dynamic:
                axes[3] = 'width'
            if axes:
                dynamic_axes[name] = axes
        
        # Output dynamic axes (typically batch and detections)
        for name in output_names:
            axes = {}
            if batch_dynamic:
                axes[0] = 'batch_size'
            # Output may have additional dynamic dimensions
            if 'output' in name.lower() or 'detections' in name.lower():
                axes[1] = 'num_detections'
            if axes:
                dynamic_axes[name] = axes
        
        return dynamic_axes
    
    @staticmethod
    def validate_onnx_dynamic_shapes(
        onnx_path: str,
        test_shapes: List[Tuple[int, int, int]] = None
    ) -> bool:
        """
        Validate that an ONNX model supports dynamic shapes.
        
        Args:
            onnx_path: Path to ONNX model file
            test_shapes: List of (batch, height, width) tuples to test
            
        Returns:
            True if model supports dynamic shapes, False otherwise
        """
        if test_shapes is None:
            test_shapes = [
                (1, 640, 640),
                (2, 320, 320),
                (4, 1280, 1280)
            ]
        
        try:
            # Load ONNX model
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
            
            # Check for dynamic dimensions in graph inputs
            graph = model.graph
            dynamic_found = False
            
            for input_tensor in graph.input:
                tensor_type = input_tensor.type.tensor_type
                if tensor_type.HasField('shape'):
                    for dim in tensor_type.shape.dim:
                        if dim.HasField('dim_param'):  # Dynamic dimension
                            dynamic_found = True
                            logger.info(f"Found dynamic dimension '{dim.dim_param}' in input '{input_tensor.name}'")
            
            if not dynamic_found:
                logger.warning("No dynamic dimensions found in ONNX model")
                return False
            
            # Test with ONNX Runtime
            try:
                session = ort.InferenceSession(onnx_path)
                input_name = session.get_inputs()[0].name
                input_shape = session.get_inputs()[0].shape
                
                # Check if any dimension is dynamic (string or -1)
                is_dynamic = any(isinstance(dim, str) or dim == -1 for dim in input_shape)
                
                if not is_dynamic:
                    logger.warning(f"Input shape {input_shape} appears static")
                    return False
                
                # Test inference with different shapes
                for batch, height, width in test_shapes:
                    try:
                        # Create random input
                        test_input = np.random.randn(batch, 3, height, width).astype(np.float32)
                        
                        # Run inference
                        outputs = session.run(None, {input_name: test_input})
                        logger.info(f"✓ Successfully tested shape: batch={batch}, height={height}, width={width}")
                        
                    except Exception as e:
                        logger.error(f"✗ Failed testing shape: batch={batch}, height={height}, width={width}")
                        logger.error(f"Error: {str(e)}")
                        return False
                
                return True
                
            except Exception as e:
                logger.error(f"ONNX Runtime validation failed: {str(e)}")
                return False
                
        except Exception as e:
            logger.error(f"ONNX model validation failed: {str(e)}")
            return False
    
    @staticmethod
    def export_with_dynamic_shapes(
        model: nn.Module,
        dummy_input: torch.Tensor,
        output_path: str,
        dynamic_batch: bool = True,
        dynamic_height: bool = True,
        dynamic_width: bool = True,
        opset_version: int = 12,
        **kwargs
    ) -> None:
        """
        Export PyTorch model to ONNX with dynamic shapes.
        
        Args:
            model: PyTorch model
            dummy_input: Example input tensor
            output_path: Path to save ONNX model
            dynamic_batch: Enable dynamic batch dimension
            dynamic_height: Enable dynamic height dimension
            dynamic_width: Enable dynamic width dimension
            opset_version: ONNX opset version
            **kwargs: Additional arguments for torch.onnx.export
        """
        # Set model to evaluation mode
        model.eval()
        
        # Get dynamic axes
        dynamic_axes = DynamicShapeHandler.get_dynamic_axes(
            batch_dynamic=dynamic_batch,
            height_dynamic=dynamic_height,
            width_dynamic=dynamic_width
        )
        
        # Default export arguments
        export_args = {
            'input_names': ['images'],
            'output_names': ['output'],
            'dynamic_axes': dynamic_axes,
            'opset_version': opset_version,
            'do_constant_folding': True,
            'export_params': True,
            **kwargs
        }
        
        # Export model
        try:
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                **export_args
            )
            logger.info(f"Successfully exported model with dynamic shapes to {output_path}")
            
            # Validate exported model
            if DynamicShapeHandler.validate_onnx_dynamic_shapes(output_path):
                logger.info("Dynamic shape validation passed")
            else:
                logger.warning("Dynamic shape validation failed")
                
        except Exception as e:
            logger.error(f"Export failed: {str(e)}")
            raise


class DynamicDetect(nn.Module):
    """
    Modified Detect layer that handles dynamic input shapes.
    Compatible with both static and dynamic batch/height/width dimensions.
    """
    
    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        """
        Initialize dynamic detection layer.
        
        Args:
            nc: Number of classes
            anchors: Anchor box sizes
            ch: Channel sizes from backbone
            inplace: Use inplace operations
        """
        super().__init__()
        self.nc = nc
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl
        self.anchor_grid = [torch.zeros(1)] * self.nl
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
        self.inplace = inplace
        
    def forward(self, x):
        """
        Forward pass with dynamic shape handling.
        
        Args:
            x: List of feature maps from backbone
            
        Returns:
            List of detection outputs
        """
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            
            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid_dynamic(nx, ny, i)
                
                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for SSD compatibility
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy = (xy * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))
        
        return x if self.training else (torch.cat(z, 1), x)
    
    def _make_grid_dynamic(self, nx: int, ny: int, i: int):
        """
        Create anchor grid with dynamic dimensions.
        
        Args:
            nx: Number of grid cells in x direction
            ny: Number of grid cells in y direction
            i: Layer index
            
        Returns:
            Tuple of (grid, anchor_grid)
        """
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        
        # Create grid
        yv, xv = torch.meshgrid([torch.arange(ny, device=d, dtype=t),
                                 torch.arange(nx, device=d, dtype=t)], indexing='ij')
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        
        # Create anchor grid
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        
        return grid, anchor_grid


class DynamicNMS:
    """
    Dynamic Non-Maximum Suppression that handles variable batch sizes.
    """
    
    @staticmethod
    def non_max_suppression_dynamic(
        prediction: torch.Tensor,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        classes: Optional[List[int]] = None,
        agnostic: bool = False,
        multi_label: bool = False,
        labels: List[torch.Tensor] = None,
        max_det: int = 300,
        max_nms: int = 30000
    ) -> List[torch.Tensor]:
        """
        Dynamic NMS supporting variable batch sizes.
        
        Args:
            prediction: Model output tensor (batch, detections, 5 + num_classes)
            conf_thres: Confidence threshold
            iou_thres: IoU threshold for NMS
            classes: Filter by class indices
            agnostic: Class-agnostic NMS
            multi_label: Allow multiple labels per box
            labels: Additional labels
            max_det: Maximum detections per image
            max_nms: Maximum boxes for torchvision.ops.nms
            
        Returns:
            List of detections per image (n, 6) - (x1, y1, x2, y2, conf, cls)
        """
        import torchvision
        
        # Check if prediction has batch dimension
        if prediction.dim() == 3:
            bs = prediction.shape[0]  # batch size
        else:
            bs = 1
            prediction = prediction.unsqueeze(0)
        
        nc = prediction.shape[2] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates
        
        # Settings
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        time_limit = 10.0  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS
        
        output = [torch.zeros((0, 6), device=prediction.device)] * bs
        
        for xi in range(bs):
            x = prediction[xi]
            
            # Apply constraints
            x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence
            
            # If none remain process next image
            if not x.shape[0]:
                continue
            
            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
            
            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = xywh2xyxy(x[:, :4])
            
            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            else:  # best class only
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
            
            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
            
            # Apply finite constraint
            # if not torch.isfinite(x).all():
            #     x = x[torch.isfinite(x).all(1)]
            
            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
            
            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            
            # Dynamic NMS - handle variable number of boxes
            try:
                i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
                if i.shape[0] > max_det:  # limit detections
                    i = i[:max_det]
                
                output[xi] = x[i]
                
            except Exception as e:
                logger.warning(f"NMS failed for batch {xi}: {str(e)}")
                # Fallback: return top-k detections
                if n > max_det:
                    _, topk_indices = scores.topk(max_det)
                    output[xi] = x[topk_indices]
                else:
                    output[xi] = x
        
        return output


def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding box coordinates from (x, y, width, height) to (x1, y1, x2, y2).
    
    Args:
        x: Bounding box tensor in (x, y, w, h) format
        
    Returns:
        Bounding box tensor in (x1, y1, x2, y2) format
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


class DynamicShapeValidator:
    """
    Validator for dynamic shape compatibility across different backends.
    """
    
    SUPPORTED_BACKENDS = ['onnxruntime', 'tensorrt', 'openvino']
    
    @staticmethod
    def validate_model_compatibility(
        model_path: str,
        backend: str = 'onnxruntime',
        test_shapes: Optional[List[Tuple[int, int, int]]] = None
    ) -> Dict[str, bool]:
        """
        Validate model compatibility with dynamic shapes for specified backend.
        
        Args:
            model_path: Path to model file
            backend: Target backend ('onnxruntime', 'tensorrt', 'openvino')
            test_shapes: List of (batch, height, width) tuples to test
            
        Returns:
            Dictionary with validation results
        """
        if backend not in DynamicShapeValidator.SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported backend: {backend}. Supported: {DynamicShapeValidator.SUPPORTED_BACKENDS}")
        
        if test_shapes is None:
            test_shapes = [
                (1, 640, 640),
                (2, 320, 320),
                (4, 1280, 1280),
                (8, 1920, 1080)
            ]
        
        results = {
            'backend': backend,
            'model_path': model_path,
            'dynamic_shapes_supported': False,
            'tested_shapes': {},
            'errors': []
        }
        
        try:
            if backend == 'onnxruntime':
                results = DynamicShapeValidator._validate_onnxruntime(
                    model_path, test_shapes, results
                )
            elif backend == 'tensorrt':
                results = DynamicShapeValidator._validate_tensorrt(
                    model_path, test_shapes, results
                )
            elif backend == 'openvino':
                results = DynamicShapeValidator._validate_openvino(
                    model_path, test_shapes, results
                )
                
        except Exception as e:
            results['errors'].append(str(e))
            logger.error(f"Validation failed: {str(e)}")
        
        return results
    
    @staticmethod
    def _validate_onnxruntime(
        model_path: str,
        test_shapes: List[Tuple[int, int, int]],
        results: Dict
    ) -> Dict:
        """Validate ONNX Runtime compatibility."""
        try:
            import onnxruntime as ort
            
            # Create session
            session = ort.InferenceSession(model_path)
            input_name = session.get_inputs()[0].name
            input_shape = session.get_inputs()[0].shape
            
            # Check for dynamic dimensions
            dynamic_dims = []
            for i, dim in enumerate(input_shape):
                if isinstance(dim, str) or dim == -1:
                    dynamic_dims.append(i)
            
            if not dynamic_dims:
                results['errors'].append("No dynamic dimensions found in model")
                return results
            
            results['dynamic_shapes_supported'] = True
            results['dynamic_dimensions'] = dynamic_dims
            
            # Test inference
            for batch, height, width in test_shapes:
                try:
                    test_input = np.random.randn(batch, 3, height, width).astype(np.float32)
                    outputs = session.run(None, {input_name: test_input})
                    results['tested_shapes'][f'{batch}x{height}x{width}'] = True
                except Exception as e:
                    results['tested_shapes'][f'{batch}x{height}x{width}'] = False
                    results['errors'].append(f"Shape {batch}x{height}x{width} failed: {str(e)}")
            
        except ImportError:
            results['errors'].append("onnxruntime not installed")
        except Exception as e:
            results['errors'].append(f"ONNX Runtime validation error: {str(e)}")
        
        return results
    
    @staticmethod
    def _validate_tensorrt(
        model_path: str,
        test_shapes: List[Tuple[int, int, int]],
        results: Dict
    ) -> Dict:
        """Validate TensorRT compatibility."""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            
            # Check if model is TensorRT engine or ONNX
            if model_path.endswith('.engine'):
                with open(model_path, 'rb') as f:
                    engine_data = f.read()
                runtime = trt.Runtime(TRT_LOGGER)
                engine = runtime.deserialize_cuda_engine(engine_data)
                
                # Check input dimensions
                input_name = engine.get_binding_name(0)
                input_shape = engine.get_binding_shape(0)
                
                # TensorRT uses -1 for dynamic dimensions
                dynamic_dims = [i for i, dim in enumerate(input_shape) if dim == -1]
                
                if not dynamic_dims:
                    results['errors'].append("No dynamic dimensions found in TensorRT engine")
                    return results
                
                results['dynamic_shapes_supported'] = True
                results['dynamic_dimensions'] = dynamic_dims
                
            else:
                results['errors'].append("TensorRT validation requires .engine file")
                return results
                
        except ImportError:
            results['errors'].append("tensorrt not installed")
        except Exception as e:
            results['errors'].append(f"TensorRT validation error: {str(e)}")
        
        return results
    
    @staticmethod
    def _validate_openvino(
        model_path: str,
        test_shapes: List[Tuple[int, int, int]],
        results: Dict
    ) -> Dict:
        """Validate OpenVINO compatibility."""
        try:
            from openvino.runtime import Core
            
            ie = Core()
            
            # Read model
            if model_path.endswith('.xml'):
                model = ie.read_model(model_path)
            else:
                results['errors'].append("OpenVINO validation requires .xml file")
                return results
            
            # Check input shape
            input_layer = model.input(0)
            input_shape = input_layer.shape
            
            # OpenVINO uses -1 or ? for dynamic dimensions
            dynamic_dims = []
            for i, dim in enumerate(input_shape):
                if dim == -1 or str(dim) == '?':
                    dynamic_dims.append(i)
            
            if not dynamic_dims:
                results['errors'].append("No dynamic dimensions found in OpenVINO model")
                return results
            
            results['dynamic_shapes_supported'] = True
            results['dynamic_dimensions'] = dynamic_dims
            
            # Test with dynamic shapes
            compiled_model = ie.compile_model(model, "CPU")
            infer_request = compiled_model.create_infer_request()
            
            for batch, height, width in test_shapes:
                try:
                    # Create input tensor
                    input_tensor = np.random.randn(batch, 3, height, width).astype(np.float32)
                    
                    # Infer
                    infer_request.infer([input_tensor])
                    results['tested_shapes'][f'{batch}x{height}x{width}'] = True
                except Exception as e:
                    results['tested_shapes'][f'{batch}x{height}x{width}'] = False
                    results['errors'].append(f"Shape {batch}x{height}x{width} failed: {str(e)}")
            
        except ImportError:
            results['errors'].append("openvino not installed")
        except Exception as e:
            results['errors'].append(f"OpenVINO validation error: {str(e)}")
        
        return results


# Integration with existing YOLOv5 modules
def patch_yolo_for_dynamic_shapes():
    """
    Patch YOLOv5 modules to support dynamic shapes.
    Call this function to enable dynamic shape support in existing models.
    """
    from models.yolo import Detect
    from models.common import NMS
    
    # Replace Detect layer with DynamicDetect
    original_detect = Detect
    
    class PatchedDetect(DynamicDetect):
        """Patched Detect layer with dynamic shape support."""
        pass
    
    # Monkey-patch the Detect class
    import models.yolo
    models.yolo.Detect = PatchedDetect
    
    logger.info("YOLOv5 modules patched for dynamic shape support")


# Utility functions for export.py integration
def export_onnx_with_dynamic_shapes(
    model: nn.Module,
    im: torch.Tensor,
    file: Union[str, Path],
    dynamic: bool = True,
    dynamic_batch: bool = True,
    dynamic_height: bool = True,
    dynamic_width: bool = True,
    **kwargs
) -> None:
    """
    Export model to ONNX with dynamic shapes.
    Drop-in replacement for export functionality in export.py.
    
    Args:
        model: PyTorch model
        im: Example input tensor
        file: Output file path
        dynamic: Enable all dynamic dimensions
        dynamic_batch: Enable dynamic batch dimension
        dynamic_height: Enable dynamic height dimension
        dynamic_width: Enable dynamic width dimension
        **kwargs: Additional export arguments
    """
    if dynamic:
        # Enable all dynamic dimensions
        dynamic_batch = dynamic_height = dynamic_width = True
    
    DynamicShapeHandler.export_with_dynamic_shapes(
        model=model,
        dummy_input=im,
        output_path=str(file),
        dynamic_batch=dynamic_batch,
        dynamic_height=dynamic_height,
        dynamic_width=dynamic_width,
        **kwargs
    )


# Example usage and testing
if __name__ == "__main__":
    # Test dynamic axes generation
    handler = DynamicShapeHandler()
    dynamic_axes = handler.get_dynamic_axes()
    print("Dynamic axes:", dynamic_axes)
    
    # Test validator
    validator = DynamicShapeValidator()
    
    # Example: Validate an existing ONNX model
    # results = validator.validate_model_compatibility("nexuss.onnx", "onnxruntime")
    # print("Validation results:", results)
    
    # Patch YOLOv5 for dynamic shapes
    # patch_yolo_for_dynamic_shapes()
    
    print("Dynamic shapes utilities loaded successfully")
```

This `utils/dynamic_shapes.py` module provides comprehensive dynamic shape support for YOLOv5 with the following key features:

1. **DynamicShapeHandler**: Core class for handling dynamic shape operations
   - Generates dynamic axes for ONNX export
   - Validates ONNX models for dynamic shape support
   - Exports models with dynamic batch, height, and width dimensions

2. **DynamicDetect**: Modified Detect layer that handles dynamic input shapes
   - Creates anchor grids dynamically based on input dimensions
   - Compatible with both static and dynamic shapes

3. **DynamicNMS**: Non-Maximum Suppression supporting variable batch sizes
   - Handles dynamic number of detections per image
   - Includes fallback mechanisms for robustness

4. **DynamicShapeValidator**: Validates dynamic shape compatibility across backends
   - Supports ONNX Runtime, TensorRT, and OpenVINO
   - Tests multiple input shapes for compatibility

5. **Integration utilities**:
   - `patch_yolo_for_dynamic_shapes()`: Monkey-patches existing YOLOv5 modules
   - `export_onnx_with_dynamic_shapes()`: Drop-in replacement for export functionality
   - `xywh2xyxy()`: Bounding box conversion utility

The module is designed to integrate seamlessly with the existing YOLOv5 codebase, particularly with:
- `export.py` for ONNX export
- `models/yolo.py` for the Detect layer
- `models/common.py` for NMS operations

To use this module, you can either:
1. Call `patch_yolo_for_dynamic_shapes()` to enable dynamic shapes in existing models
2. Use `export_onnx_with_dynamic_shapes()` instead of the standard export function
3. Manually replace the Detect layer with DynamicDetect in your model architecture