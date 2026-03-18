# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Export a YOLOv5 PyTorch model to other formats. TensorFlow exports authored by https://github.com/zldrobit.

Format                      | `export.py --include`         | Model
---                         | ---                           | ---
PyTorch                     | -                             | nexuss.pt
TorchScript                 | `torchscript`                 | nexuss.torchscript
ONNX                        | `onnx`                        | nexuss.onnx
OpenVINO                    | `openvino`                    | nexuss_openvino_model/
TensorRT                    | `engine`                      | nexuss.engine
CoreML                      | `coreml`                      | nexuss.mlmodel
TensorFlow SavedModel       | `saved_model`                 | nexuss_saved_model/
TensorFlow GraphDef         | `pb`                          | nexuss.pb
TensorFlow Lite             | `tflite`                      | nexuss.tflite
TensorFlow Edge TPU         | `edgetpu`                     | nexuss_edgetpu.tflite
TensorFlow.js               | `tfjs`                        | nexuss_web_model/
PaddlePaddle                | `paddle`                      | nexuss_paddle_model/

Requirements:
    $ pip install -r requirements.txt coremltools onnx onnx-simplifier onnxruntime openvino-dev tensorflow-cpu  # CPU
    $ pip install -r requirements.txt coremltools onnx onnx-simplifier onnxruntime-gpu openvino-dev tensorflow  # GPU

Usage:
    $ python export.py --weights nexuss.pt --include torchscript onnx openvino engine coreml tflite ...

Inference:
    $ python detect.py --weights nexuss.pt                 # PyTorch
                                 nexuss.torchscript        # TorchScript
                                 nexuss.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 nexuss_openvino_model     # OpenVINO
                                 nexuss.engine             # TensorRT
                                 nexuss.mlmodel            # CoreML (macOS-only)
                                 nexuss_saved_model        # TensorFlow SavedModel
                                 nexuss.pb                 # TensorFlow GraphDef
                                 nexuss.tflite             # TensorFlow Lite
                                 nexuss_edgetpu.tflite     # TensorFlow Edge TPU
                                 nexuss_paddle_model       # PaddlePaddle

TensorFlow.js:
    $ cd .. && git clone https://github.com/zldrobit/tfjs-nexus-example.git && cd tfjs-nexus-example
    $ npm install
    $ ln -s ../../nexus/nexuss_web_model public/nexuss_web_model
    $ npm start
"""

import argparse
import contextlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from models.yolo import ClassificationModel, Detect, DetectionModel, SegmentationModel
from utils.dataloaders import LoadImages
from utils.general import (
    LOGGER,
    Profile,
    check_dataset,
    check_img_size,
    check_requirements,
    check_version,
    check_yaml,
    colorstr,
    file_size,
    get_default_args,
    print_args,
    url2file,
    yaml_save,
)
from utils.torch_utils import select_device, smart_inference_mode

MACOS = platform.system() == "Darwin"  # macOS environment


class iOSModel(torch.nn.Module):
    """An iOS-compatible wrapper for YOLOv5 models that normalizes input images based on their dimensions."""

    def __init__(self, model, im):
        """Initializes an iOS compatible model with normalization based on image dimensions.

        Args:
            model (torch.nn.Module): The PyTorch model to be adapted for iOS compatibility.
            im (torch.Tensor): An input tensor representing a batch of images with shape (B, C, H, W).

        Returns:
            None: This method does not return any value.

        Notes:
            This initializer configures normalization based on the input image dimensions, which is critical for
            ensuring the model's compatibility and proper functionality on iOS devices. The normalization step
            involves dividing by the image width if the image is square; otherwise, additional conditions might apply.
        """
        super().__init__()
        _b, _c, h, w = im.shape  # batch, channel, height, width
        self.model = model
        self.nc = model.nc  # number of classes
        if w == h:
            self.normalize = 1.0 / w
        else:
            self.normalize = torch.tensor([1.0 / w, 1.0 / h, 1.0 / w, 1.0 / h])  # broadcast (slower, smaller)
            # np = model(im)[0].shape[1]  # number of points
            # self.normalize = torch.tensor([1. / w, 1. / h, 1. / w, 1. / h]).expand(np, 4)  # explicit (faster, larger)

    def forward(self, x):
        """Run a forward pass on the input tensor, returning class confidences and normalized coordinates.

        Args:
            x (torch.Tensor): Input tensor containing the image data with shape (batch, channels, height, width).

        Returns:
            torch.Tensor: Concatenated tensor with normalized coordinates (xywh), confidence scores (conf), and class
                probabilities (cls), having shape (N, 4 + 1 + C), where N is the number of predictions, and C is the
                number of classes.

        Examples:
            ```python
            model = iOSModel(pretrained_model, input_image)
            output = model.forward(torch_input_tensor)
            ```
        """
        xywh, conf, cls = self.model(x)[0].squeeze().split((4, 1, self.nc), 1)
        return cls * conf, xywh * self.normalize  # confidence (3780, 80), coordinates (3780, 4)


def export_formats():
    r"""Returns a DataFrame of supported YOLOv5 model export formats and their properties.

    Returns:
        pandas.DataFrame: A DataFrame containing supported export formats and their properties. The DataFrame includes
            columns for format name, CLI argument suffix, file extension or directory name, and boolean flags indicating
            if the export format supports training and detection.

    Examples:
        ```python
        formats = export_formats()
        print(f"Supported export formats:\n{formats}")
        ```

    Notes:
        The DataFrame contains the following columns:
        - Format: The name of the model format (e.g., PyTorch, TorchScript, ONNX, etc.).
        - Include Argument: The argument to use with the export script to include this format.
        - File Suffix: File extension or directory name associated with the format.
        - Supports Training: Whether the format supports training.
        - Supports Detection: Whether the format supports detection.
    """
    x = [
        ["PyTorch", "-", ".pt", True, True],
        ["TorchScript", "torchscript", ".torchscript", True, True],
        ["ONNX", "onnx", ".onnx", True, True],
        ["OpenVINO", "openvino", "_openvino_model", True, False],
        ["TensorRT", "engine", ".engine", False, True],
        ["CoreML", "coreml", ".mlpackage", True, False],
        ["TensorFlow SavedModel", "saved_model", "_saved_model", True, True],
        ["TensorFlow GraphDef", "pb", ".pb", True, True],
        ["TensorFlow Lite", "tflite", ".tflite", True, False],
        ["TensorFlow Edge TPU", "edgetpu", "_edgetpu.tflite", False, False],
        ["TensorFlow.js", "tfjs", "_web_model", False, False],
        ["PaddlePaddle", "paddle", "_paddle_model", True, True],
    ]
    return pd.DataFrame(x, columns=["Format", "Include Argument", "File Suffix", "Supports Training", "Supports Detection"])


def try_export(inner_function):
    """Attempts to export model using inner_function, logging success or failure with timing information.

    Args:
        inner_function (callable): A function that performs the actual export operation.

    Returns:
        tuple: A tuple containing the exported model (or None on failure) and a status message string.

    Examples:
        ```python
        @try_export
        def export_onnx(model, file, opset, train, dynamic, simplify, prefix=colorstr("ONNX:")):
            # Export logic here
            return model, f"exported to {file}"
        ```
    """
    inner_function.results = []
    inner_function.time = time.time()
    try:
        f, model = inner_function(*inner_function.args, **inner_function.kwargs)
        inner_function.results.append(f)
        assert f.exists(), f"Failed to export to {f}"
        LOGGER.info(f"{inner_function.prefix} export success ✅ {time.time() - inner_function.time:.1f}s, saved as {f}")
        return model, f"exported to {f}"
    except Exception as e:
        LOGGER.info(f"{inner_function.prefix} export failure ❌ {time.time() - inner_function.time:.1f}s: {e}")
        return None, f"export failure: {e}"


def prepare_qat_model(model, qat_config=None):
    """Prepare model for Quantization-Aware Training (QAT) by inserting fake quantization nodes.
    
    Args:
        model (torch.nn.Module): The model to prepare for QAT.
        qat_config (dict, optional): Configuration for QAT. Defaults to None.
    
    Returns:
        torch.nn.Module: Model prepared for QAT with fake quantization nodes.
    
    Notes:
        This function modifies the model in-place by inserting fake quantization nodes
        using PyTorch's quantization APIs. The model should be fine-tuned after this
        step to adapt to quantization effects.
    """
    import torch.quantization as quant
    
    if qat_config is None:
        qat_config = {
            'quantization_backend': 'fbgemm',  # or 'qnnpack' for mobile
            'activation_observer': quant.MinMaxObserver.with_args(
                dtype=torch.quint8, qscheme=torch.per_tensor_affine
            ),
            'weight_observer': quant.MinMaxObserver.with_args(
                dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
            ),
        }
    
    # Define quantization configuration
    qconfig = quant.QConfig(
        activation=qat_config['activation_observer'],
        weight=qat_config['weight_observer']
    )
    
    # Prepare model for QAT
    model.qconfig = qconfig
    quant.prepare_qat(model, inplace=True)
    
    LOGGER.info(f"Model prepared for QAT with backend: {qat_config['quantization_backend']}")
    return model


def calibrate_model(model, dataloader, num_batches=100):
    """Calibrate quantized model using representative dataset.
    
    Args:
        model (torch.nn.Module): Quantized model to calibrate.
        dataloader: DataLoader with calibration images.
        num_batches (int): Number of batches to use for calibration.
    
    Notes:
        This function runs the model on calibration data to collect statistics
        for quantization. It should be called after prepare_qat_model and before
        convert_to_quantized.
    """
    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            model(images)
            if i % 10 == 0:
                LOGGER.info(f"Calibration batch {i}/{num_batches}")
    
    LOGGER.info(f"Calibration completed with {min(num_batches, len(dataloader))} batches")


def convert_to_quantized(model):
    """Convert QAT model to fully quantized model.
    
    Args:
        model (torch.nn.Module): Model prepared for QAT and calibrated.
    
    Returns:
        torch.nn.Module: Fully quantized model with int8 weights and activations.
    
    Notes:
        This function converts fake quantization nodes to actual quantized operations.
        The resulting model can be exported to ONNX or other formats with quantization.
    """
    import torch.quantization as quant
    
    quant.convert(model, inplace=True)
    LOGGER.info("Model converted to quantized format")
    return model


@try_export
def export_torchscript(model, im, file, optimize, prefix=colorstr("TorchScript:")):
    """YOLOv5 TorchScript model export.

    Args:
        model (torch.nn.Module): The YOLOv5 model to export.
        im (torch.Tensor): A dummy input tensor for tracing the model with shape (1, 3, 640, 640).
        file (Path): Path to save the exported TorchScript model.
        optimize (bool): If True, applies mobile-specific optimizations to the TorchScript model.
        prefix (str): A string prefix for logging messages.

    Returns:
        tuple: A tuple containing the path to the saved model file and the TorchScript model object.

    Examples:
        ```python
        model = torch.load("nexuss.pt")
        im = torch.zeros(1, 3, 640, 640)
        export_torchscript(model, im, "nexuss.torchscript", optimize=True)
        ```

    Notes:
        This function exports the YOLOv5 model to TorchScript format. If `optimize` is True, it applies
        mobile-specific optimizations using `optimize_for_mobile`. The model is saved with a `.torchscript` extension.
    """
    LOGGER.info(f"\n{prefix} starting export with torch {torch.__version__}...")
    f = file.with_suffix(".torchscript")
    ts = torch.jit.trace(model, im, strict=False)
    d = {"shape": im.shape, "stride": int(max(model.stride)), "names": model.names}
    extra_files = {"config.txt": json.dumps(d)}  # torch._C.ExtraFilesMap does not exist
    if optimize:  # https://pytorch.org/tutorials/recipes/mobile_interpreter.html
        optimize_for_mobile(ts)._save_for_lite_interpreter(str(f), _extra_files=extra_files)
    else:
        ts.save(str(f), _extra_files=extra_files)
    return f, None


@try_export
def export_onnx(model, im, file, opset, train, dynamic, simplify, prefix=colorstr("ONNX:")):
    """YOLOv5 ONNX export.

    Args:
        model (torch.nn.Module): The YOLOv5 model to export.
        im (torch.Tensor): A dummy input tensor for tracing the model with shape (1, 3, 640, 640).
        file (Path): Path to save the exported ONNX model.
        opset (int): The ONNX opset version to use for export.
        train (bool): If True, exports the model in training mode; otherwise in inference mode.
        dynamic (bool): If True, enables dynamic axes for input and output tensors.
        simplify (bool): If True, simplifies the ONNX model using onnx-simplifier.
        prefix (str): A string prefix for logging messages.

    Returns:
        tuple: A tuple containing the path to the saved ONNX model file and None.

    Examples:
        ```python
        model = torch.load("nexuss.pt")
        im = torch.zeros(1, 3, 640, 640)
        export_onnx(model, im, "nexuss.onnx", opset=12, train=False, dynamic=True, simplify=True)
        ```

    Notes:
        This function exports the YOLOv5 model to ONNX format. It handles dynamic axes for variable batch sizes
        and can simplify the model using onnx-simplifier if requested. The model is saved with a `.onnx` extension.
    """
    check_requirements(("onnx",))
    import onnx

    LOGGER.info(f"\n{prefix} starting export with onnx {onnx.__version__}...")
    f = file.with_suffix(".onnx")

    output_names = ["output0", "output1"] if isinstance(model, SegmentationModel) else ["output0"]
    if dynamic:
        dynamic = {"images": {0: "batch", 2: "height", 3: "width"}}  # shape(1,3,640,640)
        if isinstance(model, SegmentationModel):
            dynamic["output0"] = {0: "batch", 1: "anchors"}  # shape(1,25200,85)
            dynamic["output1"] = {0: "batch", 2: "mask_height", 3: "mask_width"}  # shape(1,32,160,160)
        elif isinstance(model, DetectionModel):
            dynamic["output0"] = {0: "batch", 1: "anchors"}  # shape(1,25200,85)

    torch.onnx.export(
        model.cpu() if dynamic else model,  # --dynamic only compatible with cpu
        im.cpu() if dynamic else im,
        f,
        verbose=False,
        opset_version=opset,
        training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
        do_constant_folding=not train,
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=dynamic or None,
    )

    # Checks
    model_onnx = onnx.load(f)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    # Simplify
    if simplify:
        try:
            cuda = torch.cuda.is_available()
            check_requirements(("onnxruntime-gpu" if cuda else "onnxruntime", "onnx-simplifier>=0.4.1"))
            import onnxsim

            LOGGER.info(f"{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...")
            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check, "assert check failed"
            onnx.save(model_onnx, f)
        except Exception as e:
            LOGGER.info(f"{prefix} simplifier failure: {e}")
    onnx.save(model_onnx, f)
    return f, None


@try_export
def export_openvino(model, file, half, prefix=colorstr("OpenVINO:")):
    """YOLOv5 OpenVINO export.

    Args:
        model (torch.nn.Module): The YOLOv5 model to export.
        file (Path): Path to save the exported OpenVINO model.
        half (bool): If True, exports the model in FP16 (half-precision) format.
        prefix (str): A string prefix for logging messages.

    Returns:
        tuple: A tuple containing the path to the saved OpenVINO model directory and None.

    Examples:
        ```python
        model = torch.load("nexuss.pt")
        export_openvino(model, "nexuss_openvino_model", half=True)
        ```

    Notes:
        This function exports the YOLOv5 model to OpenVINO format. It first exports to ONNX format and then
        uses OpenVINO's Model Optimizer to convert the ONNX model to OpenVINO's Intermediate Representation (IR) format.
        The model is saved in a directory with `_openvino_model` suffix.
    """
    check_requirements("openvino-dev>=2022.3")  # requires openvino-dev: https://pypi.org/project/openvino-dev/
    import openvino.runtime as ov  # noqa
    from openvino.tools import mo

    LOGGER.info(f"\n{prefix} starting export with openvino {ov.__version__}...")
    f = str(file).replace(".pt", f"_openvino_model{os.sep}")
    f_onnx = file.with_suffix(".onnx")
    f_ov = str(Path(f) / file.with_suffix(".xml").name)

    ov_model = mo.convert_model(
        f_onnx,
        model_name=file.with_suffix(".xml"),
        framework="onnx",
        compress_to_fp16=half,  # compress to FP16
    )
    ov.serialize(ov_model, f_ov)  # export
    return f, None


@try_export
def export_paddle(model, im, file, metadata, prefix=colorstr("PaddlePaddle:")):
    """YOLOv5 PaddlePaddle export.

    Args:
        model (torch.nn.Module): The YOLOv5 model to export.
        im (torch.Tensor): A dummy input tensor for tracing the model with shape (1, 3, 640, 640).
        file (Path): Path to save the exported PaddlePaddle model.
        metadata (dict): Dictionary containing model metadata such as stride and class names.
        prefix (str): A string prefix for logging messages.

    Returns:
        tuple: A tuple containing the path to the saved PaddlePaddle model directory and None.

    Examples:
        ```python
        model = torch.load("nexuss.pt")
        im = torch.zeros(1, 3, 640, 640)
        metadata = {"stride": 32, "names": ["person", "car", "dog"]}
        export_paddle(model, im, "nexuss_paddle_model", metadata)
        ```

    Notes:
        This function exports the YOLOv5 model to PaddlePaddle format using X2Paddle. It first exports to ONNX format
        and then converts the ONNX model to PaddlePaddle format. The model is saved in a directory with `_paddle_model` suffix.
    """
    check_requirements(("x2paddle", "paddlepaddle"))
    import x2paddle
    from x2paddle.convert import onnx2paddle

    LOGGER.info(f"\n{prefix} starting export with X2Paddle {x2paddle.__version__}...")
    f = str(file).replace(".pt", f"_paddle_model{os.sep}")

    onnx2paddle(
        module=onnx.load(file.with_suffix(".onnx")),
        save_dir=f,
        jit_type="trace",
        disable_feedback=True,
    )
    yaml_save(Path(f) / file.with_suffix(".yaml").name, metadata)  # add metadata.yaml
    return f, None


@try_export
def export_coreml(model, im, file, int8, half, prefix=colorstr("CoreML:")):
    """YOLOv5 CoreML export.

    Args:
        model (torch.nn.Module): The YOLOv5 model to export.
        im (torch.Tensor): A dummy input tensor for tracing the model with shape (1, 3, 640, 640).
        file (Path): Path to save the exported CoreML model.
        int8 (bool): If True, enables INT8 quantization for CoreML.
        half (bool): If True, enables FP16 (half-precision) quantization for CoreML.
        prefix (str): A string prefix for logging messages.

    Returns:
        tuple: A tuple containing the path to the saved CoreML model file and None.

    Examples:
        ```python
        model = torch.load("nexuss.pt")
        im = torch.zeros(1, 3, 640, 640)
        export_coreml(model, im, "nexuss.mlmodel", int8=False, half=True)
        ```

    Notes:
        This function exports the YOLOv5 model to CoreML format using coremltools. It supports quantization
        to INT8 or FP16 formats. The model is saved with a `.mlpackage` extension on macOS or `.mlmodel` on other platforms.
    """
    check_requirements("coremltools")
    import coremltools as ct

    LOGGER.info(f"\n{prefix} starting export with coremltools {ct.__version__}...")
    f = file.with_suffix(".mlmodel" if MACOS else ".mlpackage")

    ct_model = ct.convert(
        torch.jit.trace(model.cpu(), im, strict=False),
        inputs=[ct.ImageType(name="image", shape=im.shape, scale=1 / 255, bias=[0, 0, 0])],
    )
    bits, mode = (8, "kmeans") if int8 else (16, "linear") if half else (32, None)
    if bits < 32:
        if MACOS:  # quantization only supported on macOS
            ct_model = ct.models.neural_network.quantization_utils.quantize_weights(ct_model, nbits=bits, mode=mode)
        else:
            print(f"\n{prefix} quantization only supported on macOS, skipping...")
    ct_model.save(f)
    return f, None


@try_export
def export_engine(model, im, file, half, dynamic, simplify, workspace=4, verbose=False, prefix=colorstr("TensorRT:")):
    """YOLOv5 TensorRT export.

    Args:
        model (torch.nn.Module): The YOLOv5 model to export.
        im (torch.Tensor): A dummy input tensor for tracing the model with shape (1, 3, 640, 640).
        file (Path): Path to save the exported TensorRT engine.
        half (bool): If True, exports the model in FP16 (half-precision) format.
        dynamic (bool): If True, enables dynamic shapes for batch, height, and width dimensions.
        simplify (bool): If True, simplifies the ONNX model using onnx-simplifier before TensorRT conversion.
        workspace (int): Maximum workspace size in GB for TensorRT engine building.
        verbose (bool): If True, enables verbose logging during TensorRT engine building.
        prefix (str): A string prefix for logging messages.

    Returns:
        tuple: A tuple containing the path to the saved TensorRT engine file and None.

    Examples:
        ```python
        model = torch.load("nexuss.pt")
        im = torch.zeros(1, 3, 640, 640)
        export_engine(model, im, "nexuss.engine", half=True, dynamic=True, simplify=True)
        ```

    Notes:
        This function exports the YOLOv5 model to TensorRT format. It first exports to ONNX format and then
        uses TensorRT's trtexec tool to build the engine. The model is saved with a `.engine` extension.
    """
    assert im.device.type != "cpu", "export running on CPU but must be on GPU, i.e. `python export.py --device 0`"
    try:
        import tensorrt as trt
    except Exception:
        if platform.system() == "Linux":
            check_requirements("nvidia-tensorrt", cmds="-U --index-url https://pypi.ngc.nvidia.com")
        import tensorrt as trt

    if dynamic:
        if im.shape[0] <= 1:
            LOGGER.warning(f"{prefix} WARNING ⚠️ --dynamic model requires maximum --batch-size argument")
        profile = trt.Profile()
        for s in range(3):
            profile.set_shape("images", [1, 3, 640, 640], [max(1, im.shape[0] // 2), 3, 640, 640], im.shape)
    check_version(trt.__version__, "7.0.0", hard=True)  # require tensorrt>=8.0.0
    batch_size = im.shape[0]  # batch size

    export_onnx(model, im, file, 12, False, dynamic, simplify)  # opset 12
    onnx = file.with_suffix(".onnx")

    LOGGER.info(f"\n{prefix} starting export with TensorRT {trt.__version__}...")
    assert onnx.exists(), f"failed to export ONNX file: {onnx}"
    f = file.with_suffix(".engine")  # TensorRT engine file
    logger = trt.Logger(trt.Logger.INFO)
    if verbose:
        logger.min_severity = trt.Logger.Severity.VERBOSE

    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.max_workspace_size = workspace * 1 << 30
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)  # fix TRT 8.4 deprecation

    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx)):
        raise RuntimeError(f"failed to load ONNX file: {onnx}")

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        LOGGER.info(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        LOGGER.info(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')

    if dynamic:
        if im.shape[0] <= 1:
            LOGGER.warning(f"\n{prefix} WARNING ⚠️ --dynamic model requires maximum --batch-size argument")
        profile = trt.Profile()
        for inp in inputs:
            profile.set_shape(inp.name, (1, *inp.shape[1:]), (max(1, im.shape[0] // 2), *inp.shape[1:]), im.shape)
        config.add_optimization_profile(profile)

    LOGGER.info(f"{prefix} building FP{16 if builder.platform_has_fast_fp16 and half else 32} engine as {f}")
    if builder.platform_has_fast_fp16 and half:
        config.set_flag(trt.BuilderFlag.FP16)
    with builder.build_engine(network, config) as engine, open(f, "wb") as t:
        t.write(engine.serialize())
    return f, None


@try_export
def export_saved_model(
    model,
    im,
    file,
    dynamic,
    tf_nms=False,
    agnostic_nms=False,
    topk_per_class=100,
    topk_all=100,
    iou_thres=0.45,
    conf_thres=0.25,
    prefix=colorstr("TensorFlow SavedModel:"),
):
    """YOLOv5 TensorFlow SavedModel export.

    Args:
        model (torch.nn.Module): The YOLOv5 model to export.
        im (torch.Tensor): A dummy input tensor for tracing the model with shape (1, 3, 640, 640).
        file (Path): Path to save the exported TensorFlow SavedModel.
        dynamic (bool): If True, enables dynamic batch size for the model.
        tf_nms (bool): If True, adds TensorFlow NMS to the exported model.
        agnostic_nms (bool): If True, enables class-agnostic NMS.
        topk_per_class (int): Maximum number of detections per class.
        topk_all (int): Maximum number of detections overall.
        iou_thres (float): IoU threshold for NMS.
        conf_thres (float): Confidence threshold for NMS.
        prefix (str): A string prefix for logging messages.

    Returns:
        tuple: A tuple containing the path to the saved TensorFlow SavedModel directory and None.

    Examples:
        ```python
        model = torch.load("nexuss.pt")
        im = torch.zeros(1, 3, 640, 640)
        export_saved_model(model, im, "nexuss_saved_model", dynamic=True, tf_nms=True)
        ```

    Notes:
        This function exports the YOLOv5 model to TensorFlow SavedModel format using TensorFlow.js. It supports
        dynamic batch sizes and optional TensorFlow NMS. The model is saved in a directory with `_saved_model` suffix.
    """
    # YOLOv5 TensorFlow SavedModel export
    check_requirements(("tensorflow>=2.4.1", "tensorflowjs>=3.9.0"))
    import tensorflow as tf
    import tensorflowjs as tfjs

    LOGGER.info(f"\n{prefix} starting export with tensorflow {tf.__version__}...")
    f = str(file).replace(".pt", f"_saved_model{os.sep}")
    batch_size, ch, *imgsz = list(im.shape)  # BCHW

    tf_model = tf.function(lambda x: model(x))
    tf_model = tf_model.get_concrete_function(
        tf.TensorSpec([None, ch, *imgsz], dtype=tf.float32, name="images")
    )
    tf.saved_model.save(model, f, signatures={"serving_default": tf_model})
    return f, None


@try_export
def export_pb(model, file, prefix=colorstr("TensorFlow GraphDef:")):
    """YOLOv5 TensorFlow GraphDef export.

    Args:
        model (torch.nn.Module): The YOLOv5 model to export.
        file (Path): Path to save the exported TensorFlow GraphDef.
        prefix (str): A string prefix for logging messages.

    Returns:
        tuple: A tuple containing the path to the saved TensorFlow GraphDef file and None.

    Examples:
        ```python
        model = torch.load("nexuss.pt")
        export_pb(model, "nexuss.pb")
        ```

    Notes:
        This function exports the YOLOv5 model to TensorFlow GraphDef format. It first exports to ONNX format
        and then converts the ONNX model to TensorFlow GraphDef using onnx-tf. The model is saved with a `.pb` extension.
    """
    import onnx
    from onnx_tf.backend import prepare

    LOGGER.info(f"\n{prefix} starting export with onnx {onnx.__version__}...")
    f = file.with_suffix(".pb")

    onnx_model = onnx.load(file.with_suffix(".onnx"))  # load onnx model
    tf_rep = prepare(onnx_model)  # prepare tf representation
    tf_rep.export_graph(str(f))  # export the model
    return f, None


@try_export
def export_tflite(file, half, prefix=colorstr("TensorFlow Lite:")):
    """YOLOv5 TensorFlow Lite export.

    Args:
        file (Path): Path to the saved TensorFlow SavedModel directory.
        half (bool): If True, enables FP16 (half-precision) quantization for TFLite.
        prefix (str): A string prefix for logging messages.

    Returns:
        tuple: A tuple containing the path to the saved TensorFlow Lite model file and None.

    Examples:
        ```python
        export_tflite("nexuss_saved_model", half=True)
        ```

    Notes:
        This function exports the TensorFlow SavedModel to TensorFlow Lite format. It supports FP16 quantization.
        The model is saved with a `.tflite` extension.
    """
    check_requirements(("tensorflow>=2.4.1",))
    import tensorflow as tf

    LOGGER.info(f"\n{prefix} starting export with tensorflow {tf.__version__}...")
    saved_model = Path(str(file).replace(".pt", f"_saved_model"))
    f = saved_model.with_suffix(".tflite")

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model))
    if half:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    f.write_bytes(tflite_model)
    return f, None


@try_export
def export_edgetpu(file, prefix=colorstr("Edge TPU:")):
    """YOLOv5 Edge TPU export.

    Args:
        file (Path): Path to the saved TensorFlow Lite model.
        prefix (str): A string prefix for logging messages.

    Returns:
        tuple: A tuple containing the path to the saved Edge TPU model file and None.

    Examples:
        ```python
        export_edgetpu("nexuss.tflite")
        ```

    Notes:
        This function exports the TensorFlow Lite model to Edge TPU format using the Edge TPU compiler.
        The model is saved with a `_edgetpu.tflite` suffix.
    """
    cmd = "edgetpu_compiler -s -d -k 10 --out_dir ../..".split()
    subprocess.run([*cmd, str(file)], check=True)
    return file.with_suffix("_edgetpu.tflite"), None


@try_export
def export_tfjs(file, prefix=colorstr("TensorFlow.js:")):
    """YOLOv5 TensorFlow.js export.

    Args:
        file (Path): Path to the saved TensorFlow SavedModel directory.
        prefix (str): A string prefix for logging messages.

    Returns:
        tuple: A tuple containing the path to the saved TensorFlow.js model directory and None.

    Examples:
        ```python
        export_tfjs("nexuss_saved_model")
        ```

    Notes:
        This function exports the TensorFlow SavedModel to TensorFlow.js format using tensorflowjs_converter.
        The model is saved in a directory with `_web_model` suffix.
    """
    check_requirements("tensorflowjs")
    import tensorflowjs as tfjs

    LOGGER.info(f"\n{prefix} starting export with tensorflowjs {tfjs.__version__}...")
    f = str(file).replace(".pt", f"_web_model{os.sep}")

    cmd = "tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default --saved_model_tags=serve".split()
    subprocess.run([*cmd, str(file), f], check=True)
    return f, None


@smart_inference_mode()
def run(
    data=ROOT / "data/coco128.yaml",  # 'dataset.yaml path'
    weights=ROOT / "nexuss.pt",  # weights path
    imgsz=(640, 640),  # image (height, width)
    batch_size=1,  # batch size
    device="cpu",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    include=("torchscript", "onnx"),  # include formats
    half=False,  # FP16 half-precision export
    inplace=False,  # set YOLOv5 Detect() inplace=True
    keras=False,  # use Keras
    optimize=False,  # TorchScript: optimize for mobile
    int8=False,  # CoreML/TF INT8 quantization
    dynamic=False,  # ONNX/TF/TensorRT: dynamic axes
    simplify=False,  # ONNX: simplify model
    opset=12,  # ONNX: opset version
    verbose=False,  # TensorRT: verbose log
    workspace=4,  # TensorRT: workspace size (GB)
    nms=False,  # TF: add NMS to model
    agnostic_nms=False,  # TF: class-agnostic NMS
    topk_per_class=100,  # TF.js NMS: topk per class to keep
    topk_all=100,  # TF.js NMS: topk for all classes to keep
    iou_thres=0.45,  # TF.js NMS: IoU threshold
    conf_thres=0.25,  # TF.js NMS: confidence threshold
    qat=False,  # Quantization-Aware Training export
    qat_backend="fbgemm",  # QAT backend: fbgemm or qnnpack
    calibrate_data=None,  # Calibration data for QAT
    calibrate_batches=100,  # Number of calibration batches
):
    """Exports a YOLOv5 model to various formats including TorchScript, ONNX, OpenVINO, TensorRT, CoreML,
    TensorFlow SavedModel, TensorFlow GraphDef, TensorFlow Lite, TensorFlow Edge TPU, TensorFlow.js, and PaddlePaddle.

    Args:
        data (str): Path to the dataset configuration YAML file.
        weights (str): Path to the YOLOv5 model weights file.
        imgsz (tuple): Image dimensions (height, width) for export.
        batch_size (int): Batch size for export.
        device (str): Device to use for export ('cpu' or 'cuda' device).
        include (tuple): Tuple of formats to include in export.
        half (bool): Enable FP16 half-precision export.
        inplace (bool): Set YOLOv5 Detect() layer inplace.
        keras (bool): Enable Keras export.
        optimize (bool): Enable optimization for TorchScript mobile export.
        int8 (bool): Enable INT8 quantization for CoreML/TFLite.
        dynamic (bool): Enable dynamic axes for ONNX/TF/TensorRT export.
        simplify (bool): Enable ONNX model simplification.
        opset (int): ONNX opset version.
        verbose (bool): Enable verbose logging for TensorRT export.
        workspace (int): TensorRT workspace size in GB.
        nms (bool): Add NMS to TensorFlow model.
        agnostic_nms (bool): Enable class-agnostic NMS for TensorFlow.
        topk_per_class (int): Topk per class for TF.js NMS.
        topk_all (int): Topk for all classes for TF.js NMS.
        iou_thres (float): IoU threshold for TF.js NMS.
        conf_thres (float): Confidence threshold for TF.js NMS.
        qat (bool): Enable Quantization-Aware Training export.
        qat_backend (str): QAT backend ('fbgemm' for x86, 'qnnpack' for ARM).
        calibrate_data (str): Path to calibration data for QAT.
        calibrate_batches (int): Number of calibration batches for QAT.

    Returns:
        None: This function does not return any value. It exports the model to the specified formats and saves them to disk.

    Examples:
        ```python
        from export import run
        run(weights="nexuss.pt", include=("torchscript", "onnx"), imgsz=(640, 640))
        ```

    Notes:
        This function loads the YOLOv5 model, sets up the export environment, and exports the model to all specified
        formats. It handles device selection, model loading, and format-specific export logic.
    """
    t = time.time()
    include = [x.lower() for x in include]
    fmts = tuple(export_formats()["Include Argument"][1:])  # available export formats
    flags = [x in include for x in fmts]
    assert sum(flags) >= 1, f"ERROR: Invalid --include {include}, valid --include arguments are {fmts}"
    (
        jit,
        onnx,
        xml,
        engine,
        coreml,
        saved_model,
        pb,
        tflite,
        edgetpu,
        tfjs,
        paddle,
    ) = flags  # export booleans
    file = Path(url2file(weights) if str(weights).startswith(("http:/", "https:/")) else weights)  # PyTorch weights

    # Load PyTorch model
    device = select_device(device)
    if half:
        assert device.type != "cpu", "--half only compatible with GPU export, i.e. use --device 0"
        assert not dynamic, "--half not compatible with --dynamic, i.e. use either --half or --dynamic but not both"
    model = attempt_load(weights, device=device, inplace=True, fuse=True)  # load FP32 model

    # Checks
    imgsz *= 2 if len(imgsz) == 1 else 1  # expand
    if optimize:
        assert device.type == "cpu", "--optimize not compatible with cuda devices, i.e. use --device cpu"

    # Input
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz = [check_img_size(x, gs) for x in imgsz]  # verify img_size are gs-multiples
    im = torch.zeros(batch_size, 3, *imgsz).to(device)  # image size(1,3,320,192) BCHW iDetection

    # Update model
    model.eval()
    for k, m in model.named_modules():
        if isinstance(m, Detect):
            m.inplace = inplace
            m.dynamic = dynamic
            m.export = True

    for _ in range(2):
        y = model(im)  # dry runs
    if half and not coreml:
        im, model = im.half(), model.half()  # to FP16
    shape = tuple((y[0] if isinstance(y, tuple) else y).shape)  # model output shape
    metadata = {"stride": int(max(model.stride)), "names": model.names}  # model metadata
    LOGGER.info(
        f"\n{colorstr('PyTorch:')} starting from {file} with output shape {shape} ({file_size(file):.1f} MB)"
    )

    # Exports
    if qat:
        LOGGER.info(f"\n{colorstr('QAT:')} preparing model for Quantization-Aware Training...")
        model = prepare_qat_model(model, qat_config={'quantization_backend': qat_backend})
        
        if calibrate_data:
            from utils.dataloaders import create_dataloader
            from utils.general import check_dataset
            
            data = check_dataset(calibrate_data)
            calibrate_loader = create_dataloader(
                data['val'], imgsz[0], batch_size, gs, 
                pad=0.5, rect=True, prefix=colorstr('calibrate:')
            )[0]
            calibrate_model(model, calibrate_loader, num_batches=calibrate_batches)
        
        model = convert_to_quantized(model)
        LOGGER.info(f"{colorstr('QAT:')} model quantized successfully")

    if jit:
        export_torchscript(model, im, file, optimize)
    if engine:  # TensorRT
        export_engine(model, im, file, half, dynamic, simplify, workspace, verbose)
    if onnx or xml:  # OpenVINO requires ONNX
        export_onnx(model, im, file, opset, False, dynamic, simplify)
    if xml:  # OpenVINO
        export_openvino(file, half)
    if paddle:  # PaddlePaddle
        export_paddle(model, im, file, metadata)
    if coreml:  # CoreML
        export_coreml(model, im, file, int8, half)

    # TensorFlow Exports
    if any((saved_model, pb, tflite, edgetpu, tfjs)):
        if int8 or edgetpu:  # TFLite --int8 bug https://github.com/ultralytics/nexus/issues/5707
            check_requirements(("flatbuffers==1.12",))  # required before `import tensorflow`
        assert not tflite or not tfjs, "TFLite and TF.js models must be exported separately, please use only one at once."
        if not (saved_model or pb):  # TensorFlow SavedModel export
            export_saved_model(
                model,
                im,
                file,
                dynamic,
                tf_nms=nms or agnostic_nms or tfjs,
                agnostic_nms=agnostic_nms,
                topk_per_class=topk_per_class,
                topk_all=topk_all,
                iou_thres=iou_thres,
                conf_thres=conf_thres,
            )
        if pb:  # TensorFlow GraphDef
            export_pb(model, file)
        if tflite:  # TensorFlow Lite
            export_tflite(file, half)
        if edgetpu:  # Edge TPU
            export_edgetpu(file)
        if tfjs:  # TensorFlow.js
            export_tfjs(file)

    # Finish
    LOGGER.info(
        f"\nExport complete ({time.time() - t:.1f}s)"
        f"\nResults saved to {colorstr('bold', file.parent.resolve())}"
        f"\nDetect:          python detect.py --weights {file.with_suffix('.engine' if engine else file.suffix)}"
        f"\nValidate:        python val.py --weights {file.with_suffix('.engine' if engine else file.suffix)}"
        f"\nPyTorch Hub:     model = torch.hub.load('ultralytics/nexus', 'custom', '{file}')"
        f"\nVisualize:       https://netron.app"
    )
    return tuple(list(zip(*[getattr(m, "results", [""]) for m in [export_torchscript, export_onnx, export_openvino, export_engine, export_coreml, export_saved_model, export_pb, export_tflite, export_edgetpu, export_tfjs, export_paddle]])))


def parse_opt(known=False):
    """Parses command line arguments for the export script.

    Args:
        known (bool): If True, parses only known arguments; otherwise, parses all arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.

    Examples:
        ```python
        opt = parse_opt()
        print(opt.weights)
        ```
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "nexuss.pt", help="model.pt path(s)")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640, 640], help="image (h, w)")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--device", default="cpu", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument(
        "--include",
        nargs="+",
        default=["torchscript", "onnx"],
        help="torchscript, onnx, openvino, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle",
    )
    parser.add_argument("--half", action="store_true", help="FP16 half-precision export")
    parser.add_argument("--inplace", action="store_true", help="set YOLOv5 Detect() inplace=True")
    parser.add_argument("--keras", action="store_true", help="TF: use Keras")
    parser.add_argument("--optimize", action="store_true", help="TorchScript: optimize for mobile")
    parser.add_argument("--int8", action="store_true", help="CoreML/TF INT8 quantization")
    parser.add_argument("--dynamic", action="store_true", help="ONNX/TF/TensorRT: dynamic axes")
    parser.add_argument("--simplify", action="store_true", help="ONNX: simplify model")
    parser.add_argument("--opset", type=int, default=12, help="ONNX: opset version")
    parser.add_argument("--verbose", action="store_true", help="TensorRT: verbose log")
    parser.add_argument("--workspace", type=int, default=4, help="TensorRT: workspace size (GB)")
    parser.add_argument("--nms", action="store_true", help="TF: add NMS to model")
    parser.add_argument("--agnostic-nms", action="store_true", help="TF: class-agnostic NMS")
    parser.add_argument("--topk-per-class", type=int, default=100, help="TF.js NMS: topk per class to keep")
    parser.add_argument("--topk-all", type=int, default=100, help="TF.js NMS: topk for all classes to keep")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="TF.js NMS: IoU threshold")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="TF.js NMS: confidence threshold")
    parser.add_argument("--qat", action="store_true", help="Quantization-Aware Training export")
    parser.add_argument("--qat-backend", type=str, default="fbgemm", help="QAT backend: fbgemm or qnnpack")
    parser.add_argument("--calibrate-data", type=str, default=None, help="Calibration data for QAT")
    parser.add_argument("--calibrate-batches", type=int, default=100, help="Number of calibration batches for QAT")
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    """Executes the YOLOv5 model export process based on command line arguments.

    Args:
        opt (argparse.Namespace): Parsed command line arguments.

    Examples:
        ```python
        opt = parse_opt()
        main(opt)
        ```

    Notes:
        This function serves as the main entry point for the export script. It processes the command line arguments
        and calls the `run` function to export the model to the specified formats.
    """
    for opt.weights in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
        run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)