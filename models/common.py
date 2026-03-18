import ast
import contextlib
import json
import math
import onnx
import os
import platform
import time
import warnings
import zipfile
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp

# Import 'ultralytics' package or install if missing
try:
    import ultralytics

    assert hasattr(ultralytics, "__version__")  # verify package is not directory
except (ImportError, AssertionError):
    os.system("pip install -U ultralytics")
    import ultralytics

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from utils import TryExcept
from utils.dataloaders import exif_transpose, letterbox
from utils.general import (
    LOGGER,
    ROOT,
    Profile,
    check_requirements,
    check_suffix,
    check_version,
    colorstr,
    increment_path,
    is_jupyter,
    make_divisible,
    non_max_suppression,
    scale_boxes,
    xywh2xyxy,
    xyxy2xywh,
    yaml_load,
)
from utils.torch_utils import copy_attr, smart_inference_mode


def autopad(k, p=None, d=1):
    """Pads kernel to 'same' output shape, adjusting for optional dilation; returns padding size.

    `k`: kernel, `p`: padding, `d`: dilation.
    """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Applies a convolution, batch normalization, and activation function to an input tensor in a neural network."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initializes a standard convolution layer with optional batch normalization and activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies a convolution followed by batch normalization and an activation function to the input tensor `x`."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Applies a fused convolution and activation function to the input tensor `x`."""
        return self.act(self.conv(x))


class DWConv(Conv):
    """Implements a depth-wise convolution layer with optional activation for efficient spatial filtering."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """Initializes a depth-wise convolution layer with optional activation; args: input channels (c1), output
        channels (c2), kernel size (k), stride (s), dilation (d), and activation flag (act).
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """A depth-wise transpose convolutional layer for upsampling in neural networks, particularly in YOLOv5 models."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """Initializes a depth-wise transpose convolutional layer for YOLOv5; args: input channels (c1), output channels
        (c2), kernel size (k), stride (s), input padding (p1), output padding (p2).
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class TransformerLayer(nn.Module):
    """Transformer layer with multihead attention and linear layers, optimized by removing LayerNorm."""

    def __init__(self, c, num_heads):
        """Initializes a transformer layer, sans LayerNorm for performance, with multihead attention and linear layers.

        See  as described in https://arxiv.org/abs/2010.11929.
        """
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        """Performs forward pass using MultiheadAttention and two linear transformations with residual connections."""
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    """A Transformer block for vision tasks with convolution, position embeddings, and Transformer layers."""

    def __init__(self, c1, c2, num_heads, num_layers):
        """Initializes a Transformer block for vision tasks, adapting dimensions if necessary and stacking specified
        layers.
        """
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        """Processes input through an optional convolution, followed by Transformer layers and position embeddings for
        object detection.
        """
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class Bottleneck(nn.Module):
    """A bottleneck layer with optional shortcut and group convolution for efficient feature extraction."""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """Initializes a standard bottleneck layer with optional shortcut and group convolution, supporting channel
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Processes input through two convolutions, optionally adds shortcut if channel dimensions match; input is a
        tensor.
        """
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP bottleneck layer for feature extraction with cross-stage partial connections and optional shortcuts."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes CSP bottleneck with optional shortcuts; args: ch_in, ch_out, number of repeats, shortcut bool,
        groups, expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Performs forward pass by applying layers, activation, and concatenation on input x, returning feature-
        enhanced output.
        """
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class CrossConv(nn.Module):
    """Implements a cross convolution layer with downsampling, expansion, and optional shortcut."""

    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        """Initializes CrossConv with downsampling, expanding, and optionally shortcutting; `c1` input, `c2` output
        channels.

        Inputs are ch_in, ch_out, kernel, stride, groups, expansion, and shortcut.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies two convolutions on input `x`, optionally adds shortcut if channels match, returns result."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions for efficient feature extraction in YOLOv5 networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize CSP bottleneck with 3 convolutions; args: ch_in, ch_out, number of repeats, shortcut, groups,
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Performs forward pass by applying cv1, cv2, cv3 and m on input x, returns concatenated output."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions for enhanced feature extraction in YOLOv5."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3x module with cross-convolutions; inherits from C3 with custom Bottleneck."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)))


class C3TR(C3):
    """C3 module with TransformerBlock for enhanced feature extraction in YOLOv5."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3TR module with TransformerBlock; inherits from C3 with custom Transformer layer."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3SPP(C3):
    """C3 module with SPP block for spatial pyramid pooling in YOLOv5."""

    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3SPP module with SPP block; inherits from C3 with custom SPP layer."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer for multi-scale feature extraction."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initializes SPP layer with multiple max-pooling kernel sizes; args: ch_in, ch_out, kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Applies convolution and multiple max-pooling operations, concatenates results, then applies final
        convolution.
        """
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """Initializes SPPF layer with given input/output channels and kernel size; equivalent to SPP(k=(5, 9, 13))."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Applies convolution and three max-pooling operations, concatenates results, then applies final
        convolution.
        """
        x = self.cv1(x)
        return self.cv2(torch.cat((x, self.m(x), self.m(self.m(x)), self.m(self.m(self.m(x)))), 1))


class Focus(nn.Module):
    """Focus wh information into c-space, slicing 2x2 to 4 channels for efficient downsampling."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus module to concentrate spatial info into channels; args: ch_in, ch_out, kernel, stride,
        padding, groups, activation.
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)

    def forward(self, x):
        """Concatenates strided slices of input along channel dim, applies convolution; input shape (b,c,w,h)."""
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))


class GhostConv(nn.Module):
    """Ghost Convolution for efficient feature generation using cheap operations."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes GhostConv with primary and cheap convolutions; args: ch_in, ch_out, kernel, stride, groups,
        activation.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Applies GhostConv by concatenating outputs of primary and cheap convolutions on input x."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck for efficient feature extraction in lightweight networks."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck with GhostConvs and optional stride convolution; args: ch_in, ch_out, kernel,
        stride.
        """
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),
            GhostConv(c_, c2, 1, 1, act=False),
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies GhostBottleneck operations with residual connection on input x, returns output tensor."""
        return self.conv(x) + self.shortcut(x)


class Contract(nn.Module):
    """Contracts spatial dimensions into channel dimension for width-height to channel transformation."""

    def __init__(self, gain=2):
        """Initializes Contract module to combine spatial info into channels with specified gain factor."""
        super().__init__()
        self.gain = gain

    def forward(self, x):
        """Reshapes input tensor by contracting spatial dims into channels using pixelshuffle; input (b,c,w,h)."""
        b, c, h, w = x.size()
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        return x.view(b, c * s * s, h // s, w // s)


class Expand(nn.Module):
    """Expands channel dimension into spatial dimensions, reversing the Contract operation."""

    def __init__(self, gain=2):
        """Initializes Expand module to transform channels back to spatial dimensions with specified gain."""
        super().__init__()
        self.gain = gain

    def forward(self, x):
        """Expands channels into spatial dims using pixelshuffle; input (b,c,h,w) -> (b,c/g^2,h*g,w*g)."""
        b, c, h, w = x.size()
        s = self.gain
        x = x.view(b, s, s, c // s**2, h, w)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        return x.view(b, c // s**2, h * s, w * s)


class Concat(nn.Module):
    """Concatenates a list of tensors along a specified dimension."""

    def __init__(self, dimension=1):
        """Initializes Concat module to concatenate tensors along specified dimension (default: 1)."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Concatenates a list of tensors along self.d dimension; input is list of tensors."""
        return torch.cat(x, self.d)


class Detect(nn.Module):
    """YOLOv5 Detect head for outputting detection predictions from feature maps."""

    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        """Initializes YOLOv5 detection layer with anchors, number of classes, and channel dimensions."""
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use inplace ops

    def forward(self, x):
        """Processes input through detection layers, applying convolution and reshaping for dynamic batch sizes."""
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x[i] shape: (bs, na*no, ny, nx)
            # Use reshape with tensor shape arg for full dynamic shape support (batch, height, width)
            x[i] = x[i].reshape(torch.tensor([x[i].shape[0], self.na, self.no, ny * nx])).permute(0, 1, 3, 2)
            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, Segment):  # (boxes + mask)
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # Detect (boxes only)
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * ny * nx, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, "1.10.0")):
        """Generates detection grid and anchor grid for specified layer index i."""
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing="ij") if torch_1_10 else torch.meshgrid(y, x)
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class Segment(Detect):
    """YOLOv5 Segment head for instance segmentation with mask predictions."""

    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True, det=None):
        """Initializes YOLOv5 Segment head with mask protos and coefficients."""
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward if det is None else det.forward

    def forward(self, x):
        """Processes input through segment head, returning detections and protos for mask generation."""
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])


class BaseModel(nn.Module):
    """Base class for YOLOv5 models providing forward pass and profiling capabilities."""

    def forward(self, x, profile=False, visualize=False):
        """Runs forward pass on input x with optional profiling and visualization of features."""
        return self._forward_once(x, profile, visualize)

    def _forward_once(self, x, profile=False, visualize=False):
        """Performs single forward pass through all layers with optional profiling and feature visualization."""
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _profile_one_layer(self, m, x, dt):
        """Profiles computation time for a single layer; modifies input x in place."""
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1e9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):
        """Fuses Conv2d and BatchNorm2d layers for optimized inference performance."""
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, "bn")  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):
        """Prints model information including GFLOPs and parameter counts."""
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        """Applies function fn to model parameters and buffers (e.g., for device transfer)."""
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


class DetectionModel(BaseModel):
    """YOLOv5 detection model with Detect head and anchor-based predictions."""

    def __init__(self, cfg="nexuss.yaml", ch=3, nc=None, anchors=None):
        """Initializes YOLOv5 detection model from config file with optional class count and anchor overrides."""
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub

            self.yaml_file = Path(cfg).name
            with open(cfg, encoding="ascii", errors="ignore") as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override yaml value
        if anchors:
            LOGGER.info(f"Overriding model.yaml anchors with anchors={anchors}")
            self.yaml["anchors"] = f"{anchors}"  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml["nc"])]  # default names
        self.inplace = self.yaml.get("inplace", True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info("")

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Runs augmented or single forward pass with optional profiling and visualization."""
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        """Performs augmented inference across multiple scales and flips, returning combined detections."""
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _descale_pred(self, p, flips, scale, img_size):
        """De-scales and optionally flips predictions for augmented inference."""
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        """Clips augmented inference tails to remove boundary artifacts from multi-scale augmentation."""
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _initialize_biases(self, cf=None):
        """Initialize biases into Detect(), cf is class frequency tensor for focal loss weighting."""
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5 : 5 + m.nc] += (
                math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())
            )  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


Model = DetectionModel  # retain old name for backwards compatibility


class SegmentationModel(DetectionModel):
    """YOLOv5 segmentation model extending DetectionModel with mask prediction head."""

    def __init__(self, cfg="nexuss-seg.yaml", ch=3, nc=None, anchors=None):
        """Initializes YOLOv5 segmentation model from config with optional class and anchor overrides."""
        super().__init__(cfg, ch, nc, anchors)


class ClassificationModel(BaseModel):
    """YOLOv5 classification model with adaptive pooling and linear classifier head."""

    def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):
        """Initializes classification model from config or detection model backbone."""
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        """Creates classification model from a detection model backbone up to cutoff layer."""
        if isinstance(model, DetectionModel):
            m = model.model[:cutoff] if cutoff > 0 else model.model
        else:
            m = model[:cutoff] if cutoff > 0 else model
        m = nn.Sequential(*m) if not isinstance(m, nn.Sequential) else m
        self.model = m
        self.stride = model.stride if isinstance(model, DetectionModel) else torch.tensor([32])
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg):
        """Creates classification model from YAML config file."""
        self.model = None

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Runs forward pass with optional augmentation, profiling, and feature visualization."""
        if augment:
            return self._forward_augment(x)
        return self._forward_once(x, profile, visualize)

    def _forward_augment(self, x):
        """Performs augmented inference with horizontal flip for classification."""
        y = []  # outputs
        for f in [None, 3]:
            xi = scale_img(x.flip(f) if f else x, 0, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            y.append(yi)
        y = torch.stack(y, 1)  # augmented output
        return y.mean(1), y  # mean augmented, individual


class autoShape(nn.Module):
    """YOLOv5 autoShape module for input-agnostic inference with automatic preprocessing and postprocessing."""

    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    agnostic = False  # NMS class-agnostic
    multi_label = False  # NMS multiple labels per box
    classes = None  # NMS filter by class: list[int]
    max_det = 1000  # maximum detections per image
    amp = False  # Automatic Mixed Precision

    def __init__(self, model, verbose=True):
        """Initializes autoShape wrapper for YOLOv5 model with automatic input preprocessing."""
        super().__init__()
        if verbose:
            LOGGER.info("Adding autoShape... ")
        self.model = model.eval()
        self.stride = model.stride
        self.names = model.names
        self.dmb = isinstance(model, DetectMultiBackend)  # DetectMultiBackend instance
        self.pt = not self.dmb or model.pt  # PyTorch model
        self._pipeline_init()

    def _pipeline_init(self):
        """Initializes the inference pipeline attributes."""
        self.done_warmup = False

    @smart_inference_mode()
    def forward(self, imgs, size=640, augment=False, profile=False):
        """Inference from various sources with automatic preprocessing, supporting dynamic input shapes."""
        if isinstance(imgs, torch.Tensor):  # torch.Tensor
            return self.model(imgs.to(self.device), augment=augment, profile=profile)

        # Inference from various sources
        if isinstance(imgs, (str, Path)):  # filename or URI
            imgs = [imgs]

        # Pre-process
        imgs, shapes_batch = self._preprocess(imgs, size)

        # Inference
        preds = self.model(imgs.to(self.device), augment=augment, profile=profile)

        # Post-process
        return self._postprocess(preds, imgs, shapes_batch)

    def _preprocess(self, imgs, size):
        """Preprocesses images: loads, letterboxes, and converts to tensor batch with dynamic shapes."""
        if isinstance(imgs, (str, Path)):
            imgs = [imgs]
        imgs_new = []
        shapes = []
        for img in imgs:
            if isinstance(img, (str, Path)):
                img = Image.open(requests.get(img, stream=True).raw if str(img).startswith("http") else img)
                img = exif_transpose(img)
            elif isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img = img.convert("RGB")
            imgs_new.append(img)
            shapes.append(img.size)

        # Letterbox
        imgs_letterboxed = []
        shapes_letterboxed = []
        for img, shape in zip(imgs_new, shapes):
            im, ratio, pad = letterbox(np.array(img), (size, size), auto=False, scaleup=True)
            imgs_letterboxed.append(im)
            shapes_letterboxed.append((shape, ratio, pad))

        # Stack
        im = np.stack(imgs_letterboxed, 0)  # assemble into batch
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im).to(self.model.device).float() / 255.0  # uint8 to fp16/32

        return im, shapes_letterboxed

    def _postprocess(self, preds, imgs, shapes_batch):
        """Post-processes predictions: applies NMS and scales boxes back to original image dimensions."""
        preds = non_max_suppression(
            preds,
            self.conf,
            self.iou,
            self.classes,
            self.agnostic,
            self.multi_label,
            max_det=self.max_det,
        )

        results = []
        for pred, (shape, ratio, pad) in zip(preds, shapes_batch):
            if pred is not None and len(pred):
                pred[:, :4] = scale_boxes(imgs.shape[2:], pred[:, :4], shape).round()
            results.append(pred)

        return results


class Classify(nn.Module):
    """YOLOv5 classification head with convolution, pooling, and linear classifier."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        """Initializes classification head with convolution, adaptive pooling, and linear layer."""
        super().__init__()
        self.conv = Conv(c1, c2, k, s, autopad(k, p), g)
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(c2, c2) if c1 != c2 else nn.Identity()

    def forward(self, x):
        """Applies convolution, average pooling, flattening, and linear transformation to input tensor."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        return self.fc(self.flat(self.avgpool(self.conv(x))))


class DetectMultiBackend(nn.Module):
    """YOLOv5 Multi-backend model wrapper supporting PyTorch, ONNX, TensorRT, and other formats."""

    def __init__(self, weights="nexuss.pt", device=torch.device("cpu"), dnn=False, data=None, fp16=False, fuse=True):
        """Initializes multi-backend model loader supporting various inference backends."""
        super().__init__()
        from models.experimental import attempt_load

        w = str(weights[0] if isinstance(weights, list) else weights)
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, triton = self._model_type(w)
        fp16 &= pt or jit or engine or onnx  # FP16
        nhwc = coreml or saved_model or pb or tflite or edgetpu  # BHWC formats (vs torch BCHW)
        stride = 32  # default stride
        cuda = torch.cuda.is_available() and device.type != "cpu"  # CUDA

        if pt:  # PyTorch
            model = attempt_load(weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse)
            stride = max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32
            names = model.module.names if hasattr(model, "module") else model.names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        elif jit:  # TorchScript
            LOGGER.info(f"Loading {w} for TorchScript inference...")
            extra_files = {"config.txt": ""}  # model metadata
            model = torch.jit.load(w, _extra_files=extra_files, map_location=device)
            model.half() if fp16 else model.float()
            if extra_files["config.txt"]:
                d = json.loads(extra_files["config.txt"])  # extra_files dict
                stride, names = int(d["stride"]), d["names"]
        elif dnn:  # ONNX OpenCV DNN
            LOGGER.info(f"Loading {w} for ONNX OpenCV DNN inference...")
            check_requirements("opencv-python>=4.5.4")
            net = cv2.dnn.readNetFromONNX(w)
        elif onnx:  # ONNX Runtime
            LOGGER.info(f"Loading {w} for ONNX Runtime inference...")
            cuda = torch.cuda.is_available() and device.type != "cpu"
            check_requirements(("onnx", "onnxruntime-gpu" if cuda else "onnxruntime"))
            import onnxruntime as ort
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]
            session = ort.InferenceSession(w, providers=providers)
            output_names = [x.name for x in session.get_outputs()]
            meta = session.get_modelmeta().custom_metadata_map  # metadata
            if "stride" in meta:
                stride, names = int(meta["stride"]), eval(meta["names"])
        elif xml:  # OpenVINO
            LOGGER.info(f"Loading {w} for OpenVINO inference...")
            check_requirements("openvino>=2023.0")  # requires openvino-dev: https://pypi.org/project/openvino-dev/
            from openvino.runtime import Core

            ie = Core()
            if not Path(w).is_file():  # if not *.xml
                w = next(Path(w).glob("*.xml"))  # get *.xml file from *_openvino_model dir
            network = ie.read_model(model=w, weights=Path(w).with_suffix(".bin"))
            if network.get_parameters()[0].get_layout().empty:
                network.get_parameters()[0].set_layout(ov.Layout("NCHW"))
            batch_dim = network.get_parameters()[0].get_partial_shape()
            if batch_dim.is_dynamic:
                pass  # dynamic shapes already configured
            exec = ie.compile_model(network, device_name="CPU")  # device_name="MYRIAD" for Intel NCS2
            output_layer = exec.output(0)
            meta = Path(w).with_suffix(".json")
            if meta.exists():
                names = json_load(meta)["names"]
        elif engine:  # TensorRT
            LOGGER.info(f"Loading {w} for TensorRT inference...")
            import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download

            if device.type == "cpu":
                device = torch.device("cuda:0")
            Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
            logger = trt.Logger(trt.Logger.INFO)
            with open(w, "rb") as f, trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
            context = model.create_execution_context()
            bindings = OrderedDict()
            output_names = []
            fp16 = False  # default updated below
            dynamic = False
            for i in range(model.num_bindings):
                name = model.get_binding_name(i)
                dtype = trt.nptype(model.get_binding_dtype(i))
                if model.binding_is_input(i):
                    if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                        dynamic = True
                        context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)))
                    shape = tuple(context.get_binding_shape(i))
                else:  # output
                    shape = tuple(context.get_binding_shape(i))
                    output_names.append(name)
                if dtype == np.float16:
                    fp16 = True
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            batch_size = bindings["images"].shape[0]  # if dynamic, this is instead max batch size
        elif coreml:  # CoreML
            LOGGER.info(f"Loading {w} for CoreML inference...")
            import coremltools as ct

            model = ct.models.MLModel(w)
        elif saved_model:  # TF SavedModel
            LOGGER.info(f"Loading {w} for TensorFlow SavedModel inference...")
            import tensorflow as tf

            keras = False  # assume TF1 saved_model
            model = tf.keras.models.load_model(w) if keras else tf.saved_model.load(w)
        elif pb:  # TF GraphDef https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            LOGGER.info(f"Loading {w} for TensorFlow GraphDef inference...")
            import tensorflow as tf

            def wrap_frozen_graph(gd, inputs, outputs):
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])
                return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
                               tf.nest.map_structure(x.graph.as_graph_element, outputs))

            gd = tf.Graph().as_graph_def()
            with open(w, "rb") as f:
                gd.ParseFromString(f.read())
            frozen_func = wrap_frozen_graph(gd, inputs="x:0", outputs="Identity:0")
        elif tflite or edgetpu:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
            try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
                from tflite_runtime.interpreter import Interpreter, load_delegate
            except ImportError:
                import tensorflow as tf

                Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate
            if edgetpu:  # TF Edge TPU https://coral.ai/software/#edgetpu-runtime
                LOGGER.info(f"Loading {w} for TensorFlow Lite Edge TPU inference...")
                delegate = {"Linux": "libedgetpu.so.1", "Darwin": "libedgetpu.1.dylib", "Windows": "edgetpu.dll"}[
                    platform.system()
                ]
                interpreter = Interpreter(model_path=w, experimental_delegates=[load_delegate(delegate)])
            else:  # TFLite
                LOGGER.info(f"Loading {w} for TensorFlow Lite inference...")
                interpreter = Interpreter(model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            if nhwc:
                _, height, width, _ = input_details[0]["shape"]  # get shape NHWC
            else:
                _, _, height, width = input_details[0]["shape"]  # get shape NCHW
        elif tfjs:  # TF.js
            raise NotImplementedError("ERROR: YOLOv5 TF.js inference is not supported")
        elif paddle:  # PaddlePaddle
            LOGGER.info(f"Loading {w} for PaddlePaddle inference...")
            check_requirements("paddlepaddle-gpu" if cuda else "paddlepaddle")
            import paddle.inference as pdi

            if not Path(w).is_file():  # if not *.pdmodel
                w = next(Path(w).glob("*.pdmodel"))  # get *.pdmodel file from *_paddle_model dir
            config = pdi.Config(str(w), str(Path(w).with_suffix(".pdiparams")))
            if cuda:
                config.enable_use_gpu(memory_pool_init_size_mb=2048, device_id=0)
            predictor = pdi.create_predictor(config)
            input_handle = predictor.get_input_handle(predictor.get_input_names()[0])
            output_names = predictor.get_output_names()
        elif triton:  # Triton Inference Server
            LOGGER.info(f"Using Triton Inference Server...")
            from utils.triton import TritonRemoteModel

            model = TritonRemoteModel(url=w)
            nhwc = model.runtime.startswith("tensorflow")
            stride = max(int(stride), 32)
            names = metadata["names"]
        else:
            from models.experimental import attempt_load

            model = attempt_load(weights, device=device, inplace=True, fuse=fuse)
            stride = max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32
            names = model.module.names if hasattr(model, "module") else model.names
            self.model = model

        # class attributes
        self.__dict__.update(locals())

    def forward(self, im, augment=False, visualize=False):
        """Runs inference on input image batch with optional augmentation and visualization."""
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC

        # PyTorch
        if self.pt:
            y = self.model(im, augment=augment, visualize=visualize)

        # TorchScript
        elif self.jit:
            y = self.model(im)

        # ONNX OpenCV DNN
        elif self.dnn:
            im = im.cpu().numpy()  # torch to numpy
            self.net.setInput(im)
            y = self.net.forward()

        # ONNX Runtime
        elif self.onnx:
            im = im.cpu().numpy()  # torch to numpy
            y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})

        # OpenVINO
        elif self.xml:
            im = im.cpu().numpy()  # FP32
            y = list(self.exec([im]).values())

        # TensorRT
        elif self.engine:
            if self.dynamic and im.shape != self.bindings["images"].shape:
                i = self.model.get_binding_index("images")
                self.context.set_binding_shape(i, im.shape)  # reshape if dynamic
                self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)
                for name in self.output_names:
                    i = self.model.get_binding_index(name)
                    self.bindings[name] = self.bindings[name]._replace(
                        shape=tuple(self.context.get_binding_shape(i))
                    )
            s = self.bindings["images"].shape
            assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
            self.binding_addrs["images"] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = [self.bindings[x].data for x in sorted(self.output_names)]

        # CoreML
        elif self.coreml:
            im = im.cpu().numpy()
            im_pil = [Image.fromarray(x) for x in im]
            y = self.model.predict({"image": im_pil})  # coordinates are xywh normalized
            if "confidence" in y:
                box = xywh2xyxy(y["coordinates"] * [[w, h, w, h]])  # xyxy pixels
                conf, cls = y["confidence"].max(1), y["confidence"].argmax(1).astype(np.float)
                y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
            else:
                y = list(y.values())

        # TF SavedModel
        elif self.saved_model:
            im = im.cpu().numpy()
            y = self.model(im, training=False) if self.keras else self.model(im)
            y = [y] if isinstance(y, np.ndarray) else y

        # TF GraphDef
        elif self.pb:
            im = im.cpu().numpy()
            y = self.frozen_func(x=self.tf.constant(im)).numpy()

        # TFLite or Edge TFLite
        elif self.tflite or self.edgetpu:
            input = self.input_details[0]
            int8 = input["dtype"] == np.uint8  # is TFLite quantized uint8 model
            if int8:
                scale, zero_point = input["quantization"]
                im = (im / scale + zero_point).astype(np.uint8)  # de-scale
            self.interpreter.set_tensor(input["index"], im)
            self.interpreter.invoke()
            y = []
            for output in self.output_details:
                x = self.interpreter.get_tensor(output["index"])
                if int8:
                    scale, zero_point = output["quantization"]
                    x = (x.astype(np.float32) - zero_point) * scale  # re-scale
                y.append(x)

        # PaddlePaddle
        elif self.paddle:
            im = im.cpu().numpy().astype(np.float32)
            self.input_handle.copy_from_cpu(im)
            self.predictor.run()
            y = [self.predictor.get_output_handle(x).copy_to_cpu() for x in self.output_names]

        # Triton Inference Server
        elif self.triton:
            y = self.model(im)

        # TF.js
        else:
            raise NotImplementedError(f"ERROR: {w} is not a supported format")

        # for y in (y if isinstance(y, list) else [y]):
        #     print(type(y), y.shape if hasattr(y, 'shape') else len(y))  # debug shapes
        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        """Converts numpy array to torch tensor with appropriate device placement."""
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, img_size=(1, 3, 640, 640)):
        """Warmup model by running inference once with dummy input of specified size."""
        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb
        if any(warmup_types) and self.device.type != "cpu":
            im = torch.empty(*img_size, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(2 if self.jit else 1):  #
                self.forward(im)  # warmup

    @staticmethod
    def _model_type(p="path/to/model.pt"):
        """Determines model type from file path or URL, returning tuple of format booleans."""
        from export import export_formats
        from utils.downloads import is_url

        sf = list(export_formats().Suffix)  # export suffixes
        if not is_url(p, check=False):
            check_suffix(p, sf)  # checks
        url = urlparse(p)  # if url may get FLAG_BASIC flag
        types = [s in Path(p).name for s in sf]
        types[8] &= not types[9]  # tflite &= not edgetpu
        triton = not any(types) and all([any([s in url.scheme, s in url.netloc]) for s in ["http", "grpc"]])
        types.append(triton)
        return types


def attempt_load(weights, device=None, inplace=True, fuse=True):
    """Loads and optionally fuses YOLOv5 model(s) from weights, supporting single or ensemble models."""
    from models.yolo import Detect, Model

    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(attempt_download(w), map_location="cpu")  # load
        args = {**DEFAULT_CFG_DICT, **ckpt["train_args"]} if "train_args" in ckpt else None  # combined args
        model.append(
            ckpt["ema" if ckpt.get("ema") else "model"].float().fuse().eval() if fuse else ckpt["model"].float().eval()
        )  # FP32 model

    # Module updates
    for m in model.modules():
        if not hasattr(m, "inplace"):
            m.inplace = False
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            m.inplace = inplace
        elif t is nn.Upsample and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(model) == 1:
        return model[-1]

    # Return detection ensemble
    LOGGER.info(f"Ensemble created with {weights}\n")
    for k in "names", "nc", "yaml":
        setattr(model, k, getattr(model[0], k))
    model.stride = model[int(torch.argmax(torch.tensor([m.stride.max() for m in model])))].stride
    assert all(model[0].nc == m.nc for m in model), f"Models have different class counts: {[m.nc for m in model]}"
    return model


class Ensemble(nn.Module):
    """Ensemble of models for combining predictions from multiple YOLOv5 model instances."""

    def __init__(self):
        """Initializes an empty ensemble to aggregate outputs from multiple models."""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Runs forward pass through all ensemble models and aggregates their outputs."""
        y = [module(x, augment, profile, visualize)[0] for module in self]
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


def attempt_download(file, repo="ultralytics/nexus", release="v7.0"):
    """Attempts to download a file from GitHub release assets if not found locally."""
    from utils.downloads import gsutil_getsize

    file = Path(str(file).strip().replace("'", ""))
    if not file.exists():
        name = Path(file).name
        assets = [f"nexus{size}{suffix}.pt" for size in "nsmlx" for suffix in ("", "6", "-cls", "-seg")]
        if str(file) in assets:
            file = ROOT / "weights" / file
            if not file.exists() and not (ROOT / ".." / file).exists():
                LOGGER.info(f"Downloading {repo}@{release}/{name} to {file}...")
                url = f"https://github.com/{repo}/releases/download"
                if name.startswith("nexus"):
                    # GitHub release assets
                    safe_download(file, url=f"{url}/{release}/{name}", min_bytes=1e5)
                else:
                    # Ultralytics assets
                    from ultralytics.hub.utils import GITHUB_ASSETS_STEMS

                    if name in GITHUB_ASSETS_STEMS:
                        safe_download(file, url=f"{url}/{release}/{name}", min_bytes=1e5)
        if not file.exists():
            LOGGER.error(f"ERROR: {file} not found. Download from https://github.com/{repo}/releases/")
            raise FileNotFoundError(f"ERROR: {file} not found. Download from https://github.com/{repo}/releases/")

    return str(file)


def scale_img(img, ratio=1.0, same_shape=False, gs=32):
    """Scales image tensor by ratio, optionally maintaining shape, padded to gs (grid size) stride."""
    if ratio == 1.0:
        return img
    h, w = img.shape[2:]
    s = (int(h * ratio), int(w * ratio))  # new size
    img = F.interpolate(img, size=s, mode="bilinear", align_corners=False)  # resize
    if not same_shape:  # pad to gs-multiple
        h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean


def copy_attr(a, b, include=(), exclude=()):
    """Copies attributes from object b to object a, with include/exclude filter lists."""
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        setattr(a, k, v)


def make_divisible(x, divisor):
    """Returns x rounded up to nearest divisor-multiple; ensures channel counts are divisible."""
    return math.ceil(x / divisor) * divisor


def initialize_weights(model):
    """Initializes model weights using default PyTorch initialization for various layer types."""
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True


def model_info(model, verbose=False, img_size=640):
    """Prints model information: parameters, gradients, GFLOPs for given image size."""
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        LOGGER.info(
            f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}"
        )
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace("module_list.", "")
            LOGGER.info(
                f"{i:>5} {name:>40} {str(p.requires_grad):>9} {p.numel():>12} {str(list(p.shape)):>20} "
                f"{p.mean():10.3g} {p.std():10.3g}"
            )

    try:  # FLOPs
        p = next(model.parameters())
        stride = max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32
        im = torch.empty((1, p.shape[1], stride, stride), device=p.device)  # input image in BCHW format
        flops = thop.profile(deepcopy(model), inputs=(im,), verbose=False)[0] / 1e9 * 2  # stride GFLOPs
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]  # expand if int/float
        fs = f", {flops * img_size[0] / stride * img_size[1] / stride:.1f} GFLOPs"  # 640x640 GFLOPs
    except Exception:
        fs = ""
    name = Path(model.yaml_file).stem if hasattr(model, "yaml_file") else model.__class__.__name__
    LOGGER.info(f"{name} summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}")


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True):
    """Rescales bounding boxes from img1_shape to img0_shape, with optional padding correction."""
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., [0, 2]] -= pad[0]  # x padding
        boxes[..., [1, 3]] -= pad[1]  # y padding
        boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def clip_boxes(boxes, shape):
    """Clips bounding box coordinates to image boundaries given shape (height, width)."""
    if isinstance(boxes, torch.Tensor):
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nm=0,
):
    """Runs Non-Maximum Suppression (NMS) on inference results, supporting dynamic batch sizes."""
    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = device.type == "mps"  # Apple Silicon
    if mps:  # MPS not fully supported yet, convert tensors to CPU
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU threshold {iou_thres}, valid values are between 0.0 and 1.0"

    # Settings
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        if not torch.isfinite(x).all():
            x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            iou = box_iou(boxes[i], boxes[i]) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            LOGGER.warning(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded

    return output


def validate_dynamic_onnx(model_path, img_size=640, device="cpu"):
    """Validates that an exported ONNX model with dynamic axes works correctly across various input shapes.

    Tests multiple batch sizes and resolutions to ensure the model handles dynamic shapes properly
    and produces consistent outputs. This is essential for deployment scenarios where input
    dimensions vary at runtime (e.g., video streams with different resolutions).

    Args:
        model_path: Path to the ONNX model file with dynamic axes.
        img_size: Base image size for validation (default: 640).
        device: Device to run validation on (default: 'cpu').

    Returns:
        bool: True if all validation tests pass, False otherwise.
    """
    import onnxruntime as ort

    LOGGER.info(f"Validating dynamic shapes for ONNX model: {model_path}")

    # Load ONNX model
    try:
        onnx_model = onnx.load(model_path)
        onnx.checker.check_model(onnx_model)
    except Exception as e:
        LOGGER.error(f"ONNX model validation failed: {e}")
        return False

    # Check dynamic axes configuration
    input_info = onnx_model.graph.input[0]
    input_name = input_info.name
    shape = [d.dim_value if d.dim_param == "" else d.dim_param for d in input_info.type.tensor_type.shape.dim]
    LOGGER.info(f"  Input shape spec: {shape}")

    has_dynamic_batch = len(shape) >= 1 and isinstance(shape[0], str)
    has_dynamic_height = len(shape) >= 3 and isinstance(shape[2], str)
    has_dynamic_width = len(shape) >= 4 and isinstance(shape[3], str)

    if not (has_dynamic_batch or has_dynamic_height or has_dynamic_width):
        LOGGER.warning("  No dynamic axes found in ONNX model. Export with dynamic_axes for full flexibility.")
        return False

    LOGGER.info(
        f"  Dynamic axes - batch: {has_dynamic_batch}, height: {has_dynamic_height}, width: {has_dynamic_width}"
    )

    # Check output dynamic axes
    if onnx_model.graph.output:
        output_info = onnx_model.graph.output[0]
        output_shape = [
            d.dim_value if d.dim_param == "" else d.dim_param for d in output_info.type.tensor_type.shape.dim
        ]
        LOGGER.info(f"  Output shape spec: {output_shape}")

    # Create ONNX Runtime session
    try:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        session = ort.InferenceSession(model_path, providers=providers)
        output_names = [o.name for o in session.get_outputs()]
    except Exception as e:
        LOGGER.error(f"Failed to create ONNX Runtime session: {e}")
        return False

    # Test various input shapes
    test_shapes = [
        (1, 3, img_size, img_size),  # single image, base size
        (2, 3, img_size, img_size),  # batch of 2
        (4, 3, img_size, img_size),  # batch of 4
        (1, 3, img_size // 2, img_size // 2),  # half resolution
        (1, 3, img_size * 2, img_size * 2),  # double resolution
        (1, 3, 320, 640),  # non-square, common video aspect
        (1, 3, 480, 640),  # non-square
        (2, 3, 384, 640),  # batch + non-square
        (1, 3, 128, 256),  # small resolution
    ]

    all_passed = True
    for test_shape in test_shapes:
        try:
            dummy_input = np.random.randn(*test_shape).astype(np.float32)
            outputs = session.run(output_names, {input_name: dummy_input})
            output_shape_actual = outputs[0].shape

            # Validate output shape is consistent with input
            if len(output_shape_actual) >= 1 and output_shape_actual[0] != test_shape[0]:
                LOGGER.warning(
                    f"  Shape mismatch for input {test_shape}: "
                    f"expected batch dim {test_shape[0]}, got {output_shape_actual[0]}"
                )
                all_passed = False
            else:
                LOGGER.info(f"  ✓ Input {test_shape} -> Output {output_shape_actual}")
        except Exception as e:
            LOGGER.error(f"  ✗ Failed for input shape {test_shape}: {e}")
            all_passed = False

    if all_passed:
        LOGGER.info(f"Dynamic shape validation PASSED for {model_path}")
    else:
        LOGGER.warning(f"Dynamic shape validation had FAILURES for {model_path}")

    return all_passed


def fuse_conv_and_bn(conv, bn):
    """Fuses a Conv2d and BatchNorm2d layer into a single Conv2d for faster inference."""
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    # Prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # Prepare spatial bias
    b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def feature_visualization(x, module_type, stage, n=32, save_dir=Path("runs/detect/exp")):
    """Visualizes intermediate feature maps from a module during inference for debugging."""
    if "Detect" not in module_type:
        batch, channels, height, width = x.shape  # batch, channels, height, width
        if height > 1 and width > 1:
            f = save_dir / f"stage{stage}_{module_type.split('.')[-1]}_features.png"  # filename
            blocks = torch.chunk(x[0].cpu(), channels // dim := min(channels, n), dim=0)  # select n features
            if not f.parent.exists():
                f.parent.mkdir(parents=True)
            LOGGER.info(f"Saving {f}... ({n}/{channels})")
            from torchvision.utils import make_grid

            Image.fromarray(make_grid(blocks, nrow=int(math.sqrt(len(blocks))), normalize=True).mul(255).byte().permute(
                1, 2, 0
            ).numpy()).save(f, quality=95)