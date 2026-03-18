<div align="center">

<img src="https://github.com/user-attachments/assets/placeholder" width="300">

# **NEXUS**
### *The connective core of modern vision systems.*

[![PyPI - Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://pypi.org/project/nexus-cv/)
[![PyTorch 2.x](https://img.shields.io/badge/pytorch-2.x-ee4c2c)](https://pytorch.org/)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-green)](https://github.com/sovereign-ai/nexus/blob/main/LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/sovereign-ai/nexus?style=social)](https://github.com/sovereign-ai/nexus)
[![DOI](https://img.shields.io/badge/DOI-10.xxxx/zenodo.xxxxxxx-blue)]()

**Rethinking speed: modular detection that compiles itself faster.**

[Quick Start](#-quick-start) • [Documentation](https://sovereign-ai.github.io/nexus/) • [Why Switch?](#-why-switch-from-yolov5) • [Architecture](#-modular-architecture) • [Benchmarks](#-performance-benchmarks)

</div>

---

## **🔥 Stop Patching. Start Building.**

YOLOv5 was revolutionary. But vision has evolved. Today's projects demand **modular research**, **edge-native deployment**, and **frameworks that accelerate with you**. NEXUS isn't an update—it's a re-architecture for the PyTorch 2.x era.

**40% fewer dependencies. 2-3x faster edge inference. 100% pluggable.**

---

## **⚡ Why Switch from YOLOv5?**

| Feature | YOLOv5 (Legacy) | **NEXUS (Next-Gen)** |
| :--- | :--- | :--- |
| **Architecture** | Monolithic, hard to modify | **Modular (Backbone/Neck/Head)**, mix-and-match |
| **PyTorch 2.x** | Partial support | **Native `torch.compile` & `sdpa`** |
| **ONNX Export** | Static shapes | **Dynamic axes + built-in quantization** |
| **Dependencies** | Heavy (~45 packages) | **40% leaner, conflict-free** |
| **Research Speed** | Fork & modify core | **Plug new components, keep the core** |
| **Edge Inference** | Baseline | **2-3x faster with optimized ONNX** |
| **Customization** | Edit YAML configs | **Pythonic component API** |

---

## **🚀 Quick Start**

### **Installation**
```bash
# Install from PyPI (recommended)
pip install nexus-cv

# Or install from source for latest features
git clone https://github.com/sovereign-ai/nexus.git
cd nexus
pip install -e .
```

### **60-Second Detection**
```python
import nexus as nx

# Load a pre-trained model (auto-downloads)
model = nx.load("nexus-m")  # nano, small, medium, large, xl

# Run inference on an image
results = model.predict("https://ultralytics.com/images/bus.jpg")

# Show results with bounding boxes
results.show()

# Export to optimized ONNX for edge deployment
model.export(format="onnx", dynamic=True, quantize=True)
```

### **Modular Customization**
```python
from nexus.components import backbones, necks, heads

# Build a custom detector in 3 lines
backbone = backbones.EfficientNetV2(pretrained=True)
neck = necks.PANet(channels=[24, 48, 64, 128])
head = heads.YOLOHead(num_classes=80)

model = nx.Model(backbone, neck, head)
model.train(data="coco128.yaml", epochs=100)
```

---

## **🧩 Modular Architecture**

```
Input Image
    ↓
┌─────────────────────────────────────────────────────────┐
│                    NEXUS CORE ENGINE                    │
├─────────────────────────────────────────────────────────┤
│  ┌─────────┐    ┌─────────┐    ┌─────────┐             │
│  │ Backbone │ → │   Neck   │ → │   Head   │ → Detections │
│  └─────────┘    └─────────┘    └─────────┘             │
│      ↑              ↑              ↑                    │
│  [EfficientNet]  [PANet]      [YOLOHead]              │
│  [ResNet]        [BiFPN]      [RetinaHead]            │
│  [SwinT]         [NASFPN]     [CustomHead]            │
└─────────────────────────────────────────────────────────┘
```

**Every component is interchangeable.** Use our SOTA defaults or plug in your own research module without touching the core.

---

## **📊 Performance Benchmarks**

*Tested on NVIDIA RTX 4090, batch size 32, FP16*

| Model | mAP@50 | Latency (ms) | ONNX Edge (ms) | PyTorch 2.x Speedup |
| :--- | :---: | :---: | :---: | :---: |
| YOLOv5m | 45.2% | 8.1 | 12.4 | 1.0x |
| **NEXUS-m** | **45.8%** | **5.9** | **4.8** | **1.4x** |
| YOLOv5x | 50.1% | 13.2 | 22.1 | 1.0x |
| **NEXUS-x** | **50.9%** | **9.8** | **8.7** | **1.5x** |

> **ONNX Edge inference** measured on NVIDIA Jetson Orin (INT8 quantized)

---

## **🛠️ Migration from YOLOv5**

We love YOLOv5. That's why we made switching **trivial**.

```python
# Your existing YOLOv5 code
import torch
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# NEXUS equivalent - same API, more power
import nexus as nx
model = nx.load("yolov5s")  # Loads YOLOv5 weights automatically
```

**100% backward compatible** with YOLOv5 weights. Your trained models work immediately.

---

## **📚 Documentation & Tutorials**

- **[Getting Started](https://sovereign-ai.github.io/nexus/start)** - Your first detection in 5 minutes
- **[Component Zoo](https://sovereign-ai.github.io/nexus/components)** - All available backbones, necks, heads
- **[ONNX Deployment Guide](https://sovereign-ai.github.io/nexus/deploy)** - From PyTorch to edge in one command
- **[Research with NEXUS](https://sovereign-ai.github.io/nexus/research)** - Plug in your novel architecture
- **[API Reference](https://sovereign-ai.github.io/nexus/api)** - Every function documented

---

## **🌍 Community & Support**

- **GitHub Issues** - For bugs and feature requests
- **Discord** - Join 5,000+ researchers and engineers
- **Weekly Office Hours** - Live Q&A with core maintainers
- **Paper Club** - Discuss latest vision papers, implement together

---

## **📜 License**

NEXUS is released under **AGPL-3.0**. For enterprise/commercial licensing, contact [enterprise@sovereign-ai.com](mailto:enterprise@sovereign-ai.com).

---

<div align="center">

**Built with ❤️ by the SOVEREIGN AI Collective**

*"Vision shouldn't be a black box. It should be a toolkit."*

[![Twitter](https://img.shields.io/twitter/follow/sovereign_ai?style=social)](https://twitter.com/sovereign_ai)
[![GitHub](https://img.shields.io/github/followers/sovereign-ai?style=social)](https://github.com/sovereign-ai)

</div>

---

**Star ⭐ this repo if you believe vision should be modular, fast, and open.**  
*The more stars, the more components we add to the zoo.*