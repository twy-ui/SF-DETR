# SF-DETR  
**Enhancing UAV Small Object Detection via Spatial-Frequency Synergy and Polarity-Aware Attention**

##  Introduction

Object detection in UAV imagery faces significant challenges, including:

- Blurred target edges — High flight altitude and complex imaging conditions cause edge information loss
-Complex background interference — Dense distribution and cluttered backgrounds submerge weak target signals
-Low resolution and poor recognizability of small targets — Objects often occupy only tens or even a few pixels
-Real-time constraints — UAV applications demand efficient inference on edge devices  

To address these issues, we propose **SF-DETR**, a transformer-based detection framework integrating:

- **Spatial-Frequency Hybrid-Enhanced Multi-scale Feature Fusion (SFH-MFF)**  
- **Dual-Linear Attention Frequency-domain Interaction Module (DL-AFIM)**   

The proposed design improves small-object perception, foreground-background separation, and computational efficiency for UAV detection scenarios.

---

##  Framework Overview

SF-DETR introduces three core components:

### 1️ Spatial-Frequency Hybrid-Enhanced Multi-scale Feature Fusion (SFH-MFF)
A lightweight backbone network embedding a learnable frequency selection module to achieve complementary representation of spatial and frequency-domain features.

### 2 Dual-Linear Attention Frequency-domain Interaction Module (DL-AFIM)
 **Dual-Channel Positive-Negative Linear Attention (DC-PNLA)**  Decomposes Query/Key into positive/negative components to suppress background noise
 **Frequency-domain Feed-Forward Network (FFT-FFN)** Low-rank quantization matrix filtering high/mid-frequency components
 **Adaptive Multi-scale Frequency-domain Convolution Dynamic Fusion (AdaptFuse-FD)** Multi-scale DWConv (3×3, 5×5, 7×7) with residual connections for robust 
 

---

##  Project Structure
SF-DETR/main  
│  
├── ultralytics              
├       ├──cfg/           # Training configuration files
├       ├──utils/         # Utilities (metrics, visualization, tools) 
├       ├──nn/            # Provides reusable neural network layers and custom operators 
├
├── dataset/           # Dataset processing scripts   
├── train.py           # Training entry point  
├── val.py             # Validation and inference entry point  
├── requirements.txt   # Dependency list  


---

## ⚙️ Requirements

- Python ≥ 3.9  
- PyTorch ≥ 1.12  
- TorchVision ≥ 0.13  
- CUDA ≥ 11.x  
- pycocotools  
- timm  
- einops  

Install dependencies:

```bash
pip install -r requirements.txt
```

---

##  Datasets

We evaluate SF-DETR on the following datasets:

- **VisDrone2019**
- **UAVDT**
- **SIMD**

Please download the datasets from their official websites.

### Dataset Directory Structure

```
data/
├── VisDrone/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   │
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
│       
├── UAVDT/
│   ├── images/
│   │   ├── train/
│   │   ├── val/   
│   │
│   └── labels/
│       ├── train/
│       ├── val/
│
├── SIMD/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    │
    └── labels/
        ├── train/
        ├── val/
        └── test/
```

After downloading, modify the dataset path in the corresponding config file if necessary.

---

##  Training

```bash
python train.py 
```

---

##  Validation and Inference

```bash
python val.py 
```
Detection results will be saved in:
```
runs/train/
```

---

##  Experimental Results

SF-DETR achieves:

- Improved AP and AP50 on VisDrone2019, UAVDT and SIMD 
- Significant gains in small-object detection  
- Competitive real-time inference performance  

Detailed quantitative comparisons are provided in the paper.

---

##  Reproducibility

To reproduce the reported results:

1. Install the required dependencies  
2. Prepare datasets following the structure above  
3. Run validation using the provided config files  

All hyperparameters and training settings are included in the configuration files.

---

##  Citation

If you use this code in your research, please cite:

```bibtex
@article{sf-detr,
  title={Enhancing UAV Small Object Detection via Spatial-Frequency Synergy and Polarity-Aware Attention},
  author={Tang, Weiyan and Sun, Fuzhen and Jing, Zihao and Zhu, Zhuangrui and Li, Yudong and Wang, Shaoqing},
  journal={The Visual Computer},
  year={2026}
}
```

---

##  Resources

- Code: https://github.com/twy-ui/SF-DETR
- DOI: To be added  

---

##  Contact

Weiyan Tang  
Email: (twy124189@163.com)
  
Fuzhen Sun 
Email: (sunfuzhen@sdut.edu.cn)
