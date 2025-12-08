# <p align=center>:fire:`Medical Foundation Models 2024-2025`:fire:</p>

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://makeapullrequest.com)

🔥🔥 Latest medical foundation models from 2024-2025 across diverse imaging modalities 🔥🔥

## Overview

This repository provides the latest curated list of foundation models in medical imaging released during 2024-2025, organized by imaging modality:
- **MRI**: Magnetic Resonance Imaging models (2D and 3D)
- **CT**: Computed Tomography models (2D and 3D)
- **Ultrasound**: Ultrasound imaging models
- **Fundus**: Retinal/Fundus imaging models
- **Histopathology**: Digital pathology and histopathology models
- **X-ray**: Chest X-ray and radiography models
- **Dermatology**: Skin disease and dermatology imaging models
- **Microscopy**: Electron and cellular microscopy models

We strongly encourage authors of relevant works to make a pull request and add their paper's information.

## Contents
- [MRI Models](#mri-models)
- [CT Models](#ct-models)
- [Ultrasound Models](#ultrasound-models)
- [Fundus Models](#fundus-models)
- [Histopathology Models](#histopathology-models)
- [X-ray Models](#x-ray-models)
- [Dermatology Models](#dermatology-models)
- [Microscopy Models](#microscopy-models)

---

## Papers

### MRI Models

| Year | Title | Journal | Model | Resolution | Scale | Code | Weights | Key Feature |
|:----:|--------|---------|---------|:----------:|--------|:---:|:-------:|---|
| 2025 | **MRI-CORE**: A Foundation Model for Magnetic Resonance Imaging | arXiv | ViT-Base | 1024×1024 | 116.8k vol, 6.9M slices | [✓](https://github.com/mazurowski-lab/mri_foundation) | [✓](https://drive.google.com/file/d/1nPkTI3H0vsujlzwY8jxjKwAbOCTJv4yW/view) | MIM+DINOv2 |
| 2025 | **Triad**: Vision Foundation Model for 3D MRI | arXiv | SwinTransformer | Adaptive | 131K volumes | [✓](https://github.com/wangshansong1/Triad) | - | VoCo v2 |

---

### CT Models

| Year | Title | Journal | Model | Resolution | Scale | Code | Weights | Key Feature |
|:----:|--------|---------|---------|:----------:|--------|:---:|:-------:|---|
| 2025 | **Curia**: Multi-Modal Foundation Model for Radiology | arXiv | ViT-B/L | 512×512 | 228M DICOM | [✓](https://huggingface.co/raidium/curia) | [✓](https://huggingface.co/raidium/curia) | DINOv2 |
| 2024 | **VoCo**: Volume Contrastive Learning for 3D CT Analysis | CVPR | SwinUNETR | 384, 64 | 1.6k scans | [✓](https://github.com/Luffy03/VoCo) | [✓](https://huggingface.co/Luffy503/VoCo) | VoCo |
| 2025 | **CT-FM**: Vision Foundation Models for CT | arXiv | SegResNet | 24×128×128 | 148K scans | [✓](https://github.com/project-lighter/CT-FM) | [✓](https://huggingface.co/project-lighter) | Sim-CLR |
| 2025 | **CT-CLIP**: Multimodal Foundation Models for 3D CT | arXiv | ViT-B | - | 25.6k | [✓](https://huggingface.co/ibrahimethemhamamci/CT-CLIP) | [✓](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) | CLIP |
| 2025 | **Merlin**: Vision-Language Model for 3D CT | arXiv | ResNet152 | - | 15.3K scans | [✓](https://github.com/StanfordMIMI/Merlin) | [✓](https://huggingface.co/stanfordmimi/Merlin) | CLIP |
| 2025 | **TAP-CT**: 3D Task-Agnostic Pretraining for CT | arXiv | ViT-Base | 12×224×224 | 105K vol | [✓](https://huggingface.co/fomofo/tap-ct-b-3d) | [✓](https://huggingface.co/fomofo/tap-ct-b-3d) | - |

---

### Ultrasound Models

| Year | Title | Journal | Model | Resolution | Scale | Code | Weights | Key Feature |
|:----:|--------|---------|---------|:----------:|--------|:---:|:-------:|---|
| 2025 | **URFM**: Ultrasound Representation Foundation Model | Iscience | ViT-Base | 224×224 | 1M images | [✓](https://github.com/sonovision-ai/URFM) | [✓](https://huggingface.co/QingboKang/URFM) | MIM+KD |
| 2024 | **USFM**: Universal Ultrasound Foundation Model | MIA | ViT-Base | 224×224 | 3M images | [✓](https://github.com/openmedlab/USFM) | [✓](https://drive.google.com/file/d/1KRwXZgYterH895Z8EpXpR1L1eSMMJo4q/view) | MIM |
| 2025 | **TinyUSFM**: Compact Ultrasound Foundation Model | arXiv | ViT-Tiny | 224×224 | 200k images | [✓](https://github.com/MacDunno/TinyUSFM) | [✓](https://drive.google.com/file/d/15R3hnH0ILO39rE1gs-UgJonRqbaYTSRB/view) | Coreset+KD |
| 2025 | **EchoCare**: Open Ultrasound Foundation Model | arXiv | Swin | 256×256 | 4.5M images | [✓](https://github.com/CAIR-HKISI/EchoCare) | Soon | MIM+Metadata |
| 2024 | **UltraFedFM**: Privacy-Preserving Federated US Model | arXiv | - | - | 1M images | [✓](https://github.com/yuncheng97/UltraFedFM) | [✓](https://cuhko365-my.sharepoint.com/) | FL+MIM |

---

### Fundus Models

| Year | Title | Journal | Model | Resolution | Scale | Code | Weights | Key Feature |
|:----:|--------|---------|---------|:----------:|--------|:---:|:-------:|---|
| 2025 | **RetiZero**: Knowledge-Rich Vision-Language Model for Fundus | NC | RETFound | 224×224 | 342K images | [✓](https://github.com/LooKing9218/RetiZero) | [✓](https://drive.google.com/file/d/14bMmnefO73_NL1Xc4x0A5qFNbuI7GqKM/view) | CLIP+MAE |
| 2024 | **MM-Retinal/KeepFIT**: Knowledge-Enhanced Fundus Pretraining | MICCAI | ResNet50 | 800×800 | 4.3K pairs | [✓](https://github.com/lxirich/MM-Retinal) | [✓](https://drive.google.com/drive/folders/1hPDt9noBnlL75mBpNKtfaYmTr4oQJBjP) | CLIP |
| 2024 | **EyeFound**: Multimodal Generalist Ophthalmic Model | arXiv | RETFound | 224×224 | 2.78M images | - | - | MIM |
| 2024 | **UrFound**: Universal Retinal Foundation Models | MICCAI | ViT-base | 224×224 | 180k images | [✓](https://huggingface.co/yyyyk/UrFound) | [✓](https://huggingface.co/yyyyk/UrFound) | MIM+MLM |
| 2024 | **Fundus-FM**: Disease-Specific Retinal Model | arXiv | ResNet50 | 256-1024 | 0.1M images | [✓](https://github.com/Jang-Boa/Research-Foundation-Retina) | - | ImageNet+SP |
| 2025 | **EyeCLIP**: Multimodal Vision-Language Model | npjDM | ViT-B/16 | 224×224 | 2.77M images | [✓](https://github.com/Michi-3000/EyeCLIP) | [✓](https://drive.google.com/file/d/1u_pUJYPppbprVQQ5jaULEKKp-eJqaVw6/view) | CLIP+MIM |
| 2024 | **VisionUnite**: Ophthalmology Model w/ Clinical Knowledge | TPAMI | eva02 | 448 | 1.24M pairs | [✓](https://github.com/HUANGLIZI/VisionUnite) | [✓](https://drive.google.com/file/d/1kbdpPklCdDxEgxcpsp4OgGjxvxKh5jpV/view) | CLIP+LLM |
| 2023 | **RETFound**: Generalizable Disease Detection Model | Nature | ViT-L | 224×224 | 1.6M images | [✓](https://github.com/rmaphoh/RETFound) | [✓](https://huggingface.co/YukunZhou) | MIM |
| 2024 | **VisionFM**: Vision Foundation Model for Ophthalmology | NEJM AI | ViT-Base | 224×224 | 3.4M images | [✓](https://github.com/ABILab-CUHK/VisionFM) | [✓](https://github.com/ABILab-CUHK/VisionFM) | iBOT |
| 2025 | **FLAIR**: Foundation Language-Image Model of Retina | MIA | ResNet-50 | 512×512 | 288k images | [✓](https://github.com/jusiro/FLAIR) | - | CLIP |

---

### Histopathology Models

| Year | Title | Journal | Model | Resolution | Scale | Code | Weights | Key Feature |
|:----:|--------|---------|---------|:----------:|--------|:---:|:-------:|---|
| 2024 | **UNI**: General-Purpose Computational Pathology Model | Nat Med | ViT-l/16 | 224×224 | 100K WSIs | [✓](https://github.com/mahmoodlab/UNI) | [✓](https://github.com/mahmoodlab/UNI) | DINOv2 |
| 2024 | **CONCH**: Visual-Language Pathology Model | Nat Med | ViT-B-16 | 224×224 | 1.17M pairs | [✓](https://github.com/mahmoodlab/CONCH) | [✓](https://huggingface.co/mahmoodlab/CONCH) | DINOv2 |
| 2024 | **TITAN**: Multimodal Whole-Slide Model | Nat Med | ViT-B-16 | 512×512 | 335K WSIs | [✓](https://github.com/mahmoodlab/TITAN) | [✓](https://huggingface.co/mahmoodlab/TITAN) | iBOT+CLIP |
| 2024 | **Virchow**: Million-Slide Digital Pathology Model | arXiv | ViT-H/14 | 224×224 | 1.5M images | [✓](https://huggingface.co/paige-ai/Virchow) | [✓](https://huggingface.co/paige-ai/Virchow) | DINOv2 |
| 2023-2024 | **Phikon v2**: Self-Supervised Histopathology Learning | arXiv | iBOT ViT | 224×224 | 450M images | [✓](https://huggingface.co/owkin/phikon-v2) | [✓](https://huggingface.co/owkin/phikon-v2) | iBOT+DINOv2 |
| 2024 | **CHIEF**: Cancer Diagnosis & Prognosis Model | Nature | ViT | 224×224 | 60K WSIs | [✓](https://github.com/hms-dbmi/CHIEF) | [✓](https://github.com/hms-dbmi/CHIEF) | CLIP |
| 2025 | **GPFM**: Generalizable Pathology Foundation Model | Nat BME | ViT | 512×512 | 190M samples | [✓](https://github.com/birkhoffkiki/GPFM/) | [✓](https://huggingface.co/majiabo/GPFM) | Distillation |
| 2023 | **Quilt-1M/Quilt-Net**: Histopathology Image-Text Dataset | NeurIPS | ViT-B | 512×512 | 802K pairs | [✓](https://github.com/wisdomikezogwo/quilt1m) | [✓](https://huggingface.co/wisdomik/QuiltNet-B-32) | CLIP |
| 2023 | **PLIP**: Pathology Language & Image Pretraining | Nat Med | ViT | 224×224 | 208K images | [✓](https://github.com/PathologyFoundation/plip) | [✓](https://huggingface.co/vinid/plip) | CLIP |

---

### X-ray Models

| Year | Title | Journal | Model | Resolution | Scale | Code | Weights | Key Feature |
|:----:|--------|---------|---------|:----------:|--------|:---:|:-------:|---|
| 2025 | **Ark+**: Open AI Foundation Model for Chest Radiography | Nature | Swin-L | 768×768 | 704k images | [✓](https://github.com/jlianglab/Ark) | [✓](https://docs.google.com/forms/d/e/1FAIpQLSfQ0XLZRNUwbFPLzQo9cDWelHvt84q2Vh5_wS0Tu9Mt8PIAwQ/viewform) | Multi-task+KD |
| 2024 | **CXR-FM**: Health AI Foundation Model for CXR | - | EfficientNet-L2 | 1024×1024 | 821k images | [✓](https://github.com/google-health/cxr-foundation) | [✓](https://huggingface.co/google/cxr-foundation) | - |
| 2024 | **RAD-DINO**: Scalable Medical Image Encoders | Nat MI | ViT-Base | 518×518 | 882k images | [✓](https://huggingface.co/microsoft/rad-dino) | [✓](https://huggingface.co/microsoft/rad-dino) | DINOv2 |
| 2022 | **MIM-CXR**: Transformers for Medical Image Classification | MICCAI | Swin-Base | 224×224 | 926k images | [✓](https://github.com/JLiangLab/BenchmarkTransformers) | [✓](https://github.com/jlianglab/BenchmarkTransformers) | SimMIM |
| 2023 | **CheSS**: Chest X-Ray Self-Supervised Model | JDI | ResNet-50 | 512×512 | 4.8M images | [✓](https://github.com/mi2rl/CheSS) | [✓](https://drive.usercontent.google.com/download?id=1IfiuQdKV7en9DFaB0NqNdsDkVbdyoVyD) | MoCo v2 |
| 2023 | **KAD**: Knowledge-Enhanced Visual-Language CXR Model | NC | ResNet-50 | 512×512 | 377k images | [✓](https://github.com/xiaoman-zhang/KAD) | [✓](https://drive.google.com/drive/folders/1ArEgk-VNKZnXd5Fjkf6tiNA4YvBDZCCF) | CLIP |
| 2024 | **CheXagent**: Vision-Language CXR Interpretation Model | arXiv | SigLIP-L | 512×512 | 8.5M samples | - | - | MLLM |
| 2025 | **CheXFound**: Vision-Language CXR Model | TMI | ViT-L | 512×512 | 987K CXRs | [✓](https://github.com/RPIDIAL/CheXFound) | [✓](https://drive.google.com/drive/folders/1GX2BWbujuVABtVpSZ4PTBykGULzrw806) | DINOv2 |

---

### Dermatology Models

| Year | Title | Journal | Model | Resolution | Scale | Code | Weights | Key Feature |
|:----:|--------|---------|---------|:----------:|--------|:---:|:-------:|---|
| 2024 | **Derm-FM**: Health AI Foundation Model | - | ResNet101x3 | 448×448 | 16K+ images | [✓](https://developers.google.com/health-ai-developer-foundations/derm-foundation) | [✓](https://huggingface.co/google/derm-foundation) | ConVIRT |
| 2025 | **PanDerm**: Multimodal Vision Model for Dermatology | Nat Med | ViT-L/16 | 224×224 | 2M images | [✓](https://github.com/SiyuanYan1/PanDerm) | [✓](https://github.com/SiyuanYan1/PanDerm) | CLIP |
| 2024 | **MONET**: Transparent Medical Image AI for Dermatology | Nat Med | ViT-L/14 | 224×224 | 105K pairs | [✓](https://github.com/suinleelab/MONET) | [✓](https://aimslab.cs.washington.edu/MONET/weight_clip.pt) | CLIP |
| 2025 | **Derm1M/DermLIP**: Million-Scale Dermatology Dataset | ICCV | ViT-B/16 | 224×224 | 1.03M pairs | [✓](https://github.com/SiyuanYan1/Derm1M) | [✓](https://huggingface.co/redlessone/DermLIP_ViT-B-16) | CLIP |

---

### Microscopy Models

| Year | Title | Journal | Model | Resolution | Scale | Code | Weights | Key Feature |
|:----:|--------|---------|---------|:----------:|--------|:---:|:-------:|---|
| 2025 | **EM-DINO**: Electron Microscopy Foundation Model | bioRxiv | - | - | 5.5M images | - | - | DINOv2 |
| 2025 | **STED-FM**: Self-Supervised STED Microscopy Model | bioRxiv | ViT | 224×224 | 37K images | [✓](https://github.com/FLClab/STED-FM) | [✓](https://github.com/FLClab/STED-FM) | MAE |
| 2024 | **OpenPhenom**: Cell Microscopy Foundation Model | CVPR | ViT-G/8 | 256×256 | 8B crops | [✓](https://github.com/recursionpharma/maes_microscopy) | [✓](https://huggingface.co/recursionpharma/OpenPhenom) | MAE |

---

## Contributing

We welcome contributions! Please feel free to submit a pull request to add new papers or update existing information. When adding new papers, please follow the existing format and ensure the information is accurate and complete.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
