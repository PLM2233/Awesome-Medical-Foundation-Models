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

| Year | Modality | Title | Journal | Model Backbone | Input Resolution | Dataset Scale | Code | Weights | Key Feature |
|------|----------|-------|---------|---|---|---|---|---|---|
| 2025 | MRI, 2D | MRI-CORE: A Foundation Model for Magnetic Resonance Imaging | arXiv | ViT-Base | 1024×1024 | 116.8k volumes, 6.9M slices | [GitHub](https://github.com/mazurowski-lab/mri_foundation) | [Google Drive](https://drive.google.com/file/d/1nPkTI3H0vsujlzwY8jxjKwAbOCTJv4yW/view) | MIM+DINOv2 |
| 2025 | MRI, 3D | Triad: Vision Foundation Model for 3D Magnetic Resonance Imaging | arXiv | SwinTransformer | Adaptive | 131K volumes | [GitHub](https://github.com/wangshansong1/Triad) | - | VoCo v2 |

---

### CT Models

| Year | Modality | Title | Journal | Model Backbone | Input Resolution | Dataset Scale | Code | Weights | Key Feature |
|------|----------|-------|---------|---|---|---|---|---|---|
| 2025 | CT/MR, 2D | Curia: A Multi-Modal Foundation Model for Radiology | arXiv | ViT-B/L | 512×512 | 228M DICOM files (164M CT, 64M MR) | [HuggingFace](https://huggingface.co/raidium/curia) | [HuggingFace](https://huggingface.co/raidium/curia) | DINOv2 |
| 2024 | CT, 3D | VoCo: A Simple-yet-Effective Volume Contrastive Learning Framework for 3D Medical Image Analysis | CVPR | SwinUNETR | 384, 64 | 1.6k CT scans | [GitHub](https://github.com/Luffy03/VoCo) | [HuggingFace](https://huggingface.co/Luffy503/VoCo) | VoCo |
| 2025 | CT, 3D | Vision Foundation Models for Computed Tomography (CT-FM) | arXiv | SegResNet | 24×128×128 | 148K scans | [GitHub](https://github.com/project-lighter/CT-FM) | [HuggingFace](https://huggingface.co/project-lighter) | Sim-CLR |
| 2025 | CT, 3D | Developing Generalist Foundation Models from a Multimodal Dataset for 3D Computed Tomography (CT-CLIP) | arXiv | ViT-B | - | 25.6k | [HuggingFace](https://huggingface.co/ibrahimethemhamamci/CT-CLIP) | [HuggingFace](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) | CLIP |
| 2025 | CT, 3D | Merlin: Vision Language Foundation Model for 3D Computed Tomography | arXiv | ResNet152 | - | 15.3K paired CT scans | [GitHub](https://github.com/StanfordMIMI/Merlin) | [HuggingFace](https://huggingface.co/stanfordmimi/Merlin) | CLIP |
| 2025 | CT, 3D | TAP-CT: 3D Task-Agnostic Pretraining of Computed Tomography Foundation Models | arXiv | ViT-Base (adaption) | 12×224×224 | 105K volumes | [HuggingFace](https://huggingface.co/fomofo/tap-ct-b-3d) | [HuggingFace](https://huggingface.co/fomofo/tap-ct-b-3d) | - |

---

### Ultrasound Models

| Year | Modality | Title | Journal | Model Backbone | Input Resolution | Dataset Scale | Code | Weights | Key Feature |
|------|----------|-------|---------|---|---|---|---|---|---|
| 2025 | US, 2D | URFM: A general Ultrasound Representation Foundation Model for advancing ultrasound image diagnosis | Iscience | ViT-Base | 224×224 | 1M images | [GitHub](https://github.com/sonovision-ai/URFM) | [HuggingFace](https://huggingface.co/QingboKang/URFM) | MIM+KD |
| 2024 | US, 2D | USFM: A universal ultrasound foundation model generalized to tasks and organs towards label efficient image analysis | MIA | ViT-Base | 224×224 | 3M | [GitHub](https://github.com/openmedlab/USFM) | [Google Drive](https://drive.google.com/file/d/1KRwXZgYterH895Z8EpXpR1L1eSMMJo4q/view) | MIM (spatial, frequency) |
| 2025 | US, 2D | TinyUSFM: Towards Compact and Efficient Ultrasound Foundation Models | arXiv | ViT-Tiny | 224×224 | 200k coreset images | [GitHub](https://github.com/MacDunno/TinyUSFM) | [Google Drive](https://drive.google.com/file/d/15R3hnH0ILO39rE1gs-UgJonRqbaYTSRB/view) | Coreset+KD |
| 2025 | US, 2D | A Fully Open and Generalizable Foundation Model for Ultrasound Clinical Applications (EchoCare) | arXiv | Swin Transformer | 256×256 | 4.5M | [GitHub](https://github.com/CAIR-HKISI/EchoCare) | Coming soon | MIM+Metadata CLS |
| 2024 | US, 2D | Privacy-Preserving Federated Foundation Model for Generalist Ultrasound Artificial Intelligence (UltraFedFM) | arXiv | - | - | 1M images | [GitHub](https://github.com/yuncheng97/UltraFedFM) | [SharePoint](https://cuhko365-my.sharepoint.com/personal/220019054_link_cuhk_edu_cn/_layouts/15/onedrive.aspx) | FL+MIM |

---

### Fundus Models

| Year | Modality | Title | Journal | Model Backbone | Input Resolution | Dataset Scale | Code | Weights | Key Feature |
|------|----------|-------|---------|---|---|---|---|---|---|
| 2025 | Fundus, 2D | Enhancing diagnostic accuracy in rare and common fundus diseases with a knowledge-rich vision-language model (RetiZero) | Nature Communications | RETFound | 224×224 | 342K images | [GitHub](https://github.com/LooKing9218/RetiZero) | [Google Drive](https://drive.google.com/file/d/14bMmnefO73_NL1Xc4x0A5qFNbuI7GqKM/view) | CLIP, MAE weight |
| 2024 | Fundus, 2D | MM-Retinal: Knowledge-Enhanced Foundational Pretraining with Fundus Image-Text Expertise (KeepFIT) | MICCAI | ResNet50 | 800×800 | 4.3K image-text pairs | [GitHub](https://github.com/lxirich/MM-Retinal) | [Google Drive](https://drive.google.com/drive/folders/1hPDt9noBnlL75mBpNKtfaYmTr4oQJBjP) | CLIP |
| 2024 | Fundus, 2D | EyeFound: A Multimodal Generalist Foundation Model for Ophthalmic Imaging | arXiv | RETFound | 224×224 | 2.78 million images | - | - | MIM |
| 2024 | Fundus, 2D | UrFound: Towards Universal Retinal Foundation Models via Knowledge-Guided Masked Modeling | MICCAI | ViT-base | 224×224 | 180k retinal images | [HuggingFace](https://huggingface.co/yyyyk/UrFound) | [HuggingFace](https://huggingface.co/yyyyk/UrFound) | MIM+MLM |
| 2024 | Fundus, 2D | A Disease-Specific Foundation Model Using Over 100K Fundus Images | arXiv | ResNet50 | 256, 512, 1024 | 0.1 million retinal images | [GitHub](https://github.com/Jang-Boa/Research-Foundation-Retina) | - | ImageNet+Fundus, SP, 2-step |
| 2025 | Fundus, 2D | A multimodal visual–language foundation model for computational ophthalmology (EyeCLIP) | npjDM | ViT-B/16 | 224×224 | 2.77 million | [GitHub](https://github.com/Michi-3000/EyeCLIP) | [Google Drive](https://drive.google.com/file/d/1u_pUJYPppbprVQQ5jaULEKKp-eJqaVw6/view) | CLIP+MIM |
| 2024 | Fundus, 2D | VisionUnite: A Vision-Language Foundation Model for Ophthalmology Enhanced with Clinical Knowledge | TPAMI | eva02_base_patch14_448 | 448 | 1.24 million image-text pairs | [GitHub](https://github.com/HUANGLIZI/VisionUnite) | [Google Drive](https://drive.google.com/file/d/1kbdpPklCdDxEgxcpsp4OgGjxvxKh5jpV/view) | CLIP+LLM |
| 2023 | Fundus, 2D | A foundation model for generalizable disease detection from retinal images (RETFound) | Nature | ViT-large | 224×224 | 1.6 million retinal images | [GitHub](https://github.com/rmaphoh/RETFound) | [HuggingFace](https://huggingface.co/YukunZhou) | MIM |
| 2024 | Fundus, 2D | VisionFM: A Vision Foundation Model for Generalist Ophthalmic Artificial Intelligence | NEJM AI | ViT-Base | 224×224 | 3.4 million ophthalmic images | [GitHub](https://github.com/ABILab-CUHK/VisionFM) | [GitHub](https://github.com/ABILab-CUHK/VisionFM) | iBOT |
| 2025 | Fundus, 2D | A Foundation Language-Image Model of the Retina (FLAIR): encoding expert knowledge in text supervision | MIA | ResNet-50 | 512×512 | 288k images | [GitHub](https://github.com/jusiro/FLAIR) | - | CLIP |

---

### Histopathology Models

| Year | Modality | Title | Journal | Model Backbone | Input Resolution | Dataset Scale | Code | Weights | Key Feature |
|------|----------|-------|---------|---|---|---|---|---|---|
| 2024 | Histopath, 2D | Towards a general-purpose foundation model for computational pathology (UNI) | Nature Medicine | ViT-l/16 | 224×224 | 100,426 WSIs | [GitHub](https://github.com/mahmoodlab/UNI) | [GitHub](https://github.com/mahmoodlab/UNI) | DINOv2 |
| 2024 | Histopath, 2D | A visual-language foundation model for computational pathology (CONCH) | Nature Medicine | conch_ViT-B-16 | 224×224 | 1.17 million image-caption pairs | [GitHub](https://github.com/mahmoodlab/CONCH) | [HuggingFace](https://huggingface.co/mahmoodlab/CONCH) | DINOv2 |
| 2024 | Histopath, 2D | A multimodal whole-slide foundation model for pathology (TITAN) | Nature Medicine | ViT-B-16 | 512×512 | 335,645 WSIs | [GitHub](https://github.com/mahmoodlab/TITAN) | [HuggingFace](https://huggingface.co/mahmoodlab/TITAN) | iBOT+CLIP |
| 2024 | Histopath, 2D | Virchow: A Million-Slide Digital Pathology Foundation Model | arXiv | ViT-H/14 | 224×224 | 1.5M whole slide histopathology images | [HuggingFace](https://huggingface.co/paige-ai/Virchow) | [HuggingFace](https://huggingface.co/paige-ai/Virchow) | DINOv2 |
| 2023-2024 | Histopath, 2D | Scaling Self-Supervised Learning for Histopathology with Masked Image Modeling (Phikon v2) | arXiv | iBOT ViT | 224×224 | 450M 20x magnification histology images | [HuggingFace](https://huggingface.co/owkin/phikon-v2) | [HuggingFace](https://huggingface.co/owkin/phikon-v2) | iBOT, DINOv2 |
| 2024 | Histopath, 2D | A pathology foundation model for cancer diagnosis and prognosis prediction (CHIEF) | Nature | ViT | 224×224 | 60,530 whole-slide images (WSIs) | [GitHub](https://github.com/hms-dbmi/CHIEF) | [GitHub](https://github.com/hms-dbmi/CHIEF) | CLIP |
| 2025 | Histopath, 2D | A generalizable pathology foundation model using a unified knowledge distillation pretraining framework (GPFM) | Nature BME | ViT | 512×512 | 190 million image-level samples | [GitHub](https://github.com/birkhoffkiki/GPFM/) | [HuggingFace](https://huggingface.co/majiabo/GPFM) | Novel Distillation |
| 2023 | Histopath, 2D | Quilt-1M: One Million Image-Text Pairs for Histopathology (Quilt-Net) | NeurIPS | ViT-B | 512×512 | 802,148 image and text pairs | [GitHub](https://github.com/wisdomikezogwo/quilt1m) | [HuggingFace](https://huggingface.co/wisdomik/QuiltNet-B-32) | CLIP |
| 2023 | Histopath, 2D | Pathology Language and Image Pre-Training (PLIP) | Nature Medicine | ViT | 224×224 | 208,414 pathology images with natural language descriptions | [GitHub](https://github.com/PathologyFoundation/plip) | [HuggingFace](https://huggingface.co/vinid/plip) | CLIP |

---

### X-ray Models

| Year | Modality | Title | Journal | Model Backbone | Input Resolution | Dataset Scale | Code | Weights | Key Feature |
|------|----------|-------|---------|---|---|---|---|---|---|
| 2025 | X-ray, 2D | A fully open AI foundation model applied to chest radiography (Ark+) | Nature | Swin-Large | 768×768 | 704k images | [GitHub](https://github.com/jlianglab/Ark) | [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSfQ0XLZRNUwbFPLzQo9cDWelHvt84q2Vh5_wS0Tu9Mt8PIAwQ/viewform) | Multi-task, KD, Supervised |
| 2024 | X-ray, 2D | Health AI Developer Foundations (CXR-FM) | - | EfficientNet-L2 | 1024×1024 | 821k images | [GitHub](https://github.com/google-health/cxr-foundation) | [HuggingFace](https://huggingface.co/google/cxr-foundation) | - |
| 2024 | X-ray, 2D | Exploring scalable medical image encoders beyond text supervision (RAD-DINO) | Nature MI | ViT-Base | 518×518 | 882k images | [HuggingFace](https://huggingface.co/microsoft/rad-dino) | [HuggingFace](https://huggingface.co/microsoft/rad-dino) | DINOv2 |
| 2022 | X-ray, 2D | Benchmarking and Boosting Transformers for Medical Image Classification (MIM-CXR) | MICCAI | Swin-Base | 224×224 | 926k images | [GitHub](https://github.com/JLiangLab/BenchmarkTransformers) | [GitHub](https://github.com/jlianglab/BenchmarkTransformers) | SimMIM |
| 2023 | X-ray, 2D | CheSS: Chest X-Ray Pre-trained Model via Self-supervised Contrastive Learning | JDI | ResNet-50 | 512×512 | 4.8M images | [GitHub](https://github.com/mi2rl/CheSS) | [Google Drive](https://drive.usercontent.google.com/download?id=1IfiuQdKV7en9DFaB0NqNdsDkVbdyoVyD) | MoCo v2 |
| 2023 | X-ray, 2D | Knowledge-enhanced Visual-Language Pre-training on Chest Radiology Images (KAD) | Nature Communications | ResNet-50 | 512×512 | 377k images | [GitHub](https://github.com/xiaoman-zhang/KAD) | [Google Drive](https://drive.google.com/drive/folders/1ArEgk-VNKZnXd5Fjkf6tiNA4YvBDZCCF) | CLIP |
| 2024 | X-ray, 2D | A Vision-Language Foundation Model to Enhance Efficiency of Chest X-ray Interpretation (CheXagent) | arXiv | SigLIP-Large | 512×512 | 8.5 million training samples | - | - | MLLM |
| 2025 | X-ray, 2D | A Vision-Language Foundation Model to Enhance Efficiency of Chest X-ray Interpretation (CheXFound) | TMI | ViT-L | 512×512 | 987K unique CXRs | [GitHub](https://github.com/RPIDIAL/CheXFound) | [Google Drive](https://drive.google.com/drive/folders/1GX2BWbujuVABtVpSZ4PTBykGULzrw806) | DINOv2 |

---

### Dermatology Models

| Year | Modality | Title | Journal | Model Backbone | Input Resolution | Dataset Scale | Code | Weights | Key Feature |
|------|----------|-------|---------|---|---|---|---|---|---|
| 2024 | Derm | Health AI Developer Foundations | - | BiT-M ResNet101x3 | 448×448 | Over 16K natural and dermatology images | [Google Developers](https://developers.google.com/health-ai-developer-foundations/derm-foundation) | [HuggingFace](https://huggingface.co/google/derm-foundation) | ConVIRT |
| 2025 | Derm | A multimodal vision foundation model for clinical dermatology (PanDerm) | Nature Medicine | ViT-L/16 | 224×224 | 2 million skin disease images | [GitHub](https://github.com/SiyuanYan1/PanDerm) | [GitHub](https://github.com/SiyuanYan1/PanDerm) | CLIP |
| 2024 | Derm | Transparent medical image AI via an image–text foundation model grounded in medical literature (MONET) | Nature Medicine | ViT-L/14 | 224×224 | 105,550 dermatological image–text pairs | [GitHub](https://github.com/suinleelab/MONET) | [AIMS Lab](https://aimslab.cs.washington.edu/MONET/weight_clip.pt) | CLIP |
| 2025 | Derm | Derm1M: A Million-scale Vision-Language Dataset Aligned with Clinical Ontology Knowledge for Dermatology (DermLIP/DermLIP-PanDerm) | ICCV | ViT-Base/16 | 224×224 | 1,029,761 dermatological image–text pairs | [GitHub](https://github.com/SiyuanYan1/Derm1M) | [HuggingFace](https://huggingface.co/redlessone/DermLIP_ViT-B-16) | CLIP |

---

### Microscopy Models

| Year | Modality | Title | Journal | Model Backbone | Input Resolution | Dataset Scale | Code | Weights | Key Feature |
|------|----------|-------|---------|---|---|---|---|---|---|
| 2025 | Microscopy | Unifying the Electron Microscopy Multiverse through a Large-scale Foundation Model (EM-DINO) | bioRxiv | - | - | 5.5 million 2D EM images | - | - | DINOv2 |
| 2025 | Microscopy | A Self-Supervised Foundation Model for Robust and Generalizable Representation Learning in STED Microscopy | bioRxiv | ViT | 224×224 | 37,387 images | [GitHub](https://github.com/FLClab/STED-FM) | [GitHub](https://github.com/FLClab/STED-FM) | MAE |
| 2024 | Microscopy | ViTally Consistent: Scaling Biological Representation Learning for Cell Microscopy | CVPR | ViT-G/8 | 256×256 | 8 billion microscopy image crops | [GitHub](https://github.com/recursionpharma/maes_microscopy) | [HuggingFace](https://huggingface.co/recursionpharma/OpenPhenom) | MAE |

---

## Contributing

We welcome contributions! Please feel free to submit a pull request to add new papers or update existing information. When adding new papers, please follow the existing format and ensure the information is accurate and complete.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
