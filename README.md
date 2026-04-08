# <p align=center>:fire:`Medical Foundation Models 2024-2026`:fire:</p>

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://makeapullrequest.com)

🔥🔥 Latest medical foundation models from 2024-2026 across diverse imaging modalities 🔥🔥


We strongly encourage authors of relevant works to make a pull request and add their paper's information.

## Contents
- [MRI Models](#mri-models)
- [CT Models](#ct-models)
- [3D Multimodal Models](#3d-models)
- [Ultrasound Models](#ultrasound-models)
- [Fundus Models](#fundus-models)
- [Histopathology Models](#histopathology-models)
- [X-ray Models](#x-ray-models)
- [Dermatology Models](#dermatology-models)
- [Microscopy Models](#microscopy-models)
- [Endoscopy Models](#endoscopy-models)
- [Generalist Medical Models](#generalist-medical-models)
- [Medical MLLM Models](#medical-mllm-models)
- [3D Medical MLLLM Models](#3d-medical-mllm-models)
---

## Papers

### MRI Models

| Year | Title | Journal | Model | Resolution | Scale | Code | Weights | Key Feature |
|:----:|--------|---------|---------|:----------:|--------|:---:|:-------:|---|
| 2025 | **MRI-CORE**: A Foundation Model for Magnetic Resonance Imaging | arXiv | ViT-Base | 1024×1024 | 116.8k volumes, 6.9M slices | [✓](https://github.com/mazurowski-lab/mri_foundation) | [✓](https://drive.google.com/file/d/1nPkTI3H0vsujlzwY8jxjKwAbOCTJv4yW/view) | MIM+DINOv2 |
| 2025 | **Triad**: Vision Foundation Model for 3D Magnetic Resonance Imaging | arXiv | SwinTransformer | Adaptive | 131K volumes | [✓](https://github.com/wangshansong1/Triad) | - | VoCo v2 |
| 2026 | **Decipher-MR**: A Vision-Language Foundation Model for 3D MRI Representations | arXiv | 3D-ViT-B | 8x8x8 | 200,000 MRI series from over 22,000 studies | [✓]() | - | Dinov2 for vision, CLIP for text |

---

### CT Models

| Year | Title | Journal | Model | Resolution | Scale | Code | Weights | Key Feature |
|:----:|--------|---------|---------|:----------:|--------|:---:|:-------:|---|
| 2025 | **Curia**: A Multi-Modal Foundation Model for Radiology | arXiv | ViT-B/L | 512×512 | 228M DICOM files (164M CT and 64M MR DICOM files) | [✓](https://huggingface.co/raidium/curia) | [✓](https://huggingface.co/raidium/curia) | DINOv2 |
| 2024 | **VoCo**: A Simple-yet-Effective Volume Contrastive Learning Framework for 3D Medical Image Analysis | CVPR | SwinUNETR | 384, 64 | 1.6k CT scans | [✓](https://github.com/Luffy03/VoCo) | [✓](https://huggingface.co/Luffy503/VoCo) | VoCo |
| 2025 | **CT-FM**: Vision Foundation Models for Computed Tomography | arXiv | SegResNet | 24×128×128 | 148K scans | [✓](https://github.com/project-lighter/CT-FM) | [✓](https://huggingface.co/project-lighter) | Sim-CLR |
| 2025 | **CT-CLIP**: Developing Generalist Foundation Models from a Multimodal Dataset for 3D Computed Tomography | arXiv | ViT-B | - | 25.6k | [✓](https://github.com/ibrahimethemhamamci/CT-CLIP) | [✓](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) | CLIP |
| 2025 | **Merlin**: Vision Language Foundation Model for 3D Computed Tomography | arXiv | ResNet152 | - | 15.3K paired CT scans | [✓](https://github.com/StanfordMIMI/Merlin) | [✓](https://huggingface.co/stanfordmimi/Merlin) | CLIP |
| 2025 | **TAP-CT**: 3D Task-Agnostic Pretraining of Computed Tomography Foundation Models | arXiv | ViT-Base (adaptation) | 12×224×224 | 105K volumes | [✓](https://huggingface.co/fomofo/tap-ct-b-3d) | [✓](https://huggingface.co/fomofo/tap-ct-b-3d) | - |

---
### 3D Models
| Year | Title | Journal | Model | Resolution | Scale | Code | Weights | Key Feature |
|:----:|--------|---------|---------|:----------:|--------|:---:|:-------:|---|
| 2025 | **3DINO** A generalizable 3D framework and model for self-supervised learning in medical imaging | npj Digital Medicine | ViT-L | 112 | ~100,000 3D scans from over 10 organs | [✓](https://github.com/AICONSlab/3DINO) | [✓](https://huggingface.co/AICONSlab/3DINO-ViT) | DINOv2+3D Adaptor|

### Ultrasound Models

| Year | Title | Journal | Model | Resolution | Scale | Code | Weights | Key Feature |
|:----:|--------|---------|---------|:----------:|--------|:---:|:-------:|---|
| 2025 | **URFM**: A general Ultrasound Representation Foundation Model for advancing ultrasound image diagnosis | Iscience | ViT-Base | 224×224 | 1M images | [✓](https://github.com/sonovision-ai/URFM) | [✓](https://huggingface.co/QingboKang/URFM) | MIM+KD |
| 2024 | **USFM**: A universal ultrasound foundation model generalized to tasks and organs towards label efficient image analysis | MIA | ViT-Base | 224×224 | 3M images | [✓](https://github.com/openmedlab/USFM) | [✓](https://drive.google.com/file/d/1KRwXZgYterH895Z8EpXpR1L1eSMMJo4q/view) | MIM (spatial, frequency) |
| 2025 | **TinyUSFM**: Towards Compact and Efficient Ultrasound Foundation Models | arXiv | ViT-Tiny | 224×224 | 200k coreset images | [✓](https://github.com/MacDunno/TinyUSFM) | [✓](https://drive.google.com/file/d/15R3hnH0ILO39rE1gs-UgJonRqbaYTSRB/view) | Coreset+KD |
| 2025 | **EchoCare**: A Fully Open and Generalizable Foundation Model for Ultrasound Clinical Applications | arXiv | Swin Transformer | 256×256 | 4.5M images | [✓](https://github.com/CAIR-HKISI/EchoCare) | [✓](https://huggingface.co/CAIR-HKISI/EchoCare) | MIM+Metadata CLS |
| 2024 | **UltraFedFM**: Privacy-Preserving Federated Foundation Model for Generalist Ultrasound Artificial Intelligence | arXiv | - | - | 1M images | [✓](https://github.com/yuncheng97/UltraFedFM) | [✓](https://cuhko365-my.sharepoint.com/personal/220019054_link_cuhk_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F220019054%5Flink%5Fcuhk%5Fedu%5Fcn%2FDocuments%2Fusfm%2FUltraFedFM%2Foutput%5Fdir&ga=1) | FL+MIM |

---

### Fundus Models

| Year | Title | Journal | Model | Resolution | Scale | Code | Weights | Key Feature |
|:----:|--------|---------|---------|:----------:|--------|:---:|:-------:|---|
| 2025 | **RetiZero**: Enhancing diagnostic accuracy in rare and common fundus diseases with a knowledge-rich vision-language model | NC | RETFound | 224×224 | 342K images | [✓](https://github.com/LooKing9218/RetiZero) | [✓](https://drive.google.com/file/d/14bMmnefO73_NL1Xc4x0A5qFNbuI7GqKM/view) | CLIP+MAE |
| 2024 | **MM-Retinal/KeepFIT**: Knowledge-Enhanced Foundational Pretraining with Fundus Image-Text Expertise | MICCAI | ResNet50 | 800×800 | 4.3K image-text pairs | [✓](https://github.com/lxirich/MM-Retinal) | [✓](https://drive.google.com/drive/folders/1hPDt9noBnlL75mBpNKtfaYmTr4oQJBjP) | CLIP |
| 2024 | **EyeFound**: A Multimodal Generalist Foundation Model for Ophthalmic Imaging | arXiv | RETFound | 224×224 | 2.78 million images | - | - | MIM |
| 2024 | **UrFound**: Towards Universal Retinal Foundation Models via Knowledge-Guided Masked Modeling | MICCAI | ViT-base | 224×224 | 180k retinal images | [✓](https://huggingface.co/yyyyk/UrFound) | [✓](https://huggingface.co/yyyyk/UrFound) | MIM+MLM |
| 2024 | **Fundus-FM**: A Disease-Specific Foundation Model Using Over 100K Fundus Images: Release and Validation for Abnormality and Multi-Disease Classification on Downstream Tasks | arXiv | ResNet50 | 256-1024 | 0.1 million retinal images | [✓](https://github.com/Jang-Boa/Research-Foundation-Retina) | - | ImageNet+Fundus, SP, 2-step |
| 2025 | **EyeCLIP**: A multimodal visual–language foundation model for computational ophthalmology | npjDM | ViT-B/16 | 224×224 | 2.77 million images | [✓](https://github.com/Michi-3000/EyeCLIP) | [✓](https://drive.google.com/file/d/1u_pUJYPppbprVQQ5jaULEKKp-eJqaVw6/view) | CLIP+MIM |
| 2024 | **VisionUnite**: A Vision-Language Foundation Model for Ophthalmology Enhanced with Clinical Knowledge | TPAMI | eva02_base_patch14_448 | 448×448 | 1.24 million image-text pairs | [✓](https://github.com/HUANGLIZI/VisionUnite) | [✓](https://drive.google.com/file/d/1kbdpPklCdDxEgxcpsp4OgGjxvxKh5jpV/view) | CLIP+LLM |
| 2023 | **RETFound**: A foundation model for generalizable disease detection from retinal images | Nature | ViT-L | 224×224 | 1.6 million retinal images | [✓](https://github.com/rmaphoh/RETFound) | [✓](https://huggingface.co/YukunZhou) | MIM |
| 2024 | **VisionFM**: A Vision Foundation Model for Generalist Ophthalmic Artificial Intelligence | NEJM AI | ViT-Base | 224×224 | 3.4 million ophthalmic images | [✓](https://github.com/ABILab-CUHK/VisionFM) | [✓](https://github.com/ABILab-CUHK/VisionFM) | iBOT |
| 2025 | **FLAIR**: A Foundation Language-Image Model of the Retina: encoding expert knowledge in text supervision | MIA | ResNet-50 | 512×512 | 288k images | [✓](https://github.com/jusiro/FLAIR) | - | CLIP |

---

### Histopathology Models

| Year | Title | Journal | Model | Resolution | Scale | Code | Weights | Key Feature |
|:----:|--------|---------|---------|:----------:|--------|:---:|:-------:|---|
| 2024 | **UNI**: Towards a general-purpose foundation model for computational pathology | Nat Med | ViT-l/16 | 224×224 | 100,426 WSIs | [✓](https://github.com/mahmoodlab/UNI) | [✓](https://github.com/mahmoodlab/UNI) | DINOv2 |
| 2024 | **CONCH**: A visual-language foundation model for computational pathology | Nat Med | ViT-B-16 | 224×224 | 1.17 million image-caption pairs | [✓](https://github.com/mahmoodlab/CONCH) | [✓](https://huggingface.co/mahmoodlab/CONCH) | DINOv2 |
| 2024 | **TITAN**: A multimodal whole-slide foundation model for pathology | Nat Med | ViT-B-16 | 512×512 | 335,645 WSIs | [✓](https://github.com/mahmoodlab/TITAN) | [✓](https://huggingface.co/mahmoodlab/TITAN) | iBOT+CLIP |
| 2024 | **Virchow**: A foundation model for clinical-grade computational pathology and rare cancers detection | Nat Med | ViT-H/14 | 224×224 | 1.5M whole slide histopathology images | [✓](https://huggingface.co/paige-ai/Virchow) | [✓](https://huggingface.co/paige-ai/Virchow) | DINOv2 |
| 2024 | **Virchow2**: Scaling Self-Supervised Mixed Magnification Models in Pathology | arXiv | ViT-H/14 | 224×224 | 3M whole slide images | - | [✓](https://huggingface.co/paige-ai/Virchow2) | DINOv2 |
| 2024 | **H-optimus-0**: Foundation model for histopathology | - | ViT | 224×224 | 500,000 H&E stained WSIs | [✓](https://github.com/bioptimus/releases/tree/main/models/h-optimus/v0) | [✓](https://huggingface.co/bioptimus/H-optimus-0) | - |
| 2025 | **H-optimus-1**: Foundation model for histopathology | - | ViT | 224×224 | 500,000 H&E stained WSIs | - | [✓](https://huggingface.co/bioptimus/H-optimus-1) | DINOv2+iBOT |
| 2025 | **mSTAR**: mSTAR: A Multimodal Knowledge-enhanced Whole-slide Pathology Foundation Model | arXiv | ViT | 224×224 | 26,169 slide-level modality pairs | [✓](https://github.com/Innse/mSTAR) | [✓](https://huggingface.co/Wangyh/mSTAR) | CLIP |
| 2023-2024 | **Phikon v2**: Scaling Self-Supervised Learning for Histopathology with Masked Image Modeling | arXiv | iBOT ViT | 224×224 | 450M 20x magnification histology images | [✓](https://huggingface.co/owkin/phikon-v2) | [✓](https://huggingface.co/owkin/phikon-v2) | iBOT+DINOv2 |
| 2024 | **CHIEF**: A pathology foundation model for cancer diagnosis and prognosis prediction | Nature | ViT | 224×224 | 60,530 whole-slide images (WSIs) | [✓](https://github.com/hms-dbmi/CHIEF) | [✓](https://github.com/hms-dbmi/CHIEF) | CLIP |
| 2025 | **GPFM**: A generalizable pathology foundation model using a unified knowledge distillation pretraining framework | Nat BME | ViT | 512×512 | 190 million image-level samples | [✓](https://github.com/birkhoffkiki/GPFM/) | [✓](https://huggingface.co/majiabo/GPFM) | Novel Distillation |
| 2023 | **Quilt-1M/Quilt-Net**: Quilt-1M: One Million Image-Text Pairs for Histopathology | NeurIPS | ViT-B | 512×512 | 802,148 image and text pairs | [✓](https://github.com/wisdomikezogwo/quilt1m) | [✓](https://huggingface.co/wisdomik/QuiltNet-B-32) | CLIP |
| 2023 | **PLIP**: Pathology Language and Image Pre-Training | Nat Med | ViT | 224×224 | 208,414 pathology images paired with natural language descriptions | [✓](https://github.com/PathologyFoundation/plip) | [✓](https://huggingface.co/vinid/plip) | CLIP |

---

### X-ray Models

| Year | Title | Journal | Model | Resolution | Scale | Code | Weights | Key Feature |
|:----:|--------|---------|---------|:----------:|--------|:---:|:-------:|---|
| 2025 | **Ark+**: A fully open AI foundation model applied to chest radiography | Nature | Swin-Large | 768×768 | 704k images | [✓](https://github.com/jlianglab/Ark) | [✓](https://docs.google.com/forms/d/e/1FAIpQLSfQ0XLZRNUwbFPLzQo9cDWelHvt84q2Vh5_wS0Tu9Mt8PIAwQ/viewform) | Multi-task, KD, Supervised |
| 2024 | **CXR-FM**: Health AI Developer Foundations | - | EfficientNet-L2 | 1024×1024 | 821k images | [✓](https://github.com/google-health/cxr-foundation) | [✓](https://huggingface.co/google/cxr-foundation) | - |
| 2024 | **RAD-DINO**: Exploring scalable medical image encoders beyond text supervision | Nat MI | ViT-Base | 518×518 | 882k images | [✓](https://huggingface.co/microsoft/rad-dino) | [✓](https://huggingface.co/microsoft/rad-dino) | DINOv2 |
| 2022 | **MIM-CXR**: Benchmarking and Boosting Transformers for Medical Image Classification | MICCAI | Swin-Base | 224×224 | 926k images | [✓](https://github.com/JLiangLab/BenchmarkTransformers) | [✓](https://github.com/jlianglab/BenchmarkTransformers) | SimMIM |
| 2023 | **CheSS**: Chest X-Ray Pre-trained Model via Self-supervised Contrastive Learning | JDI | ResNet-50 | 512×512 | 4.8M images | [✓](https://github.com/mi2rl/CheSS) | [✓](https://drive.usercontent.google.com/download?id=1IfiuQdKV7en9DFaB0NqNdsDkVbdyoVyD&export=download&authuser=0) | MoCo v2 |
| 2023 | **KAD**: Knowledge-enhanced Visual-Language Pre-training on Chest Radiology Images | NC | ResNet-50 | 512×512 | 377k images | [✓](https://github.com/xiaoman-zhang/KAD) | [✓](https://drive.google.com/drive/folders/1ArEgk-VNKZnXd5Fjkf6tiNA4YvBDZCCF?usp=sharing) | CLIP |
| 2024 | **CheXagent**: A Vision-Language Foundation Model to Enhance Efficiency of Chest X-ray Interpretation | arXiv | SigLIP-Large | 512×512 | 8.5 million training samples | - | - | MLLM |
| 2025 | **CheXFound**: Chest X-ray Foundation Model with Global and Local Representations Integration | TMI | ViT-L | 512×512 | 987K unique CXRs | [✓](https://github.com/RPIDIAL/CheXFound) | [✓](https://drive.google.com/drive/folders/1GX2BWbujuVABtVpSZ4PTBykGULzrw806) | DINOv2 |
| 2022 | **MedCLIP**: Contrastive Learning from Unpaired Medical Images and Text | EMNLP | Swin Transformer | 224×224 | 200K data | [✓](https://github.com/RyanWangZf/MedCLIP) | [✓](https://pypi.org/project/medclip/) | CLIP |

---

### Dermatology Models

| Year | Title | Journal | Model | Resolution | Scale | Code | Weights | Key Feature |
|:----:|--------|---------|---------|:----------:|--------|:---:|:-------:|---|
| 2024 | **Derm-FM**: Health AI Developer Foundations | - | BiT-M ResNet101x3 | 448×448 | over 16K natural and dermatology images | [✓](https://developers.google.com/health-ai-developer-foundations/derm-foundation) | [✓](https://huggingface.co/google/derm-foundation) | ConVIRT |
| 2025 | **PanDerm**: A multimodal vision foundation model for clinical dermatology | Nat Med | ViT-L/16 | 224×224 | 2 million skin disease images | [✓](https://github.com/SiyuanYan1/PanDerm) | [✓](https://github.com/SiyuanYan1/PanDerm) | CLIP |
| 2024 | **MONET**: Transparent medical image AI via an image–text foundation model grounded in medical literature | Nat Med | ViT-L/14 | 224×224 | 105,550 dermatological image–text pairs | [✓](https://github.com/suinleelab/MONET) | [✓](https://aimslab.cs.washington.edu/MONET/weight_clip.pt) | CLIP |
| 2025 | **Derm1M/DermLIP**: Derm1M: A Million-scale Vision-Language Dataset Aligned with Clinical Ontology Knowledge for Dermatology | ICCV | ViT-B/16 | 224×224 | 1,029,761 dermatological image–text pairs | [✓](https://github.com/SiyuanYan1/Derm1M) | [✓](https://huggingface.co/redlessone/DermLIP_ViT-B-16) | CLIP |

---

### Microscopy Models

| Year | Title | Journal | Model | Resolution | Scale | Code | Weights | Key Feature |
|:----:|--------|---------|---------|:----------:|--------|:---:|:-------:|---|
| 2025 | **EM-DINO**: Unifying the Electron Microscopy Multiverse through a Large-scale Foundation Model | bioRxiv | - | - | 5.5 million 2D EM images | - | - | DINOv2 |
| 2025 | **STED-FM**: A Self-Supervised Foundation Model for Robust and Generalizable Representation Learning in STED Microscopy | bioRxiv | ViT | 224×224 | 37,387 images | [✓](https://github.com/FLClab/STED-FM) | [✓](https://github.com/FLClab/STED-FM) | MAE |
| 2024 | **OpenPhenom**: ViTally Consistent: Scaling Biological Representation Learning for Cell Microscopy | CVPR | ViT-G/8 | 256×256 | 8 billion microscopy image crops | [✓](https://github.com/recursionpharma/maes_microscopy) | [✓](https://huggingface.co/recursionpharma/OpenPhenom) | MAE |

---

### Endoscopy Models

| Year | Title | Journal | Model | Resolution | Scale | Code | Weights | Key Feature |
|:----:|--------|---------|---------|:----------:|--------|:---:|:-------:|---|
| 2024 | **Endo-FM**: Foundation Model for Endoscopy Video Analysis via Large-scale Self-supervised Pre-train | MICCAI | ViT-Base, T | 224×224 | 5 million frames | [✓](https://github.com/med-air/Endo-FM) | [✓](https://mycuhk-my.sharepoint.com/personal/1155167044_link_cuhk_edu_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F1155167044%5Flink%5Fcuhk%5Fedu%5Fhk%2FDocuments%2FEndo%2DFM%2Fpretrain%5Fweights%2Fendo%5Ffm%2Epth&parent=%2Fpersonal%2F1155167044%5Flink%5Fcuhk%5Fedu%5Fhk%2FDocuments%2FEndo%2DFM%2Fpretrain%5Fweights&ga=1) | DINO |
| 2025 | **SurgVISTA**: Large-scale Self-supervised Video Foundation Model for Intelligent Surgery | arXiv | S-T ViT | - | - | - | - | - |
| 2024 | **EndoViT**: Pretraining vision transformers on a large collection of endoscopic images | IJCARS | ViT-Base | 224×224 | 700,000 images | [✓](https://github.com/DominikBatic/EndoViT) | [✓](https://huggingface.co/egeozsoy/EndoViT) | MAE |

---

### Generalist Medical Models

| Year | Title | Journal | Model | Resolution | Scale | Code | Weights | Key Feature |
|:----:|--------|---------|---------|:----------:|--------|:---:|:-------:|---|
| 2025 | **BIOMEDICA**: An Open Biomedical Image-Caption Archive, Dataset, and Vision-Language Models Derived from Scientific Literature (BMCA-CLIP) | arXiv | ViT-L-14 | 224×224 | 24 million unique image-text pairs from over 6 million articles | [✓](https://github.com/Ale9806/open_clip_with_biomedica) | [✓](https://huggingface.co/BIOMEDICA) | CLIP |
| 2025 | **MEDICALNARRATIVES**: Connecting Medical Vision and Language with Localized Narratives (GenMedClip) | arXiv | ViT-Base | 224×224 | 4.7M image-text pairs from videos and articles, with 1M samples containing dense annotations | - | [✓](https://huggingface.co/wisdomik/GenMedClip) | CLIP |
| 2025 | **BiomedCLIP**: A multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs | arXiv | ViT-Base | 224×224 | 15 million figure-caption pairs | [✓](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) | [✓](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) | CLIP |
| 2021 | **PubMedCLIP**: Does CLIP Benefit Visual Question Answering in the Medical Domain as Much as it Does in the General Domain? | EACL | ViT-B, RN50 | 224×224 | 80K samples | [✓](https://github.com/sarahESL/PubMedCLIP) | [✓](https://huggingface.co/sarahESL/PubMedCLIP) | CLIP |
| 2023 | **PMC-CLIP**: Contrastive Language-Image Pre-training using Biomedical Documents | arXiv | ViT-L, RN50 | 224×224 | 1.6M image-caption pairs | [✓](https://github.com/WeixiongLin/PMC-CLIP) | [✓](https://huggingface.co/datasets/axiong/pmc_oa) | CLIP |
| 2024 | **UniMed-CLIP**: Towards a Unified Image-Text Pretraining Paradigm for Diverse Medical Imaging Modalities | arXiv | ViT-Base/Large | 224, 336 | 5.3 million image-text pairs | [✓](https://github.com/mbzuai-oryx/UniMed-CLIP) | [✓](https://github.com/mbzuai-oryx/UniMed-CLIP) | CLIP |

---

### Medical MLLM Models

| Year | Title | Journal | Model | Resolution | Scale | Code | Weights | Key Feature |
|:----:|--------|---------|---------|:----------:|--------|:---:|:-------:|---|
| 2025 | **Lingshu**: A Generalist Foundation Model for Unified Multimodal Medical Understanding and Reasoning | arXiv | ViT-Huge-like | Adaptive | 5.05M medical multimodal and textual data | [✓](https://github.com/QwenLM/Qwen3-VL) | [✓](https://huggingface.co/collections/lingshu-medical-mllm/lingshu-mllms) | MLLM |
| 2025 | **UniMedVL**: Unifying Medical Multimodal Understanding and Generation through Observation-Knowledge-Analysis | arXiv | MoT | Adaptive | UniMed-5M | [✓](https://github.com/uni-medical/UniMedVL) | [✓](https://huggingface.co/General-Medical-AI/UniMedVL) | MLLM |
| 2025 | **Hulu-Med**: A Transparent Generalist Model towards Holistic Medical Vision-Language Understanding | arXiv | SigLIP-2D/3D | Adaptive | multimodal dataset of 16.7 million samples | [✓](https://github.com/ZJUI-AI4H/Hulu-Med) | [✓](https://huggingface.co/ZJU-AI4H/Hulu-Med-32B) | MLLM |

---

### 3D Medical MLLM Models
| Year | Title | Journal | Model | Scale | Code | Weights | Key Feature |
|:----:|--------|---------|---------|--------|:---:|:-------:|---|
| 2024 | **M3D**: Advancing 3D Medical Image Analysis with Multi-Modal Large Language Models | arXiv | 3D-ViT+LLaMA-2-7B |  120K image-text pairs and 662K instruction-response pairs | [✓](https://github.com/BAAI-DCAI/M3D) | [✓](https://huggingface.co/GoodBaiBai88) | 3D MLLM |
| 2024 | **RadFM ** Towards generalist foundation model for radiology by leveraging web-scale 2D&3D medical data | NC | 3D-ViT+MedLLaMA-13B |  13M 2D images and 615K 3D scans | [✓](https://github.com/chaoyi-wu/RadFM) | [✓](https://huggingface.co/chaoyi-wu/RadFM) | 3D MLLM |


## Contributing

We welcome contributions! Please feel free to submit a pull request to add new papers or update existing information. When adding new papers, please follow the existing format and ensure the information is accurate and complete.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
