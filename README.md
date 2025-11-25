# <p align=center>:fire:`Awesome Medical Foundation Models`:fire:</p>

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://makeapullrequest.com)

🔥🔥 A comprehensive collection of medical foundation models organized by learning paradigm and architecture 🔥🔥

## Overview

This repository provides a curated list of foundation models in medical imaging and healthcare, organized by their core architectural approaches:
- **Visual Self-Supervised Learning**: Models pretrained using self-supervision on medical images
- **Contrastive Learning**: Vision-language models using contrastive objectives
- **Mixed/Hybrid Architecture**: Models combining multiple learning paradigms
- **Medical MLLMs**: Multimodal large language models for medical applications

We strongly encourage authors of relevant works to make a pull request and add their paper's information.

## Contents
- [Visual Self-Supervised Learning](#visual-self-supervised-learning)
- [Contrastive Learning](#contrastive-learning)
- [Mixed/Hybrid Architecture](#mixedhybrid-architecture)
- [Medical Multimodal Large Language Models (MLLM)](#medical-multimodal-large-language-models-mllm)
  - [General Medical MLLMs](#general-medical-mllms)
  - [Specialized Medical MLLMs](#specialized-medical-mllms)

---

## Papers

### Visual Self-Supervised Learning

| Year | Modality | Title | Paper | Code | Weights | Method |
|------|----------|-------|-------|------|---------|--------|
| 2023 | Pathology | Virchow: A Million-Slide Digital Pathology Foundation Model | [arXiv](https://arxiv.org/abs/2309.07778) | - | ✅ | Development |
| 2023 | Retinal | A foundation model for generalizable disease detection from retinal images | [Nature](https://www.nature.com/articles/s41586-023-06555-x) | [GitHub](https://github.com/rmaphoh/RETFound_MAE) | ✅ | Development |
| 2023 | Multi-modal | LVM-Med: Learning Large-Scale Self-Supervised Vision Models for Medical Imaging via Second-order Graph Matching | [arXiv](https://arxiv.org/abs/2306.11925) | [GitHub](https://github.com/duyhominhnguyen/LVM-Med) | ✅ | Development |
| 2022 | X-ray | Expert-level detection of pathologies from unannotated chest X-ray images via self-supervised learning | [Nature BME](https://www.nature.com/articles/s41551-022-00936-9) | - | ✅ | Development |
| 2023 | Multi-modal | Segment Anything in Medical Images (MedSAM) | [arXiv](https://arxiv.org/abs/2304.12306) | [GitHub](https://github.com/bowang-lab/MedSAM) | ✅ | Adaptation |
| 2023 | Multi-modal | Medical SAM Adapter | [arXiv](https://arxiv.org/pdf/2304.12620.pdf) | [GitHub](https://github.com/WuJunde/Medical-SAM-Adapter) | ✅ | Adaptation |
| 2023 | Multi-modal | SAMed: Customized Segment Anything Model for Medical Image Segmentation | [arXiv](https://arxiv.org/pdf/2304.13785.pdf) | [GitHub](https://github.com/hitachinsk/SAMed) | ✅ | Adaptation |
| 2023 | Multi-modal | SAM-Med2D | [arXiv](https://arxiv.org/pdf/2308.16184.pdf) | [GitHub](https://github.com/uni-medical/SAM-Med2D) | ✅ | Adaptation |
| 2023 | Multi-modal | SAM-U: Multi-box prompts triggered uncertainty estimation for reliable SAM | [arXiv](https://arxiv.org/pdf/2307.04973.pdf) | - | ✅ | Improvement |
| 2023 | Multi-modal | AutoSAM: How to Efficiently Adapt Large Segmentation Model to Medical Images | [arXiv](https://arxiv.org/pdf/2306.13731.pdf) | [GitHub](https://github.com/xhu248/AutoSAM) | ✅ | Adaptation |

---

### Contrastive Learning

| Year | Modality | Title | Paper | Code | Weights | Method |
|------|----------|-------|-------|------|---------|--------|
| 2023 | X-ray | Enhancing Representation in Radiography-Reports Foundation Model: Masked Contrastive Learning | [arXiv](https://arxiv.org/pdf/2309.05904.pdf) | - | ✅ | Development |
| 2023 | Pathology | PLIP: A visual-language foundation model for pathology image analysis using medical Twitter | [Nature Medicine](https://www.nature.com/articles/s41591-023-02504-3) | [GitHub](https://tinyurl.com/webplip) | ✅ | Development |
| 2023 | X-ray | ELIXR: Towards a general purpose X-ray AI system | [arXiv](https://arxiv.org/abs/2308.01317) | - | ✅ | Development |
| 2023 | Multi-modal | KoBo: Knowledge Boosting Medical Contrastive Vision-Language Pre-Training | [MICCAI](https://arxiv.org/pdf/2307.07246.pdf) | [GitHub](https://github.com/ChenXiaoFei-CS/KoBo) | ✅ | Development |
| 2023 | Pathology | CITE: Text-guided Foundation Model Adaptation for Pathological Image Classification | [MICCAI](https://arxiv.org/abs/2307.14901) | [GitHub](https://github.com/Yunkun-Zhang/CITE) | ✅ | Adaptation |
| 2023 | Pathology | Visual Language Pretrained Multiple Instance Zero-Shot Transfer for Histopathology | [CVPR](https://openaccess.thecvf.com/content/CVPR2023/papers/Lu_Visual_Language_Pretrained_Multiple_Instance_Zero-Shot_Transfer_for_Histopathology_Images_CVPR_2023_paper.pdf) | - | ✅ | Application |
| 2023 | Multi-modal | BiomedCLIP: Large-Scale Domain-Specific Pretraining for Biomedical Vision-Language | [arXiv](https://arxiv.org/abs/2303.00915) | [GitHub](https://aka.ms/biomedclip) | ✅ | Development |
| 2023 | Multi-modal | PTUnifier: Towards Unifying Medical Vision-and-Language Pre-training via Soft Prompts | [arXiv](https://arxiv.org/pdf/2302.08958.pdf) | [GitHub](https://github.com/zhjohnchan/PTUnifier) | ✅ | Development |
| 2023 | Multi-modal | Learning to Exploit Temporal Structure for Biomedical Vision Language Processing | [CVPR](https://arxiv.org/pdf/2301.04558.pdf) | - | ✅ | Development |
| 2023 | Multi-modal | CLIP-Driven Universal Model for Organ Segmentation and Tumor Detection | [ICCV](https://arxiv.org/abs/2301.00785) | [GitHub](https://github.com/ljwztc/CLIP-Driven-Universal-Model) | ✅ | Application |
| 2022 | Multi-modal | MedCLIP: Contrastive Learning from Unpaired Medical Images and Text | [EMNLP](https://arxiv.org/pdf/2210.10163.pdf) | [GitHub](https://github.com/RyanWangZf/MedCLIP) | ✅ | Development |

---

### Mixed/Hybrid Architecture

| Year | Modality | Title | Paper | Code | Weights | Method |
|------|----------|-------|-------|------|---------|--------|
| 2023 | 3D Medical | MedBLIP: Bootstrapping Language-Image Pre-training from 3D Medical Images and Texts | [arXiv](https://arxiv.org/pdf/2305.10799.pdf) | [GitHub](https://github.com/Qybc/MedBLIP) | ✅ | Development |
| 2023 | Multi-modal | Med-Flamingo: a Multimodal Medical Few-shot Learner | [arXiv](https://arxiv.org/abs/2307.15189) | [GitHub](https://github.com/snap-stanford/med-flamingo) | ✅ | Development |
| 2022 | X-ray | Clinical-BERT: Vision-Language Pre-training for Radiograph Diagnosis and Reports Generation | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/20204) | - | ✅ | Development |
| 2023 | Multi-modal | Vision-Language Model for Visual Question Answering in Medical Imagery | [Bioengineering](https://www.mdpi.com/2306-5354/10/3/380) | - | ✅ | Application |

---

### Medical Multimodal Large Language Models (MLLM)

#### General Medical MLLMs

| Year | Modality | Title | Paper | Code | Weights | Method |
|------|----------|-------|-------|------|---------|--------|
| 2023 | Multi-modal | Med-PaLM 2: Towards Generalist Biomedical AI | [arXiv](https://arxiv.org/abs/2307.14334) | - | ✅ | Development |
| 2023 | Multi-modal | BiomedGPT: A Unified and Generalist Biomedical Generative Pre-trained Transformer | [arXiv](https://arxiv.org/pdf/2305.17100.pdf) | [GitHub](https://github.com/taokz/BiomedGPT) | ✅ | Development |
| 2023 | Text | Med-PaLM: Towards Expert-Level Medical Question Answering with Large Language Models | [arXiv](https://arxiv.org/pdf/2305.09617.pdf) | - | ✅ | Development |
| 2023 | Multi-modal | Foundation models for generalist medical artificial intelligence | [Nature](https://www.nature.com/articles/s41586-023-05881-4) | - | ✅ | Review |
| 2023 | Multi-modal | Generalist Vision Foundation Models for Medical Imaging: SAM Case Study | [Diagnostics](https://arxiv.org/pdf/2304.12637v2.pdf) | [GitHub](https://github.com/hwei-hw/Generalist_Vision_Foundation_Models_for_Medical_Imaging) | ✅ | Application |

#### Specialized Medical MLLMs

| Year | Modality | Title | Paper | Code | Weights | Method |
|------|----------|-------|-------|------|---------|--------|
| 2023 | Radiology | Radiology-Llama2: Best-in-Class Large Language Model for Radiology | [arXiv](https://arxiv.org/abs/2309.06419) | - | ✅ | Development |
| 2023 | Radiology | RadFM: Towards Generalist Foundation Model for Radiology | [arXiv](https://arxiv.org/abs/2308.02463) | [GitHub](https://github.com/chaoyi-wu/RadFM) | ✅ | Development |
| 2023 | Clinical | ClinicalGPT: Large Language Models Finetuned with Diverse Medical Data | [arXiv](https://arxiv.org/abs/2306.09968) | - | ✅ | Development |
| 2023 | X-ray | XrayGPT: Chest Radiographs Summarization using Medical Vision-Language Models | [arXiv](https://arxiv.org/abs/2306.07971) | [GitHub](https://github.com/mbzuai-oryx/XrayGPT) | ✅ | Development |
| 2023 | Multi-modal | LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day | [arXiv](https://arxiv.org/abs/2306.00890) | [GitHub](https://github.com/microsoft/LLaVA-Med) | ✅ | Development |
| 2023 | Text | PMC-LLaMA: Towards Building Open-source Language Models for Medicine | [arXiv](https://arxiv.org/abs/2304.14454) | [GitHub](https://github.com/chaoyi-wu/PMC-LLaMA) | ✅ | Development |
| 2023 | Multi-modal | Visual Med-Alpaca: A Parameter-Efficient Biomedical LLM with Visual Capabilities | [GitHub](https://github.com/cambridgeltl/visual-med-alpaca) | [GitHub](https://github.com/cambridgeltl/visual-med-alpaca) | ✅ | Development |
| 2023 | Text | ChatDoctor: A Medical Chat Model Fine-Tuned on LLaMA | [Cureus](https://arxiv.org/abs/2303.14070) | [GitHub](https://github.com/Kent0n-Li/ChatDoctor) | ✅ | Development |
| 2023 | Text | DeID-GPT: Zero-shot Medical Text De-Identification by GPT-4 | [arXiv](https://arxiv.org/pdf/2303.11032.pdf) | [GitHub](https://github.com/yhydhx/ChatGPT-API) | ✅ | Application |
| 2023 | Multi-modal | ChatCAD: Interactive Computer-Aided Diagnosis using Large Language Models | [arXiv](https://arxiv.org/abs/2302.07257) | - | ✅ | Application |

---

## Contributing

We welcome contributions! Please feel free to submit a pull request to add new papers or update existing information. When adding new papers, please follow the existing format and ensure the information is accurate and complete.

## Acknowledgements

This repository is inspired by and partially inherits content from [Awesome Foundation Models in Medical Imaging](https://github.com/xmindflow/Awesome-Foundation-Models-in-Medical-Imaging). We thank the original authors for their excellent survey work:

**Foundational Models in Medical Imaging: A Comprehensive Survey and Future Vision**<br>
*Bobby Azad, Reza Azad, Sania Eskandari, Afshin Bozorgpour, Amirhossein Kazerouni, Islem Rekik, Dorit Merhof*<br>
[28th Oct., 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2310.18689)]

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
