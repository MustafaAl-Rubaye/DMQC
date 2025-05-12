# DMQC
**DeepMammographyQualityControl** – a Python package for automatic quality assessment in mammography using deep learning.


## 📄 Related Publication

For more details, see the full thesis:  
[Deep learning-based lower back pain classification and detection from T2-weighted magnetic resonance images](https://oulurepo.oulu.fi/bitstream/handle/10024/18084/nbnfioulu-202106178375.pdf?sequence=1&isAllowed=y)

## 📝 Project Overview

DMQC automates quality assessment in mammography by segmenting anatomical features and imaging artifacts using deep learning, followed by rule-based evaluation aligned with PGMI criteria. This system was developed as part of a Bachelor's thesis at the University of Oulu.

## 🧠 Methods

- **Dataset**: 503 manually annotated mediolateral oblique (MLO) view mammograms from Oulu University Hospital.
- **Annotations**: CVAT was used to label five pixel-level classes: whole breast, breast, pectoral muscle, nipple, and skin-folds.
- **Architecture**: U-Net and Feature Pyramid Network (FPN) decoders tested with VGG11, ResNet34, and ResNet50 backbones.
- **Training**:
  - Input size: 512×256 (resized and padded)
  - Augmentations: horizontal flips, cropping
  - Loss: multi-class cross-entropy
  - Evaluation: Dice coefficient (per class)

## ⚙️ Package Contents

- `dmqc/` – Core segmentation and quality assessment code
- `conf/` – YAML configs for model and rules
- `env.yaml` – Conda dependencies
- `setup.py` – Install script

## 🚀 Getting Started

```bash
# Create environment
conda env create -f env.yaml
conda activate dmqc


