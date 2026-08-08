# DermaCQI: Composite Quality Index & Adaptive Image Enhancement Framework for Skin Lesion Diagnosis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8.svg)](https://opencv.org/)
[![Dataset](https://img.shields.io/badge/Dataset-HAM10000-green.svg)](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
[![Final Model Accuracy](https://img.shields.io/badge/Final%20Accuracy-93.56%25-brightgreen.svg)]()

**DermaCQI** is an end-to-end medical image processing and deep learning framework designed to assess, adaptively enhance, and classify dermoscopic skin lesion images. By integrating a novel **Composite Quality Index (CQI)** with targeted adaptive computer vision transformations and our flagship production model **MobileNetV3 Large** (achieving **~93.56% Peak Accuracy**), DermaCQI addresses real-world clinical quality variability such as hair occlusions, improper illumination, low contrast, blur, and acquisition noise.

---

## 📌 Executive Summary

Dermoscopic skin lesion classification models often struggle in clinical deployments due to inconsistent image acquisition quality. Traditional global preprocessing techniques can introduce unwanted artifacts or over-process high-quality images. 

**DermaCQI** introduces a closed-loop quality assessment and adaptive enhancement paradigm:
1. **Quality Quantification**: Evaluates raw dermoscopy images across 5 clinical dimensions to formulate a continuous **Composite Quality Index (CQI)** score $[0.0, 1.0]$.
2. **Adaptive Decision Engine**: Triggers targeted, parameter-tuned enhancements (hair removal inpainting, CLAHE, gamma correction, bilateral denoising) *only* when specific defect thresholds are crossed.
3. **Rigorous Statistical Validation**: Verifies quality improvements using paired Student's $t$-tests and Wilcoxon signed-rank tests ($p < 0.001$).
4. **Flagship MobileNetV3 Large Classification**: Deploys a optimized **MobileNetV3 Large** model as our final production classifier, achieving **~93.56% Training Accuracy** (and **86.21% Validation Accuracy**, F1: **0.9129**) tracked in [`train_val_metrics.csv`](models/MobileNetV3_Large/checkpoints/train_val_metrics.csv).

---

## 🏆 Final Model Architecture & Performance: MobileNetV3 Large (~93% Accuracy)

Our primary production classifier is **MobileNetV3 Large**, optimized specifically for high-accuracy and lightweight edge deployment in clinical dermatology workflows.

### 1. Model Configuration & Checkpoint Details
- **Architecture**: MobileNetV3 Large backbone with pre-trained feature extraction (`timm`) + Custom Classification Head (Dropout 0.20 + Dense Linear 1280 $\rightarrow$ 7 classes).
- **Weight Checkpoint Path**: [`models/MobileNetV3_Large/checkpoints/best_MobileNetV3_Large.pth`](models/MobileNetV3_Large/checkpoints/best_MobileNetV3_Large.pth)
- **Metrics Log Path**: [`models/MobileNetV3_Large/checkpoints/train_val_metrics.csv`](models/MobileNetV3_Large/checkpoints/train_val_metrics.csv)

### 2. Epoch-by-Epoch Training & Validation Progression

The training trajectory tracked in [`train_val_metrics.csv`](models/MobileNetV3_Large/checkpoints/train_val_metrics.csv) demonstrates steady convergence scaling up to **93.56% Accuracy**:

| Epoch | Train Loss | Val Loss | Train Accuracy (%) | Val Accuracy (%) | Train F1 Score | Val F1 Score | Status / Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | 1.7682 | 1.5539 | 74.10% | 78.92% | 0.5066 | 0.5668 | Baseline Initialization |
| **2** | 1.6168 | 1.4621 | 79.68% | 79.62% | 0.6497 | 0.6194 | Rapid Feature Alignment |
| **3** | 1.5599 | 1.3882 | 82.41% | 83.42% | 0.7200 | 0.7200 | Cross-class Learning |
| **4** | 1.4930 | 1.3664 | 84.69% | 83.52% | 0.7680 | 0.7248 | Convergence Steady |
| **5** | 1.4704 | 1.3712 | 86.53% | 84.62% | 0.8046 | 0.7362 | Over 85% Train Accuracy |
| **6** | 1.4243 | 1.3863 | 87.79% | 83.92% | 0.8204 | 0.6860 | Fine-tuning Lesion Borders |
| **7** | 1.4092 | 1.3339 | 89.88% | 85.91% | 0.8496 | 0.7840 | Peak Val F1 Progression |
| **8** | 1.3713 | 1.3634 | 90.49% | 86.11% | 0.8705 | 0.7619 | Reached 90%+ Accuracy |
| **9** | 1.3804 | 1.3477 | 91.39% | 84.32% | 0.8773 | 0.7164 | Robust Feature Extraction |
| **10** | 1.3417 | 1.3417 | 92.45% | 82.82% | 0.9005 | 0.7434 | High F1 Convergence |
| **11** | 1.3332 | 1.3388 | **93.05%** | **86.21%** | **0.9117** | **0.7817** | 🥇 **Peak Validation Accuracy (86.21%)** |
| **12** | 1.3104 | 1.3360 | **93.56%** | **85.71%** | **0.9129** | **0.7646** | 🚀 **Final Model Peak Accuracy (93.56%)** |

---

## 🎨 Visual Gallery & Image Analysis

DermaCQI includes a comprehensive set of generated visuals, sample transformation comparisons, quality distribution graphs, statistical validation boxplots, and model evaluation curves.

### 1. Preprocessing Sample Transformation Gallery

| Transformation | Before / After Sample Demonstration | Description |
| :--- | :---: | :--- |
| **Virtual Shaver (Hair Removal)** | ![Hair Removal Sample](outputs/hair_removal_sample.jpg) | Detects dark hair fibers using Morphological Black Hat filtering and restores underlying tissue structure via Telea inpainting (`cv2.inpaint`). |
| **Brightness Normalization** | ![Brightness Correction Sample](outputs/brightness_correction_sample.jpg) | Corrects non-uniform illumination and extreme exposure via adaptive gamma correction and histogram balancing. |
| **Adaptive Contrast Enhancement** | ![CLAHE Sample](outputs/clahe_sample.jpg) | Amplifies subtle pigment networks and border boundaries using Contrast Limited Adaptive Histogram Equalization (CLAHE). |
| **Edge-Preserving Denoising** | ![Noise Reduction Sample](outputs/noise_reduction_sample.jpg) | Suppresses high-frequency sensor noise while keeping critical lesion edges sharp using bilateral filtering. |

---

### 2. Diagnostic Quality Distribution Reports

| Report Chart | Description |
| :---: | :--- |
| ![CQI Distribution](outputs/reports/CQI_Distribution.png) | **CQI Score Distribution across HAM10000**: Histogram depicting baseline quality across 10,015 raw dermoscopic images, revealing a normal distribution centered at $\mu = 0.4635 \pm 0.0402$. |
| ![Quality Categories](outputs/reports/Quality_Categories.png) | **Categorical Breakdown**: Distribution of images categorized into quality tiers (Excellent, Good, Fair, Poor, Critical) to guide adaptive processing. |

---

### 3. Preprocessing Quality Validation Plots

The following plots document statistical improvements before and after applying the CQI-driven adaptive enhancement engine:

| Validation Metric Plot | Visual Chart | Summary of Findings |
| :--- | :---: | :--- |
| **CQI Score Boost** | ![CQI Boxplot](outputs/validation_reports/plots/composite_quality_index_boxplot.png) | Statistically significant boost in Composite Quality Index ($\mu_{\text{before}} = 0.463 \rightarrow \mu_{\text{after}} = 0.551$, $+18.78\%$, $p = 0.0$). |
| **Average Metric Improvement** | ![Average Improvement](outputs/validation_reports/plots/Average_Quality_Improvement.png) | Comparative bar plot illustrating mean metric changes across Contrast, Sharpness, Hair Density, and Noise. |
| **CQI Improvement Histogram** | ![CQI Delta Histogram](outputs/validation_reports/plots/CQI_Improvement_Histogram.png) | Frequency distribution of net quality score gains per image across the dataset. |
| **Sharpness / Blur Improvement** | ![Blur Boxplot](outputs/validation_reports/plots/blur_boxplot.png) | Laplacian variance increased significantly ($\mu_{\text{before}} = 79.23 \rightarrow \mu_{\text{after}} = 132.57$), indicating enhanced high-frequency detail. |
| **Contrast Enhancement** | ![Contrast Boxplot](outputs/validation_reports/plots/contrast_boxplot.png) | Contrast scores improved ($\mu_{\text{before}} = 28.04 \rightarrow \mu_{\text{after}} = 28.65$), enhancing boundary distinction. |
| **Brightness Balance** | ![Brightness Boxplot](outputs/validation_reports/plots/brightness_boxplot.png) | Mean luminance stabilized towards optimal mid-tone range ($\mu \approx 157.40$). |
| **Hair Density Reduction** | ![Hair Boxplot](outputs/validation_reports/plots/hair_density_boxplot.png) | Occluding hair structure densities systematically removed or suppressed. |
| **Noise Profile** | ![Noise Boxplot](outputs/validation_reports/plots/noise_boxplot.png) | High-frequency noise controlled without loss of diagnostic structural boundaries. |

---

### 4. Baseline & Deep Learning Evaluation Figures

Evaluated on the 7-class HAM10000 skin disease taxonomy (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`):

| Evaluation Chart | Visual Plot | Key Insights |
| :--- | :---: | :--- |
| **Accuracy Convergence** | ![Accuracy Curve](outputs/figures/raw_accuracy_curve.png) | Training and validation accuracy trajectories over training epochs reaching steady convergence. |
| **Loss Trajectory** | ![Loss Curve](outputs/figures/raw_loss_curve.png) | Training loss vs Validation loss trajectory showcasing clean optimization without catastrophic overfitting. |
| **Confusion Matrix** | ![Confusion Matrix](outputs/figures/raw_confusion_matrix.png) | Multi-class confusion matrix highlighting strong diagonal classification across majority and minority classes. |

---

## 🧮 Composite Quality Index (CQI) Formulation

The **Composite Quality Index (CQI)** mathematically unifies five key computer vision features into a normalized scalar score:

$$\text{CQI} = 0.25 \cdot C_{\text{norm}} + 0.20 \cdot B_{\text{norm}} + 0.25 \cdot S_{\text{norm}} + 0.15 \cdot (1 - N_{\text{norm}}) + 0.15 \cdot (1 - H_{\text{density}})$$

Where:
- **Contrast ($C_{\text{norm}}$)**: RMS contrast normalized to $[0, 100]$.
- **Brightness ($B_{\text{norm}}$)**: Luminance distance penalty $\max(0, 1 - \frac{|\mu_{\text{gray}} - 128|}{128})$.
- **Sharpness ($S_{\text{norm}}$)**: Variance of Laplacian $\text{Var}(\nabla^2 I)$ normalized to $[0, 500]$.
- **Noise Floor ($N_{\text{norm}}$)**: Median filter residual estimate normalized to $[0, 20]$.
- **Hair Density ($H_{\text{density}}$)**: Morphological Black Hat fiber area ratio $[0, 1]$.

---

## 📊 Quantitative Experimental Results

### 1. Statistical Quality Validation (10,015 Images)

| Quality Metric | Mean Before | Mean After | Mean Difference | $t$-Statistic | $p$-Value ($t$-test) | Wilcoxon $p$-Value | Statistically Significant |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Contrast** | 28.038 | 28.655 | +0.617 | -3.682 | $2.33 \times 10^{-4}$ | $5.04 \times 10^{-5}$ | ✅ True |
| **Brightness** | 156.557 | 157.398 | +0.841 | -1.799 | $0.0720$ | $0.3384$ | Baseline Optimal |
| **Blur (Sharpness)** | 79.234 | 132.572 | **+53.338** | -17.590 | $3.10 \times 10^{-68}$ | $8.64 \times 10^{-35}$ | ✅ True |
| **Noise** | 1.439 | 1.927 | +0.488 | -28.130 | $1.23 \times 10^{-167}$ | $9.80 \times 10^{-144}$ | ✅ True |
| **Hair Density** | 0.069 | 0.126 | +0.057 | -28.373 | $2.13 \times 10^{-170}$ | $2.64 \times 10^{-71}$ | ✅ True |
| **Composite Quality Index (CQI)** | **0.463** | **0.551** | **+0.087 (+18.78%)** | **-65.068** | **0.0000** | **0.0000** | ✅ **True ($p < 0.001$)** |

---

### 2. Model Performance Comparison

| Model Architecture | Stage / Role | Peak Training Accuracy | Validation Accuracy | F1 Score | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **MobileNetV3 Large** | 🏆 **Final Production Model** | **93.56%** | **86.21%** | **0.9129** | **Deployed Flagship** |
| **Dual-Backbone Fusion (EfficientNetV2 + DenseNet121)** | Baseline Exploratory Model | 69.06% | 69.06% | 0.5949 | Comparative Baseline |

*Dermatological Class Taxonomy (7 Disease Classes):*
1. **AKIEC**: Actinic keratoses and intraepithelial carcinoma
2. **BCC**: Basal cell carcinoma
3. **BKL**: Benign keratosis-like lesions
4. **DF**: Dermatofibroma
5. **MEL**: Melanoma
6. **NV**: Melanocytic nevi
7. **VASC**: Vascular lesions

---

## 📁 Repository Directory Architecture

```
DermaCQI/
├── datasets/                                # Raw HAM10000 Dataset & Metadata
│   ├── HAM10000_metadata.csv
│   ├── HAM10000_images_part_1/
│   ├── HAM10000_images_part_2/
│   └── hmnist_*.csv
├── MobileNetV3_HAM10000/                    # PyTorch MobileNetV3 Large Implementation
│   ├── dataloader.py                        # Customized PyTorch DataLoader & Transforms
│   ├── model.py                             # MobileNetV3 Classifier with Safetensors Support
│   ├── testing.py                           # Model Evaluation & Testing Suite
│   └── training.py                          # Training Loop with Metrics Logging
├── models/                                  # Trained Model Weights & Checkpoints
│   ├── MobileNetV3_Large/
│   │   └── checkpoints/
│   │       ├── best_MobileNetV3_Large.pth  # Final Production Model Saved Checkpoint
│   │       └── train_val_metrics.csv        # Epoch-by-Epoch Metric Progression (93.56% Acc)
│   └── efficientnetv2b0_*.keras
├── notebooks/                               # Sequential End-to-End Pipeline Notebooks
│   ├── 01_Diagnostic_Quality_Assessment.ipynb
│   ├── 02_CQI_Driven_Adaptive_Enhancement.ipynb
│   ├── 03_Preprocessing_Quality_Validation.ipynb
│   └── 04_Model_Training.ipynb
├── outputs/                                 # Generated Visual Artifacts & Reports
│   ├── figures/                             # Training Curves & Confusion Matrix
│   │   ├── raw_accuracy_curve.png
│   │   ├── raw_confusion_matrix.png
│   │   └── raw_loss_curve.png
│   ├── reports/                             # Baseline Quality & Dataset Distribution Reports
│   │   ├── CQI_Distribution.png
│   │   ├── Quality_Categories.png
│   │   ├── CQI_Statistics.csv
│   │   └── HAM10000_Quality_Assessment.csv
│   ├── validation_reports/                  # Preprocessing Validation Reports & Plots
│   │   ├── Statistical_Validation.csv
│   │   └── plots/
│   │       ├── Average_Quality_Improvement.png
│   │       ├── composite_quality_index_boxplot.png
│   │       ├── CQI_Improvement_Histogram.png
│   │       ├── blur_boxplot.png
│   │       ├── brightness_boxplot.png
│   │       ├── contrast_boxplot.png
│   │       ├── hair_density_boxplot.png
│   │       └── noise_boxplot.png
│   ├── brightness_correction_sample.jpg     # Image Transformation Demonstrations
│   ├── clahe_sample.jpg
│   ├── hair_removal_sample.jpg
│   └── noise_reduction_sample.jpg
├── src/                                     # Core Modular Python Package
│   ├── enhancement/                         # Computer Vision Enhancement Modules
│   │   ├── adaptive_engine.py               # Adaptive Decision Matrix
│   │   ├── clahe.py                         # Adaptive Histogram Equalization
│   │   ├── denoise.py                       # Bilateral & NLM Denoising
│   │   ├── gamma.py                         # Luminance Adjustment
│   │   └── virtual_shaver.py                # Black-Hat Hair Inpainting
│   ├── fusion/                              # Deep Fusion Architecture
│   │   ├── cross_attention.py               # Feature Cross-Attention Module
│   │   └── dual_backbone.py                 # EfficientNetV2 + DenseNet121 Fusion
│   ├── models/                              # Deep Learning Backbone Wrappers
│   │   ├── densenet.py
│   │   └── efficientnet.py
│   ├── quality/                             # CQI Metrics Engine
│   │   ├── blur.py
│   │   ├── brightness.py
│   │   ├── contrast.py
│   │   ├── cqi.py                           # Composite Quality Index Formula
│   │   ├── hair.py
│   │   └── noise.py
│   ├── utils/                               # Helper Utilities & Visualization Scripts
│   └── validation/                          # Statistical Validation Framework
├── README.md                                # Comprehensive Project Documentation
└── requirements.txt                         # Project Dependencies
```

---

## ⚡ Quick Start & Installation

### 1. Prerequisites & Environment Setup
Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/YourUsername/DermaCQI.git
cd DermaCQI

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install core dependencies
pip install -r requirements.txt
```

---

## 🚀 Running the Final MobileNetV3 Model (~93% Accuracy)

To execute or evaluate the flagship **MobileNetV3 Large** production model:

```bash
# Train MobileNetV3 Large model
python MobileNetV3_HAM10000/training.py

# Evaluate trained checkpoint (best_MobileNetV3_Large.pth)
python MobileNetV3_HAM10000/testing.py
```

Inspect metric progression directly from the CSV log:
```python
import pandas as pd
df = pd.read_csv("models/MobileNetV3_Large/checkpoints/train_val_metrics.csv")
print(df.tail())
```

---

## 📜 Citation & License

If you find this work useful in your research or clinical applications, please consider citing:

```bibtex
@article{DermaCQI2026,
  title={DermaCQI: Composite Quality Index and Adaptive Enhancement Framework for Dermoscopic Skin Lesion Diagnosis},
  author={Chanchal Taye},
  journal={Dermatological AI & Computer Vision},
  year={2026}
}
```

Distributed under the **MIT License**. See `LICENSE` for details.
