# Paper 2 Implementation: Classification of Alzheimer's Disease using MRI Data

## Overview

This repository contains the complete implementation of **Paper 2: "Classification of Alzheimer's Disease using MRI data based on Deep Learning Techniques"** published in the Journal of King Saud University - Computer and Information Sciences (2024).

## Dataset

**OASIS-2 (Open Access Series of Imaging Studies)**
- Raw 3D NIfTI MRI volumes: 1,367 brain scans
- Extracted 2D slices: ~6,400 images
- Binary classification: Demented vs Non-Demented
- Labels based on CDR (Clinical Dementia Rating) scores
- Train/Test split: 80/20 (5,120 / 1,280 images)

## Models Implemented

All 5 models from Paper 2 with exact specifications:

| Model | Architecture | Input Size | Epochs | Batch | Target Accuracy |
|-------|--------------|------------|--------|-------|-----------------|
| 1. CNNs-without-Aug | 13-layer CNN | 224×224×3 | 100 | 30 | 99.22% |
| 2. CNNs-with-Aug | 13-layer CNN + Aug | 128×128×3 | 100 | 65 | 99.61% |
| 3. CNN-LSTM-with-Aug ⭐ | 7-layer CNN-LSTM | 128×128×3 | 25 | 16 | **99.92%** |
| 4. CNN-SVM-with-Aug | 6-layer CNN-SVM | 224×224×3 | 20 | 32 | 99.14% |
| 5. VGG16-SVM-with-Aug | Transfer Learning | 224×224×3 | - | 32 | 98.67% |

## Repository Structure

```
Paper2/
├── Raw_Data/                    # OASIS-2 raw NIfTI volumes
│   ├── OAS2_RAW_PART1/         # 209 subject-scan directories
│   ├── OAS2_RAW_PART2/         # 164 subject-scan directories
│   └── OASIS_demographic.xlsx  # CDR scores and demographics
│
├── notebooks/                   # Jupyter notebooks (main implementation)
│   ├── 00_utils_and_config.ipynb         # Utilities and configuration
│   ├── 01_data_preparation.ipynb         # Data extraction and preprocessing
│   ├── 02_model1_cnn_without_aug.ipynb   # Model 1: CNN without augmentation
│   ├── 03_model2_cnn_with_aug.ipynb      # Model 2: CNN with augmentation
│   ├── 04_model3_cnn_lstm_with_aug.ipynb # Model 3: CNN-LSTM (BEST) ⭐
│   ├── 05_model4_cnn_svm_with_aug.ipynb  # Model 4: CNN-SVM
│   ├── 06_model5_vgg16_svm_with_aug.ipynb # Model 5: VGG16-SVM
│   └── 07_results_comparison.ipynb       # Comprehensive comparison
│
├── processed_data/              # Preprocessed image arrays
│   ├── X_train_224.npy         # Training images (224×224)
│   ├── X_test_224.npy          # Test images (224×224)
│   ├── X_train_128.npy         # Training images (128×128)
│   ├── X_test_128.npy          # Test images (128×128)
│   ├── y_train.npy             # Training labels
│   ├── y_test.npy              # Test labels
│   └── dataset_metadata.json   # Dataset metadata
│
├── saved_models/                # Trained model weights
│   ├── model1_cnn_without_aug_final.h5
│   ├── model2_cnn_with_aug_best.h5
│   ├── model3_cnn_lstm_best.h5
│   ├── model4_cnn_svm_best.h5
│   └── model5_vgg16_svm_best.h5
│
├── results/                     # Evaluation results
│   ├── confusion_matrices/     # Confusion matrix plots
│   ├── training_curves/        # Training history plots
│   ├── model1_results.json     # Model 1 metrics
│   ├── model2_results.json     # Model 2 metrics
│   ├── model3_results.json     # Model 3 metrics
│   ├── model4_results.json     # Model 4 metrics
│   ├── model5_results.json     # Model 5 metrics
│   ├── all_models_comparison.csv   # Comparison table
│   └── all_models_comparison.png   # Comparison charts
│
└── README.md                    # This file
```

## Quick Start

### Prerequisites

```bash
# Python 3.8+
pip install tensorflow>=2.10.0
pip install keras>=2.10.0
pip install numpy pandas matplotlib seaborn
pip install scikit-learn
pip install nibabel  # For reading NIfTI files
pip install openpyxl  # For reading Excel
pip install pillow opencv-python
```

### Running the Notebooks

Execute notebooks in order:

1. **00_utils_and_config.ipynb** - Load all utilities (run this first in each notebook)
2. **01_data_preparation.ipynb** - Extract 2D slices from 3D NIfTI volumes
3. **02-06** - Train each of the 5 models independently
4. **07_results_comparison.ipynb** - Compare all models

### Example: Training Model 3 (Best Model)

```python
# In Jupyter notebook
%run 00_utils_and_config.ipynb

# Load preprocessed data
X_train = np.load('processed_data/X_train_128.npy')
X_test = np.load('processed_data/X_test_128.npy')
y_train = np.load('processed_data/y_train.npy')
y_test = np.load('processed_data/y_test.npy')

# Build and train CNN-LSTM model
# (See notebook 04 for complete implementation)
```

## Methodology

### 1. Data Preprocessing (4 Steps from Paper 2)

- **Step 1: Resize** - Convert to 224×224 or 128×128
- **Step 2: Labeling** - Binary labels from CDR scores (0=Non-Demented, 1=Demented)
- **Step 3: Normalization** - Rescale pixels from [0, 255] to [0, 1]
- **Step 4: Color Modification** - Convert grayscale to RGB (3-channel)

### 2. Data Augmentation (Models 2-5)

- Random rotations: 0-90 degrees
- Random horizontal/vertical flips
- Random zoom/magnification
- Random spatial shifting

### 3. Model Architectures

#### Model 1 & 2: CNN (13 layers)
```
Input → Conv2D(16) → MaxPool → Conv2D(32) → MaxPool → Dropout(0.25)
→ Conv2D(64) → MaxPool → Dropout(0.20) → Flatten
→ Dense(128) → Dense(64) → Dense(2, Softmax)
```

#### Model 3: CNN-LSTM (7 layers) ⭐
```
Input(1, 128, 128, 3) → TimeDistributed(Conv2D(64)) → TimeDistributed(MaxPool)
→ TimeDistributed(Conv2D(32)) → TimeDistributed(MaxPool)
→ TimeDistributed(Flatten) → LSTM(100) → Dense(2, Sigmoid)
```

#### Model 4: CNN-SVM (6 layers)
```
Input → Conv2D(64) → MaxPool → Conv2D(32) → MaxPool
→ Flatten → Dense(2, L2 reg) + SVM → Softmax
```

#### Model 5: VGG16-SVM
```
Pre-trained VGG16 → Flatten → SVM (Linear kernel) → Dense(2, Softmax)
```

### 4. Evaluation Metrics (from Paper 2)

- **Accuracy** = (TP + TN) / (TP + TN + FP + FN)
- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **F1-Score** = 2 × (Precision × Recall) / (Precision + Recall)
- **Specificity** = TN / (TN + FP)
- **Training Time** (seconds)
- **Testing Time** (milliseconds)

## Expected Results

### Model Performance (Paper 2 Targets)

| Model | Accuracy | Precision | Recall | F1-Score | Specificity |
|-------|----------|-----------|--------|----------|-------------|
| CNN-LSTM ⭐ | 99.92% | 100.00% | 99.50% | 99.70% | 100.00% |
| CNN-with-Aug | 99.61% | 100.00% | 97.39% | 98.70% | 100.00% |
| CNN-without-Aug | 99.22% | 100.00% | 95.00% | 97.39% | 100.00% |
| CNN-SVM | 99.14% | 100.00% | 94.00% | 97.10% | 100.00% |
| VGG16-SVM | 98.67% | 100.00% | 91.20% | 95.39% | 100.00% |

## Key Features

✅ **Exact Paper Replication** - All models match Paper 2 specifications precisely
✅ **Binary Classification** - Demented vs Non-Demented (from CDR scores)
✅ **Real Medical Data** - OASIS-2 dataset with 1,367 3D brain scans
✅ **5 Deep Learning Models** - From simple CNN to transfer learning
✅ **Complete Pipeline** - Data extraction → Training → Evaluation
✅ **Jupyter Notebooks** - Interactive, reproducible, well-documented
✅ **Comprehensive Evaluation** - All metrics from Paper 2
✅ **Visualization** - Confusion matrices, training curves, comparisons

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{sorour2024classification,
  title={Classification of Alzheimer's disease using MRI data based on Deep Learning Techniques},
  author={Sorour, Shaymaa E and Abd El-Mageed, Amr A and Albarrak, Khalied M and Alnaim, Abdulrahman K and Wafa, Abeer A and El-Shafeiy, Engy},
  journal={Journal of King Saud University-Computer and Information Sciences},
  volume={36},
  number={1},
  pages={101940},
  year={2024},
  publisher={Elsevier}
}
```

## License

This implementation is for educational and research purposes. The OASIS-2 dataset has its own usage terms.

## Acknowledgments

- **Paper 2 Authors**: Shaymaa E. Sorour et al.
- **Dataset**: OASIS-2 (Open Access Series of Imaging Studies)
- **Institution**: King Faisal University, Saudi Arabia

## Contact

For questions about this implementation, please refer to the original paper or OASIS-2 dataset documentation.

---

**Paper 2 Implementation Complete** ✅
*All 5 models implemented with exact specifications from the paper*
