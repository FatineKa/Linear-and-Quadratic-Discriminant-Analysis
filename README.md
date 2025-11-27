# LDA & QDA Classification Analysis

## Overview

This notebook demonstrates **Linear Discriminant Analysis (LDA)** and **Quadratic Discriminant Analysis (QDA)** for classification tasks on two datasets:

1. **Diabetes Dataset**: Binary classification (diabetic vs non-diabetic)
2. **Handwritten Digits Dataset**: Multi-class classification (digits 0-9)

## Datasets

### Diabetes Dataset (`diabetes.csv`)
- **Features**: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
- **Target**: Outcome (0 = non-diabetic, 1 = diabetic)
- **Samples**: ~768 patients

### Handwritten Digits Dataset
- **Files**: `digitsX.csv`, `digitsXt.csv`, `digitsY.csv`, `digitsYt.csv`
- **Features**: 784 pixels (28×28 grayscale images)
- **Target**: Digits 0-9 (10 classes)

## What's Inside

### Part 1: Diabetes Classification
1. **Data Exploration**: Scatter plots showing Glucose vs Age, BMI vs Age
2. **LDA Model**: Linear decision boundary, ~77% accuracy
3. **QDA Model**: Quadratic decision boundary, ~75% accuracy
4. **2D Visualization**: Decision boundaries plotted for Glucose and Age features
5. **Evaluation**: Confusion matrices and classification reports

### Part 2: Digit Recognition
1. **Data Loading**: Visualize handwritten digit images
2. **PCA Analysis**: Dimensionality reduction (784 → K dimensions)
3. **LDA Projection**: Supervised dimensionality reduction
4. **Comparison**: PCA vs LDA visualization (LDA shows better class separation)
5. **Classification**: LDA digit recognition with ~85-90% accuracy

## Key Concepts

### LDA (Linear Discriminant Analysis)
- **Assumption**: All classes share the same covariance matrix
- **Boundary**: Linear (straight line in 2D)
- **Best for**: Similar class distributions

### QDA (Quadratic Discriminant Analysis)
- **Assumption**: Each class has its own covariance matrix
- **Boundary**: Quadratic (curved)
- **Best for**: Classes with different shapes/spreads

### PCA vs LDA
- **PCA**: Unsupervised, maximizes variance → classes may overlap
- **LDA**: Supervised, maximizes class separation → clearer clusters

## Installation

```bash
pip install pandas numpy matplotlib scipy scikit-learn mlxtend
```

## Usage

```python
# Run in Jupyter Notebook
jupyter notebook lda_qda_analysis.ipynb
```

Ensure all CSV files are in the same directory as the notebook.

## Key Results

| Dataset | Model | Accuracy |
|---------|-------|----------|
| Diabetes (8 features) | LDA | ~77% |
| Diabetes (8 features) | QDA | ~75% |
| Digits (784 features) | LDA | ~85-90% |

## Files Required

- `diabetes.csv`
- `digitsX.csv` (training features)
- `digitsXt.csv` (test features)
- `digitsY.csv` (training labels)
- `digitsYt.csv` (test labels)

## Main Libraries

- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `matplotlib`: Visualization
- `scikit-learn`: LDA, QDA, PCA implementations
- `mlxtend`: Enhanced decision boundary plots
