# Breast Cancer Classification using Machine Learning

## Overview

This project applies supervised machine learning techniques to classify breast cancer tumors as **Malignant (M)** or **Benign (B)** using the Breast Cancer Diagnostic Dataset. The pipeline covers end-to-end data science workflow — from exploratory data analysis and preprocessing through dimensionality reduction, model training, hyperparameter tuning, and final evaluation.

The objective is not only to build accurate classifiers but to rigorously compare model performance before and after dimensionality reduction, providing interpretable insights that are relevant in a clinical diagnostic context.

---

## Table of Contents

- [Project Motivation](#project-motivation)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Methodology](#methodology)
- [Results Summary](#results-summary)
- [Key Insights](#key-insights)
- [Future Improvements](#future-improvements)

---

## Project Motivation

Breast cancer is one of the most prevalent cancers globally, and early, accurate diagnosis is critical for improving patient outcomes. Machine learning models trained on cell nucleus features extracted from fine needle aspirate (FNA) images can serve as a reliable second-opinion tool for pathologists. This project explores whether high classification accuracy can be maintained even after significantly reducing the feature space through PCA.

---

## Dataset

- **Source:** Breast Cancer Diagnostic Dataset (Wisconsin Diagnostic Breast Cancer — WDBC)
- **File:** `Breast_Cancer_Diagnostic_Dataset.csv`
- **Samples:** 569
- **Original Features:** 33 columns (id, diagnosis, 30 numeric features, 1 unnamed column dropped)
- **Final Features Used:** 30 numeric features
- **Target Variable:** `diagnosis` — encoded as `1` (Malignant) and `0` (Benign)
- **Class Distribution:** 357 Malignant, 212 Benign
- **Missing Values:** None
- **Duplicate Rows:** 0

The 30 features describe cell nucleus characteristics computed from digitized FNA images — including radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension — each measured as mean, standard error, and worst (largest) value.

---

## Project Structure

```
Breast_Cancer_Classification/
|
|-- Breast_Cancer_Classification.ipynb             # Main notebook
|-- Breast_Cancer_Classification_Visualizations    # Visualizations
|-- Breast_Cancer_Diagnostic_Dataset.csv           # Dataset
|-- README.md                                      # Project documentation
```

---

## Tech Stack

| Category              | Library / Tool                         |
|-----------------------|----------------------------------------|
| Language              | Python 3                               |
| Data Manipulation     | NumPy, Pandas                          |
| Visualization         | Matplotlib, Seaborn                    |
| Preprocessing         | StandardScaler (scikit-learn)          |
| Dimensionality Reduction | PCA, LDA (scikit-learn)             |
| Models                | Logistic Regression, Random Forest, SVM |
| Evaluation            | Accuracy, AUC-ROC, Confusion Matrix, Classification Report |
| Hyperparameter Tuning | GridSearchCV                           |
| Validation            | StratifiedKFold Cross-Validation (cv=5) |
| Environment           | Google Colab / Jupyter Notebook        |

---

## Methodology

### 1. Data Loading and Inspection
- Loaded the CSV dataset using Pandas
- Inspected shape, dtypes, null values, and duplicates
- Dropped the unnamed trailing column

### 2. Exploratory Data Analysis (EDA)
- Analyzed class distribution — dataset is moderately imbalanced (357 M vs 212 B)
- Generated a full correlation heatmap across all 30 features
- Identified highly correlated feature clusters (radius, perimeter, area — near perfect correlation)

### 3. Preprocessing
- Encoded target variable: Malignant = 1, Benign = 0
- Split data: 80% train, 20% test with `stratify=y` and `random_state=42`
- Applied `StandardScaler` fitted only on training data to prevent data leakage

### 4. Baseline Model Training
Trained three classifiers on the full 30-feature scaled dataset:
- Logistic Regression (max_iter=1000)
- Random Forest Classifier
- Support Vector Machine (SVC with probability=True)

### 5. Cross-Validation
5-fold stratified cross-validation on the training set to assess generalization.

### 6. PCA (Principal Component Analysis)
- Fitted PCA on training data only
- Identified 7 components explain 90% of variance (from Scree Plot)
- Retrained all three classifiers on PCA-reduced data (76.67% feature reduction)
- Evaluated Accuracy vs. Number of PCA Components curve

### 7. LDA (Linear Discriminant Analysis)
- Applied LDA as a supervised dimensionality reduction technique
- Retrained Logistic Regression on LDA-transformed features

### 8. Hyperparameter Tuning
- Applied `GridSearchCV` on Logistic Regression
- Searched over `C`: [0.01, 0.1, 1, 10] and `solver`: ['liblinear', 'lbfgs']
- Best Parameters: `C=0.1`, `solver='lbfgs'`
- Best CV Score: 0.9802

### 9. Final Evaluation
- Confusion Matrix for Logistic Regression
- ROC Curves for all three models
- Full Classification Report (precision, recall, F1-score)
- Before vs After PCA comparison table

---

## Results Summary

### Baseline Performance (Full 30 Features)

| Model               | Accuracy | AUC-ROC |
|---------------------|----------|---------|
| Logistic Regression | 98.25%   | 0.9954  |
| Random Forest       | 94.74%   | 0.9932  |
| SVM                 | 98.25%   | 0.9950  |

### Cross-Validation Accuracy (5-Fold)

| Model               | CV Accuracy |
|---------------------|-------------|
| Logistic Regression | 98.02%      |
| Random Forest       | 95.60%      |
| SVM                 | 97.14%      |

### After PCA (7 Components — 90% Variance Retained, 76.67% Dimension Reduction)

| Model               | Accuracy | AUC-ROC |
|---------------------|----------|---------|
| Logistic Regression | 94.74%   | 0.9937  |
| Random Forest       | 90.35%   | 0.9840  |
| SVM                 | 95.61%   | 0.9947  |

### LDA Result

| Model                         | Accuracy |
|-------------------------------|----------|
| Logistic Regression (LDA)     | 97.37%   |

### Classification Report — Logistic Regression (Best Model)

| Class     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| 0 (Benign)    | 0.98  | 0.98   | 0.98     | 42      |
| 1 (Malignant) | 0.99  | 0.99   | 0.99     | 72      |
| Accuracy  |           |        | 0.98     | 114     |
| Macro Avg | 0.98      | 0.98   | 0.98     | 114     |
| Weighted Avg | 0.98   | 0.98   | 0.98     | 114     |

### Confusion Matrix — Logistic Regression

|                  | Predicted Benign | Predicted Malignant |
|------------------|------------------|---------------------|
| Actual Benign    | 41               | 1                   |
| Actual Malignant | 1                | 71                  |

Only 2 misclassifications out of 114 test samples.

---

## Key Insights

1. **Logistic Regression is the top performer** — achieving 98.25% accuracy and 0.9954 AUC with a fully linear decision boundary, suggesting the classes are nearly linearly separable in the scaled feature space.

2. **PCA trades a small accuracy drop for massive dimensionality reduction** — going from 30 to 7 features (76.67% reduction) costs only ~3-4% accuracy. This is highly beneficial for deployment in resource-constrained environments.

3. **High feature multicollinearity exists** — radius, perimeter, and area are near-perfectly correlated (>0.99). PCA effectively handles this by decorrelating features into principal components.

4. **The dataset has no missing values or duplicates**, which is unusually clean for a medical dataset — reducing preprocessing complexity and increasing confidence in the results.

5. **ROC AUC near 1.0 for all models** indicates exceptional class discriminability across all classifiers, with Logistic Regression and SVM both reaching AUC = 1.00 on the test set.

6. **LDA achieves 97.37% accuracy** using supervised linear projections — slightly behind the full-feature Logistic Regression but a compelling result for a single-component projection.

7. **GridSearchCV identified C=0.1, solver=lbfgs** as optimal Logistic Regression hyperparameters, confirming that slight regularization improves generalization over the default.

---

## Future Improvements

1. **Ensemble and Boosting Methods:** Incorporate XGBoost, LightGBM, or CatBoost to potentially push classification accuracy beyond 99% and provide feature importance rankings.

2. **SHAP / LIME Explainability:** Apply SHAP (SHapley Additive exPlanations) values to explain individual predictions — critical for clinical trust and model transparency.

3. **Handling Class Imbalance Formally:** While the current imbalance (357 vs 212) is moderate, applying SMOTE or class-weighted loss functions could improve recall for the minority Benign class.

4. **Feature Selection Before PCA:** Use Recursive Feature Elimination (RFE) or mutual information-based selection to identify the most diagnostically meaningful features prior to dimensionality reduction.

5. **Deep Learning Baseline:** Implement a simple feedforward neural network (MLP) using TensorFlow/Keras to benchmark deep learning against classical ML on this tabular dataset.

6. **Clinical Threshold Optimization:** In medical diagnosis, false negatives (missing a malignant tumor) are more costly than false positives. Tune the classification threshold to optimize recall for the Malignant class specifically.

7. **Model Deployment:** Package the best model (Logistic Regression with StandardScaler) using Flask or FastAPI and deploy as a REST API or Streamlit web application for real-time inference.

8. **External Validation:** Evaluate the trained model on an independent external dataset to assess true generalizability beyond the WDBC benchmark.

---


