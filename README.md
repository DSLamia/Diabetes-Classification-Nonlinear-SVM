# Diabetes Classification with Nonlinear SVM

An individual academic machine learning project exploring nonlinear Support Vector Machines and comparative classification techniques on the **Pima Indians Diabetes dataset**. The project evaluates polynomial and RBF SVM kernels alongside multiple traditional machine learning classifiers, combining data preprocessing, class imbalance handling, feature selection, model tuning, and performance visualization.

**Stack:** Python · pandas · NumPy · scikit-learn · imbalanced-learn · Matplotlib · Seaborn · Jupyter Notebook

## Project Overview

| Component | Description |
|---|---|
| Objective | Binary classification of diabetes outcomes |
| Dataset | Pima Indians Diabetes dataset |
| Samples | 768 records |
| Features | 8 input features |
| Target | `Outcome` — 0 (negative), 1 (positive) |
| Primary Models | Polynomial SVM, RBF SVM |
| Comparison Models | Random Forest, Logistic Regression, KNN, Decision Tree |
| Imbalance Handling | SMOTE |
| Feature Selection | `SelectKBest` with mutual information |
| Hyperparameter Tuning | GridSearchCV with 5-fold cross-validation |

## Project Highlights

- **Nonlinear Classification:** Compared polynomial and RBF kernels to explore different decision boundaries.
- **Model Benchmarking:** Evaluated six classification approaches across multiple performance metrics.
- **Class Imbalance Handling:** Applied SMOTE to the training data to balance class representation.
- **Feature Selection:** Used mutual information to identify informative features for classification.
- **Hyperparameter Optimization:** Applied GridSearchCV to tune K-Nearest Neighbors and Decision Tree models.
- **Performance Visualization:** Created comparative visualizations to examine model accuracy and classification performance.

## Methodology

The project follows a structured machine learning workflow:

1. **Exploratory Data Analysis (EDA)** — examined dataset structure, feature distributions, descriptive statistics, and class balance.
2. **Data Preprocessing** — standardized numerical features using `StandardScaler`.
3. **Train-Test Split** — used a stratified 80/20 split with `random_state=16`.
4. **Class Imbalance Handling** — applied SMOTE to the training data.
5. **Feature Selection** — used `SelectKBest` with `mutual_info_classif` to select informative features.
6. **Nonlinear SVM Modeling** — trained Support Vector Classifiers using polynomial and RBF kernels.
7. **Comparative Evaluation** — assessed SVMs against Random Forest, Logistic Regression, KNN, and Decision Tree classifiers.
8. **Hyperparameter Tuning** — optimized KNN and Decision Tree configurations using GridSearchCV.
9. **Prediction Demonstration** — demonstrated predictions for a sample input using the trained preprocessing and classification components.

## Model Performance

The following results were reported by the original notebook run on its hold-out test set.

| Model | Test Accuracy | Precision (Class 1) | Recall (Class 1) |
|---|---:|---:|---:|
| Random Forest | 82.00% | 0.73 | 0.76 |
| SVM — RBF Kernel | 81.17% | 0.69 | 0.83 |
| SVM — Polynomial Kernel | 77.92% | 0.75 | 0.56 |
| Logistic Regression | 79.00% | 0.66 | 0.80 |
| K-Nearest Neighbors | 75.00% | 0.62 | 0.78 |
| Decision Tree | 68.00% | 0.53 | 0.57 |
| Decision Tree — Tuned | 74.68% | — | — |
| KNN — Tuned | 72.08% | — | — |

*Note: Metrics are reproduced from the notebook's saved outputs and have not been independently re-evaluated.*

### Visualizations

**SVM Kernel Comparison**

SVM Kernel Comparison



Classifier Performance Comparison



Feature Distributions



## Technologies & Libraries

- **Programming:** Python
- **Data Manipulation:** pandas, NumPy
- **Machine Learning:** scikit-learn
- **Class Imbalance:** imbalanced-learn (SMOTE)
- **Feature Selection:** `SelectKBest`, `mutual_info_classif`
- **Model Optimization:** `GridSearchCV`
- **Visualization:** Matplotlib, Seaborn
- **Development Environment:** Jupyter Notebook, Google Colab

## Repository Structure

```text
Diabetes-Classification-Nonlinear-SVM/
├── Diabetes-Classification-Nonlinear-SVM.ipynb
├── assets/
│   ├── svm_kernel_comparison.png
│   └── classifier_comparison.png
├── requirements.txt
├── .gitignore
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/DSLamia/Diabetes-Classification-Nonlinear-SVM.git
cd Diabetes-Classification-Nonlinear-SVM
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset

The original notebook reads a local CSV file named `diabetes - Copy.csv`, which is not included in this repository.

Obtain the Pima Indians Diabetes dataset from a reputable public source, review its terms of use, and place the CSV file in the notebook directory. Alternatively, update the file path in the notebook's data-loading cell.

### 4. Launch Jupyter

```bash
jupyter lab
```

Open `Diabetes-Classification-Nonlinear-SVM.ipynb` and execute the cells in sequence.

## Key Takeaways

This project demonstrates practical experience with nonlinear classification, comparative model evaluation, feature selection, imbalanced datasets, and hyperparameter optimization. It provides a hands-on exploration of how different machine learning algorithms approach a binary classification task.

## Disclaimer

This is an **academic machine learning project**, intended for educational and experimental purposes only. It is not a clinically validated model and must not be used for medical diagnosis or treatment decisions.

---

**Developed by Lamia** · Data Science & Machine Learning
