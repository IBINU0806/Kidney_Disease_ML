# Kidney Disease Prediction using Machine Learning

This project builds a machine learning model to predict whether a patient needs dialysis treatment, based on medical indicators in a kidney disease dataset.

## 📁 Dataset

- `kidney_disease_dataset.csv`: contains health data of patients.
- Target column: `Dialysis_Needed` (binary classification).

## ⚙️ Features

- Data preprocessing with `SMOTE` to handle class imbalance
- Feature scaling using `StandardScaler`
- Hyperparameter tuning with `GridSearchCV` on `RandomForestClassifier`
- Evaluation using `classification_report` (precision, recall, f1-score)

## 📦 Requirements

```bash
pip install pandas scikit-learn imbalanced-learn
