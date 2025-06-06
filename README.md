# Credit Card Default Prediction ML

## Overview

This project aims to predict whether a customer will default on their credit card payment using machine learning. It utilizes various classifiers to train on a dataset containing customer demographic and credit information, and compares the performance of the models using metrics such as accuracy, precision, recall, and F1 score.

The project includes multiple machine learning models:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)

### Key Features:
- Data preprocessing and feature scaling
- Model training and evaluation
- Comparison of multiple models with performance metrics
- Visualizations of confusion matrices and model performance

## Dataset

The dataset used in this project is from a dataset of credit card clients. It contains demographic and financial information of customers, and the target variable indicates whether the customer defaulted on their credit card payment in the next month.

**Data Source:**
- File: `default of credit card clients.xls`
- Columns:
  - Various customer demographic and credit details.
  - Target: `default payment next month` (1 for default, 0 for no default).

## Requirements

This project requires Python 3.x and several libraries. To set up the environment, follow these instructions:

### 1. Install Required Libraries

Make sure you have the following libraries installed:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`

You can install them using pip:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn

