# Customer Churn Prediction Using Interpretable Machine Learning Models

This repository contains the code and analysis for Assignment #3: Interpretable ML, where we analyze a telecommunications dataset to predict customer churn. We explore different models including Linear Regression, Logistic Regression, and Generalized Additive Models (GAM), focusing on both model performance and interpretability.

Dataset: [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

# Models Implemented
## Linear Regression:

1. Churn is treated as a continuous variable (0 for not churned, 1 for churned).
2. Coefficients are interpreted to understand the linear effect of each feature.
3. Model performance metrics: R², Mean Squared Error (MSE).

## Logistic Regression:

1. Churn is treated as a binary classification task.
2. The model provides odds ratios for feature importance.
3. Model performance metrics: Accuracy, Classification Report, ROC-AUC score.

## Generalized Additive Model (GAM):

1. Captures non-linear relationships between features and churn.
2. The model fits splines to each feature for better flexibility.
3. Model performance metrics: Accuracy, Classification Report, ROC-AUC score.

## Usage
Open the notebook in Google Colab and run all cells to reproduce the results.

To Import Data from Kaggle
- Go to your account, Scroll to API section and Click Expire API Token to remove previous tokens
- Click on Create New API Token - It will download kaggle.json file on your machine.
- Upload kaggle.json to the folder directory

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1X0DZqMjzM6sEKjRa73NIsYH7wtKpxEkQ#scrollTo=YUEUl_U78ORX)


Reference: 
1. https://github.com/AIPI-590-XAI/Duke-AI-XAI/tree/main/interpretable-ml-example-notebooks
2. https://github.com/sandipanpaul21/Logistic-regression-in-python/tree/main
