# customer-churn-gridsearch-L1
# Customer Churn Prediction with Optimized Logistic Regression

This repository contains a complete implementation of a customer churn prediction system based on the Telco Customer Churn dataset. The project focuses on improving a baseline L1-regularized logistic regression model through feature engineering and systematic hyperparameter tuning using GridSearchCV. The goal is to identify at-risk customers more effectively and support data-driven retention strategies.

---

## Project Overview

Customer churn prediction is a central problem in customer analytics. Businesses often rely on churn models to anticipate which users are likely to discontinue their services. In this project, logistic regression is used as a transparent, interpretable model, while feature transformations and regularization help capture meaningful patterns.

This project includes:

- Data preprocessing and cleaning
- Feature engineering to capture behavioral and financial indicators
- One-hot encoding for categorical variables
- Training logistic regression with L1 regularization
- Hyperparameter tuning using GridSearchCV
- Modular Python scripts for reproducible experimentation

---

## Dataset

**Source:** Telco Customer Churn dataset (Kaggle)  
**Size:** 7,043 customer records with 21 input features  
**Feature groups include:**

- **Demographics:** gender, senior citizen, partner, dependents  
- **Account information:** tenure, contract type, billing mode, payment method  
- **Services:** phone service, internet type, online security, backup, streaming  
- **Financial:** monthly charges, total charges  

**Note:**  
The dataset is **not included** due to licensing constraints.


