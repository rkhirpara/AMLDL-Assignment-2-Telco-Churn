# AMLDL Assignment 2 - Telecom Customer Churn Prediction

MBA ZG582 - **Applied Machine Learning and Deep Learning**  
BITS Pilani Work Integrated Learning Programmes  
Assignment 2 - Phase 2: Machine Learning Phase

This repository contains the reproducible implementation for a telecom customer churn prediction project. The work continues the Phase 1 concept note and covers data preparation, model development, hyperparameter tuning, evaluation, model comparison, and business interpretation.

## Open in Google Colab

[Open the notebook in Google Colab](https://colab.research.google.com/github/rkhirpara/AMLDL-Assignment-2-Telco-Churn/blob/main/2024BC26597_AMLDL_Assignment_2.ipynb)

The notebook first looks for the CSV locally. If the file is not present, it loads the dataset directly from this public GitHub repository, so the Colab version can be run with **Runtime -> Run all**.

## Repository contents

- `2024BC26597_AMLDL_Assignment_2.ipynb` - executable analysis notebook
- `WA_Fn-UseC_-Telco-Customer-Churn.csv` - Telco Customer Churn dataset used in the analysis
- `requirements.txt` - Python package requirements for local execution

## Problem statement

The objective is to predict whether a telecom customer is likely to churn using demographic, account, service-subscription, and billing-related attributes. The intended business use is to rank customers by churn risk so that retention teams can intervene proactively.

## Dataset

The analysis uses the Telco Customer Churn sample dataset with **7,043 customer records and 21 columns**. The target variable is `Churn` (`Yes`/`No`).

Key preparation steps include:

- converting `TotalCharges` to numeric and handling 11 blank values
- dropping `customerID` from model features
- median imputation for numerical missing values
- most-frequent imputation and one-hot encoding for categorical features
- standardization of numerical variables
- stratified 80/20 train-test split (`random_state=42`)

## Models evaluated

1. Logistic Regression
2. Random Forest
3. Gradient Boosting

Hyperparameter tuning uses `GridSearchCV` and `RandomizedSearchCV` with cross-validation.

## Tuned hold-out results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.805 | 0.655 | 0.559 | 0.603 | 0.841 |
| Random Forest | 0.766 | 0.542 | **0.765** | **0.634** | 0.842 |
| Gradient Boosting | 0.796 | 0.651 | 0.495 | 0.562 | **0.846** |

The **tuned Random Forest** is selected as the operational model because it identifies a substantially larger proportion of actual churners and gives the strongest F1 trade-off at the default 0.50 threshold. On the hold-out test set it identifies **286 of 374 churners** (recall **76.5%**).

## Business interpretation

Important predictive signals include month-to-month contracts, tenure, total charges, lack of online security/technical support, monthly charges, fiber-optic service, and electronic-check payment. These are predictive associations rather than causal conclusions and should be validated through controlled retention experiments.

## Run locally

```bash
pip install -r requirements.txt
jupyter notebook 2024BC26597_AMLDL_Assignment_2.ipynb
```

Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the same directory as the notebook, then run all cells.

## Reproducibility

The notebook fixes random seeds where applicable and performs preprocessing inside scikit-learn pipelines to reduce data-leakage risk. Reported figures and metrics are generated from the supplied dataset and the executed workflow.

## Dataset source

IBM Telco Customer Churn sample dataset; also distributed through public dataset mirrors such as Kaggle.
