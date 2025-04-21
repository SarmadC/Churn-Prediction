# Customer Churn Prediction Project

## Overview

This project aims to predict customer churn within a 90-day window using historical customer data. The analysis involves data cleaning, extensive feature engineering, customer segmentation based on tenure, model building (RandomForest, XGBoost), hyperparameter tuning, and model interpretation using SHAP. A key finding was the need to model newly signed-up customers separately from established customers due to differing churn behaviors and data availability.

## Data

The analysis uses customer data encompassing:
* Signup information (date, demographics like age group)
* Product/Subscription details (plan type, price)
* Billing history (number of cycles completed)
* Support ticket interactions (dates, total cases)

Data was soruced from Kaggle: [Customer Subscription Data](https://www.kaggle.com/datasets/gsagar12/dspp1)

## Methodology

The project follows these steps within the `churn-prediction.ipynb` Jupyter Notebook:

1.  **Data Loading & Initial Exploration:** Loading datasets for signups, billing, and support tickets. Initial checks for data quality and missing values.
2.  **Data Preprocessing & Cleaning:** Handling missing values (especially interpreting NaNs in cancellation/support data), converting data types (e.g., dates), joining datasets.
3.  **Feature Engineering:** Creating new features relevant to churn, such as:
    * Customer tenure (days, months, groups)
    * Support interaction metrics (ever contacted, days since last contact, monthly rate, days to first contact)
    * Billing cycle counts
    * Derived demographics (age groups)
4.  **Target Variable Definition:** Defining churn (`will_churn_next_90d`) based on whether a customer cancels within 90 days after a specific `cutoff_date`.
5.  **Customer Segmentation:** Identifying a high churn rate among customers with < 90 days tenure ("new customers") and splitting the dataset into "new" and "established" segments for separate modeling.
6.  **Model Development:**
    * **Established Customers:** Trained baseline RandomForest and tuned an XGBoost model using `RandomizedSearchCV`. Evaluated using classification reports, ROC AUC, and Precision-Recall (PRC) AUC.
    * **New Customers:** Attempted modeling using features available at signup, but models performed poorly.
7.  **Model Interpretation:** Used SHAP (SHapley Additive exPlanations) to understand feature importance and their impact on the predictions for the established customer XGBoost model.

## Key Findings & Results

* **Established Customers:**
    * Successfully developed and tuned an XGBoost model to predict churn within the next 90 days.
    * Achieved high precision (0.95) and moderate recall (0.40) with good AUC scores (ROC AUC: 0.83, PRC AUC: 0.51) on the test set. This indicates the model is effective at identifying likely churners among established customers with few false positives.
    * SHAP analysis revealed key factors influencing churn risk for established customers:
        * *Decreasing Risk:* More completed billing cycles, higher price points (likely indicating annual plans), longer tenure, higher monthly support contact rates (potentially indicating engagement or successful issue resolution).
        * *Increasing Risk:* Monthly billing cycles, very recent support contact.
* **New Customers (0-90 Days Tenure):**
    * Attempts to model early churn using features available near signup were unsuccessful.
    * Models performed near random chance (e.g., ROC AUC ~0.56, PRC AUC ~0.04), indicating the current features lack sufficient predictive power for this high-churn group.
    * Addressing early churn effectively would likely require additional data related to customer acquisition channels or initial onboarding experiences.

## Usage

To explore the analysis or replicate the results, run the Jupyter Notebook:
`churn-prediction.ipynb`

## Libraries Required

The analysis primarily uses the following Python libraries:

* pandas
* numpy
* scikit-learn
* xgboost
* shap
* matplotlib
* seaborn
