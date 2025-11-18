# EMIPredict AI – Intelligent Financial Risk Assessment Platform

## Overview

This project implements the full spec from the *EMIPredict AI – Intelligent Financial Risk Assessment Platform* document:

- Dual ML:
  - **Classification**: `emi_eligibility` (Eligible / High_Risk / Not_Eligible)
  - **Regression**: `max_monthly_emi`
- 22+ input features with **feature engineering** (financial ratios & risk scores)
- **MLflow** experiment tracking and model comparison
- **Streamlit** multi-page web app with:
  - Overview & scenario rules
  - EDA & model summary
  - Real-time EMI prediction
  - CRUD for user records
- Streamlit Cloud–ready
DEPLOYED LINK IS :-  https://emi-prediction-ai.streamlit.app/
## Dataset Schema

Expected columns in `data/EMI_dataset.csv`:

```text
age,gender,marital_status,education,monthly_salary,employment_type,
years_of_employment,company_type,house_type,monthly_rent,family_size,
dependents,school_fees,college_fees,travel_expenses,groceries_utilities,
other_monthly_expenses,existing_loans,current_emi_amount,credit_score,
bank_balance,emergency_fund,emi_scenario,requested_amount,requested_tenure,
emi_eligibility,max_monthly_emi

