import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Basic totals
    df["total_education_fees"] = df["school_fees"] + df["college_fees"]
    df["total_living_expenses"] = (
        df["travel_expenses"]
        + df["groceries_utilities"]
        + df["other_monthly_expenses"]
        + df["monthly_rent"]
    )

    df["total_expenses"] = (
        df["total_education_fees"]
        + df["total_living_expenses"]
        + df["current_emi_amount"]
    )

    # Disposable income
    df["disposable_income"] = df["monthly_salary"] - df["total_expenses"]

    # Ratios (avoid division by zero)
    df["dti_ratio"] = (df["current_emi_amount"] + 1) / (df["monthly_salary"] + 1)
    df["expense_to_income_ratio"] = df["total_expenses"] / (df["monthly_salary"] + 1)
    df["savings_rate"] = (df["bank_balance"] + df["emergency_fund"]) / (
        df["monthly_salary"] * 6 + 1
    )

    # Loan burden vs expenses
    df["emi_burden_ratio"] = (df["current_emi_amount"] + 1) / (df["total_expenses"] + 1)

    # Binary flags
    df["existing_loans_flag"] = (df["existing_loans"].astype(str).str.lower() == "yes").astype(int)
    df["high_credit_score_flag"] = (df["credit_score"] >= 750).astype(int)
    df["medium_credit_score_flag"] = (
        (df["credit_score"] >= 650) & (df["credit_score"] < 750)
    ).astype(int)

    # Simple affordability proxy (monthly capacity estimate)
    df["affordability_score"] = (
        df["disposable_income"] * 0.6 + (df["savings_rate"] * 10000)
    )

    return df
