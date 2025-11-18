import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import joblib

from src.config import (
    DATA_PATH,
    PREPROCESSOR_PATH,
    TARGET_CLASS,
    TARGET_REG,
    RANDOM_STATE,
)
from src.feature_engineering import engineer_features
from src.data_quality import data_quality_report


def load_data(path: str = str(DATA_PATH)) -> pd.DataFrame:
    # low_memory=False avoids weird chunk-based dtype guesses
    df = pd.read_csv(path, low_memory=False)
    return df



def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1) Strip spaces from object columns
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip()

    # 2) Force known numeric columns to numeric
    numeric_like_cols = [
        "age",
        "monthly_salary",
        "years_of_employment",
        "monthly_rent",
        "family_size",
        "dependents",
        "school_fees",
        "college_fees",
        "travel_expenses",
        "groceries_utilities",
        "other_monthly_expenses",
        "current_emi_amount",
        "credit_score",
        "bank_balance",
        "emergency_fund",
        "requested_amount",
        "requested_tenure",
        "max_monthly_emi",
    ]

    for col in numeric_like_cols:
        if col in df.columns:
            # Turn "38800.0", " 7200 ", "" etc. into real floats (or NaN)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3) Recompute numeric & categorical columns after coercion
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    # 4) Fill numeric missing with median
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # 5) Fill categorical missing with "Unknown"
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    # 6) Drop duplicate rows if any
    df = df.drop_duplicates().reset_index(drop=True)

    return df


def build_preprocessor(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in [TARGET_CLASS, TARGET_REG]]
    X = df[feature_cols]

    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor, feature_cols, numeric_features, categorical_features


def prepare_data(test_size: float = 0.2):
    # Load + clean
    df_raw = load_data()
    df_clean = basic_cleaning(df_raw)

    # Feature engineering
    df_feat = engineer_features(df_clean)

    # Data quality (you can print or log this in notebooks / training)
    dq = data_quality_report(df_feat)
    print("Data quality report (summary):")
    print(f"Rows: {dq['row_count']}, Columns: {dq['column_count']}")
    print(f"Duplicate rows: {dq['duplicate_rows']}")

    # Build preprocessor on full data (features only)
    preprocessor, feature_cols, num_cols, cat_cols = build_preprocessor(df_feat)

    X = df_feat[feature_cols]
    y_class = df_feat[TARGET_CLASS]
    y_reg = df_feat[TARGET_REG]

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_class, test_size=test_size, random_state=RANDOM_STATE, stratify=y_class
    )

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y_reg, test_size=test_size, random_state=RANDOM_STATE
    )

    # Fit preprocessor on training (classification split)
    preprocessor.fit(X_train_c)

    # Save preprocessor
    MODELS_DIR = PREPROCESSOR_PATH.parent
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "preprocessor": preprocessor,
            "feature_cols": feature_cols,
            "num_cols": num_cols,
            "cat_cols": cat_cols,
        },
        PREPROCESSOR_PATH,
    )

    return (
        preprocessor,
        feature_cols,
        X_train_c,
        X_test_c,
        y_train_c,
        y_test_c,
        X_train_r,
        X_test_r,
        y_train_r,
        y_test_r,
        df_feat,
    )
