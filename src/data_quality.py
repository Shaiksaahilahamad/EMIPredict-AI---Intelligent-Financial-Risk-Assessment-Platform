import pandas as pd

def data_quality_report(df: pd.DataFrame) -> dict:
    report = {}

    report["row_count"] = len(df)
    report["column_count"] = len(df.columns)
    report["missing_values_per_column"] = df.isna().sum().to_dict()
    report["duplicate_rows"] = int(df.duplicated().sum())

    # Simple numeric stats
    num_cols = df.select_dtypes(include=["number"]).columns
    report["numeric_summary"] = df[num_cols].describe().to_dict()

    # Simple categorical stats
    cat_cols = df.select_dtypes(include=["object"]).columns
    report["categorical_unique_values"] = {
        col: df[col].nunique() for col in cat_cols
    }

    return report
