from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

DATA_PATH = DATA_DIR / "EMI_dataset.csv"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
CLASSIFIER_MODEL_PATH = MODELS_DIR / "classifier_best.pkl"
REGRESSOR_MODEL_PATH = MODELS_DIR / "regressor_best.pkl"

# MLflow experiment names
MLFLOW_EXPERIMENT_CLASSIFICATION = "EMI_Eligibility_Classification"
MLFLOW_EXPERIMENT_REGRESSION = "Max_EMI_Regression"

RANDOM_STATE = 42

# EMI scenario constraints (amount + tenure)
EMI_SCENARIO_RULES = {
    "E-commerce Shopping EMI": {
        "amount_min": 10000,
        "amount_max": 200000,
        "tenure_min": 3,
        "tenure_max": 24,
    },
    "Home Appliances EMI": {
        "amount_min": 20000,
        "amount_max": 300000,
        "tenure_min": 6,
        "tenure_max": 36,
    },
    "Vehicle EMI": {
        "amount_min": 80000,
        "amount_max": 1500000,
        "tenure_min": 12,
        "tenure_max": 84,
    },
    "Personal Loan EMI": {
        "amount_min": 50000,
        "amount_max": 1000000,
        "tenure_min": 12,
        "tenure_max": 60,
    },
    "Education EMI": {
        "amount_min": 50000,
        "amount_max": 500000,
        "tenure_min": 6,
        "tenure_max": 48,
    },
}

# Target column names (as in dataset)
TARGET_CLASS = "emi_eligibility"
TARGET_REG = "max_monthly_emi"
