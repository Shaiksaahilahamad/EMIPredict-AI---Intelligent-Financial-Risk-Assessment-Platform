import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import joblib

from src.data_preprocessing import prepare_data
from src.config import CLASSIFIER_MODEL_PATH, MLFLOW_EXPERIMENT_CLASSIFICATION
from src.mlflow_utils import configure_mlflow
import mlflow
import mlflow.sklearn


def evaluate_classification(y_true, y_proba, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    try:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr")
    except Exception:
        auc = float("nan")

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc_ovr": auc,
    }


def train_classification_models():
    (
        preprocessor,
        feature_cols,
        X_train_c,
        X_test_c,
        y_train_c,
        y_test_c,
        _X_train_r,
        _X_test_r,
        _y_train_r,
        _y_test_r,
        df_feat,
    ) = prepare_data()

    # 🔑 Encode string labels to integers 0,1,2 for ALL models (incl. XGBoost)
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_c)
    y_test = label_encoder.transform(y_test_c)

    configure_mlflow(MLFLOW_EXPERIMENT_CLASSIFICATION)

    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, n_jobs=-1),
        "random_forest_classifier": RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
        ),
        "xgb_classifier": XGBClassifier(
            n_estimators=400,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        ),
        # "gradient_boosting_classifier": GradientBoostingClassifier(
        #     n_estimators=200, learning_rate=0.05, random_state=42
        # ),
    }
    X_train_trans = preprocessor.transform(X_train_c)
    X_test_trans = preprocessor.transform(X_test_c)

    best_model = None
    best_name = None
    best_f1 = -1

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_type", name)

            model.fit(X_train_trans, y_train)
            y_proba = model.predict_proba(X_test_trans)
            y_pred = model.predict(X_test_trans)

            metrics = evaluate_classification(y_test, y_proba, y_pred)

            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))

            mlflow.sklearn.log_model(model, artifact_path="model")

            print(f"[{name}] Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_model = model
                best_name = name

    print(f"\nBest classification model: {best_name} | F1={best_f1:.4f}")

    # 🔥 Save both model AND label encoder so Streamlit can decode 0/1/2 back to text
    CLASSIFIER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best_model,
            "label_encoder": label_encoder,
        },
        CLASSIFIER_MODEL_PATH,
    )


if __name__ == "__main__":
    train_classification_models()
