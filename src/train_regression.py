import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import mlflow
import mlflow.sklearn

from src.data_preprocessing import prepare_data
from src.config import REGRESSOR_MODEL_PATH, MLFLOW_EXPERIMENT_REGRESSION
from src.mlflow_utils import configure_mlflow


def evaluate_regression(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100.0)
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


def train_regression_models():
    (
        preprocessor,
        feature_cols,
        _X_train_c,
        _X_test_c,
        _y_train_c,
        _y_test_c,
        X_train_r,
        X_test_r,
        y_train_r,
        y_test_r,
        df_feat,
    ) = prepare_data()

    configure_mlflow(MLFLOW_EXPERIMENT_REGRESSION)

    models = {
        # 1) Fast baseline – good for comparison
        "linear_regression": LinearRegression(),

        # 2) Random Forest – reduced depth & trees for balance
        "random_forest_regressor": RandomForestRegressor(
            n_estimators=150,   # moderate number of trees
            max_depth=14,       # limits complexity so it doesn't explode
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1,
        ),

        # 3) XGBoost – powerful but with controlled n_estimators
        "xgb_regressor": XGBRegressor(
            n_estimators=250,   # less than your earlier 500
            learning_rate=0.08, # slightly smaller LR for stability
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            reg_lambda=1.0,
            reg_alpha=0.0,
            random_state=42,
            n_jobs=-1,
        ),
    }

    X_train_trans = preprocessor.transform(X_train_r)
    X_test_trans = preprocessor.transform(X_test_r)

    best_model = None
    best_name = None
    best_rmse = float("inf")

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_type", name)

            model.fit(X_train_trans, y_train_r)
            y_pred = model.predict(X_test_trans)
            metrics = evaluate_regression(y_test_r, y_pred)

            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))

            mlflow.sklearn.log_model(model, artifact_path="model")

            print(
                f"[{name}] RMSE={metrics['rmse']:.2f} | MAE={metrics['mae']:.2f} | R2={metrics['r2']:.3f}"
            )

            if metrics["rmse"] < best_rmse:
                best_rmse = metrics["rmse"]
                best_model = model
                best_name = name

    print(f"\nBest regression model: {best_name} | RMSE={best_rmse:.2f}")

    REGRESSOR_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, REGRESSOR_MODEL_PATH)


if __name__ == "__main__":
    train_regression_models()
