import mlflow
from pathlib import Path
from src.config import BASE_DIR

def configure_mlflow(experiment_name: str):
    mlruns_dir = BASE_DIR / "mlruns"
    mlruns_dir.mkdir(exist_ok=True)

    # Local file-based backend is fine for capstone
    mlflow.set_tracking_uri(f"file:{mlruns_dir}")
    mlflow.set_experiment(experiment_name)
