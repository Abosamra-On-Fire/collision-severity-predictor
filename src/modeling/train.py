#train.py
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.utils.class_weight import compute_class_weight

import mlflow

from src.data.load_data import load_csv
from src import config as cfg
from src.utils import log_action, save_stage_report, setup_logging

# Import all model training functions
from src.modeling.models import (
    train_random_forest,
    train_xgboost,
    train_catboost,
    train_lightgbm,
    train_neural_network,
    train_cascade_rf,
)

warnings.filterwarnings("ignore")


# ============================================================
# MLFLOW SETUP
# ============================================================

def setup_mlflow() -> None:
    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(cfg.MLFLOW_EXPERIMENT_NAME)
    log_action(
        step="MLflowSetup",
        action="MLflow configured",
        rationale=f"tracking_uri={cfg.MLFLOW_TRACKING_URI}, experiment={cfg.MLFLOW_EXPERIMENT_NAME}",
        stage="training",
    )


# ============================================================
# DATA LOADING
# ============================================================

def load_trainset() -> tuple[
    pd.DataFrame, pd.Series
]:
    """
    Load feature-engineered train

    Returns
    -------
    X_train, y_train
    """
    train_df = load_csv(str(cfg.PROCESSED_DATA_DIR / cfg.TRAIN_OUTPUT_FILE))


    X_train, y_train =  train_df.drop(columns=[cfg.TARGET_COL]), train_df[cfg.TARGET_COL]

    log_action(
        step="LoadData",
        action="Loaded Train set",
        records_affected=X_train.shape[0],
        rationale=f"shape={X_train.shape}, class_dist={y_train.value_counts().to_dict()}",
        stage="training",
    )

    return X_train, y_train


# ============================================================
# COMPARISON TABLE
# ============================================================

def _log_cv_comparison(cv_results: dict[str, dict]) -> pd.DataFrame:
    """Create and save a comparison table of CV results."""
    rows = []
    for name, scores in cv_results.items():
        row = {
            "model": name,
            "cv_f1_weighted_mean": round(scores.get("cv_f1_weighted_mean", float("nan")), 4),
            "cv_f1_weighted_std": round(scores.get("cv_f1_weighted_std", float("nan")), 4),
            "cv_fatal_recall_mean": round(scores.get("cv_fatal_recall_mean", float("nan")), 4),
            "cv_accuracy_mean": round(scores.get("cv_accuracy_mean", float("nan")), 4),
            "cv_roc_auc_mean": round(scores.get("cv_roc_auc_ovr_weighted_mean", float("nan")), 4),
        }
        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")
    df = df.sort_values("cv_fatal_recall_mean", ascending=False)
    
    log_action(
        step="ModelComparison",
        action="CV comparison table generated",
        rationale=f"Saved to {cfg.MODELS_DIR / 'cv_model_comparison.csv'}\n{df.to_string()}",
        stage="training",
    )

    cmp_path = cfg.MODELS_DIR / "cv_model_comparison.csv"
    df.to_csv(cmp_path)
    
    return df


# ============================================================
# MAIN PIPELINE
# ============================================================

def train_all_models() -> None:
    """
    Full training pipeline:
      1. Train all models on X_train using cross-validation
      2. Select the best model based on CV f1_weighted score
      3. Save the best model for later evaluation
    """
    setup_logging()
    setup_mlflow()

    X_train, y_train = load_trainset()

    # Class weights
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = dict(zip(map(int, classes), weights.tolist()))
    log_action(
        step="ClassWeights",
        action="Computed class weights",
        rationale=str(class_weights),
        stage="training",
    )

    # ── Train all models using CV only ─────────────────────────────────────
    log_action(
        step="TrainAllModels",
        action="Starting training of all models with cross-validation",
        stage="training",
    )

    trained: dict[str, tuple[object, dict]] = {}
    
    try:
        log_action(step="TrainModel", action="Training Random Forest...", stage="training")
        trained["RandomForest"] = train_random_forest(X_train, y_train, class_weights)
    except Exception as e:
        log_action(step="TrainModel", action=f"Random Forest failed: {e}", stage="training")

    try:
        log_action(step="TrainModel", action="Training XGBoost...", stage="training")
        trained["XGBoost"] = train_xgboost(X_train, y_train, class_weights)
    except Exception as e:
        log_action(step="TrainModel", action=f"XGBoost failed: {e}", stage="training")

    try:
        log_action(step="TrainModel", action="Training CatBoost...", stage="training")
        trained["CatBoost"] = train_catboost(X_train, y_train, class_weights)
    except Exception as e:
        log_action(step="TrainModel", action=f"CatBoost failed: {e}", stage="training")

    try:
        log_action(step="TrainModel", action="Training LightGBM...", stage="training")
        trained["LightGBM"] = train_lightgbm(X_train, y_train, class_weights)
    except Exception as e:
        log_action(step="TrainModel", action=f"LightGBM failed: {e}", stage="training")

    try:
        log_action(step="TrainModel", action="Training Neural Network...", stage="training")
        trained["NeuralNetwork"] = train_neural_network(X_train, y_train, class_weights)
    except Exception as e:
        log_action(step="TrainModel", action=f"Neural Network failed: {e}", stage="training")

    try:
        log_action(step="TrainModel", action="Training Cascade RF...", stage="training")
        trained["CascadeRF"] = train_cascade_rf(X_train, y_train, class_weights)
    except Exception as e:
        log_action(step="TrainModel", action=f"Cascade RF failed: {e}", stage="training")

    # ── Compare CV results ─────────────────────────────────────────────────
    cv_results = {name: scores for name, (_, scores) in trained.items()}
    comparison_df = _log_cv_comparison(cv_results)

    # ── Select best model by CV f1_weighted ───────────────────────────────
    best_model_name = max(
        cv_results,
        key=lambda n: cv_results[n].get("cv_fatal_recall_mean", -1.0),
    )
    best_model, best_cv_scores = trained[best_model_name]

    log_action(
        step="ModelSelection",
        action="Selected best model based on CV cv_fatal_recall_mean",
        rationale=(
            f"model={best_model_name}, "
            f"cv_f1_weighted={best_cv_scores.get("cv_fatal_recall_mean", 0.0):.4f}"
        ),
        stage="training",
    )

    # ── Save best model info ───────────────────────────────────────────────
    best_model_info = {
        "model_name": best_model_name,
        "cv_fatal_recall_mean": best_cv_scores.get("cv_fatal_recall_mean", 0.0),
    }
    
    import json
    best_model_path = cfg.MODELS_DIR / "best_model_info.json"
    with open(best_model_path, "w") as f:
        json.dump(best_model_info, f, indent=2)
    
    log_action(
        step="SaveBestModel",
        action="Saved best model information",
        rationale=f"Saved to {best_model_path}",
        stage="training",
    )

    save_stage_report("training")

    log_action(
        step="PipelineComplete",
        action="All models trained with CV",
        rationale=f"Best model: {best_model_name} (f1={best_cv_scores.get('cv_f1_weighted_mean', 0.0):.4f})",
        stage="training",
    )
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nBest Model: {best_model_name}")
    print(f"CV F1 Weighted: {best_cv_scores.get('cv_f1_weighted_mean', 0.0):.4f}")
    print(f"\nAll models comparison saved to: {cfg.MODELS_DIR / 'cv_model_comparison.csv'}")
    print(f"Best model info saved to: {best_model_path}")
    print(f"\nRun eval.py to evaluate the best model on the test set and perform error analysis.")
    print("=" * 80)


def train_single_model(model_name: str) -> None:
    """Train a single model using CV only."""
    setup_mlflow()
    X_train, y_train = load_trainset()

    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = dict(zip(map(int, classes), weights.tolist()))

    model_map = {
        "rf": train_random_forest,
        "xgb": train_xgboost,
        "catboost": train_catboost,
        "lgb": train_lightgbm,
        "nn": train_neural_network,
        "cascade": train_cascade_rf,
    }

    if model_name not in model_map:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(model_map)}")

    model, cv_scores = model_map[model_name](X_train, y_train, class_weights)
    save_stage_report("training")
    
    log_action(
        step="TrainModel",
        action=f"{model_name.upper()} training complete",
        rationale=f"cv_f1_weighted={cv_scores.get('cv_f1_weighted_mean', 0.0):.4f}",
        stage="training",
    )


if __name__ == "__main__":
    import sys

    # Usage:
    #   python train.py                # train all models
    #   python train.py --model rf     # train one model
    setup_logging()
    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        train_single_model(sys.argv[idx + 1])
    else:
        train_all_models()