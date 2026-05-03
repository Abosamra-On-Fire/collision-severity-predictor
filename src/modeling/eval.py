#eval.py
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_fscore_support,
)

import torch
import torch.nn as nn

import mlflow
import mlflow.sklearn
import mlflow.pytorch

from src.data.load_data import load_csv
from src import config as cfg
from src.utils import log_action, save_stage_report, setup_logging
from src.modeling.models import CascadeRFModel
from src.modeling.models import MLPClassifier



warnings.filterwarnings("ignore")


# ============================================================
# SETUP
# ============================================================

def setup_mlflow() -> None:
    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(cfg.MLFLOW_EXPERIMENT_NAME)


def load_test() -> tuple[pd.DataFrame, pd.Series]:
    """Load feature-engineered test split only."""
    test_df = load_csv(str(cfg.PROCESSED_DATA_DIR / cfg.TEST_OUTPUT_FILE))
    X_test = test_df.drop(columns=[cfg.TARGET_COL])
    y_test = test_df[cfg.TARGET_COL]
    return X_test, y_test


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    prefix: str = "",
) -> dict[str, float]:
    """Standard classification metrics."""
    p = prefix
    metrics: dict[str, float] = {}

    metrics[f"{p}accuracy"] = accuracy_score(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0
    )
    for i, cls in enumerate(["slight", "serious", "fatal"]):
        metrics[f"{p}precision_{cls}"] = float(precision[i])
        metrics[f"{p}recall_{cls}"] = float(recall[i])
        metrics[f"{p}f1_{cls}"] = float(f1[i])
        metrics[f"{p}support_{cls}"] = int(support[i])

    pw, rw, f1w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    metrics[f"{p}precision_weighted"] = float(pw)
    metrics[f"{p}recall_weighted"] = float(rw)
    metrics[f"{p}f1_weighted"] = float(f1w)

    try:
        metrics[f"{p}roc_auc_weighted"] = roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="weighted"
        )
        metrics[f"{p}roc_auc_macro"] = roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="macro"
        )
    except Exception:
        metrics[f"{p}roc_auc_weighted"] = 0.0
        metrics[f"{p}roc_auc_macro"] = 0.0

    return metrics


def compute_business_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> dict[str, float]:
    """Business-oriented metrics for road-safety collision severity prediction."""
    p = prefix
    metrics: dict[str, float] = {}
    
    # Force plain Python arrays to avoid numpy scalar type issues
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Fatal recall
    fatal_mask = y_true == 2
    if fatal_mask.sum() > 0:
        metrics[f"{p}business_fatal_recall"] = float(
            (y_pred[fatal_mask] == 2).sum() / fatal_mask.sum()
        )
    else:
        metrics[f"{p}business_fatal_recall"] = 0.0

    # High-severity detection rate
    high_mask = y_true >= 1
    if high_mask.sum() > 0:
        metrics[f"{p}business_high_severity_detection_rate"] = float(
            ((y_pred >= 1) & high_mask).sum() / high_mask.sum()
        )
    else:
        metrics[f"{p}business_high_severity_detection_rate"] = 0.0

    # Average misclassification cost
    # Use .item() to safely convert numpy scalars to Python ints
    costs = np.array([
        cfg.SEVERITY_COST_MATRIX.get((int(np.array(p_).item()), int(np.array(t_).item())), 0.0)
        for p_, t_ in zip(y_pred, y_true)
    ])
    avg_cost = float(costs.mean())
    metrics[f"{p}business_avg_misclassification_cost"] = avg_cost

    # Weighted safety score
    max_cost = max(cfg.SEVERITY_COST_MATRIX.values())
    metrics[f"{p}business_weighted_safety_score"] = float(
        100.0 * (1.0 - avg_cost / max_cost)
    )

    return metrics


def _all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    prefix: str = "",
) -> dict[str, float]:
    m = compute_metrics(y_true, y_pred, y_proba, prefix=prefix)
    m.update(compute_business_metrics(y_true, y_pred, prefix=prefix))
    return m


# ============================================================
# LOAD MODELS
# ============================================================

def load_all_models() -> dict[str, Any]:
    """Load all trained models from disk."""
    import joblib

    models = {}
    model_files = {
        "RandomForest": "random_forest_model.pkl",
        "XGBoost": "xgboost_model.pkl",
        "CatBoost": "catboost_model.pkl",
        "LightGBM": "lightgbm_model.pkl",
        "CascadeRF": "cascade_rf_model.pkl",
    }

    for name, filename in model_files.items():
        model_path = cfg.MODELS_DIR / filename
        if model_path.exists():
            try:
                if name == "CascadeRF":
                    cascade_dict = joblib.load(model_path)
                    models[name] = CascadeRFModel(**cascade_dict)
                else:
                    models[name] = joblib.load(model_path)
                log_action(
                    step="LoadModel",
                    action=f"Loaded {name}",
                    rationale=f"from {model_path}",
                    stage="evaluation",
                )
            except Exception as e:
                log_action(
                    step="LoadModel",
                    action=f"Failed to load {name}",
                    rationale=str(e),
                    stage="evaluation",
                )
        else:
            log_action(
                step="LoadModel",
                action=f"Model file not found: {name}",
                rationale=f"Expected at {model_path}",
                stage="evaluation",
            )

    # Load Neural Network separately
    nn_path = cfg.MODELS_DIR / "best_nn_model.pt"
    if nn_path.exists():
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Need to know input_dim - infer from test data
            X_test, _ = load_test()
            input_dim = X_test.shape[1]

            model = MLPClassifier(
                input_dim, cfg.MLP_HIDDEN_DIM, cfg.MLP_NUM_CLASSES, cfg.MLP_DROPOUT
            ).to(device)
            model.load_state_dict(torch.load(nn_path, map_location=device))
            model.eval()
            models["NeuralNetwork"] = model
            log_action(
                step="LoadModel",
                action="Loaded Neural Network",
                rationale=f"from {nn_path}",
                stage="evaluation",
            )
        except Exception as e:
            log_action(
                step="LoadModel",
                action="Failed to load Neural Network",
                rationale=str(e),
                stage="evaluation",
            )

    return models


# ============================================================
# TEST SET EVALUATION (ALL MODELS)
# ============================================================

def evaluate_all_on_test(
    models: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, dict]:
    """Evaluate all models on the test set."""
    results = {}

    for name, model in models.items():
        log_action(
            step="EvaluateModel",
            action=f"Evaluating {name} on test set",
            stage="evaluation",
        )

        # Get predictions
        if name == "NeuralNetwork":
            device = next(model.parameters()).device
            X_t = torch.FloatTensor(np.array(X_test, copy=True)).to(device)
            with torch.no_grad():
                proba = torch.softmax(model(X_t), dim=1).cpu().numpy()
            y_pred = proba.argmax(axis=1)
        else:
            y_pred = model.predict(X_test)
            proba = model.predict_proba(X_test)

        # Compute metrics
        metrics = _all_metrics(y_test, y_pred, proba, prefix="test_")
        results[name] = metrics

        # Log to MLflow
        with mlflow.start_run(run_name=f"{name}_TEST_EVAL"):
            mlflow.log_param("evaluated_model", name)
            mlflow.log_param("dataset", "test")
            mlflow.log_metrics(metrics)

        log_action(
            step="EvaluateModel",
            action=f"{name} test evaluation complete",
            rationale=(
                f"accuracy={metrics['test_accuracy']:.4f}, "
                f"f1_weighted={metrics['test_f1_weighted']:.4f}"
            ),
            stage="evaluation",
        )

    return results


# ============================================================
# COMPARISON VISUALIZATIONS (TEST SET)
# ============================================================

def create_comparison_plots(
    test_results: dict[str, dict],
    output_dir: Path,
) -> None:
    """Create comparison visualizations for all models on test set."""
    output_dir.mkdir(parents=True, exist_ok=True)

    models = list(test_results.keys())

    # 1. Overall Performance Comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    metrics_to_plot = [
        ("test_accuracy", "Accuracy"),
        ("test_f1_weighted", "F1 Weighted"),
        ("test_roc_auc_weighted", "ROC AUC Weighted"),
        ("test_business_weighted_safety_score", "Safety Score"),
    ]

    for idx, (metric_key, metric_name) in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        values = [test_results[m].get(metric_key, 0) for m in models]
        bars = ax.bar(models, values, color='steelblue', alpha=0.7)
        ax.set_title(f"{metric_name} Comparison (Test Set)", fontsize=14, fontweight='bold')
        ax.set_ylabel(metric_name)
        ax.set_ylim([0, max(values) * 1.1])
        ax.tick_params(axis='x', rotation=45)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "test_overall_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Per-Class Performance
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    classes = ["slight", "serious", "fatal"]

    for idx, cls in enumerate(classes):
        ax = axes[idx]

        f1_scores = [test_results[m].get(f"test_f1_{cls}", 0) for m in models]
        recall_scores = [test_results[m].get(f"test_recall_{cls}", 0) for m in models]
        precision_scores = [test_results[m].get(f"test_precision_{cls}", 0) for m in models]

        x = np.arange(len(models))
        width = 0.25

        ax.bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
        ax.bar(x, recall_scores, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1_scores, width, label='F1', alpha=0.8)

        ax.set_title(f"{cls.capitalize()} Class Performance (Test)", fontsize=14, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "test_per_class_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Business Metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    business_metrics = [
        ("test_business_fatal_recall", "Fatal Recall"),
        ("test_business_high_severity_detection_rate", "High Severity Detection"),
        ("test_business_avg_misclassification_cost", "Avg Misclassification Cost"),
    ]

    for idx, (metric_key, metric_name) in enumerate(business_metrics):
        ax = axes[idx]
        values = [test_results[m].get(metric_key, 0) for m in models]
        bars = ax.bar(models, values, color='coral', alpha=0.7)
        ax.set_title(f"{metric_name} (Test)", fontsize=14, fontweight='bold')
        ax.set_ylabel('Score')
        ax.tick_params(axis='x', rotation=45)

        if "cost" in metric_key.lower():
            ax.set_ylim([0, max(values) * 1.1])
            best_idx = np.argmin(values)
            bars[best_idx].set_color('green')
        else:
            ax.set_ylim([0, max(values) * 1.1])
            best_idx = np.argmax(values)
            bars[best_idx].set_color('green')

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "test_business_metrics_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    log_action(
        step="CreateVisualizations",
        action="Test set comparison plots created",
        rationale=f"Saved to {output_dir}",
        stage="evaluation",
    )


# ============================================================
# BEST MODEL TEST REPORT
# ============================================================

def save_best_model_report(
    model_name: str,
    test_metrics: dict[str, float],
    y_test: pd.Series,
    y_pred: np.ndarray,
) -> None:
    """Save detailed classification report for best model."""
    report = classification_report(
        y_test, y_pred,
        target_names=["Slight", "Serious", "Fatal"],
        digits=4,
    )

    report_path = cfg.MODELS_DIR / "test_classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write("Dataset: TEST SET\n")
        f.write("=" * 80 + "\n")
        f.write(report)

    # Also save metrics JSON
    metrics_path = cfg.MODELS_DIR / "test_metrics_summary.json"
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)

    log_action(
        step="SaveReport",
        action=f"Saved test report for {model_name}",
        rationale=str(report_path),
        stage="evaluation",
    )


# ============================================================
# ERROR ANALYSIS
# ============================================================

def perform_error_analysis(
    model: Any,
    model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    output_dir: Path,
) -> None:
    """Comprehensive error analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)

    log_action(
        step="ErrorAnalysis",
        action="Starting comprehensive error analysis",
        stage="evaluation",
    )

    # 1. Confusion Matrix with detailed annotations
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=["Slight", "Serious", "Fatal"],
        yticklabels=["Slight", "Serious", "Fatal"],
        cbar_kws={'label': 'Count'},
        ax=ax
    )
    ax.set_title(f"Confusion Matrix - {model_name} (Test)", fontsize=16, fontweight='bold')
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)

    for i in range(3):
        for j in range(3):
            if i != j:
                if cm[i, :].sum() > 0:
                    error_rate = cm[i, j] / cm[i, :].sum() * 100
                    ax.text(j + 0.5, i + 0.7, f'({error_rate:.1f}%)',
                           ha='center', va='center', fontsize=9, color='red')

    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix_detailed.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Misclassification Analysis
    misclassified_mask = y_test != y_pred
    misclassified_indices = np.where(misclassified_mask)[0]

    error_df = pd.DataFrame({
        'true_label': y_test.iloc[misclassified_indices].values,
        'pred_label': y_pred[misclassified_indices],
        'confidence': y_proba[misclassified_indices].max(axis=1),
    })

    error_patterns = error_df.groupby(['true_label', 'pred_label']).size().reset_index(name='count')
    error_patterns['pattern'] = error_patterns.apply(
        lambda x: f"{['Slight', 'Serious', 'Fatal'][int(x['true_label'])]} → "
                  f"{['Slight', 'Serious', 'Fatal'][int(x['pred_label'])]}",
        axis=1
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(error_patterns['pattern'], error_patterns['count'], color='coral')
    ax.set_xlabel('Number of Misclassifications', fontsize=12)
    ax.set_title('Misclassification Patterns (Test Set)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
               f'{int(width)}', ha='left', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "misclassification_patterns.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Confidence Distribution by Correctness
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    correct_mask = y_test == y_pred
    correct_conf = y_proba[correct_mask].max(axis=1)
    incorrect_conf = y_proba[~correct_mask].max(axis=1)

    axes[0].hist(correct_conf, bins=30, alpha=0.7, color='green', label='Correct')
    axes[0].hist(incorrect_conf, bins=30, alpha=0.7, color='red', label='Incorrect')
    axes[0].set_xlabel('Prediction Confidence', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Confidence Distribution (Test)', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].boxplot(
        [correct_conf, incorrect_conf],
        labels=['Correct', 'Incorrect'],
        patch_artist=True,
        boxprops=dict(facecolor='lightblue', alpha=0.7)
    )
    axes[1].set_ylabel('Prediction Confidence', fontsize=12)
    axes[1].set_title('Confidence Comparison (Test)', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "confidence_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Error by True Class
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    class_names = ["Slight", "Serious", "Fatal"]

    for i, cls_name in enumerate(class_names):
        ax = axes[i]
        cls_mask = y_test == i
        cls_correct = (y_test[cls_mask] == y_pred[cls_mask]).sum()
        cls_total = cls_mask.sum()
        cls_error_rate = (1 - cls_correct / cls_total) * 100 if cls_total > 0 else 0

        cls_errors = y_pred[cls_mask & ~correct_mask]
        if len(cls_errors) > 0:
            error_dist = pd.Series(cls_errors).value_counts()
            colors = ['lightcoral' if idx != i else 'lightgreen' for idx in range(3)]

            ax.bar(range(3), [error_dist.get(j, 0) for j in range(3)],
                  color=colors, alpha=0.7)
            ax.set_xticks(range(3))
            ax.set_xticklabels(class_names, rotation=0)
            ax.set_ylabel('Error Count', fontsize=11)
            ax.set_title(f'{cls_name} Errors\n(Error Rate: {cls_error_rate:.1f}%)',
                        fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

            for j, count in enumerate([error_dist.get(k, 0) for k in range(3)]):
                if count > 0:
                    ax.text(j, count, f'{int(count)}',
                           ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "error_distribution_by_class.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 5. High Confidence Errors
    high_conf_errors = error_df[error_df['confidence'] > 0.8].sort_values('confidence', ascending=False)

    if len(high_conf_errors) > 0:
        print("\n" + "=" * 80)
        print("HIGH CONFIDENCE ERRORS (Confidence > 0.8)")
        print("=" * 80)
        print(f"Total: {len(high_conf_errors)} errors")
        print("\nTop 10 most confident errors:")
        for idx, row in high_conf_errors.head(10).iterrows():
            true_cls = class_names[int(row['true_label'])]
            pred_cls = class_names[int(row['pred_label'])]
            print(f"  True: {true_cls:8s} → Predicted: {pred_cls:8s} (Confidence: {row['confidence']:.3f})")

    # 6. Summary Statistics
    summary = {
        "total_errors": int(misclassified_mask.sum()),
        "error_rate": float(misclassified_mask.sum() / len(y_test) * 100),
        "avg_confidence_correct": float(correct_conf.mean()),
        "avg_confidence_incorrect": float(incorrect_conf.mean()),
        "high_confidence_errors": int((error_df['confidence'] > 0.8).sum()),
    }

    for i, cls_name in enumerate(class_names):
        cls_mask = y_test == i
        cls_errors = (y_test[cls_mask] != y_pred[cls_mask]).sum()
        cls_total = cls_mask.sum()
        summary[f"{cls_name.lower()}_errors"] = int(cls_errors)
        summary[f"{cls_name.lower()}_error_rate"] = float(cls_errors / cls_total * 100) if cls_total > 0 else 0.0

    summary_path = output_dir / "error_analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    report_path = output_dir / "error_analysis_report.txt"
    with open(report_path, "w") as f:
        f.write(f"ERROR ANALYSIS REPORT - {model_name} (TEST SET)\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total Test Samples: {len(y_test)}\n")
        f.write(f"Total Errors: {summary['total_errors']} ({summary['error_rate']:.2f}%)\n\n")

        f.write("Per-Class Error Rates:\n")
        for cls_name in class_names:
            f.write(f"  {cls_name}: {summary[f'{cls_name.lower()}_error_rate']:.2f}% ")
            f.write(f"({summary[f'{cls_name.lower()}_errors']} errors)\n")

        f.write(f"\nAverage Confidence (Correct): {summary['avg_confidence_correct']:.3f}\n")
        f.write(f"Average Confidence (Incorrect): {summary['avg_confidence_incorrect']:.3f}\n")
        f.write(f"High Confidence Errors (>0.8): {summary['high_confidence_errors']}\n\n")

        f.write("Misclassification Patterns:\n")
        for _, row in error_patterns.iterrows():
            f.write(f"  {row['pattern']}: {row['count']} errors\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("KEY FINDINGS:\n\n")

        max_error_pattern = error_patterns.loc[error_patterns['count'].idxmax()]
        f.write(f"1. Most common error: {max_error_pattern['pattern']} ")
        f.write(f"({max_error_pattern['count']} occurrences)\n")

        error_rates = {cls: summary[f'{cls.lower()}_error_rate'] for cls in class_names}
        worst_class = max(error_rates, key=error_rates.get)
        f.write(f"2. Class with highest error rate: {worst_class} ")
        f.write(f"({error_rates[worst_class]:.2f}%)\n")

        if summary['high_confidence_errors'] > 0:
            f.write(f"3. WARNING: {summary['high_confidence_errors']} high-confidence errors detected.\n")
            f.write("   Model may be overconfident in some predictions.\n")

        fatal_mask = y_test == 2
        fatal_correct = (y_test[fatal_mask] == y_pred[fatal_mask]).sum()
        fatal_recall = fatal_correct / fatal_mask.sum() if fatal_mask.sum() > 0 else 0
        f.write(f"4. Fatal Recall: {fatal_recall:.2f} ")
        if fatal_recall < 0.7:
            f.write("(CRITICAL: Low fatal recall - high risk of missing fatal accidents)\n")
        else:
            f.write("(Acceptable)\n")

    log_action(
        step="ErrorAnalysis",
        action="Error analysis complete",
        rationale=f"Results saved to {output_dir}",
        stage="evaluation",
    )

    print("\n" + "=" * 80)
    print("ERROR ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Total Errors: {summary['total_errors']} / {len(y_test)} ({summary['error_rate']:.2f}%)")
    print(f"\nPer-Class Error Rates:")
    for cls_name in class_names:
        print(f"  {cls_name:8s}: {summary[f'{cls_name.lower()}_error_rate']:6.2f}%")
    print(f"\nMost Common Error: {max_error_pattern['pattern']} ({max_error_pattern['count']} occurrences)")
    print("=" * 80)


# ============================================================
# MAIN EVALUATION PIPELINE (TEST SET ONLY)
# ============================================================

def eval():
    """Main evaluation pipeline using test set only."""
    setup_mlflow()

    print("\n" + "=" * 80)
    print("MODEL EVALUATION ON TEST SET ONLY")
    print("=" * 80)

    # Load test data only
    print("\n1. Loading test data...")
    X_test, y_test = load_test()
    print(f"   Test samples: {len(X_test)}")

    # Load all models
    print("\n2. Loading trained models...")
    models = load_all_models()

    if not models:
        print("ERROR: No models found. Please run train.py first.")
        return

    print(f"   Loaded {len(models)} models: {', '.join(models.keys())}")

    # Evaluate all models on test set
    print("\n3. Evaluating all models on test set...")
    test_results = evaluate_all_on_test(models, X_test, y_test)

    # Print results table
    print("\n" + "-" * 80)
    print(f"{'Model':<20} {'Accuracy':>10} {'F1 Weighted':>12} {'ROC AUC':>10} {'Fatal Recall':>12} {'Safety Score':>12}")
    print("-" * 80)
    for name, metrics in test_results.items():
        print(f"{name:<20} "
              f"{metrics['test_accuracy']:>10.4f} "
              f"{metrics['test_f1_weighted']:>12.4f} "
              f"{metrics['test_roc_auc_weighted']:>10.4f} "
              f"{metrics['test_business_fatal_recall']:>12.4f} "
              f"{metrics['test_business_weighted_safety_score']:>12.2f}")
    print("-" * 80)

    # Create comparison visualizations
    print("\n4. Creating comparison visualizations...")
    viz_dir = cfg.MODELS_DIR / "evaluation_visualizations"
    create_comparison_plots(test_results, viz_dir)
    print(f"   Visualizations saved to: {viz_dir}")

    # Select best model from test results
    best_model_info_path = cfg.MODELS_DIR / "best_model_info.json"
    if best_model_info_path.exists():
        with open(best_model_info_path, "r") as f:
            best_model_info = json.load(f)
        best_model_name = best_model_info["model_name"]
        print(f"\n5. Best Model (from train.py): {best_model_name}")
        
        # Safety check: if that model failed to load, fall back to test metrics
        if best_model_name not in models:
            print(f"   WARNING: {best_model_name} not loaded. Falling back to test-set F1-weighted.")
            best_model_name = max(
                test_results,
                key=lambda n: test_results[n].get("test_f1_weighted", 0.0)
            )
            print(f"   Fallback Best Model: {best_model_name}")
    else:
        best_model_name = max(
            test_results,
            key=lambda n: test_results[n].get("test_f1_weighted", 0.0)
        )
        print(f"\n5. Best Model (by test F1-weighted): {best_model_name}")

    print(f"\n5. Best Model (by test F1-weighted): {best_model_name}")

    # Get predictions for best model
    best_model = models[best_model_name]
    if best_model_name == "NeuralNetwork":
        device = next(best_model.parameters()).device
        X_t = torch.FloatTensor(np.array(X_test, copy=True)).to(device)
        with torch.no_grad():
            proba = torch.softmax(best_model(X_t), dim=1).cpu().numpy()
        y_pred = proba.argmax(axis=1)
    else:
        y_pred = best_model.predict(X_test)
        proba = best_model.predict_proba(X_test)

    # Save best model report
    save_best_model_report(best_model_name, test_results[best_model_name], y_test, y_pred)

    # Print best model results
    print("\nBest Model Test Results:")
    print(f"  Accuracy: {test_results[best_model_name]['test_accuracy']:.4f}")
    print(f"  F1 Weighted: {test_results[best_model_name]['test_f1_weighted']:.4f}")
    print(f"  ROC AUC: {test_results[best_model_name]['test_roc_auc_weighted']:.4f}")
    print(f"  Fatal Recall: {test_results[best_model_name]['test_business_fatal_recall']:.4f}")
    print(f"  Safety Score: {test_results[best_model_name]['test_business_weighted_safety_score']:.2f}")

    # Perform error analysis on best model
    print(f"\n6. Performing error analysis on {best_model_name}...")
    error_dir = cfg.MODELS_DIR / "error_analysis"
    perform_error_analysis(
        best_model, best_model_name,
        X_test, y_test, y_pred, proba,
        error_dir
    )
    print(f"   Error analysis saved to: {error_dir}")

    # Save final summary
    save_stage_report("evaluation")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nResults Summary:")
    print(f"  - Model comparisons: {viz_dir}")
    print(f"  - Error analysis: {error_dir}")
    print(f"  - Test classification report: {cfg.MODELS_DIR / 'test_classification_report.txt'}")
    print(f"  - Test metrics JSON: {cfg.MODELS_DIR / 'test_metrics_summary.json'}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    setup_logging()
    eval()