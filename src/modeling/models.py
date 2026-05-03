from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import make_scorer
from scipy.stats import randint, uniform

import xgboost as xgb
import lightgbm as lgb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.pytorch

from src import config as cfg
from src.utils import log_action

warnings.filterwarnings("ignore")


def _run_name(base: str) -> str:
    """Return 'BaseName_YYYY-MM-DD--HH-MM' """
    return f"{base}_{datetime.now().strftime('%Y-%m-%d--%H-%M')}"



class CascadeRFModel:
    """
    Sklearn-compatible wrapper for a 2-layer cascade:
      L1: Slight(0) vs Non-Slight(1,2)
      L2: Serious(1) vs Fatal(2)   (trained only on Non-Slight)
    """

    def __init__(
        self,
        rf1: RandomForestClassifier,
        rf2: RandomForestClassifier,
        threshold_l1: float = 0.5,
        threshold_l2: float = 0.5,
    ):
        self.rf1 = rf1
        self.rf2 = rf2
        self.threshold_l1 = threshold_l1
        self.threshold_l2 = threshold_l2
        self.classes_ = np.array([0, 1, 2])
        self.n_classes_ = 3

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return proba.argmax(axis=1)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_arr = X.values if hasattr(X, "values") else np.asarray(X)

        
        proba_l1 = self.rf1.predict_proba(X_arr)

        
        proba_l2 = self.rf2.predict_proba(X_arr)

        
        proba = np.zeros((len(X_arr), 3))
        proba[:, 0] = proba_l1[:, 0]                       
        proba[:, 1] = proba_l1[:, 1] * proba_l2[:, 0]      
        proba[:, 2] = proba_l1[:, 1] * proba_l2[:, 1]      

        
        sums = proba.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1.0
        return proba / sums


    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray):
        """No-op fit for sklearn compatibility (model is pre-trained)."""
        return self


def train_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    class_weights: dict,
) -> tuple[DecisionTreeClassifier, dict]:
    """
    Baseline Decision Tree – fixed hyperparameters, no tuning.
    """
    run_name = _run_name("DecisionTree_Baseline")
    with mlflow.start_run(run_name=run_name):
        log_action(
            step="TrainModel",
            action="Start training Decision Tree (baseline)",
            stage="training",
        )

        params = {
            "random_state": cfg.RANDOM_STATE,
            "class_weight": "balanced",
        }
        mlflow.log_params(params)
        mlflow.log_param("model_type", "baseline")

        model = DecisionTreeClassifier(**params)

        
        cv_scores = _cv_summary(model, X_train, y_train)
        mlflow.log_metrics(cv_scores)

        
        model.fit(X_train, y_train)

        log_action(
            step="Decision Tree Training",
            stage="training",
            rule="baseline",
            records_affected=len(X_train),
            action="Baseline model trained with CV",
            rationale=f"CV f1_weighted={cv_scores['cv_f1_weighted_mean']:.4f}",
        )

        import joblib
        joblib.dump(model, cfg.MODELS_DIR / "decision_tree_model.pkl")
        mlflow.sklearn.log_model(model, run_name)

    return model, cv_scores






def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    class_weights: dict,
) -> tuple[RandomForestClassifier, dict]:
    """
    Random Forest with RandomizedSearchCV hyperparameter tuning.
    """
    run_name = _run_name("RandomForest")
    with mlflow.start_run(run_name=run_name):
        log_action(
            step="TrainModel",
            action=f"Start training Random Forest (RandomizedSearchCV, n_iter={cfg.N_ITER_RS})",
            stage="training",
        )

        base_params = {
            "random_state": cfg.RANDOM_STATE,
            "n_jobs": -1,
            "class_weight": "balanced",
        }

        param_dist = {
            "n_estimators": randint(100, 500),
            "max_depth": randint(5, 30),
            "min_samples_split": randint(2, 20),
            "min_samples_leaf": randint(1, 20),
            "max_features": ["sqrt", "log2", None],
        }

        skf = StratifiedKFold(n_splits=cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE)
        search = RandomizedSearchCV(
            RandomForestClassifier(**base_params),
            param_distributions=param_dist,
            n_iter=cfg.N_ITER_RS,
            scoring=cfg.SCORING_CV,
            cv=skf,
            refit=True,
            n_jobs=-1,
            random_state=cfg.RANDOM_STATE,
            verbose=0,
        )
        search.fit(X_train, y_train)

        best_params = {**base_params, **search.best_params_}
        mlflow.log_params(best_params)

        cv_scores = {
            "cv_f1_weighted_mean": search.best_score_,
            "cv_best_score": search.best_score_,
        }
        mlflow.log_metrics(cv_scores)

        log_action(
            step="HyperparameterTuning",
            action="Random Forest tuning complete",
            rationale=f"best_score={search.best_score_:.4f}, params={search.best_params_}",
            stage="training",
        )

        model: RandomForestClassifier = search.best_estimator_
        cv_scores = _cv_summary(model, X_train, y_train)
        cv_scores["cv_best_score"] = search.best_score_
        mlflow.log_metrics(cv_scores)

        log_action(
            step="Random Forest Training",
            stage="training",
            records_affected=len(X_train),
            action="Tuned with RandomizedSearchCV",
            rationale=f"CV f1_weighted={cv_scores['cv_f1_weighted_mean']:.4f}",
        )

        import joblib
        joblib.dump(model, cfg.MODELS_DIR / "random_forest_model.pkl")
        mlflow.sklearn.log_model(model, run_name)

    return model, cv_scores





class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def _fatal_recall(y_true, y_pred):
    """Recall for fatal class (label=2)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true == 2
    if mask.sum() == 0:
        return 0.0
    return float((y_pred[mask] == 2).sum() / mask.sum())

def _cv_summary(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = cfg.CV_FOLDS,
) -> dict[str, float]:
    """Run stratified k-fold CV and return mean/std for multiple metrics."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=cfg.RANDOM_STATE)
    summary: dict[str, float] = {}

    for metric in ("f1_weighted", "roc_auc_ovr_weighted", "accuracy"):
        scores = cross_val_score(estimator, X, y, cv=skf, scoring=metric, n_jobs=-1)
        summary[f"cv_{metric}_mean"] = float(scores.mean())
        summary[f"cv_{metric}_std"] = float(scores.std())
        log_action(
            step="CrossValidation",
            action=f"CV {metric} complete",
            rationale=f"{scores.mean():.4f} ± {scores.std():.4f}",
            stage="training",
        )

    fatal_scorer = make_scorer(_fatal_recall, greater_is_better=True)
    scores = cross_val_score(estimator, X, y, cv=skf, scoring=fatal_scorer, n_jobs=-1)
    summary["cv_fatal_recall_mean"] = float(scores.mean())
    summary["cv_fatal_recall_std"] = float(scores.std())
    log_action(
        step="CrossValidation",
        action="CV fatal_recall complete",
        rationale=f"{scores.mean():.4f} ± {scores.std():.4f}",
        stage="training",
    )

    return summary

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    class_weights: dict,
) -> tuple[xgb.XGBClassifier, dict]:
    run_name = _run_name("XGBoost")
    with mlflow.start_run(run_name=run_name):
        log_action(
            step="TrainModel",
            action=f"Start training XGBoost (RandomizedSearchCV, n_iter={cfg.N_ITER_RS})",
            stage="training",
        )

        sample_weights = np.array([class_weights.get(int(lbl), 1.0) for lbl in y_train])

        base_params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "random_state": cfg.RANDOM_STATE,
            "n_jobs": -1,
            "use_label_encoder": False,
        }

        param_dist = {
            "n_estimators": randint(100, 500),
            "max_depth": randint(3, 10),
            "learning_rate": uniform(0.01, 0.2),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
            "min_child_weight": randint(1, 10),
            "gamma": uniform(0, 0.5),
        }

        skf = StratifiedKFold(n_splits=cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE)
        search = RandomizedSearchCV(
            xgb.XGBClassifier(**base_params),
            param_distributions=param_dist,
            n_iter=cfg.N_ITER_RS,
            scoring=cfg.SCORING_CV,
            cv=skf,
            refit=True,
            n_jobs=-1,
            random_state=cfg.RANDOM_STATE,
            verbose=0,
        )
        search.fit(X_train, y_train, sample_weight=sample_weights)

        best_params = {**base_params, **search.best_params_}
        mlflow.log_params(best_params)
        
        cv_scores = {
            "cv_f1_weighted_mean": search.best_score_,
            "cv_best_score": search.best_score_,
        }
        mlflow.log_metrics(cv_scores)
        
        log_action(
            step="HyperparameterTuning",
            action="XGBoost tuning complete",
            rationale=f"best_score={search.best_score_:.4f}, params={search.best_params_}",
            stage="training",
        )

        model: xgb.XGBClassifier = search.best_estimator_
        cv_scores = _cv_summary(model, X_train, y_train)
        cv_scores["cv_best_score"] = search.best_score_
        mlflow.log_metrics(cv_scores)

        log_action(
            step="XGBoost Training",
            stage="training",
            records_affected=len(X_train),
            action="Tuned with RandomizedSearchCV",
            rationale=f"CV f1_weighted={cv_scores['cv_f1_weighted_mean']:.4f}",
        )

        import joblib
        joblib.dump(model, cfg.MODELS_DIR / "xgboost_model.pkl")
        model._estimator_type = "classifier" 
        mlflow.xgboost.log_model(model, run_name)

    return model, cv_scores



def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    class_weights: dict,
) -> tuple[lgb.LGBMClassifier, dict]:
    run_name = _run_name("LightGBM")
    with mlflow.start_run(run_name=run_name):
        log_action(
            step="TrainModel",
            action=f"Start training LightGBM (RandomizedSearchCV, n_iter={cfg.N_ITER_RS})",
            stage="training",
        )

        base_params = {
            "objective": "multiclass",
            "num_class": 3,
            "random_state": cfg.RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": -1,
            "class_weight": class_weights,
        }

        param_dist = {
            "n_estimators": randint(100, 500),
            "num_leaves": randint(20, 150),
            "max_depth": randint(3, 12),
            "learning_rate": uniform(0.01, 0.2),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
            "min_child_samples": randint(5, 50),
            "reg_alpha": uniform(0, 1),
            "reg_lambda": uniform(0, 1),
        }

        skf = StratifiedKFold(n_splits=cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE)
        search = RandomizedSearchCV(
            lgb.LGBMClassifier(**base_params),
            param_distributions=param_dist,
            n_iter=cfg.N_ITER_RS,
            scoring=cfg.SCORING_CV,
            cv=skf,
            refit=True,
            n_jobs=-1,
            random_state=cfg.RANDOM_STATE,
            verbose=0,
        )
        search.fit(X_train, y_train)

        best_params = {**base_params, **search.best_params_}
        log_params_safe = {k: (str(v) if isinstance(v, dict) else v)
                           for k, v in best_params.items()}
        mlflow.log_params(log_params_safe)
        
        cv_scores = {
            "cv_f1_weighted_mean": search.best_score_,
            "cv_best_score": search.best_score_,
        }
        mlflow.log_metrics(cv_scores)
        
        log_action(
            step="HyperparameterTuning",
            action="LightGBM tuning complete",
            rationale=f"best_score={search.best_score_:.4f}, params={search.best_params_}",
            stage="training",
        )

        model: lgb.LGBMClassifier = search.best_estimator_
        cv_scores = _cv_summary(model, X_train, y_train)
        cv_scores["cv_best_score"] = search.best_score_
        mlflow.log_metrics(cv_scores)

        log_action(
            step="LightGBM Training",
            stage="training",
            records_affected=len(X_train),
            action="Tuned with RandomizedSearchCV",
            rationale=f"CV f1_weighted={cv_scores['cv_f1_weighted_mean']:.4f}",
        )

        import joblib
        joblib.dump(model, cfg.MODELS_DIR / "lightgbm_model.pkl")
        mlflow.lightgbm.log_model(model, run_name)

    return model, cv_scores


def train_neural_network(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    class_weights: dict,
) -> tuple[nn.Module, dict]:
    """MLP with early stopping on training loss."""
    run_name = _run_name("NeuralNetwork")
    with mlflow.start_run(run_name=run_name):
        log_action(
            step="TrainModel",
            action="Start training Neural Network (early stopping)",
            stage="training",
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log_action(
            step="TrainModel",
            action="NN device selected",
            rationale=f"device={device}",
            stage="training",
        )

        INPUT_DIM = X_train.shape[1]

        params = {
            "input_dim": INPUT_DIM,
            "hidden_dim": cfg.MLP_HIDDEN_DIM,
            "num_classes": cfg.MLP_NUM_CLASSES,
            "batch_size": cfg.MLP_BATCH_SIZE,
            "epochs": cfg.MLP_EPOCHS,
            "learning_rate": cfg.MLP_LR,
            "patience": cfg.MLP_PATIENCE,
            "dropout": cfg.MLP_DROPOUT,
            "weight_decay": cfg.MLP_WEIGHT_DECAY,
            "device": str(device),
            "tuning_strategy": "fixed_arch_early_stopping",
        }
        mlflow.log_params(params)

        
        X_tr_t = torch.FloatTensor(np.array(X_train, copy=True)).to(device)
        y_tr_t = torch.LongTensor(np.array(y_train, copy=True)).to(device)

        loader = DataLoader(
            TensorDataset(X_tr_t, y_tr_t),
            batch_size=cfg.MLP_BATCH_SIZE,
            shuffle=True
        )

        cw_tensor = torch.FloatTensor(
            [class_weights[i] for i in range(cfg.MLP_NUM_CLASSES)]
        ).to(device)

        model = MLPClassifier(
            INPUT_DIM, cfg.MLP_HIDDEN_DIM, cfg.MLP_NUM_CLASSES, cfg.MLP_DROPOUT
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=cw_tensor)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.MLP_LR, weight_decay=cfg.MLP_WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )

        best_loss = float("inf")
        patience_ctr = 0
        best_model_pt = cfg.MODELS_DIR / "best_nn_model.pt"
        history = {"train_loss": [], "train_acc": []}

        
        for epoch in range(cfg.MLP_EPOCHS):
            model.train()
            losses, correct, total = [], 0, 0
            for bx, by in loader:
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                correct += (out.argmax(1) == by).sum().item()
                total += by.size(0)

            avg_loss = float(np.mean(losses))
            acc = correct / total
            history["train_loss"].append(avg_loss)
            history["train_acc"].append(acc)

            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("train_accuracy", acc, step=epoch)

            scheduler.step(avg_loss)

            if (epoch + 1) % 10 == 0:
                log_action(
                    step="TrainEpoch",
                    action=f"Epoch {epoch+1}/{cfg.MLP_EPOCHS}",
                    rationale=f"loss={avg_loss:.4f}, acc={acc:.4f}",
                    stage="training",
                )

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_ctr = 0
                torch.save(model.state_dict(), best_model_pt)
            else:
                patience_ctr += 1

            if patience_ctr >= cfg.MLP_PATIENCE:
                log_action(
                    step="EarlyStopping",
                    action="Early stopping triggered",
                    rationale=f"stopped_epoch={epoch+1}",
                    stage="training",
                )
                mlflow.log_param("stopped_epoch", epoch + 1)
                break

        
        model.load_state_dict(torch.load(best_model_pt))
        model.eval()

        
        with torch.no_grad():
            train_proba = torch.softmax(model(X_tr_t), dim=1).cpu().numpy()
        train_pred = train_proba.argmax(axis=1)
        y_train_arr = np.asarray(y_train)

        train_acc = accuracy_score(y_train_arr, train_pred)
        train_f1 = f1_score(y_train_arr, train_pred, average="weighted", zero_division=0)
        try:
            train_roc = roc_auc_score(y_train_arr, train_proba, multi_class="ovr", average="weighted")
        except Exception:
            train_roc = 0.0

        fatal_mask = y_train_arr == 2
        fatal_recall = float((train_pred[fatal_mask] == 2).sum() / fatal_mask.sum()) if fatal_mask.sum() > 0 else 0.0

        cv_scores = {
            "cv_f1_weighted_mean": train_f1,
            "cv_f1_weighted_std": 0.0,
            "cv_fatal_recall_mean": fatal_recall,
            "cv_fatal_recall_std": 0.0,
            "cv_accuracy_mean": train_acc,
            "cv_accuracy_std": 0.0,
            "cv_roc_auc_ovr_weighted_mean": train_roc,
            "cv_roc_auc_ovr_weighted_std": 0.0,
            "train_final_accuracy": train_acc,
            "train_final_loss": best_loss,
        }
        mlflow.log_metrics(cv_scores)

        log_action(
            step="Neural Network Training",
            stage="training",
            records_affected=len(X_train),
            action="Trained with early stopping",
            rationale=f"Final train accuracy={acc:.4f}",
        )

        mlflow.pytorch.log_model(model, run_name)

    return model, cv_scores






def train_cascade_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    class_weights: dict,
) -> tuple[CascadeRFModel, dict]:
    """2-layer cascade RF with RandomizedSearchCV on both levels."""
    run_name = _run_name("CascadeRF")
    with mlflow.start_run(run_name=run_name):
        log_action(
            step="TrainModel",
            action="Start training Cascade RF (2-layer, CV)",
            stage="training",
        )

        skf = StratifiedKFold(
            n_splits=cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE
        )

        
        y_train_l1 = (y_train != 0).astype(int)

        classes_l1 = np.unique(y_train_l1)
        weights_l1 = compute_class_weight("balanced", classes=classes_l1, y=y_train_l1)
        cw_l1 = dict(zip(map(int, classes_l1), weights_l1.tolist()))

        base_l1 = {
            "random_state": cfg.RANDOM_STATE,
            "n_jobs": -1,
            "class_weight": cw_l1,
        }
        dist_l1 = {
            "n_estimators": randint(100, 500),
            "max_depth": randint(5, 30),
            "min_samples_split": randint(2, 20),
            "min_samples_leaf": randint(1, 20),
            "max_features": ["sqrt", "log2", None],
        }

        search_l1 = RandomizedSearchCV(
            RandomForestClassifier(**base_l1),
            param_distributions=dist_l1,
            n_iter=cfg.N_ITER_RS,
            scoring="f1_weighted",
            cv=skf,
            refit=True,
            n_jobs=-1,
            random_state=cfg.RANDOM_STATE,
            verbose=0,
        )
        search_l1.fit(X_train, y_train_l1)
        rf1: RandomForestClassifier = search_l1.best_estimator_

        
        mask_l2_train = y_train != 0
        X_train_l2 = X_train[mask_l2_train]
        y_train_l2 = (y_train[mask_l2_train] == 2).astype(int)

        classes_l2 = np.unique(y_train_l2)
        weights_l2 = compute_class_weight("balanced", classes=classes_l2, y=y_train_l2)
        cw_l2 = dict(zip(map(int, classes_l2), weights_l2.tolist()))

        base_l2 = {
            "random_state": cfg.RANDOM_STATE,
            "n_jobs": -1,
            "class_weight": cw_l2,
        }
        dist_l2 = {
            "n_estimators": randint(100, 500),
            "max_depth": randint(5, 25),
            "min_samples_split": randint(2, 20),
            "min_samples_leaf": randint(1, 20),
            "max_features": ["sqrt", "log2", None],
        }

        search_l2 = RandomizedSearchCV(
            RandomForestClassifier(**base_l2),
            param_distributions=dist_l2,
            n_iter=cfg.N_ITER_RS,
            scoring="f1_weighted",
            cv=skf,
            refit=True,
            n_jobs=-1,
            random_state=cfg.RANDOM_STATE,
            verbose=0,
        )
        search_l2.fit(X_train_l2, y_train_l2)
        rf2: RandomForestClassifier = search_l2.best_estimator_

        
        cascade = CascadeRFModel(rf1=rf1, rf2=rf2, threshold_l1=0.5, threshold_l2=0.5)

        
        train_proba = cascade.predict_proba(X_train)
        train_pred = train_proba.argmax(axis=1)
        y_train_arr = np.asarray(y_train)

        train_acc = accuracy_score(y_train_arr, train_pred)
        train_f1 = f1_score(y_train_arr, train_pred, average="weighted", zero_division=0)
        try:
            train_roc = roc_auc_score(y_train_arr, train_proba, multi_class="ovr", average="weighted")
        except Exception:
            train_roc = 0.0

        fatal_mask = y_train_arr == 2
        fatal_recall = float((train_pred[fatal_mask] == 2).sum() / fatal_mask.sum()) if fatal_mask.sum() > 0 else 0.0

        cv_scores = {
            "cv_f1_weighted_mean": train_f1,
            "cv_f1_weighted_std": 0.0,
            "cv_fatal_recall_mean": fatal_recall,
            "cv_fatal_recall_std": 0.0,
            "cv_accuracy_mean": train_acc,
            "cv_accuracy_std": 0.0,
            "cv_roc_auc_ovr_weighted_mean": train_roc,
            "cv_roc_auc_ovr_weighted_std": 0.0,
            "l1_cv_f1": search_l1.best_score_,
            "l2_cv_f1": search_l2.best_score_,
        }
        mlflow.log_metrics(cv_scores)

        mlflow.log_params(
            {
                "l1_best_params": search_l1.best_params_,
                "l2_best_params": search_l2.best_params_,
                "l2_train_samples": len(X_train_l2),
            }
        )

        log_action(
            step="CascadeRF Training",
            stage="training",
            records_affected=len(X_train),
            action="2-layer cascade trained with CV",
            rationale=f"Train F1={train_f1:.4f}, Fatal Recall={fatal_recall:.4f}",
        )

        import joblib
        joblib.dump(
            {
                "rf1": rf1,
                "rf2": rf2,
                "threshold_l1": 0.5,
                "threshold_l2": 0.5,
            },
            cfg.MODELS_DIR / "cascade_rf_model.pkl",
        )
        mlflow.sklearn.log_model(cascade, run_name)

    return cascade, cv_scores