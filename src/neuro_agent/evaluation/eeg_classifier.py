"""Classical EEG movement-state baseline models and evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

from neuro_agent.data.eeg_baseline import (
    EegSplit,
    LABEL_DISPLAY,
    MOVEMENT_LABELS,
    load_all_splits,
    verify_subject_splits,
)
from neuro_agent.paths import RESULTS_DIR

RANDOM_SEED = 42


@dataclass
class ModelResult:
    """Evaluation results for one model on one split."""

    model_name: str
    split: str
    accuracy: float
    balanced_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class: dict[str, dict[str, float]]
    confusion_matrix: list[list[int]]
    confusion_labels: list[str]
    n_samples: int


@dataclass
class BaselineRunSummary:
    """Full baseline run summary."""

    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    seed: int = RANDOM_SEED
    n_features: int = 0
    feature_names: list[str] = field(default_factory=list)
    label_mapping: dict[str, str] = field(default_factory=lambda: dict(LABEL_DISPLAY))
    subject_split_check: dict[str, Any] = field(default_factory=dict)
    models_trained: list[str] = field(default_factory=list)
    validation_best_model: str = ""
    test_best_model: str = ""
    validation_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    test_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _build_models() -> dict[str, Any]:
    models: dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced_subsample",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
    }
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            objective="multi:softprob",
            num_class=len(MOVEMENT_LABELS),
            eval_metric="mlogloss",
            random_state=RANDOM_SEED,
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            n_jobs=-1,
            verbosity=0,
        )
    except ImportError:
        pass
    return models


def _encode_labels(y: np.ndarray, encoder: LabelEncoder) -> np.ndarray:
    return encoder.transform(y)


def _evaluate(
    model_name: str,
    model: Any,
    X: np.ndarray,
    y_true_enc: np.ndarray,
    split: str,
    label_encoder: LabelEncoder,
) -> tuple[ModelResult, np.ndarray]:
    y_pred_enc = model.predict(X)
    class_labels = label_encoder.classes_
    display_labels = [LABEL_DISPLAY.get(lbl, lbl) for lbl in class_labels]

    report = classification_report(
        y_true_enc,
        y_pred_enc,
        labels=np.arange(len(class_labels)),
        target_names=display_labels,
        output_dict=True,
        zero_division=0,
    )

    per_class = {
        display_labels[i]: {
            "precision": report[display_labels[i]]["precision"],
            "recall": report[display_labels[i]]["recall"],
            "f1": report[display_labels[i]]["f1-score"],
            "support": int(report[display_labels[i]]["support"]),
        }
        for i in range(len(display_labels))
    }

    cm = confusion_matrix(y_true_enc, y_pred_enc, labels=np.arange(len(class_labels)))
    result = ModelResult(
        model_name=model_name,
        split=split,
        accuracy=float(accuracy_score(y_true_enc, y_pred_enc)),
        balanced_accuracy=float(balanced_accuracy_score(y_true_enc, y_pred_enc)),
        macro_precision=float(
            precision_score(y_true_enc, y_pred_enc, average="macro", zero_division=0)
        ),
        macro_recall=float(
            recall_score(y_true_enc, y_pred_enc, average="macro", zero_division=0)
        ),
        macro_f1=float(f1_score(y_true_enc, y_pred_enc, average="macro", zero_division=0)),
        per_class=per_class,
        confusion_matrix=cm.tolist(),
        confusion_labels=display_labels,
        n_samples=len(y_true_enc),
    )
    return result, y_pred_enc


def run_baseline_classifier(output_dir: Path | None = None) -> BaselineRunSummary:
    """Train and evaluate classical EEG baselines."""
    out_dir = output_dir or (RESULTS_DIR / "baseline_classifier")
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = load_all_splits()
    subject_check = verify_subject_splits(splits)

    train, val, test = splits["train"], splits["validation"], splits["test"]

    label_encoder = LabelEncoder()
    label_encoder.fit(np.array(MOVEMENT_LABELS, dtype=object))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train.X)
    X_val = scaler.transform(val.X)
    X_test = scaler.transform(test.X)

    y_train = _encode_labels(train.y, label_encoder)
    y_val = _encode_labels(val.y, label_encoder)
    y_test = _encode_labels(test.y, label_encoder)

    models = _build_models()
    all_metrics: dict[str, dict[str, dict[str, Any]]] = {}
    val_scores: dict[str, float] = {}
    test_predictions: dict[str, pd.DataFrame] = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)

        model_metrics: dict[str, dict[str, Any]] = {}
        for split_name, X_split, y_split, split_obj in [
            ("validation", X_val, y_val, val),
            ("test", X_test, y_test, test),
        ]:
            result, y_pred = _evaluate(model_name, model, X_split, y_split, split_name, label_encoder)
            model_metrics[split_name] = asdict(result)
            if split_name == "validation":
                val_scores[model_name] = result.balanced_accuracy

            cm_df = pd.DataFrame(
                result.confusion_matrix,
                index=[f"true_{l}" for l in result.confusion_labels],
                columns=[f"pred_{l}" for l in result.confusion_labels],
            )
            cm_df.to_csv(out_dir / f"confusion_matrix_{model_name}_{split_name}.csv")

            if split_name == "test":
                pred_df = split_obj.metadata.copy()
                pred_df["true_label"] = split_obj.y
                pred_df["pred_label"] = label_encoder.inverse_transform(y_pred)
                pred_df["pred_label_display"] = pred_df["pred_label"].map(LABEL_DISPLAY)
                pred_df["true_label_display"] = pred_df["true_label"].map(LABEL_DISPLAY)
                pred_df["correct"] = pred_df["true_label"] == pred_df["pred_label"]
                pred_df["model"] = model_name
                test_predictions[model_name] = pred_df

        all_metrics[model_name] = model_metrics

    best_val = max(val_scores, key=val_scores.get)
    best_test = max(
        models.keys(),
        key=lambda m: all_metrics[m]["test"]["balanced_accuracy"],
    )

    # Save test predictions for best validation model
    best_pred_path = out_dir / "predictions_test.csv"
    test_predictions[best_val].to_csv(best_pred_path, index=False)

    summary = BaselineRunSummary(
        n_features=train.X.shape[1],
        feature_names=train.feature_names.tolist(),
        subject_split_check=subject_check,
        models_trained=list(models.keys()),
        validation_best_model=best_val,
        test_best_model=best_test,
        validation_metrics={
            m: {
                "accuracy": all_metrics[m]["validation"]["accuracy"],
                "balanced_accuracy": all_metrics[m]["validation"]["balanced_accuracy"],
                "macro_f1": all_metrics[m]["validation"]["macro_f1"],
            }
            for m in models
        },
        test_metrics={
            m: {
                "accuracy": all_metrics[m]["test"]["accuracy"],
                "balanced_accuracy": all_metrics[m]["test"]["balanced_accuracy"],
                "macro_f1": all_metrics[m]["test"]["macro_f1"],
            }
            for m in models
        },
        notes=[
            "Features standardized with StandardScaler fit on train only.",
            "Logistic regression uses class_weight='balanced'.",
            "Random forest uses class_weight='balanced_subsample'.",
            "No oversampling applied to validation or test.",
        ],
    )
    if "xgboost" not in models:
        summary.notes.append("XGBoost skipped (package not installed).")

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "timestamp": summary.timestamp,
                "seed": summary.seed,
                "n_features": summary.n_features,
                "feature_names": summary.feature_names,
                "label_mapping": summary.label_mapping,
                "subject_split_check": summary.subject_split_check,
                "models": all_metrics,
            },
            indent=2,
        )
    )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(asdict(summary), indent=2))

    return summary
