
"""
training_pipeline.py

Company Bankruptcy Prediction - Lab 5 end-to-end pipeline.

Implements, in order:
- EDA (imbalance, correlations, boxplots, skewness, inf check)
- Preprocessing (stratified split, winsorizing outliers, PSI snapshot)
- Feature Selection: correlation filter for all; VIF only for LR
- Hyperparameter Tuning : RandomizedSearchCV on PR-AUC
- Training (OOF CV + final refit + save)
- Evaluation (ROC-AUC, PR-AUC, Brier, F1, F2, Recall@P80; calibration/PR curves)
- SHAP/coefficients for the best model (positive class; safe fallback)
- PSI on best model's features
- Markdown report with jot notes per component
- Auto-organize artifacts into subfolders

Reproducibility:
- Fixed SEED=42 for all randomized steps (NumPy + model random_state)
- All outputs saved under ./artifacts
- Minimal external state; dataset path auto-detected

Dependencies (documented only, not installed by the script):
- python >= 3.11
- numpy >= 1.26
- pandas >= 2.2
- matplotlib >= 3.8
- seaborn >= 0.13
- scikit-learn >= 1.5
- xgboost >= 2.0
- shap >= 0.46
- joblib >= 1.4
- statsmodels >= 0.14   (for VIF)
- scipy >= 1.13         (for winsorize)
"""

from __future__ import annotations

import json
import shutil
import warnings
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy.stats.mstats import winsorize
from sklearn.base import clone as sk_clone
from sklearn.calibration import CalibrationDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBClassifier

# Config / Globals

SEED: int = 42
np.random.seed(SEED)
warnings.filterwarnings("ignore", category=FutureWarning, module="shap")

ARTIFACTS: Path = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)

TARGET_COL: str = "Bankrupt?"

# Threshold/constants
PR_THRESHOLD: float = 0.5
F2_BETA: float = 2.0
PSI_SMALL: float = 0.10
PSI_LARGE: float = 0.25
MAX_VIF_ITERS: int = 60
MIN_FEATURES_AFTER_VIF: int = 2


# Small utilities

def _savefig(path: Path) -> None:
    """Save a matplotlib figure to disk with tight layout."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _percent(x: float) -> str:
    """Format a float as percentage with two decimals."""
    return f"{100.0 * x:.2f}%"


def _df_to_markdown_safe(df: pd.DataFrame, index: bool = False) -> str:
    """
    Convert DataFrame to Markdown, falling back to plain text when 'tabulate'
    isn't installed (pandas.to_markdown requires it).
    """
    try:
        return df.to_markdown(index=index)
    except Exception:
        return "```\n" + df.to_string(index=index) + "\n```"


def find_dataset() -> Path:
    """Find the dataset CSV in common locations."""
    candidates = [
        Path("data") / "data.csv",
        Path("company_bankruptcy_prediction.csv"),
        Path("bankruptcy.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Dataset not found. Place CSV at ./data/data.csv "
    )


# 1) EDA

def perform_eda(df: pd.DataFrame) -> None:
    """Perform EDA and save artifacts: imbalance, corr heatmap, boxplots, skewness."""
    # Class imbalance
    counts = df[TARGET_COL].value_counts().sort_index()
    ratios = counts / counts.sum()
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.bar(counts.index.astype(str), counts.values)
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts.values) * 0.02, _percent(ratios.iloc[i]), ha="center")
    ax.set_title("Class Imbalance - Overall")
    ax.set_xlabel("Class (0 = Non-bankrupt, 1 = Bankrupt)")
    ax.set_ylabel("Count")
    _savefig(ARTIFACTS / "eda_class_imbalance.png")

    # Correlation heatmap (top 20 most target-correlated)
    num_df = df.select_dtypes(include=[np.number])
    top_feats = (
        num_df.corr(numeric_only=True)[TARGET_COL]
        .abs()
        .sort_values(ascending=False)
        .index[1:21]  # exclude target
    )
    plt.figure(figsize=(10, 8))
    sns.heatmap(num_df[top_feats].corr(numeric_only=True), cmap="coolwarm", center=0)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.title("Correlation Heatmap - Top 20 vs. Target", fontsize=12)
    plt.tight_layout()
    _savefig(ARTIFACTS / "eda_corr_heatmap.png")

    # Missing values
    miss = df.isnull().sum()
    missing = miss[miss > 0].sort_values(ascending=False)
    if not missing.empty:
        missing.to_csv(ARTIFACTS / "eda_missing_values.csv")
    else:
        (ARTIFACTS / "eda_missing_values.txt").write_text(
            "No missing values.\n", "utf-8"
        )

    # Inf check
    inf_counts = df.select_dtypes(include=[np.number]).apply(
        lambda x: np.isinf(x).sum()
    )
    inf_counts = inf_counts[inf_counts > 0].sort_values(ascending=False)
    if not inf_counts.empty:
        inf_counts.to_csv(ARTIFACTS / "eda_inf_counts.csv")
    else:
        (ARTIFACTS / "eda_inf_counts.txt").write_text("No inf values.\n", "utf-8")

    # Sample boxplots (first 20 features)
    sample_cols = num_df.drop(columns=[TARGET_COL]).columns[:20]
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[sample_cols], orient="h", fliersize=1)
    plt.title("Boxplots - Sample of Numeric Features")
    _savefig(ARTIFACTS / "eda_boxplots_sample.png")

    # Skewness analysis
    skewness = (
        num_df.drop(columns=[TARGET_COL], errors="ignore")
        .skew(numeric_only=True)
        .sort_values(ascending=False)
    )
    pd.DataFrame({"Skewness": skewness}).to_csv(ARTIFACTS / "eda_skewness.csv")
    plt.figure(figsize=(8, 5))
    skewness.head(10).plot(kind="barh")
    plt.title("Top 10 Skewed Features")
    plt.xlabel("Skewness")
    _savefig(ARTIFACTS / "eda_skewness_bar.png")



# 2) Preprocessing (split + PSI snapshot)

def stratified_split(
    df: pd.DataFrame, test_size: float = 0.25
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a stratified train/test split, winsorize outliers, and save class ratio
    plots for both splits.
    """
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df[TARGET_COL], random_state=SEED
    )
    # Winsorize numeric features at 5% to handle outliers
    for col in train_df.select_dtypes(include=[np.number]).columns:
        if col != TARGET_COL:
            train_df[col] = winsorize(train_df[col], limits=[0.05, 0.05])
            test_df[col] = winsorize(test_df[col], limits=[0.05, 0.05])

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, part, title in zip(axes, [train_df, test_df], ["Train", "Test"], strict=False):
        c = part[TARGET_COL].value_counts().sort_index()
        ax.bar(c.index.astype(str), c.values)
        for i, v in enumerate(c.values):
            ax.text(i, v + max(c.values) * 0.02, _percent(v / c.sum()), ha="center")
        ax.set_title(f"{title} Class Distribution")
        ax.set_xlabel("Class")
    fig.suptitle("Stratified Split Check - Train vs Test")
    _savefig(ARTIFACTS / "split_class_ratios.png")

    return train_df, test_df


def _psi_single(base: np.ndarray, curr: np.ndarray, bins: int = 10) -> float:
    """Compute PSI for a single feature using train-quantile bins."""
    base = np.asarray(base, dtype=float)
    curr = np.asarray(curr, dtype=float)

    # Filter NaN/inf from the base (for quantiles)
    valid_mask = ~np.isnan(base) & ~np.isinf(base)
    if not np.any(valid_mask):
        print("[WARN] No valid values in base array after filtering NaN/inf")
        return 0.0
    base_valid = base[valid_mask]

    qs = np.linspace(0, 1, bins + 1)
    edges = np.quantile(base_valid, qs)
    edges[0], edges[-1] = -np.inf, np.inf
    b_cnt, _ = np.histogram(base, bins=edges)
    c_cnt, _ = np.histogram(curr, bins=edges)
    b_pct = np.clip(b_cnt / max(1, b_cnt.sum()), 1e-6, 1.0)
    c_pct = np.clip(c_cnt / max(1, c_cnt.sum()), 1e-6, 1.0)
    return float(np.sum((c_pct - b_pct) * np.log(c_pct / b_pct)))


def compute_psi_snapshot(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Compute PSI for all numeric features (train->test) and save a bar chart."""
    feats = train.select_dtypes(include=[np.number]).columns.drop(TARGET_COL)
    rows: list[dict[str, float]] = []
    for f in feats:
        psi = _psi_single(train[f].values, test[f].values)
        rows.append({"feature": f, "psi": psi})

    psi_df = pd.DataFrame(rows).sort_values("psi", ascending=False)
    psi_df.to_csv(ARTIFACTS / "psi_snapshot.csv", index=False)

    top_n = min(30, len(psi_df))
    plt.figure(figsize=(9, 6))
    plt.barh(psi_df["feature"].head(top_n)[::-1], psi_df["psi"].head(top_n)[::-1])
    plt.axvline(
        PSI_SMALL, linestyle="--", color="orange", label=f"{PSI_SMALL:.2f} small"
    )
    plt.axvline(PSI_LARGE, linestyle="--", color="red", label=f"{PSI_LARGE:.2f} large")
    plt.title("PSI Snapshot - Train -> Test (Top)")
    plt.legend()
    _savefig(ARTIFACTS / "psi_snapshot_bar.png")
    return psi_df

# 3) Feature Selection

def correlation_filter(train_df: pd.DataFrame, threshold: float = 0.95) -> list[str]:
    """Drop one from each highly-correlated pair (|r| > threshold)."""
    X = train_df.drop(columns=[TARGET_COL]).select_dtypes(include=[np.number])
    corr = X.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
    kept = [c for c in X.columns if c not in to_drop]
    pd.Series(kept, name="feature").to_csv(
        ARTIFACTS / "features_after_corr_filter.csv", index=False
    )
    return kept


def vif_prune_for_lr(
    train_df: pd.DataFrame, base_features: list[str], vif_threshold: float = 10.0
) -> list[str]:
    """Iteratively drop the highest-VIF feature until all VIF <= threshold."""
    feats = base_features.copy()
    for _ in range(MAX_VIF_ITERS):
        X = (
            train_df[feats]
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .dropna(axis=0)
        )
        if X.shape[1] <= MIN_FEATURES_AFTER_VIF:
            break
        # Tiny noise to avoid singularities in VIF calc.
        X = X + np.random.normal(0, 1e-12, size=X.shape)
        vifs = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif_df = pd.DataFrame({"feature": X.columns, "vif": vifs}).sort_values(
            "vif", ascending=False
        )
        worst = vif_df.iloc[0]
        if worst["vif"] <= vif_threshold:
            break
        feats.remove(str(worst["feature"]))

    pd.Series(feats, name="feature").to_csv(
        ARTIFACTS / "features_for_lr_after_vif.csv", index=False
    )
    return feats


def select_features_pipeline(train_df: pd.DataFrame) -> dict[str, list[str]]:
    """SIMPLE: correlation filter for all; VIF only for LR."""
    kept_base = correlation_filter(train_df, threshold=0.95)
    kept_lr = vif_prune_for_lr(train_df, base_features=kept_base, vif_threshold=10.0)
    (ARTIFACTS / "feature_selection_summary.txt").write_text(
        "\n".join(
            [
                f"Base after corr filter (n={len(kept_base)}): {kept_base}",
                f"LR after VIF (n={len(kept_lr)}): {kept_lr}",
                "Trees use correlation-filtered base features (no extra pruning).",
            ]
        ),
        "utf-8",
    )
    return {"for_trees": kept_base, "for_lr": kept_lr, "base_after_corr": kept_base}


# 4) Hyperparameter Tuning : RandomizedSearchCV

@dataclass
class ModelSpec:
    name: str
    pipeline: Pipeline
    param_space: dict[str, list | np.ndarray]


def get_param_space(name: str) -> dict[str, list | np.ndarray]:
    """Lean, high-impact spaces (fast + strong on this dataset)."""
    if name == "LogisticRegression":
        return {
            "clf__C": np.logspace(-3, 2, 15).tolist(),
            "clf__penalty": ["l2"],
            "clf__solver": ["lbfgs"],
            "clf__max_iter": [300, 600],
        }

    if name == "RandomForest":
        return {
            "clf__n_estimators": [200, 300, 400, 500, 600],
            "clf__max_depth": [None, 6, 10, 14],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__max_features": ["sqrt", "log2"],
        }

    if name == "XGBoost":
        return {
            "clf__n_estimators": [100, 150, 200, 250, 300, 350, 400],
            "clf__learning_rate": [0.05, 0.075, 0.10, 0.12],
            "clf__max_depth": [3, 4, 5, 6],
            "clf__subsample": [0.8, 0.9, 1.0],
            "clf__colsample_bytree": [0.8, 0.9, 1.0],
            "clf__reg_lambda": [0.1, 1.0, 10.0],
        }
    raise ValueError(f"Unknown model: {name}")


def tune_random_search(
    name: str, pipe: Pipeline, X: pd.DataFrame, y: np.ndarray
) -> tuple[Pipeline, dict, float]:
    """
    RandomizedSearchCV (5-fold Stratified), scoring=PR-AUC.
    No early stopping inside CV (simple & robust).
    """
    n_iter = {"LogisticRegression": 20, "RandomForest": 30, "XGBoost": 30}[name]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=get_param_space(name),
        n_iter=n_iter,
        scoring="average_precision",
        n_jobs=-1,
        cv=cv,
        random_state=SEED,
        verbose=0,
        refit=True,
    )

    search.fit(X, y)

    with open(ARTIFACTS / f"params_{name}.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                k: (v.item() if hasattr(v, "item") else v)
                for k, v in search.best_params_.items()
            },
            f,
            indent=2,
        )
    pd.DataFrame(search.cv_results_).to_csv(
        ARTIFACTS / f"cv_results_{name}.csv", index=False
    )

    print(f"[TUNE] {name}: best CV AP={search.best_score_:.4f}")
    return search.best_estimator_, search.best_params_, float(search.best_score_)



# 5) Training

def oof_scores(
    name: str, est: Pipeline, X: pd.DataFrame, y: np.ndarray
) -> dict[str, float | str]:
    """Compute OOF probabilities with StratifiedKFold and save CSV."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros_like(y, dtype=float)
    for tr_idx, va_idx in skf.split(X, y):
        clone = sk_clone(est)
        clone.fit(X.iloc[tr_idx], y[tr_idx])
        oof[va_idx] = clone.predict_proba(X.iloc[va_idx])[:, 1]

    oof_ap = average_precision_score(y, oof)
    oof_roc = roc_auc_score(y, oof)

    pd.DataFrame({"y": y, "oof_proba": oof}).to_csv(
        ARTIFACTS / f"oof_{name}.csv", index=False
    )
    print(f"[TRAIN] {name}: OOF AP={oof_ap:.4f} ROC={oof_roc:.4f}")
    return {
        "oof_ap": float(oof_ap),
        "oof_roc": float(oof_roc),
        "oof_csv": f"artifacts/oof_{name}.csv",
    }


def final_fit_and_save(name: str, est: Pipeline, X: pd.DataFrame, y: np.ndarray) -> str:
    """
    Refit on all training data and save the model.
    For XGBoost: try early stopping with a small validation split; fall back if unsupported.
    """
    est_to_fit = sk_clone(est)

    if name == "XGBoost":
        # small validation split (15%) from the training set
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
        tr_idx, va_idx = next(sss.split(X, y))
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X.iloc[va_idx], y[va_idx]

        try:
            # Large cap; early stopping will pick best iteration if supported.
            est_to_fit.set_params(clf__n_estimators=2000)
            est_to_fit.fit(
                X_tr,
                y_tr,
                **{
                    "clf__eval_set": [(X_va, y_va)],
                    "clf__early_stopping_rounds": 50,
                    "clf__verbose": False,
                },
            )
            best_iter = getattr(est_to_fit.named_steps["clf"], "best_iteration", None)
            print(
                f"[FINAL] XGBoost fit with early stopping (best_iteration={best_iter})."
            )
        except TypeError:
            print("[FINAL] Early stopping not supported here; fitting without it.")
            est_to_fit.fit(X, y)
    else:
        est_to_fit.fit(X, y)

    path = ARTIFACTS / f"model_{name}.joblib"
    joblib.dump(est_to_fit, path)
    print(f"[SAVE] {name} -> {path}")
    return str(path)


# 6) Evaluation (train vs test curves + metrics table)

def _plot_pr_curves(
    y_tr: np.ndarray, p_tr: np.ndarray, y_te: np.ndarray, p_te: np.ndarray, name: str
) -> None:
    """Plot Precision-Recall curves (train & test overlay) and save."""
    pr_tr_prec, pr_tr_rec, _ = precision_recall_curve(y_tr, p_tr)
    pr_te_prec, pr_te_rec, _ = precision_recall_curve(y_te, p_te)
    ap_tr = average_precision_score(y_tr, p_tr)
    ap_te = average_precision_score(y_te, p_te)

    plt.figure(figsize=(6.2, 4.0))
    plt.plot(pr_tr_rec, pr_tr_prec, label=f"Train (AP={ap_tr:.3f})")
    plt.plot(pr_te_rec, pr_te_prec, label=f"Test  (AP={ap_te:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall - {name}")
    plt.legend()
    _savefig(ARTIFACTS / f"pr_{name}.png")


def recall_at_precision(
    y_true: np.ndarray, proba: np.ndarray, target_precision: float = 0.80
) -> float:
    """Compute max recall at or above target precision."""
    prec, rec, _ = precision_recall_curve(y_true, proba)
    mask = prec >= target_precision
    return float(rec[mask].max()) if np.any(mask) else 0.0


def eval_model(  # noqa: PLR0913
    name: str,
    est: Pipeline,
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_te: pd.DataFrame,
    y_te: np.ndarray,
) -> dict[str, float]:
    """Save ROC, PR, and calibration (train+test overlay); return key metrics."""
    p_tr = est.predict_proba(X_tr)[:, 1]
    p_te = est.predict_proba(X_te)[:, 1]

    # ROC curves
    fpr_tr, tpr_tr, _ = roc_curve(y_tr, p_tr)
    fpr_te, tpr_te, _ = roc_curve(y_te, p_te)
    roc_tr, roc_te = auc(fpr_tr, tpr_tr), auc(fpr_te, tpr_te)
    plt.figure(figsize=(6.2, 4.0))
    plt.plot(fpr_tr, tpr_tr, label=f"Train (AUC={roc_tr:.3f})")
    plt.plot(fpr_te, tpr_te, label=f"Test  (AUC={roc_te:.3f})")
    plt.plot([0, 1], [0, 1], "--", lw=1)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC - {name}")
    plt.legend()
    _savefig(ARTIFACTS / f"roc_{name}.png")

    # PR curves
    _plot_pr_curves(y_tr, p_tr, y_te, p_te, name)

    # Calibration curves
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    CalibrationDisplay.from_predictions(y_tr, p_tr, n_bins=10, name="Train", ax=ax)
    CalibrationDisplay.from_predictions(y_te, p_te, n_bins=10, name="Test", ax=ax)
    ax.set_title(f"Calibration - {name}")
    _savefig(ARTIFACTS / f"calibration_{name}.png")

    # Metrics: Brier, PR-AUC, F1@0.5, F2@0.5, Recall@P80
    brier_tr, brier_te = brier_score_loss(y_tr, p_tr), brier_score_loss(y_te, p_te)
    pr_tr, pr_te = (
        average_precision_score(y_tr, p_tr),
        average_precision_score(y_te, p_te),
    )
    f1_te = f1_score(y_te, (p_te >= PR_THRESHOLD).astype(int), zero_division=0)
    f2_te = fbeta_score(
        y_te, (p_te >= PR_THRESHOLD).astype(int), beta=F2_BETA, zero_division=0
    )
    recall_p80 = recall_at_precision(y_te, p_te)

    return {
        "roc_auc_train": float(roc_tr),
        "roc_auc_test": float(roc_te),
        "pr_auc_train": float(pr_tr),
        "pr_auc_test": float(pr_te),
        "brier_train": float(brier_tr),
        "brier_test": float(brier_te),
        "f1_test": float(f1_te),
        "f2_test": float(f2_te),
        "recall_at_p80": float(recall_p80),
    }


# 7) SHAP / Coefficients (robust for trees)

def explain_model(name: str, pipe: Pipeline, X_train: pd.DataFrame) -> None:
    """
    Create SHAP (trees) or coefficient (LR) plots for interpretability.

    Robustness: If SHAP plotting fails for tree models, fall back to plotting
    model.feature_importances_ (if available).
    """
    if name == "LogisticRegression":
        clf = pipe.named_steps["clf"]
        coefs = pd.Series(clf.coef_.ravel(), index=X_train.columns).sort_values()
        plt.figure(figsize=(8, 6))
        coefs.tail(20).plot(kind="barh")
        plt.title("Top Positive Coefficients - Logistic Regression")
        _savefig(ARTIFACTS / "interpretability_lr_coeffs.png")
        return

    clf = pipe.named_steps["clf"]
    try:
        X_sample = X_train.sample(n=min(1000, len(X_train)), random_state=SEED)
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X_sample)
        if isinstance(sv, list):
            sv_plot = sv[1] if len(sv) > 1 else sv[0]
        else:
            sv_plot = sv
        plt.figure(figsize=(8, 6))
        shap.summary_plot(sv_plot, X_sample, show=False, max_display=20)
        plt.title(f"SHAP Summary - {name}")
        _savefig(ARTIFACTS / f"interpretability_shap_{name}.png")
    except Exception as exc:
        # Fallback: simple feature_importances
        if hasattr(clf, "feature_importances_"):
            importances = pd.Series(
                clf.feature_importances_, index=X_train.columns
            ).sort_values(ascending=False)
            plt.figure(figsize=(8, 6))
            importances.head(20).iloc[::-1].plot(kind="barh")
            plt.title(f"Feature Importances (fallback) - {name}")
            _savefig(ARTIFACTS / f"interpretability_fallback_{name}.png")
            print(f"[WARN] SHAP failed for {name}, plotted importances instead: {exc}")
        else:
            msg = f"SHAP failed for {name} and no importances available: {exc}\n"
            (ARTIFACTS / f"interpretability_error_{name}.txt").write_text(msg, "utf-8")
            print("[WARN] " + msg)


# 8) PSI for Best Model Features

def compute_psi_for_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    """Compute PSI (train->test) for the final feature set and save plot/report."""
    rows = [
        {"feature": f, "psi": _psi_single(train_df[f].values, test_df[f].values)}
        for f in features
    ]
    psi_df = pd.DataFrame(rows).sort_values("psi", ascending=False)
    psi_df.to_csv(ARTIFACTS / "psi_final.csv", index=False)

    plt.figure(figsize=(9, 6))
    plt.barh(psi_df["feature"].head(30)[::-1], psi_df["psi"].head(30)[::-1])
    plt.axvline(
        PSI_SMALL, linestyle="--", color="orange", label=f"{PSI_SMALL:.2f} small"
    )
    plt.axvline(PSI_LARGE, linestyle="--", color="red", label=f"{PSI_LARGE:.2f} large")
    plt.title("PSI - Final Feature Set (Top)")
    plt.legend()
    _savefig(ARTIFACTS / "psi_final_bar.png")

    n_large = int((psi_df["psi"] > PSI_LARGE).sum())
    n_mod = int(((psi_df["psi"] >= PSI_SMALL) & (psi_df["psi"] <= PSI_LARGE)).sum())
    n_small = int((psi_df["psi"] < PSI_SMALL).sum())
    (ARTIFACTS / "psi_final_summary.txt").write_text(
        "\n".join(
            [
                f"Features: {len(psi_df)}",
                f"Small (<{PSI_SMALL:.2f}): {n_small}",
                f"Moderate ({PSI_SMALL:.2f}-{PSI_LARGE:.2f}): {n_mod}",
                f"Large (>{PSI_LARGE:.2f}): {n_large}",
            ]
        ),
        "utf-8",
    )
    print("[PSI] Final PSI saved (csv, bar, summary).")
    return psi_df



# 9) Report writer

def write_report(
    metrics_table: pd.DataFrame,
    best_name: str,
    psi_snap: pd.DataFrame,
    psi_final: pd.DataFrame,
    feature_sets: dict[str, list[str]],
) -> None:
    """Create a compact Markdown report referencing all saved artifacts + jot notes."""
    report = ARTIFACTS / "report.md"
    metrics_md = _df_to_markdown_safe(metrics_table.round(4), index=False)
    psi_snap_md = _df_to_markdown_safe(psi_snap.head(10), index=False)
    psi_final_md = _df_to_markdown_safe(psi_final.head(10), index=False)

    md = f"""# Lab 5 Report - Company Bankruptcy Prediction

## Evidence (saved under `artifacts/`)
- Class imbalance: `eda_class_imbalance.png`
- Correlation heatmap: `eda_corr_heatmap.png`
- Boxplots sample: `eda_boxplots_sample.png`
- Skewness analysis: `eda_skewness.csv`, `eda_skewness_bar.png`
- Inf values: `eda_inf_counts.csv` or `eda_inf_counts.txt`
- Stratified split check: `split_class_ratios.png`
- PSI snapshot: `psi_snapshot_bar.png`
- Final model curves: `roc_*.png`, `pr_*.png`, `calibration_*.png`
- SHAP/coeffs: `interpretability_*`
- PSI final: `psi_final_bar.png`
- Model artifacts: `model_*.joblib`

## Jot Notes per Component (<=4 each)

### 1) EDA
- Checked class imbalance; confirmed minority ~ a few % => treat as imbalanced.
- Inspected correlations; used as input to correlation filter.
- Boxplots/skewness to motivate winsorization (5%) for outliers.
- Verified missing/inf values; none material problems.

### 2) Preprocessing
- Stratified train/test split; tracked class ratio plot.
- Winsorized numeric features at 5% tails (train/test separately).
- Handled imbalance via class_weight="balanced" (LR, RF) & scale_pos_weight (XGB).
- Computed PSI snapshot to check train->test drift.

### 3) Feature Selection (SIMPLE)
- Applied correlation filtering (|r|>0.95) for all models.
- VIF pruning only for LR to address multicollinearity.
- Trees use post-correlation features (no extra pruning).
- Saved selected feature lists for transparency.

### 4) Hyperparameter Tuning (SIMPLE)
- Used RandomizedSearchCV (5-fold Stratified CV), scoring=PR-AUC.
- Lean search spaces focused on impactful knobs.
- Saved JSON of best params and full CV results per model.
- Balanced compute time vs performance.

### 5) Training
- 5-fold OOF estimation for sanity check.
- Final refit on full train; joblib-saved models.
- Pipelines prevent leakage (scaler only in LR).
- Reproducible with fixed SEED=42.

### 6) Evaluation & Comparison
- Generated ROC, PR, Calibration plots (train+test overlay).
- Tabulated ROC-AUC, PR-AUC, Brier, F1, F2, Recall@P80.
- Selected best model by Test PR-AUC (tiebreak: Test ROC-AUC).
- Curves reviewed for over/underfitting.

### 7) SHAP (Interpretability)
- Trees: TreeExplainer with SHAP summary; LR: coefficients.
- Focus on top features and directionality.
- Useful for risk/compliance review.
- Saved plots in artifacts.

### 8) PSI (Stability)
- Train->test PSI for all numeric and final feature set.
- Flag thresholds: 0.10 (small), 0.25 (large).
- Summarized counts by drift severity.
- Guides re-sampling/retraining plans if drift grows.

### 9) Challenges & Reflections
- Imbalance managed via weights/scale_pos_weight (no SMOTE to keep simple).
- Avoided GUI backend issues via matplotlib.use("Agg").
- Kept selection/tuning simple to meet runtime constraints.
- Documented artifacts for Airflow-friendly reproducibility.

## Selected Features
- For LR (after VIF): {len(feature_sets["for_lr"])} features
- For Trees (after corr filter): {len(feature_sets["for_trees"])} features

## Metrics Table (higher AUC/PR-AUC/F1/F2 better; lower Brier better)
{metrics_md}

## Best Model
- **{best_name}** chosen by **Test PR-AUC** (primary) with ROC-AUC tiebreak + calibration check.

## PSI Snapshot (train->test, all numerics)
Top 10 by PSI:
{psi_snap_md}

## PSI - Final Feature Set (train->test)
Top 10 by PSI:
{psi_final_md}

*Notes*: PSI < 0.10 small/no shift; 0.10-0.25 moderate; >0.25 large (investigate).
"""
    report.write_text(md, "utf-8")
    print(f"[REPORT] Wrote {report}")


# 10) Artifact organizer (simple)

def organize_artifacts() -> None:
    """Auto-sort files in artifacts/ into subfolders."""
    root = ARTIFACTS
    folders: dict[str, list[str]] = {
        "EDA": ["eda_*", "split_class_ratios.png"],
        "Features": ["features_*", "feature_selection_summary.txt"],
        "Tuning": ["cv_results_*", "params_*"],
        "Models": ["model_*"],
        "Evaluation": ["roc_*", "pr_*", "calibration_*", "metrics_table.csv", "oof_*"],
        "PSI": ["psi_*"],
        "Meta": ["manifest.json", "report.md"],
    }

    for file in root.glob("*"):
        if not file.is_file():
            continue
        moved = False
        for folder, patterns in folders.items():
            if any(fnmatch(file.name.lower(), p.lower()) for p in patterns):
                (root / folder).mkdir(parents=True, exist_ok=True)
                shutil.move(str(file), root / folder / file.name)
                moved = True
                break
        if not moved:
            (root / "Meta").mkdir(parents=True, exist_ok=True)
            shutil.move(str(file), root / "Meta" / file.name)

    print("[ORGANIZE] Artifacts sorted into folders.")



# 11) Orchestration

def _scale_pos_weight(y: np.ndarray) -> float:
    """Compute XGBoost scale_pos_weight for imbalanced labels."""
    neg = int((y == 0).sum())
    pos = max(1, int((y == 1).sum()))
    return max(1.0, neg / pos)


def main() -> None:  # noqa: PLR0915
    """Run the full Lab 5 pipeline end-to-end."""
    # Load
    data_path = find_dataset()
    df = pd.read_csv(data_path)
    if df[TARGET_COL].dtype != int and df[TARGET_COL].dtype != np.int64:
        df[TARGET_COL] = df[TARGET_COL].astype(int)

    # EDA
    perform_eda(df)

    # Split + PSI snapshot
    train_df, test_df = stratified_split(df, test_size=0.25)
    psi_snapshot = compute_psi_snapshot(train_df, test_df)

    # Feature selection (SIMPLE)
    feature_sets = select_features_pipeline(train_df)

    # Build pipelines (scaling only for LR; class weights / scale_pos_weight)
    lr_pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    random_state=SEED,
                    class_weight="balanced",
                    max_iter=500,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    rf_pipe = Pipeline(
        [
            (
                "clf",
                RandomForestClassifier(
                    random_state=SEED, n_jobs=-1, class_weight="balanced_subsample"
                ),
            )
        ]
    )
    spw = _scale_pos_weight(train_df[TARGET_COL].values)
    xgb_pipe = Pipeline(
        [
            (
                "clf",
                XGBClassifier(
                    random_state=SEED,
                    n_estimators=300,  # default; tuner explores 100-400
                    learning_rate=0.10,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    max_depth=4,
                    reg_lambda=1.0,
                    n_jobs=-1,
                    tree_method="hist",
                    scale_pos_weight=spw,
                    eval_metric="aucpr",  # align with PR-AUC
                ),
            )
        ]
    )

    # Matrices per family
    X_train_lr = train_df[feature_sets["for_lr"]].copy()
    X_train_trees = train_df[feature_sets["for_trees"]].copy()
    y_train = train_df[TARGET_COL].values

    X_test_lr = test_df[feature_sets["for_lr"]].copy()
    X_test_trees = test_df[feature_sets["for_trees"]].copy()
    y_test = test_df[TARGET_COL].values

    # Tuning (RandomizedSearchCV only)
    best_lr, _, _ = tune_random_search(
        "LogisticRegression", lr_pipe, X_train_lr, y_train
    )
    best_rf, _, _ = tune_random_search("RandomForest", rf_pipe, X_train_trees, y_train)
    best_xgb, _, _ = tune_random_search("XGBoost", xgb_pipe, X_train_trees, y_train)

    # OOF + final fit
    _ = oof_scores("LogisticRegression", best_lr, X_train_lr, y_train)
    _ = oof_scores("RandomForest", best_rf, X_train_trees, y_train)
    _ = oof_scores("XGBoost", best_xgb, X_train_trees, y_train)

    model_paths = {
        "LogisticRegression": final_fit_and_save(
            "LogisticRegression", best_lr, X_train_lr, y_train
        ),
        "RandomForest": final_fit_and_save(
            "RandomForest", best_rf, X_train_trees, y_train
        ),
        "XGBoost": final_fit_and_save("XGBoost", best_xgb, X_train_trees, y_train),
    }

    # Evaluation on TRAIN & TEST (curves + metrics)
    evals: dict[str, dict[str, float]] = {}
    evals["LogisticRegression"] = eval_model(
        "LogisticRegression", best_lr, X_train_lr, y_train, X_test_lr, y_test
    )
    evals["RandomForest"] = eval_model(
        "RandomForest", best_rf, X_train_trees, y_train, X_test_trees, y_test
    )
    evals["XGBoost"] = eval_model(
        "XGBoost", best_xgb, X_train_trees, y_train, X_test_trees, y_test
    )

    metrics_table = (
        pd.DataFrame.from_dict(evals, orient="index")
        .reset_index()
        .rename(columns={"index": "model"})
        .sort_values("pr_auc_test", ascending=False)
    )

    # Rank (primary: Test PR-AUC; tiebreak: Test ROC-AUC)
    metrics_table["_rank_key_pr"] = -metrics_table["pr_auc_test"]
    metrics_table["_rank_key_roc"] = -metrics_table["roc_auc_test"]
    metrics_table = metrics_table.sort_values(
        by=["_rank_key_pr", "_rank_key_roc"], ascending=True
    )
    best_name = str(metrics_table.iloc[0]["model"])

    # SHAP/coeffs on best
    if best_name == "LogisticRegression":
        X_used = X_train_lr
        best_pipe = best_lr
    elif best_name == "RandomForest":
        X_used = X_train_trees
        best_pipe = best_rf
    else:
        X_used = X_train_trees
        best_pipe = best_xgb
    explain_model(best_name, best_pipe, X_used)

    # PSI for final feature set used by best model
    feats_best = (
        feature_sets["for_lr"]
        if best_name == "LogisticRegression"
        else feature_sets["for_trees"]
    )
    psi_final = compute_psi_for_features(train_df, test_df, feats_best)

    # Report
    metrics_table = metrics_table.drop(columns=["_rank_key_pr", "_rank_key_roc"])
    write_report(
        metrics_table,
        best_name,
        psi_snapshot,
        psi_final,
        feature_sets,
    )

    # Manifest (for reproducibility)
    manifest = {
        "seed": SEED,
        "dataset": str(find_dataset()),
        "models": model_paths,
        "features": feature_sets,
        "cv": {"n_splits": 5, "strategy": "StratifiedKFold", "seed": SEED},
        "metrics_table_csv": str(ARTIFACTS / "metrics_table.csv"),
    }
    metrics_table.to_csv(ARTIFACTS / "metrics_table.csv", index=False)
    (ARTIFACTS / "manifest.json").write_text(json.dumps(manifest, indent=2), "utf-8")

    # --- organize artifacts into folders ---
    organize_artifacts()

    print("\n[DONE] Artifacts saved in ./artifacts")
    print(f"- Best model: {best_name}")
    print(f"- Report: {ARTIFACTS / 'report.md'}")


if __name__ == "__main__":
    main()
