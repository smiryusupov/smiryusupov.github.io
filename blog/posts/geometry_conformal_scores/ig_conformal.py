"""
Information-Geometric Score Design for Conformal Prediction
==========================================================

Core thesis
-----------
Conformal prediction gives model-free marginal validity under exchangeability.
Information geometry / local geometry affects the nonconformity score, hence
efficiency, conditional balance, and robustness.

This script runs synthetic regression experiments across:
    - random forests
    - kernel ridge regression
    - neural networks / MLP

and score families:
    - absolute residual conformal
    - local normalized residual conformal
    - stabilized local normalization
    - blended local/global normalization
    - optional Student-t / Huber normalized score functions
      Note: these are not run by default because, in scalar symmetric intervals,
      they are rank-equivalent to normalized residual scores if they use the same
      scale. They are kept as diagnostics / extensions.
    - conformalized quantile regression as a separate GBRT-CQR model family
    - oracle local normalization, diagnostic only
    - residual-density GMM level-set conformal
    - conditional two-mode density level-set conformal for multimodal data
    - mixture-specific diagnostics: mode probability, entropy, margin, separation

It also includes a framework-agnostic image-classification conformal scaffold
for ResNet/VGG-style logits and embeddings.

Dependencies
------------
pip install numpy pandas scikit-learn matplotlib
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Synthetic data
# =============================================================================


def true_mean(X: np.ndarray) -> np.ndarray:
    """Nonlinear regression mean."""
    x = X[:, 0]
    return np.sin(2 * np.pi * x) + 0.5 * x


def true_sigma(X: np.ndarray, setting: str) -> np.ndarray:
    """Conditional noise scale."""
    x = X[:, 0]
    if setting == "homoskedastic":
        return 0.25 * np.ones_like(x)
    if setting == "heteroskedastic":
        return 0.10 + 0.80 * np.abs(x)
    if setting == "heavy_tail":
        return 0.20 + 0.50 * np.abs(x)
    if setting == "skewed":
        return 0.15 + 0.60 * np.abs(x)
    if setting == "multimodal":
        return 0.12 + 0.25 * np.abs(x)
    raise ValueError(f"Unknown setting: {setting}")


def make_synthetic_regression(
    n: int = 6000,
    d: int = 5,
    setting: str = "heteroskedastic",
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Generate synthetic regression data."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, d))
    mu = true_mean(X)
    sigma = true_sigma(X, setting)

    if setting == "heavy_tail":
        eps = rng.standard_t(df=3, size=n) / math.sqrt(3)
        y = mu + sigma * eps
    elif setting == "skewed":
        # Centered exponential noise: mean zero but strongly right-skewed.
        eps = rng.exponential(scale=1.0, size=n) - 1.0
        y = mu + sigma * eps
    elif setting == "multimodal":
        signs = rng.choice([-1.0, 1.0], size=n)
        eps = rng.normal(0, 1, size=n)
        y = mu + signs * 0.8 + sigma * eps
    else:
        eps = rng.normal(0, 1, size=n)
        y = mu + sigma * eps

    return X, y, {"mu": mu, "sigma": sigma}


# =============================================================================
# Conformal utilities
# =============================================================================


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split conformal quantile."""
    scores = np.asarray(scores)
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return float(np.sort(scores)[k - 1])


@dataclass
class IntervalResult:
    name: str
    coverage: float
    avg_length: float
    median_length: float
    qhat: float
    conditional_coverage_table: pd.DataFrame


@dataclass
class DensitySetResult:
    name: str
    coverage: float
    avg_grid_length: float
    median_grid_length: float
    qhat: float
    conditional_coverage_table: pd.DataFrame


def make_conditional_table(
    covered: np.ndarray,
    lengths: np.ndarray,
    bin_variable: np.ndarray,
    n_bins: int = 8,
) -> pd.DataFrame:
    """Coverage/length by bins of a difficulty variable.

    If the bin variable is constant, use one global bin. This matters for the
    homoskedastic setting.
    """
    rows = []
    bin_variable = np.asarray(bin_variable)

    if np.nanmax(bin_variable) == np.nanmin(bin_variable):
        rows.append(
            {
                "bin_lo": float(np.nanmin(bin_variable)),
                "bin_hi": float(np.nanmax(bin_variable)),
                "n": int(len(covered)),
                "coverage": float(np.mean(covered)),
                "avg_length": float(np.mean(lengths)),
            }
        )
    else:
        quantiles = np.quantile(bin_variable, np.linspace(0, 1, n_bins + 1))
        quantiles = np.unique(quantiles)
        for lo, hi in zip(quantiles[:-1], quantiles[1:]):
            mask = (bin_variable >= lo) & (bin_variable <= hi)
            if mask.sum() == 0:
                continue
            rows.append(
                {
                    "bin_lo": float(lo),
                    "bin_hi": float(hi),
                    "n": int(mask.sum()),
                    "coverage": float(covered[mask].mean()),
                    "avg_length": float(lengths[mask].mean()),
                }
            )

    return pd.DataFrame(
        rows, columns=["bin_lo", "bin_hi", "n", "coverage", "avg_length"]
    )


def evaluate_intervals(
    name: str,
    y_test: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    qhat: float,
    bin_variable: np.ndarray,
    n_bins: int = 8,
) -> IntervalResult:
    covered = (y_test >= lower) & (y_test <= upper)
    lengths = upper - lower
    table = make_conditional_table(covered, lengths, bin_variable, n_bins=n_bins)

    return IntervalResult(
        name=name,
        coverage=float(covered.mean()),
        avg_length=float(lengths.mean()),
        median_length=float(np.median(lengths)),
        qhat=float(qhat),
        conditional_coverage_table=table,
    )


def conditional_imbalance(table: pd.DataFrame, target: float) -> Dict[str, float]:
    """Deviation of binned conditional coverage from target."""
    if table.empty or "coverage" not in table.columns:
        return {
            "max_abs_cond_error": np.nan,
            "mean_abs_cond_error": np.nan,
            "worst_undercoverage": np.nan,
            "worst_overcoverage": np.nan,
        }

    errors = table["coverage"].to_numpy() - target
    return {
        "max_abs_cond_error": float(np.max(np.abs(errors))),
        "mean_abs_cond_error": float(np.mean(np.abs(errors))),
        "worst_undercoverage": float(np.min(errors)),
        "worst_overcoverage": float(np.max(errors)),
    }


# =============================================================================
# Models
# =============================================================================


def fit_mean_model(model_family: str, X_train: np.ndarray, y_train: np.ndarray):
    if model_family == "rf":
        return RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=5,
            random_state=0,
            n_jobs=-1,
        ).fit(X_train, y_train)

    if model_family in {"gbrt", "cqr_gbrt"}:
        return GradientBoostingRegressor(random_state=0).fit(X_train, y_train)

    if model_family == "krr":
        return make_pipeline(
            StandardScaler(),
            KernelRidge(kernel="rbf", alpha=1e-2, gamma=2.0),
        ).fit(X_train, y_train)

    if model_family == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(128, 128),
                activation="relu",
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=500,
                random_state=0,
                early_stopping=True,
            ),
        ).fit(X_train, y_train)

    raise ValueError(f"Unknown model family: {model_family}")


def fit_scale_model(model_family: str, X_train: np.ndarray, residual_abs: np.ndarray):
    """Estimate local scale E[|residual| | X=x]."""
    target = np.maximum(residual_abs, 1e-6)

    if model_family in {"rf", "gbrt"}:
        return RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=10,
            random_state=1,
            n_jobs=-1,
        ).fit(X_train, target)

    if model_family == "krr":
        return make_pipeline(
            StandardScaler(),
            KernelRidge(kernel="rbf", alpha=1e-2, gamma=2.0),
        ).fit(X_train, target)

    if model_family == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 64),
                activation="relu",
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=400,
                random_state=1,
                early_stopping=True,
            ),
        ).fit(X_train, target)

    raise ValueError(f"Unknown model family: {model_family}")



def fit_quantile_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
    random_state: int = 3,
) -> Tuple[GradientBoostingRegressor, GradientBoostingRegressor]:
    """Fit lower/upper conditional quantile models.

    This gives a genuinely different conformal geometry from symmetric residual
    scores. The resulting interval can be asymmetric around the mean and is
    therefore useful under skewed or heteroskedastic noise.
    """
    lower_alpha = alpha / 2
    upper_alpha = 1 - alpha / 2

    lower_model = GradientBoostingRegressor(
        loss="quantile",
        alpha=lower_alpha,
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        min_samples_leaf=10,
        random_state=random_state,
    ).fit(X_train, y_train)

    upper_model = GradientBoostingRegressor(
        loss="quantile",
        alpha=upper_alpha,
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        min_samples_leaf=10,
        random_state=random_state + 1,
    ).fit(X_train, y_train)

    return lower_model, upper_model


def predict_positive(model, X: np.ndarray, floor: float = 1e-4) -> np.ndarray:
    return np.maximum(np.asarray(model.predict(X)), floor)


def stabilized_scales(
    scale_model,
    X_cal: np.ndarray,
    X_test: np.ndarray,
    scale_floor_quantile: float = 0.05,
    scale_ceiling_quantile: float = 0.95,
    ridge_fraction: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Clip/ridge local scale estimates using calibration scale distribution."""
    sig_cal_raw = predict_positive(scale_model, X_cal)
    sig_test_raw = predict_positive(scale_model, X_test)

    floor = float(np.quantile(sig_cal_raw, scale_floor_quantile))
    ceiling = float(np.quantile(sig_cal_raw, scale_ceiling_quantile))
    global_scale = float(np.median(sig_cal_raw))
    ridge = ridge_fraction * global_scale

    sig_cal = np.clip(sig_cal_raw, floor, ceiling) + ridge
    sig_test = np.clip(sig_test_raw, floor, ceiling) + ridge
    return sig_cal, sig_test, global_scale


# =============================================================================
# Regression conformal scores
# =============================================================================


def residual_conformal(
    model,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
    bin_variable: np.ndarray,
) -> IntervalResult:
    mu_cal = model.predict(X_cal)
    mu_test = model.predict(X_test)
    scores = np.abs(y_cal - mu_cal)
    qhat = conformal_quantile(scores, alpha)
    return evaluate_intervals(
        "absolute residual",
        y_test,
        mu_test - qhat,
        mu_test + qhat,
        qhat,
        bin_variable,
    )


def normalized_conformal(
    mean_model,
    scale_model,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
    bin_variable: np.ndarray,
) -> IntervalResult:
    mu_cal = mean_model.predict(X_cal)
    mu_test = mean_model.predict(X_test)
    sig_cal = predict_positive(scale_model, X_cal)
    sig_test = predict_positive(scale_model, X_test)

    scores = np.abs(y_cal - mu_cal) / sig_cal
    qhat = conformal_quantile(scores, alpha)
    return evaluate_intervals(
        "locally normalized residual",
        y_test,
        mu_test - qhat * sig_test,
        mu_test + qhat * sig_test,
        qhat,
        bin_variable,
    )


def stabilized_normalized_conformal(
    mean_model,
    scale_model,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
    bin_variable: np.ndarray,
) -> IntervalResult:
    mu_cal = mean_model.predict(X_cal)
    mu_test = mean_model.predict(X_test)
    sig_cal, sig_test, _ = stabilized_scales(scale_model, X_cal, X_test)

    scores = np.abs(y_cal - mu_cal) / sig_cal
    qhat = conformal_quantile(scores, alpha)
    return evaluate_intervals(
        "stabilized normalized residual",
        y_test,
        mu_test - qhat * sig_test,
        mu_test + qhat * sig_test,
        qhat,
        bin_variable,
    )


def blended_normalized_conformal(
    mean_model,
    scale_model,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
    bin_variable: np.ndarray,
    gamma: float = 0.5,
) -> IntervalResult:
    """Blend local and global scale.

    gamma=0 is fully local; gamma=1 is global scale.
    """
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be between 0 and 1")

    mu_cal = mean_model.predict(X_cal)
    mu_test = mean_model.predict(X_test)
    sig_cal_local, sig_test_local, global_scale = stabilized_scales(
        scale_model, X_cal, X_test
    )

    sig_cal = (1 - gamma) * sig_cal_local + gamma * global_scale
    sig_test = (1 - gamma) * sig_test_local + gamma * global_scale

    scores = np.abs(y_cal - mu_cal) / sig_cal
    qhat = conformal_quantile(scores, alpha)
    return evaluate_intervals(
        f"blended normalized residual gamma={gamma:.2f}",
        y_test,
        mu_test - qhat * sig_test,
        mu_test + qhat * sig_test,
        qhat,
        bin_variable,
    )


def student_t_normalized_conformal(
    mean_model,
    scale_model,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
    bin_variable: np.ndarray,
    nu: float = 3.0,
    gamma: float = 0.25,
) -> IntervalResult:
    """Student-t-style robust normalized score.

    S = log(1 + r^2 / nu), r=(y-mu)/sigma_blend.
    Since this score is monotone in |r|, the set is still an interval.
    """
    if nu <= 0:
        raise ValueError("nu must be positive")

    mu_cal = mean_model.predict(X_cal)
    mu_test = mean_model.predict(X_test)
    sig_cal_local, sig_test_local, global_scale = stabilized_scales(
        scale_model, X_cal, X_test
    )

    sig_cal = (1 - gamma) * sig_cal_local + gamma * global_scale
    sig_test = (1 - gamma) * sig_test_local + gamma * global_scale

    r_cal = (y_cal - mu_cal) / sig_cal
    scores = np.log1p((r_cal**2) / nu)
    qhat = conformal_quantile(scores, alpha)
    radius = np.sqrt(nu * (np.exp(qhat) - 1.0))

    return evaluate_intervals(
        f"student-t normalized residual nu={nu:g} gamma={gamma:.2f}",
        y_test,
        mu_test - radius * sig_test,
        mu_test + radius * sig_test,
        qhat,
        bin_variable,
    )


def huber_normalized_conformal(
    mean_model,
    scale_model,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
    bin_variable: np.ndarray,
    delta: float = 1.5,
    gamma: float = 0.25,
) -> IntervalResult:
    """Huber-style robust normalized score."""
    if delta <= 0:
        raise ValueError("delta must be positive")

    mu_cal = mean_model.predict(X_cal)
    mu_test = mean_model.predict(X_test)
    sig_cal_local, sig_test_local, global_scale = stabilized_scales(
        scale_model, X_cal, X_test
    )

    sig_cal = (1 - gamma) * sig_cal_local + gamma * global_scale
    sig_test = (1 - gamma) * sig_test_local + gamma * global_scale

    r = np.abs((y_cal - mu_cal) / sig_cal)
    scores = np.where(r <= delta, 0.5 * r**2, delta * (r - 0.5 * delta))
    qhat = conformal_quantile(scores, alpha)

    if qhat <= 0.5 * delta**2:
        radius = np.sqrt(2 * qhat)
    else:
        radius = qhat / delta + 0.5 * delta

    return evaluate_intervals(
        f"huber normalized residual delta={delta:g} gamma={gamma:.2f}",
        y_test,
        mu_test - radius * sig_test,
        mu_test + radius * sig_test,
        qhat,
        bin_variable,
    )



def conformalized_quantile_regression(
    lower_model,
    upper_model,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
    bin_variable: np.ndarray,
) -> IntervalResult:
    """Conformalized Quantile Regression (CQR).

    Score:
        S(x,y) = max{ q_low(x) - y, y - q_high(x) }

    Set:
        [q_low(x) - qhat, q_high(x) + qhat]

    This is not rank-equivalent to symmetric residual conformal. It can produce
    asymmetric intervals and is a better candidate for skewed distributions.
    """
    qlo_cal = lower_model.predict(X_cal)
    qhi_cal = upper_model.predict(X_cal)
    qlo_test = lower_model.predict(X_test)
    qhi_test = upper_model.predict(X_test)

    # Ensure lower <= upper even if models cross.
    lo_cal = np.minimum(qlo_cal, qhi_cal)
    hi_cal = np.maximum(qlo_cal, qhi_cal)
    lo_test = np.minimum(qlo_test, qhi_test)
    hi_test = np.maximum(qlo_test, qhi_test)

    scores = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
    qhat = conformal_quantile(scores, alpha)

    return evaluate_intervals(
        "conformalized quantile regression",
        y_test,
        lo_test - qhat,
        hi_test + qhat,
        qhat,
        bin_variable,
    )


def oracle_normalized_conformal(
    mean_model,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sigma_cal: np.ndarray,
    sigma_test: np.ndarray,
    alpha: float,
    bin_variable: np.ndarray,
) -> IntervalResult:
    """Diagnostic only: uses true sigma(x)."""
    mu_cal = mean_model.predict(X_cal)
    mu_test = mean_model.predict(X_test)

    scores = np.abs(y_cal - mu_cal) / np.maximum(sigma_cal, 1e-6)
    qhat = conformal_quantile(scores, alpha)

    return evaluate_intervals(
        "oracle local normalization",
        y_test,
        mu_test - qhat * sigma_test,
        mu_test + qhat * sigma_test,
        qhat,
        bin_variable,
    )



# =============================================================================
# Conditional mixture-density conformal for 1D regression
# =============================================================================


@dataclass
class ConditionalDensitySetResult:
    name: str
    coverage: float
    avg_grid_length: float
    median_grid_length: float
    qhat: float
    conditional_coverage_table: pd.DataFrame


def fit_conditional_two_mode_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 0,
) -> Dict[str, object]:
    """Fit a simple conditional two-mode density model.

    This is designed for the synthetic multimodal setting.

    Model:
        p(y | x) = pi(x) N(y; mu_+(x), sigma_+^2)
                + (1-pi(x)) N(y; mu_-(x), sigma_-^2)

    Fitting strategy:
        1. Fit a central mean model.
        2. Cluster residual signs into two modes using residual >= median residual.
        3. Fit one regression model for the upper mode and one for the lower mode.
        4. Fit a logistic classifier pi(x) for mode probability.
        5. Estimate mode-specific residual scales.

    This is intentionally simple and transparent. It is not meant to be a
    state-of-the-art mixture density network. Its purpose is to test whether
    conditional density geometry improves conformal sets in multimodal data.
    """
    central_model = GradientBoostingRegressor(random_state=seed).fit(X_train, y_train)
    residuals = y_train - central_model.predict(X_train)

    threshold = np.median(residuals)
    mode_labels = (residuals >= threshold).astype(int)

    # If something degenerates, fall back to sign split.
    if len(np.unique(mode_labels)) < 2:
        mode_labels = (residuals >= 0).astype(int)

    upper_model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        min_samples_leaf=10,
        random_state=seed + 10,
    ).fit(X_train[mode_labels == 1], y_train[mode_labels == 1])

    lower_model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        min_samples_leaf=10,
        random_state=seed + 11,
    ).fit(X_train[mode_labels == 0], y_train[mode_labels == 0])

    # Logistic gating network. Standardize because logistic regression can be
    # sensitive to feature scale.
    gate_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=seed + 12),
    ).fit(X_train, mode_labels)

    resid_upper = y_train[mode_labels == 1] - upper_model.predict(X_train[mode_labels == 1])
    resid_lower = y_train[mode_labels == 0] - lower_model.predict(X_train[mode_labels == 0])

    sigma_upper = float(np.std(resid_upper) + 1e-4)
    sigma_lower = float(np.std(resid_lower) + 1e-4)

    return {
        "central_model": central_model,
        "upper_model": upper_model,
        "lower_model": lower_model,
        "gate_model": gate_model,
        "sigma_upper": sigma_upper,
        "sigma_lower": sigma_lower,
    }


def normal_pdf_np(y: np.ndarray, mean: np.ndarray, sigma: float) -> np.ndarray:
    z = (y - mean) / sigma
    return np.exp(-0.5 * z ** 2) / (np.sqrt(2 * np.pi) * sigma)


def conditional_two_mode_density(
    model: Dict[str, object],
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Evaluate p_hat(y | x) for vector y aligned with rows of X."""
    upper_model = model["upper_model"]
    lower_model = model["lower_model"]
    gate_model = model["gate_model"]
    sigma_upper = model["sigma_upper"]
    sigma_lower = model["sigma_lower"]

    mu_upper = upper_model.predict(X)
    mu_lower = lower_model.predict(X)
    pi_upper = gate_model.predict_proba(X)[:, 1]

    dens_upper = normal_pdf_np(y, mu_upper, sigma_upper)
    dens_lower = normal_pdf_np(y, mu_lower, sigma_lower)

    dens = pi_upper * dens_upper + (1 - pi_upper) * dens_lower
    return np.maximum(dens, 1e-300)


def conditional_two_mode_diagnostics(
    model: Dict[str, object],
    X: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Mixture-specific difficulty diagnostics.

    Returns:
        pi_upper:
            Estimated probability of the upper mode.
        mode_entropy:
            Binary entropy of the mode distribution. Large values mean mode
            ambiguity, i.e. pi_upper close to 0.5.
        mode_margin:
            Distance from the decision boundary, |pi_upper - 0.5|. Small values
            mean ambiguous mode assignment.
        mode_separation:
            Distance between predicted upper/lower conditional means.
    """
    gate_model = model["gate_model"]
    upper_model = model["upper_model"]
    lower_model = model["lower_model"]

    pi = gate_model.predict_proba(X)[:, 1]
    pi_clip = np.clip(pi, 1e-8, 1 - 1e-8)
    entropy = -(pi_clip * np.log(pi_clip) + (1 - pi_clip) * np.log(1 - pi_clip))
    margin = np.abs(pi - 0.5)
    separation = np.abs(upper_model.predict(X) - lower_model.predict(X))

    return {
        "pi_upper": pi,
        "mode_entropy": entropy,
        "mode_margin": margin,
        "mode_separation": separation,
    }


def conditional_two_mode_conformal(
    mixture_model: Dict[str, object],
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
    bin_variable: np.ndarray,
    grid_size: int = 1200,
    grid_padding: float = 1.0,
    n_bins: int = 8,
) -> ConditionalDensitySetResult:
    """Conditional density-level-set conformal prediction.

    Score:
        S(x,y) = -log p_hat(y | x)

    Set:
        C_alpha(x) = {y : -log p_hat(y | x) <= qhat}

    The set is evaluated on a one-dimensional y-grid. For multimodal conditional
    densities, the accepted grid points may form disconnected intervals.
    """
    cal_density = conditional_two_mode_density(mixture_model, X_cal, y_cal)
    cal_scores = -np.log(cal_density)
    qhat = conformal_quantile(cal_scores, alpha)

    test_density = conditional_two_mode_density(mixture_model, X_test, y_test)
    test_scores = -np.log(test_density)
    covered = test_scores <= qhat

    # Global y-grid for length approximation. A more precise implementation
    # could use x-specific grids centered around predicted modes.
    y_min = float(min(y_cal.min(), y_test.min()) - grid_padding)
    y_max = float(max(y_cal.max(), y_test.max()) + grid_padding)
    y_grid = np.linspace(y_min, y_max, grid_size)
    dy = y_grid[1] - y_grid[0]

    lengths = np.empty(len(X_test), dtype=float)
    for i in range(len(X_test)):
        X_rep = np.repeat(X_test[i : i + 1], grid_size, axis=0)
        dens_grid = conditional_two_mode_density(mixture_model, X_rep, y_grid)
        scores_grid = -np.log(dens_grid)
        accepted = scores_grid <= qhat
        lengths[i] = accepted.sum() * dy

    table = make_conditional_table(covered, lengths, bin_variable, n_bins=n_bins)

    return ConditionalDensitySetResult(
        name="conditional two-mode density level set",
        coverage=float(covered.mean()),
        avg_grid_length=float(lengths.mean()),
        median_grid_length=float(np.median(lengths)),
        qhat=float(qhat),
        conditional_coverage_table=table,
    )


def conditional_two_mode_extra_tables(
    mixture_model: Dict[str, object],
    X_test: np.ndarray,
    y_test: np.ndarray,
    qhat: float,
    lengths: Optional[np.ndarray] = None,
    n_bins: int = 8,
) -> Dict[str, pd.DataFrame]:
    """Build conditional coverage tables using mixture-specific diagnostics.

    This recomputes coverage from the conditional density score and bins by:
        - pi_upper
        - mode entropy
        - mode margin
        - mode separation
    """
    dens = conditional_two_mode_density(mixture_model, X_test, y_test)
    scores = -np.log(dens)
    covered = scores <= qhat

    if lengths is None:
        lengths = np.ones_like(y_test, dtype=float)

    diagnostics = conditional_two_mode_diagnostics(mixture_model, X_test)
    return {
        name: make_conditional_table(covered, lengths, values, n_bins=n_bins)
        for name, values in diagnostics.items()
    }


def summarize_extra_conditional_tables(
    tables: Dict[str, pd.DataFrame],
    target: float,
) -> Dict[str, float]:
    """Flatten mixture-specific conditional diagnostics into summary metrics."""
    out: Dict[str, float] = {}
    for name, table in tables.items():
        imb = conditional_imbalance(table, target=target)
        for key, value in imb.items():
            out[f"{name}_{key}"] = value
    return out


def conditional_density_result_row_and_tables(
    setting: str,
    model_family: str,
    conditional_density_result: ConditionalDensitySetResult,
    conditional_mixture: Dict[str, object],
    X_test: np.ndarray,
    y_test: np.ndarray,
    target: float,
) -> Tuple[Dict[str, float], Dict[str, pd.DataFrame]]:
    """Build a summary row and diagnostic tables for conditional density CP.

    This centralizes the logic so cqr_gbrt and RF/KRR/MLP branches get identical
    mixture-specific diagnostics.
    """
    imb = conditional_imbalance(
        conditional_density_result.conditional_coverage_table,
        target=target,
    )
    extra_tables = conditional_two_mode_extra_tables(
        conditional_mixture,
        X_test,
        y_test,
        qhat=conditional_density_result.qhat,
        n_bins=8,
    )
    extra_imb = summarize_extra_conditional_tables(
        extra_tables,
        target=target,
    )

    row = {
        "setting": setting,
        "model_family": model_family,
        "method": conditional_density_result.name,
        "coverage": conditional_density_result.coverage,
        "coverage_error": conditional_density_result.coverage - target,
        "avg_length": conditional_density_result.avg_grid_length,
        "median_length": conditional_density_result.median_grid_length,
        "qhat": conditional_density_result.qhat,
        **imb,
        **extra_imb,
    }

    tables = {
        conditional_density_result.name: (
            conditional_density_result.conditional_coverage_table
        )
    }
    for diag_name, diag_table in extra_tables.items():
        tables[f"{conditional_density_result.name} by {diag_name}"] = diag_table

    return row, tables


# =============================================================================
# Density-level-set conformal for 1D regression
# =============================================================================


def fit_residual_gmm(
    residuals: np.ndarray, n_components: int = 2, seed: int = 0
) -> GaussianMixture:
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=seed,
        reg_covar=1e-5,
    )
    gmm.fit(residuals.reshape(-1, 1))
    return gmm


def gmm_negative_log_density(gmm: GaussianMixture, residuals: np.ndarray) -> np.ndarray:
    return -gmm.score_samples(residuals.reshape(-1, 1))


def evaluate_density_level_sets_on_grid(
    name: str,
    mean_model,
    density_model: GaussianMixture,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
    bin_variable: np.ndarray,
    grid_radius: Optional[float] = None,
    grid_size: int = 1200,
    n_bins: int = 8,
) -> DensitySetResult:
    """Conformal density-level set for one-dimensional regression.

    Score: S(x,y) = -log p_hat(y - mu_hat(x)).
    Set:   C(x) = {y : S(x,y) <= qhat}.

    The set can be disconnected on the residual grid.
    """
    mu_cal = mean_model.predict(X_cal)
    mu_test = mean_model.predict(X_test)

    resid_cal = y_cal - mu_cal
    resid_test = y_test - mu_test

    scores = gmm_negative_log_density(density_model, resid_cal)
    qhat = conformal_quantile(scores, alpha)

    test_scores = gmm_negative_log_density(density_model, resid_test)
    covered = test_scores <= qhat

    if grid_radius is None:
        grid_radius = float(max(4.0, 1.25 * np.quantile(np.abs(resid_cal), 0.995)))

    residual_grid = np.linspace(-grid_radius, grid_radius, grid_size)
    grid_scores = gmm_negative_log_density(density_model, residual_grid)
    accepted = grid_scores <= qhat

    dx = residual_grid[1] - residual_grid[0]
    grid_length = float(accepted.sum() * dx)
    lengths = np.full_like(y_test, grid_length, dtype=float)

    table = make_conditional_table(covered, lengths, bin_variable, n_bins=n_bins)

    return DensitySetResult(
        name=name,
        coverage=float(covered.mean()),
        avg_grid_length=float(lengths.mean()),
        median_grid_length=float(np.median(lengths)),
        qhat=float(qhat),
        conditional_coverage_table=table,
    )


# =============================================================================
# Full synthetic experiment
# =============================================================================


def run_one_experiment(
    setting: str = "heteroskedastic",
    model_family: str = "rf",
    n: int = 6000,
    d: int = 5,
    alpha: float = 0.1,
    seed: int = 0,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    X, y, info = make_synthetic_regression(n=n, d=d, setting=setting, seed=seed)

    idx = np.arange(n)
    X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
        X, y, idx, train_size=0.5, random_state=seed
    )
    X_cal, X_test, y_cal, y_test, idx_cal, idx_test = train_test_split(
        X_temp, y_temp, idx_temp, train_size=0.5, random_state=seed + 1
    )

    sigma_cal = info["sigma"][idx_cal]
    sigma_test = info["sigma"][idx_test]

    bin_variable = sigma_test
    target = 1 - alpha

    # ------------------------------------------------------------------
    # Special branch: CQR as its own model family.
    #
    # Earlier versions inserted the same GBRT-CQR method under rf/krr/mlp,
    # which made CQR rows identical across model families and confounded
    # model-family comparisons. Here CQR is evaluated as its own family:
    #   model_family = "cqr_gbrt"
    # with a GBRT mean residual baseline plus the CQR interval.
    # ------------------------------------------------------------------
    if model_family == "cqr_gbrt":
        mean_model = fit_mean_model("cqr_gbrt", X_train, y_train)
        quantile_lower_model, quantile_upper_model = fit_quantile_models(
            X_train, y_train, alpha=alpha, random_state=seed + 100
        )

        methods: List[IntervalResult] = [
            residual_conformal(
                mean_model, X_cal, y_cal, X_test, y_test, alpha, bin_variable
            ),
            conformalized_quantile_regression(
                quantile_lower_model,
                quantile_upper_model,
                X_cal,
                y_cal,
                X_test,
                y_test,
                alpha,
                bin_variable,
            ),
        ]

        results = []
        conditional_tables: Dict[str, pd.DataFrame] = {}

        for res in methods:
            imb = conditional_imbalance(res.conditional_coverage_table, target=target)
            results.append(
                {
                    "setting": setting,
                    "model_family": model_family,
                    "method": res.name,
                    "coverage": res.coverage,
                    "coverage_error": res.coverage - target,
                    "avg_length": res.avg_length,
                    "median_length": res.median_length,
                    "qhat": res.qhat,
                    **imb,
                }
            )
            conditional_tables[res.name] = res.conditional_coverage_table

        if setting == "multimodal":
            conditional_mixture = fit_conditional_two_mode_model(
                X_train, y_train, seed=seed + 200
            )
            conditional_density_result = conditional_two_mode_conformal(
                mixture_model=conditional_mixture,
                X_cal=X_cal,
                y_cal=y_cal,
                X_test=X_test,
                y_test=y_test,
                alpha=alpha,
                bin_variable=bin_variable,
            )
            row, tables = conditional_density_result_row_and_tables(
                setting=setting,
                model_family=model_family,
                conditional_density_result=conditional_density_result,
                conditional_mixture=conditional_mixture,
                X_test=X_test,
                y_test=y_test,
                target=target,
            )
            results.append(row)
            conditional_tables.update(tables)

        return pd.DataFrame(results), conditional_tables

    # ------------------------------------------------------------------
    # Standard RF/KRR/MLP branch.
    # ------------------------------------------------------------------
    mean_model = fit_mean_model(model_family, X_train, y_train)
    train_residual_abs = np.abs(y_train - mean_model.predict(X_train))
    scale_model = fit_scale_model(model_family, X_train, train_residual_abs)

    methods: List[IntervalResult] = [
        residual_conformal(
            mean_model, X_cal, y_cal, X_test, y_test, alpha, bin_variable
        ),
        normalized_conformal(
            mean_model, scale_model, X_cal, y_cal, X_test, y_test, alpha, bin_variable
        ),
        stabilized_normalized_conformal(
            mean_model, scale_model, X_cal, y_cal, X_test, y_test, alpha, bin_variable
        ),
        blended_normalized_conformal(
            mean_model,
            scale_model,
            X_cal,
            y_cal,
            X_test,
            y_test,
            alpha,
            bin_variable,
            gamma=0.25,
        ),
        blended_normalized_conformal(
            mean_model,
            scale_model,
            X_cal,
            y_cal,
            X_test,
            y_test,
            alpha,
            bin_variable,
            gamma=0.50,
        ),
        blended_normalized_conformal(
            mean_model,
            scale_model,
            X_cal,
            y_cal,
            X_test,
            y_test,
            alpha,
            bin_variable,
            gamma=0.75,
        ),
        oracle_normalized_conformal(
            mean_model,
            X_cal,
            y_cal,
            X_test,
            y_test,
            sigma_cal,
            sigma_test,
            alpha,
            bin_variable,
        ),
    ]

    results = []
    conditional_tables: Dict[str, pd.DataFrame] = {}

    for res in methods:
        imb = conditional_imbalance(res.conditional_coverage_table, target=target)
        results.append(
            {
                "setting": setting,
                "model_family": model_family,
                "method": res.name,
                "coverage": res.coverage,
                "coverage_error": res.coverage - target,
                "avg_length": res.avg_length,
                "median_length": res.median_length,
                "qhat": res.qhat,
                **imb,
            }
        )
        conditional_tables[res.name] = res.conditional_coverage_table

    # Density-level-set conformal
    train_residuals = y_train - mean_model.predict(X_train)
    n_components = 2 if setting == "multimodal" else 1
    residual_density = fit_residual_gmm(
        train_residuals, n_components=n_components, seed=seed
    )
    density_result = evaluate_density_level_sets_on_grid(
        name=f"gmm density level set ({n_components} comp)",
        mean_model=mean_model,
        density_model=residual_density,
        X_cal=X_cal,
        y_cal=y_cal,
        X_test=X_test,
        y_test=y_test,
        alpha=alpha,
        bin_variable=bin_variable,
    )

    imb = conditional_imbalance(density_result.conditional_coverage_table, target=target)
    results.append(
        {
            "setting": setting,
            "model_family": model_family,
            "method": density_result.name,
            "coverage": density_result.coverage,
            "coverage_error": density_result.coverage - target,
            "avg_length": density_result.avg_grid_length,
            "median_length": density_result.median_grid_length,
            "qhat": density_result.qhat,
            **imb,
        }
    )
    conditional_tables[density_result.name] = density_result.conditional_coverage_table

    # Conditional mixture-density geometry for multimodal data.
    # This is the stronger density-level-set method: p_hat(y | x), not merely
    # p_hat(y - mu_hat(x)).
    if setting == "multimodal":
        conditional_mixture = fit_conditional_two_mode_model(
            X_train, y_train, seed=seed + 200
        )
        conditional_density_result = conditional_two_mode_conformal(
            mixture_model=conditional_mixture,
            X_cal=X_cal,
            y_cal=y_cal,
            X_test=X_test,
            y_test=y_test,
            alpha=alpha,
            bin_variable=bin_variable,
        )

        row, tables = conditional_density_result_row_and_tables(
            setting=setting,
            model_family=model_family,
            conditional_density_result=conditional_density_result,
            conditional_mixture=conditional_mixture,
            X_test=X_test,
            y_test=y_test,
            target=target,
        )
        results.append(row)
        conditional_tables.update(tables)

    return pd.DataFrame(results), conditional_tables


def run_rank_equivalence_diagnostics(
    setting: str = "heavy_tail",
    model_family: str = "rf",
    n: int = 6000,
    d: int = 5,
    alpha: float = 0.1,
    seed: int = 0,
) -> pd.DataFrame:
    """Optional diagnostic: show Student-t/Huber rank equivalence.

    In scalar symmetric interval prediction, Student-t, Huber, and absolute
    normalized residual scores are monotone functions of the same normalized
    residual when they use the same scale. Therefore conformal sets are often
    identical. This function is kept to demonstrate that fact, but these methods
    are not included in the default benchmark tables.
    """
    X, y, info = make_synthetic_regression(n=n, d=d, setting=setting, seed=seed)

    idx = np.arange(n)
    X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
        X, y, idx, train_size=0.5, random_state=seed
    )
    X_cal, X_test, y_cal, y_test, idx_cal, idx_test = train_test_split(
        X_temp, y_temp, idx_temp, train_size=0.5, random_state=seed + 1
    )

    bin_variable = info["sigma"][idx_test]

    mean_model = fit_mean_model(model_family, X_train, y_train)
    train_residual_abs = np.abs(y_train - mean_model.predict(X_train))
    scale_model = fit_scale_model(model_family, X_train, train_residual_abs)

    methods = [
        blended_normalized_conformal(
            mean_model,
            scale_model,
            X_cal,
            y_cal,
            X_test,
            y_test,
            alpha,
            bin_variable,
            gamma=0.25,
        ),
        student_t_normalized_conformal(
            mean_model,
            scale_model,
            X_cal,
            y_cal,
            X_test,
            y_test,
            alpha,
            bin_variable,
            nu=3.0,
            gamma=0.25,
        ),
        student_t_normalized_conformal(
            mean_model,
            scale_model,
            X_cal,
            y_cal,
            X_test,
            y_test,
            alpha,
            bin_variable,
            nu=5.0,
            gamma=0.25,
        ),
        huber_normalized_conformal(
            mean_model,
            scale_model,
            X_cal,
            y_cal,
            X_test,
            y_test,
            alpha,
            bin_variable,
            delta=1.5,
            gamma=0.25,
        ),
    ]

    rows = []
    for res in methods:
        rows.append(
            {
                "setting": setting,
                "model_family": model_family,
                "method": res.name,
                "coverage": res.coverage,
                "avg_length": res.avg_length,
                "median_length": res.median_length,
                "qhat": res.qhat,
            }
        )
    return pd.DataFrame(rows)


def add_relative_efficiency_columns(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["length_reduction_vs_abs_pct"] = np.nan

    for _, group in out.groupby(["setting", "model_family"]):
        baseline = group.loc[group["method"] == "absolute residual", "avg_length_mean"]
        if len(baseline) != 1:
            continue
        base_len = float(baseline.iloc[0])
        idx = group.index
        out.loc[idx, "length_reduction_vs_abs_pct"] = (
            100 * (base_len - out.loc[idx, "avg_length_mean"]) / base_len
        )

    return out


def run_grid(
    settings: Optional[List[str]] = None,
    model_families: Optional[List[str]] = None,
    n: int = 6000,
    d: int = 5,
    alpha: float = 0.1,
    seeds: Optional[List[int]] = None,
    save_csv: bool = True,
    output_prefix: str = "ig_conformal_results",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if settings is None:
        settings = ["homoskedastic", "heteroskedastic", "heavy_tail", "skewed", "multimodal"]
    if model_families is None:
        model_families = ["rf", "krr", "mlp", "cqr_gbrt"]
    if seeds is None:
        seeds = [0, 1, 2]

    all_rows = []
    for setting in settings:
        for model_family in model_families:
            for seed in seeds:
                print(f"Running setting={setting}, model={model_family}, seed={seed}")
                try:
                    df, _ = run_one_experiment(
                        setting=setting,
                        model_family=model_family,
                        n=n,
                        d=d,
                        alpha=alpha,
                        seed=seed,
                    )
                    df["seed"] = seed
                    all_rows.append(df)
                except Exception as e:
                    warnings.warn(
                        f"Failed setting={setting}, model={model_family}, seed={seed}: {e}"
                    )

    raw = pd.concat(all_rows, ignore_index=True)

    agg_spec = {
        "coverage_mean": ("coverage", "mean"),
        "coverage_sd": ("coverage", "std"),
        "coverage_error_mean": ("coverage_error", "mean"),
        "avg_length_mean": ("avg_length", "mean"),
        "avg_length_sd": ("avg_length", "std"),
        "median_length_mean": ("median_length", "mean"),
        "max_abs_cond_error_mean": ("max_abs_cond_error", "mean"),
        "mean_abs_cond_error_mean": ("mean_abs_cond_error", "mean"),
        "worst_undercoverage_mean": ("worst_undercoverage", "mean"),
        "worst_overcoverage_mean": ("worst_overcoverage", "mean"),
    }

    # Include mixture-specific conditional diagnostics when present.
    for col in raw.columns:
        if (
            col.endswith("_mean_abs_cond_error")
            or col.endswith("_max_abs_cond_error")
            or col.endswith("_worst_undercoverage")
            or col.endswith("_worst_overcoverage")
        ):
            agg_spec[f"{col}_mean"] = (col, "mean")

    summary = (
        raw.groupby(["setting", "model_family", "method"])
        .agg(**agg_spec)
        .reset_index()
        .sort_values(["setting", "model_family", "avg_length_mean"])
    )

    summary = add_relative_efficiency_columns(summary)

    if save_csv:
        raw.to_csv(f"{output_prefix}_raw.csv", index=False)
        summary.to_csv(f"{output_prefix}_summary.csv", index=False)

    return raw, summary



def summarize_multimodal_density_diagnostics(summary: pd.DataFrame) -> pd.DataFrame:
    """Compact table for conditional two-mode density diagnostics.

    All model families should now have non-NaN mixture-specific diagnostics in
    multimodal rows, because the row construction is shared across branches.
    """
    rows = []
    method_name = "conditional two-mode density level set"
    subset = summary[
        (summary["setting"] == "multimodal")
        & (summary["method"] == method_name)
    ]

    for _, row in subset.iterrows():
        rows.append(
            {
                "model_family": row["model_family"],
                "coverage": row["coverage_mean"],
                "avg_length": row["avg_length_mean"],
                "sigma_bin_cond_error": row.get("mean_abs_cond_error_mean", np.nan),
                "pi_upper_cond_error": row.get(
                    "pi_upper_mean_abs_cond_error_mean", np.nan
                ),
                "mode_entropy_cond_error": row.get(
                    "mode_entropy_mean_abs_cond_error_mean", np.nan
                ),
                "mode_margin_cond_error": row.get(
                    "mode_margin_mean_abs_cond_error_mean", np.nan
                ),
                "mode_separation_cond_error": row.get(
                    "mode_separation_mean_abs_cond_error_mean", np.nan
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("model_family")


def summarize_geometry_effect(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []

    geo_methods = [
        "locally normalized residual",
        "stabilized normalized residual",
        "blended normalized residual gamma=0.25",
        "blended normalized residual gamma=0.50",
        "blended normalized residual gamma=0.75",
    ]

    for (setting, model_family), group in summary.groupby(["setting", "model_family"]):
        base = group[group["method"] == "absolute residual"]
        candidates = group[group["method"].isin(geo_methods)]
        oracle = group[group["method"] == "oracle local normalization"]
        density = group[group["method"].str.contains("density level set", regex=False)]

        if len(base) != 1:
            continue

        base_row = base.iloc[0]

        if model_family == "cqr_gbrt":
            cqr = group[group["method"] == "conformalized quantile regression"]
            if cqr.empty:
                continue
            geo_row = cqr.iloc[0]
        else:
            if candidates.empty:
                continue
            geo_row = candidates.sort_values("avg_length_mean").iloc[0]

        row = {
            "setting": setting,
            "model_family": model_family,
            "baseline_coverage": base_row["coverage_mean"],
            "geo_method": geo_row["method"],
            "geo_coverage": geo_row["coverage_mean"],
            "baseline_length": base_row["avg_length_mean"],
            "geo_length": geo_row["avg_length_mean"],
            "geo_length_reduction_pct": geo_row["length_reduction_vs_abs_pct"],
            "baseline_cond_error": base_row["mean_abs_cond_error_mean"],
            "geo_cond_error": geo_row["mean_abs_cond_error_mean"],
            "cond_error_reduction_pct": 100
            * (
                base_row["mean_abs_cond_error_mean"]
                - geo_row["mean_abs_cond_error_mean"]
            )
            / max(base_row["mean_abs_cond_error_mean"], 1e-12),
        }

        if len(oracle) == 1:
            oracle_row = oracle.iloc[0]
            row.update(
                {
                    "oracle_length": oracle_row["avg_length_mean"],
                    "oracle_length_reduction_pct": oracle_row[
                        "length_reduction_vs_abs_pct"
                    ],
                }
            )

        if len(density) >= 1:
            density_row = density.sort_values("avg_length_mean").iloc[0]
            row.update(
                {
                    "density_method": density_row["method"],
                    "density_coverage": density_row["coverage_mean"],
                    "density_length": density_row["avg_length_mean"],
                    "density_length_reduction_pct": density_row[
                        "length_reduction_vs_abs_pct"
                    ],
                    "density_cond_error": density_row["mean_abs_cond_error_mean"],
                }
            )

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["setting", "model_family"])


def plot_conditional_coverage(
    conditional_tables: Dict[str, pd.DataFrame], alpha: float = 0.1
) -> None:
    import matplotlib.pyplot as plt

    target = 1 - alpha
    for name, table in conditional_tables.items():
        centers = 0.5 * (table["bin_lo"].to_numpy() + table["bin_hi"].to_numpy())
        plt.figure()
        plt.plot(centers, table["coverage"].to_numpy(), marker="o")
        plt.axhline(target, linestyle="--")
        plt.xlabel("difficulty bin")
        plt.ylabel("empirical conditional coverage")
        plt.title(name)
        plt.show()


# =============================================================================
# Classification conformal utilities for ResNet/VGG-style experiments
# =============================================================================


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def softmax_conformal_sets(
    cal_logits: np.ndarray,
    cal_y: np.ndarray,
    test_logits: np.ndarray,
    test_y: np.ndarray,
    alpha: float = 0.1,
    temperature: float = 1.0,
) -> Dict[str, float]:
    """Classification conformal using S(x,y)=1-p(y|x)."""
    p_cal = softmax_np(cal_logits, temperature=temperature)
    p_test = softmax_np(test_logits, temperature=temperature)

    cal_scores = 1.0 - p_cal[np.arange(len(cal_y)), cal_y]
    qhat = conformal_quantile(cal_scores, alpha)

    included = (1.0 - p_test) <= qhat
    covered = included[np.arange(len(test_y)), test_y]
    sizes = included.sum(axis=1)

    return {
        "method": f"softmax conformal temp={temperature:g}",
        "coverage": float(covered.mean()),
        "avg_size": float(sizes.mean()),
        "median_size": float(np.median(sizes)),
        "qhat": float(qhat),
    }


def aps_conformal_sets(
    cal_logits: np.ndarray,
    cal_y: np.ndarray,
    test_logits: np.ndarray,
    test_y: np.ndarray,
    alpha: float = 0.1,
    temperature: float = 1.0,
) -> Dict[str, float]:
    """Adaptive prediction sets.

    Score for label y is cumulative probability mass of all labels at least as
    probable as y.
    """
    p_cal = softmax_np(cal_logits, temperature=temperature)
    p_test = softmax_np(test_logits, temperature=temperature)

    def aps_scores_all(p: np.ndarray) -> np.ndarray:
        n, _ = p.shape
        order = np.argsort(-p, axis=1)
        sorted_p = np.take_along_axis(p, order, axis=1)
        cum = np.cumsum(sorted_p, axis=1)
        scores = np.empty_like(p)
        for i in range(n):
            scores[i, order[i]] = cum[i]
        return scores

    cal_scores_all = aps_scores_all(p_cal)
    cal_scores = cal_scores_all[np.arange(len(cal_y)), cal_y]
    qhat = conformal_quantile(cal_scores, alpha)

    test_scores_all = aps_scores_all(p_test)
    included = test_scores_all <= qhat
    covered = included[np.arange(len(test_y)), test_y]
    sizes = included.sum(axis=1)

    return {
        "method": f"aps conformal temp={temperature:g}",
        "coverage": float(covered.mean()),
        "avg_size": float(sizes.mean()),
        "median_size": float(np.median(sizes)),
        "qhat": float(qhat),
    }


def fit_class_conditional_gaussians(
    features: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    shrinkage: float = 0.10,
    diagonal: bool = False,
) -> Dict[str, np.ndarray]:
    """Fit class-conditional Gaussian geometry in embedding space."""
    d = features.shape[1]
    means = np.zeros((n_classes, d))
    covs = np.zeros((n_classes, d, d))

    global_cov = np.cov(features.T) + 1e-5 * np.eye(d)
    tau_global = np.trace(global_cov) / d

    for c in range(n_classes):
        Xc = features[labels == c]

        if len(Xc) == 0:
            means[c] = features.mean(axis=0)
            cov = global_cov.copy()
        elif len(Xc) == 1:
            means[c] = Xc[0]
            cov = global_cov.copy()
        else:
            means[c] = Xc.mean(axis=0)
            cov = np.cov(Xc.T) + 1e-5 * np.eye(d)

        if diagonal:
            cov = np.diag(np.diag(cov))

        tau = np.trace(cov) / d if np.isfinite(np.trace(cov)) else tau_global
        covs[c] = (1 - shrinkage) * cov + shrinkage * tau * np.eye(d)

    precisions = np.linalg.pinv(covs)
    return {"means": means, "precisions": precisions}


def mahalanobis_scores_all(
    features: np.ndarray, gaussian_geometry: Dict[str, np.ndarray]
) -> np.ndarray:
    means = gaussian_geometry["means"]
    precisions = gaussian_geometry["precisions"]
    n = features.shape[0]
    k = means.shape[0]
    scores = np.empty((n, k), dtype=float)

    for c in range(k):
        diff = features - means[c]
        scores[:, c] = np.einsum("ij,jk,ik->i", diff, precisions[c], diff)

    return scores


def mahalanobis_conformal_sets(
    train_features: np.ndarray,
    train_y: np.ndarray,
    cal_features: np.ndarray,
    cal_y: np.ndarray,
    test_features: np.ndarray,
    test_y: np.ndarray,
    alpha: float = 0.1,
    shrinkage: float = 0.10,
    diagonal: bool = False,
) -> Dict[str, float]:
    """Feature-geometric conformal classification."""
    n_classes = int(max(train_y.max(), cal_y.max(), test_y.max()) + 1)
    geometry = fit_class_conditional_gaussians(
        train_features,
        train_y,
        n_classes=n_classes,
        shrinkage=shrinkage,
        diagonal=diagonal,
    )

    cal_scores_all = mahalanobis_scores_all(cal_features, geometry)
    cal_scores = cal_scores_all[np.arange(len(cal_y)), cal_y]
    qhat = conformal_quantile(cal_scores, alpha)

    test_scores_all = mahalanobis_scores_all(test_features, geometry)
    included = test_scores_all <= qhat
    covered = included[np.arange(len(test_y)), test_y]
    sizes = included.sum(axis=1)

    return {
        "method": f"feature mahalanobis shrinkage={shrinkage:g} diagonal={diagonal}",
        "coverage": float(covered.mean()),
        "avg_size": float(sizes.mean()),
        "median_size": float(np.median(sizes)),
        "qhat": float(qhat),
    }


def run_image_conformal_from_arrays(
    train_features: np.ndarray,
    train_y: np.ndarray,
    cal_logits: np.ndarray,
    cal_features: np.ndarray,
    cal_y: np.ndarray,
    test_logits: np.ndarray,
    test_features: np.ndarray,
    test_y: np.ndarray,
    alpha: float = 0.1,
) -> pd.DataFrame:
    """Run image classification conformal methods after feature extraction."""
    rows = [
        softmax_conformal_sets(
            cal_logits, cal_y, test_logits, test_y, alpha=alpha, temperature=1.0
        ),
        softmax_conformal_sets(
            cal_logits, cal_y, test_logits, test_y, alpha=alpha, temperature=2.0
        ),
        aps_conformal_sets(
            cal_logits, cal_y, test_logits, test_y, alpha=alpha, temperature=1.0
        ),
        aps_conformal_sets(
            cal_logits, cal_y, test_logits, test_y, alpha=alpha, temperature=2.0
        ),
        mahalanobis_conformal_sets(
            train_features,
            train_y,
            cal_features,
            cal_y,
            test_features,
            test_y,
            alpha=alpha,
            shrinkage=0.10,
            diagonal=False,
        ),
        mahalanobis_conformal_sets(
            train_features,
            train_y,
            cal_features,
            cal_y,
            test_features,
            test_y,
            alpha=alpha,
            shrinkage=0.50,
            diagonal=True,
        ),
    ]
    return pd.DataFrame(rows).sort_values("avg_size")


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    alpha = 0.1

    # Focused example.
    df, cond = run_one_experiment(
        setting="heteroskedastic",
        model_family="rf",
        n=6000,
        d=5,
        alpha=alpha,
        seed=0,
    )
    print("\nSingle experiment summary:")
    print(df.to_string(index=False))

    print("\nConditional coverage tables:")
    for name, table in cond.items():
        print(f"\n{name}")
        print(table.to_string(index=False))

    # Repeated benchmark.
    raw, summary = run_grid(
        settings=["homoskedastic", "heteroskedastic", "heavy_tail", "skewed", "multimodal"],
        model_families=["rf", "krr", "mlp", "cqr_gbrt"],
        n=6000,
        d=5,
        alpha=alpha,
        seeds=[0, 1, 2],
        save_csv=True,
        output_prefix="ig_conformal_results",
    )

    print("\nRepeated benchmark summary:")
    print(summary.to_string(index=False))

    compact = summarize_geometry_effect(summary)
    compact.to_csv("ig_conformal_geometry_effect.csv", index=False)

    print("\nCompact geometry-effect summary:")
    print(compact.to_string(index=False))

    multimodal_diag = summarize_multimodal_density_diagnostics(summary)
    multimodal_diag.to_csv("ig_conformal_multimodal_density_diagnostics.csv", index=False)

    print("\nMultimodal conditional-density diagnostics:")
    print(multimodal_diag.to_string(index=False))

    # Just to demonstrate why Student-t/Huber are omitted from the default table.
    # diag = run_rank_equivalence_diagnostics(setting="heavy_tail", model_family="rf")
    # print("\nRank-equivalence diagnostic:")
    # print(diag.to_string(index=False))


# =============================================================================
# PyTorch extraction sketch for ResNet/VGG
# =============================================================================
"""
Example usage with PyTorch / torchvision:

    import torch
    import torchvision.models as models

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.eval().cuda()

    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    classifier = model.fc

    def extract_logits_features(loader):
        logits_list, features_list, y_list = [], [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.cuda()
                feat = feature_extractor(x).flatten(1)
                logits = classifier(feat)
                logits_list.append(logits.cpu().numpy())
                features_list.append(feat.cpu().numpy())
                y_list.append(y.numpy())
        return (
            np.concatenate(logits_list),
            np.concatenate(features_list),
            np.concatenate(y_list),
        )

    train_logits, train_features, train_y = extract_logits_features(train_loader)
    cal_logits, cal_features, cal_y = extract_logits_features(cal_loader)
    test_logits, test_features, test_y = extract_logits_features(test_loader)

    results = run_image_conformal_from_arrays(
        train_features=train_features,
        train_y=train_y,
        cal_logits=cal_logits,
        cal_features=cal_features,
        cal_y=cal_y,
        test_logits=test_logits,
        test_features=test_features,
        test_y=test_y,
        alpha=0.1,
    )

For VGG, use the activations before the final classifier as features. The same
conformal functions apply unchanged.
"""
