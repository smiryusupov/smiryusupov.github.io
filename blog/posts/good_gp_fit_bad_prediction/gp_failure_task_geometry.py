
"""
Helper code for the Quarto note:
Good GP fit, bad prediction: a task-geometry view of Gaussian processes.

this file intentionally keeps everything in one place:
- kernels
- GP fitting
- Fisher/Rao diagnostics
- hidden-gap and extrapolation experiments
- figure/table generation for the QMD note

Usage from the QMD:
    import gp_failure_task_geometry as gp
    results = gp.run_all(output_dir="figures")
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import eigvalsh


# ---------------------------------------------------------------------
# Kernels and covariance geometry
# ---------------------------------------------------------------------

KERNELS = ["RBF", "Exponential", "Matern32", "Matern52"]


def kernel_R_and_dlogell(name: str, D: np.ndarray, ell: float) -> tuple[np.ndarray, np.ndarray]:
    """Return correlation R(D; ell) and derivative dR/d log ell."""
    r = D

    if name == "RBF":
        q = (r / ell) ** 2
        R = np.exp(-0.5 * q)
        dR = R * q

    elif name == "Exponential":
        a = r / ell
        R = np.exp(-a)
        dR = R * a

    elif name == "Matern32":
        a = np.sqrt(3.0) * r / ell
        R = (1.0 + a) * np.exp(-a)
        dR = (a ** 2) * np.exp(-a)

    elif name == "Matern52":
        a = np.sqrt(5.0) * r / ell
        R = (1.0 + a + a * a / 3.0) * np.exp(-a)
        dR = ((a * a + a ** 3) / 3.0) * np.exp(-a)

    else:
        raise ValueError(f"Unknown kernel: {name}")

    return R, dR


def cov_and_derivs(name: str, X: np.ndarray, logtheta: np.ndarray, jitter: float = 1e-8):
    """
    Training covariance C_X(theta) and derivatives with respect to
    theta=(log sigma_f, log ell, log sigma_n).
    """
    log_sf, log_ell, log_sn = logtheta

    sf2 = np.exp(2.0 * log_sf)
    ell = np.exp(log_ell)
    sn2 = np.exp(2.0 * log_sn)

    D = np.abs(X[:, None] - X[None, :])
    R, dR = kernel_R_and_dlogell(name, D, ell)

    K = sf2 * R
    I = np.eye(len(X))

    C = K + (sn2 + jitter) * I

    d_log_sf = 2.0 * K
    d_log_ell = sf2 * dR
    d_log_sn = 2.0 * sn2 * I

    return C, [d_log_sf, d_log_ell, d_log_sn]


def fisher_metric_from_cov(C: np.ndarray, derivs: list[np.ndarray]) -> np.ndarray:
    """Fisher--Rao metric for a zero-mean Gaussian covariance model."""
    CinvD = [np.linalg.solve(C, D) for D in derivs]
    p = len(derivs)
    g = np.zeros((p, p))

    for a in range(p):
        for b in range(a, p):
            g[a, b] = g[b, a] = 0.5 * np.trace(CinvD[a] @ CinvD[b])

    return 0.5 * (g + g.T)


def fisher_metric(name: str, X: np.ndarray, logtheta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    C, derivs = cov_and_derivs(name, X, logtheta)
    return fisher_metric_from_cov(C, derivs), C


def joint_cov_and_derivs(
    name: str,
    X: np.ndarray,
    T: np.ndarray,
    logtheta: np.ndarray,
    jitter: float = 1e-8,
):
    """
    Joint covariance of (y_X, f_T), where y_X has observation noise
    and f_T is latent/noiseless.
    """
    log_sf, log_ell, log_sn = logtheta

    sf2 = np.exp(2.0 * log_sf)
    ell = np.exp(log_ell)
    sn2 = np.exp(2.0 * log_sn)

    U = np.r_[X, T]
    D = np.abs(U[:, None] - U[None, :])
    R, dR = kernel_R_and_dlogell(name, D, ell)

    K = sf2 * R
    C = K.copy()

    n = len(X)
    C[:n, :n] += (sn2 + jitter) * np.eye(n)
    C += jitter * np.eye(len(U))

    d_log_sf = 2.0 * K
    d_log_ell = sf2 * dR
    d_log_sn = np.zeros_like(C)
    d_log_sn[:n, :n] = 2.0 * sn2 * np.eye(n)

    return C, [d_log_sf, d_log_ell, d_log_sn]


# ---------------------------------------------------------------------
# GP fitting and prediction
# ---------------------------------------------------------------------

def nll(name: str, X: np.ndarray, y: np.ndarray, logtheta: np.ndarray) -> float:
    C, _ = cov_and_derivs(name, X, logtheta)

    try:
        L = np.linalg.cholesky(C)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        value = 0.5 * y @ alpha
        value += np.sum(np.log(np.diag(L)))
        value += 0.5 * len(X) * np.log(2.0 * np.pi)
        return float(value)
    except np.linalg.LinAlgError:
        return float("inf")


def fit_gp(name: str, X: np.ndarray, y: np.ndarray):
    bounds = [(-4.0, 2.2), (-6.0, 1.5), (-7.0, -0.7)]
    empirical_sf = max(float(np.std(y)), 1e-3)

    starts = [
        [np.log(empirical_sf), np.log(0.025), np.log(0.04)],
        [np.log(empirical_sf), np.log(0.060), np.log(0.04)],
        [np.log(empirical_sf), np.log(0.120), np.log(0.04)],
        [np.log(empirical_sf), np.log(0.250), np.log(0.04)],
        [0.0, np.log(0.100), np.log(0.03)],
    ]

    best = None

    for start in starts:
        result = minimize(
            lambda th: nll(name, X, y, th),
            np.array(start),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 220, "ftol": 1e-8},
        )

        if best is None or result.fun < best.fun:
            best = result

    return best


def gp_predict(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    Xs: np.ndarray,
    logtheta: np.ndarray,
):
    """Posterior mean and marginal variance of latent f(Xs)."""
    C, _ = cov_and_derivs(name, X, logtheta)
    L = np.linalg.cholesky(C)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))

    log_sf, log_ell, _ = logtheta
    sf2 = np.exp(2.0 * log_sf)
    ell = np.exp(log_ell)

    Dsx = np.abs(Xs[:, None] - X[None, :])
    Rtx, _ = kernel_R_and_dlogell(name, Dsx, ell)
    Ktx = sf2 * Rtx

    mean = Ktx @ alpha
    v = np.linalg.solve(L, Ktx.T)
    var = np.maximum(sf2 - np.sum(v * v, axis=0), 1e-12)

    return mean, var


def gp_predict_joint(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    logtheta: np.ndarray,
):
    """Posterior mean and full covariance of latent f_T | y_X."""
    Cx, _ = cov_and_derivs(name, X, logtheta)
    L = np.linalg.cholesky(Cx)

    log_sf, log_ell, _ = logtheta
    sf2 = np.exp(2.0 * log_sf)
    ell = np.exp(log_ell)

    Dtx = np.abs(T[:, None] - X[None, :])
    Rtx, _ = kernel_R_and_dlogell(name, Dtx, ell)
    Ktx = sf2 * Rtx

    Dtt = np.abs(T[:, None] - T[None, :])
    Rtt, _ = kernel_R_and_dlogell(name, Dtt, ell)
    Ktt = sf2 * Rtt + 1e-8 * np.eye(len(T))

    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    mean = Ktx @ alpha

    V = np.linalg.solve(L, Ktx.T)
    cov = Ktt - V.T @ V
    cov = 0.5 * (cov + cov.T) + 1e-8 * np.eye(len(T))

    return mean, cov, Ktt


# ---------------------------------------------------------------------
# Task-geometry diagnostics
# ---------------------------------------------------------------------

def logdet_spd(A: np.ndarray) -> float:
    sign, val = np.linalg.slogdet(0.5 * (A + A.T))
    return float(val) if sign > 0 else np.nan


def generalized_eigs(A: np.ndarray, B: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """Eigenvalues of B^{-1/2} A B^{-1/2} via generalized eigenproblem."""
    Areg = 0.5 * (A + A.T)
    Breg = 0.5 * (B + B.T) + floor * np.eye(B.shape[0])
    vals = eigvalsh(Areg, Breg)
    return np.maximum(vals, 0.0)


def predictive_metric(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    logtheta: np.ndarray,
    h: float = 1e-3,
):
    """
    Fisher metric of the posterior predictive Gaussian
    p_theta(f_T | y_X) = N(m_T(theta), S_T(theta)).
    """
    m0, S0, _ = gp_predict_joint(name, X, y, T, logtheta)
    Sinv = np.linalg.inv(S0)

    dm = []
    dS = []

    for a in range(3):
        step = np.zeros(3)
        step[a] = h

        mp, Sp, _ = gp_predict_joint(name, X, y, T, logtheta + step)
        mm, Sm, _ = gp_predict_joint(name, X, y, T, logtheta - step)

        dm.append((mp - mm) / (2.0 * h))
        dS.append((Sp - Sm) / (2.0 * h))

    H = np.zeros((3, 3))

    for a in range(3):
        for b in range(a, 3):
            mean_part = dm[a].T @ Sinv @ dm[b]
            cov_part = 0.5 * np.trace(Sinv @ dS[a] @ Sinv @ dS[b])
            H[a, b] = H[b, a] = mean_part + cov_part

    return 0.5 * (H + H.T)


def kernel_connection(name: str, X: np.ndarray, T: np.ndarray, logtheta: np.ndarray) -> np.ndarray:
    """Maximum normalized kernel correlation from each target point to the design."""
    ell = np.exp(logtheta[1])
    D = np.abs(T[:, None] - X[None, :])
    R, _ = kernel_R_and_dlogell(name, D, ell)
    return np.max(R, axis=1)


def task_geometry_diagnostics(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    truth_T: np.ndarray,
    logtheta: np.ndarray,
) -> dict:
    """
    Diagnostics computed from the fitted GP and the target region T.

    The truth_T quantities are used only for evaluation metrics, not for
    the geometry warning itself.
    """
    Cx, Dx = cov_and_derivs(name, X, logtheta)
    gX = fisher_metric_from_cov(Cx, Dx)

    Caug, Daug = joint_cov_and_derivs(name, X, T, logtheta)
    gXT = fisher_metric_from_cov(Caug, Daug)

    delta_g = 0.5 * ((gXT - gX) + (gXT - gX).T)
    missing_vals = generalized_eigs(delta_g, gX)

    Hpred = predictive_metric(name, X, y, T, logtheta)
    pred_vals = generalized_eigs(Hpred, gX)

    mT, post_cov, prior_cov = gp_predict_joint(name, X, y, T, logtheta)
    stdT = np.sqrt(np.diag(post_cov))

    prior_ld = logdet_spd(prior_cov)
    post_ld = logdet_spd(post_cov)

    coverage = np.mean((truth_T >= mT - 1.96 * stdT) & (truth_T <= mT + 1.96 * stdT))
    rmse_T = np.sqrt(np.mean((mT - truth_T) ** 2))

    train_m, _ = gp_predict(name, X, y, X, logtheta)
    train_rmse = np.sqrt(np.mean((train_m - y) ** 2))

    conn = kernel_connection(name, X, T, logtheta)

    return {
        "train_rmse": float(train_rmse),
        "target_rmse": float(rmse_T),
        "target_95_coverage": float(coverage),
        "cond_g_train": float(np.linalg.cond(gX)),
        "missing_info_lambda_max": float(np.max(missing_vals)),
        "missing_info_trace": float(np.sum(missing_vals)),
        "predictive_sensitivity_lambda_max": float(np.max(pred_vals)),
        "predictive_sensitivity_trace": float(np.sum(pred_vals)),
        "target_info_gain_per_point": float(0.5 * (prior_ld - post_ld) / len(T)),
        "avg_kernel_connection_target": float(np.mean(conn)),
        "min_kernel_connection_target": float(np.min(conn)),
        "max_kernel_connection_target": float(np.max(conn)),
    }


# ---------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------

def f_hidden_bump(x: np.ndarray) -> np.ndarray:
    return (
        np.sin(2.0 * np.pi * x)
        + 0.35 * np.sin(4.0 * np.pi * x)
        + 2.35 * np.exp(-0.5 * ((x - 0.50) / 0.045) ** 2)
    )


def make_hidden_gap():
    rng = np.random.default_rng(11)
    X = np.r_[np.linspace(0.00, 0.37, 24), np.linspace(0.63, 1.00, 24)]
    y = f_hidden_bump(X) + 0.045 * rng.normal(size=len(X))

    Xs = np.linspace(0, 1, 360)
    truth = f_hidden_bump(Xs)

    T = np.linspace(0.40, 0.60, 35)
    truth_T = f_hidden_bump(T)

    return X, y, Xs, truth, T, truth_T


def f_train_world(x: np.ndarray) -> np.ndarray:
    return np.sin(2.0 * np.pi * x) + 0.25 * np.sin(5.0 * np.pi * x) + 0.35 * x


def f_extrapolation(x: np.ndarray) -> np.ndarray:
    return f_train_world(x) + 5.0 * np.maximum(x - 1.0, 0.0) ** 2 + 2.3 * np.maximum(x - 1.0, 0.0)


def make_extrapolation():
    rng = np.random.default_rng(33)
    X = np.linspace(0, 1, 72)
    y = f_extrapolation(X) + 0.045 * rng.normal(size=len(X))

    Xs = np.linspace(0, 1.5, 420)
    truth = f_extrapolation(Xs)

    T = np.linspace(1.02, 1.50, 35)
    truth_T = f_extrapolation(T)

    return X, y, Xs, truth, T, truth_T


def run_experiment(experiment: str, output_dir: str | Path = "figures") -> pd.DataFrame:
    """
    Run one experiment and save figures.

    experiment is one of:
        "hidden_gap", "extrapolation"
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if experiment == "hidden_gap":
        X, y, Xs, truth, T, truth_T = make_hidden_gap()
        title = "Hidden-gap failure"
        shade = (0.40, 0.60)
        target_label = "hidden gap"

    elif experiment == "extrapolation":
        X, y, Xs, truth, T, truth_T = make_extrapolation()
        title = "Extrapolation failure"
        shade = (1.00, 1.50)
        target_label = "extrapolation region"

    else:
        raise ValueError("experiment must be 'hidden_gap' or 'extrapolation'")

    rows = []
    predictions = {}

    for kernel in KERNELS:
        fit = fit_gp(kernel, X, y)
        theta = fit.x

        mean, var = gp_predict(kernel, X, y, Xs, theta)
        predictions[kernel] = (mean, var, theta)

        diag = task_geometry_diagnostics(kernel, X, y, T, truth_T, theta)
        diag.update(
            {
                "experiment": experiment,
                "kernel": kernel,
                "sigma_f_hat": float(np.exp(theta[0])),
                "ell_hat": float(np.exp(theta[1])),
                "sigma_n_hat": float(np.exp(theta[2])),
                "nll": float(fit.fun),
            }
        )
        rows.append(diag)

    df = pd.DataFrame(rows)

    # Prediction figure.
    plt.figure(figsize=(9.2, 5.5))
    plt.plot(Xs, truth, label="truth")
    plt.axvspan(shade[0], shade[1], alpha=0.15, label=target_label)
    plt.scatter(X, y, s=14, label="observations")

    for kernel in KERNELS:
        plt.plot(Xs, predictions[kernel][0], label=kernel)

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(title + ": good observed fit, bad task prediction")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(output_dir / f"{experiment}_prediction_means.png", dpi=180)
    plt.close()

    # Train versus target RMSE.
    xloc = np.arange(len(df))
    width = 0.36

    plt.figure(figsize=(8.4, 5.2))
    plt.bar(xloc - width / 2, df["train_rmse"], width, label="training RMSE")
    plt.bar(xloc + width / 2, df["target_rmse"], width, label="target-region RMSE")
    plt.xticks(xloc, df["kernel"])
    plt.ylabel("RMSE")
    plt.title(title + ": fit quality is not task safety")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{experiment}_train_vs_target_rmse.png", dpi=180)
    plt.close()

    # Projection loss.
    plt.figure(figsize=(8.4, 5.2))
    plt.bar(df["kernel"], df["missing_info_lambda_max"])
    plt.yscale("log")
    plt.ylabel(r"$\lambda_{\max}(g_X^{-1}(g_{X,T}-g_X))$")
    plt.title(title + ": projection-loss score")
    plt.tight_layout()
    plt.savefig(output_dir / f"{experiment}_projection_loss.png", dpi=180)
    plt.close()

    # Predictive sensitivity.
    plt.figure(figsize=(8.4, 5.2))
    plt.bar(df["kernel"], df["predictive_sensitivity_lambda_max"])
    plt.yscale("log")
    plt.ylabel(r"$\lambda_{\max}(g_X^{-1}h_T)$")
    plt.title(title + ": predictive sensitivity per training Rao unit")
    plt.tight_layout()
    plt.savefig(output_dir / f"{experiment}_predictive_sensitivity.png", dpi=180)
    plt.close()

    # Target coverage.
    plt.figure(figsize=(8.4, 5.2))
    plt.bar(df["kernel"], df["target_95_coverage"])
    plt.ylim(0, 1.05)
    plt.ylabel("95% target-region coverage")
    plt.title(title + ": coverage in the target region")
    plt.tight_layout()
    plt.savefig(output_dir / f"{experiment}_target_coverage.png", dpi=180)
    plt.close()

    return df


def run_all(output_dir: str | Path = "figures") -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hidden = run_experiment("hidden_gap", output_dir=output_dir)
    extra = run_experiment("extrapolation", output_dir=output_dir)

    df = pd.concat([hidden, extra], ignore_index=True)
    df.to_csv(output_dir / "task_geometry_summary.csv", index=False)

    # Summary scatter.
    plt.figure(figsize=(7.5, 5.2))

    for experiment in ["hidden_gap", "extrapolation"]:
        sub = df[df["experiment"] == experiment]
        plt.scatter(
            sub["predictive_sensitivity_lambda_max"],
            sub["target_rmse"],
            s=70,
            label=experiment,
        )

        for _, row in sub.iterrows():
            plt.annotate(
                row["kernel"],
                (row["predictive_sensitivity_lambda_max"], row["target_rmse"]),
                fontsize=8,
            )

    plt.xscale("log")
    plt.xlabel(r"$\lambda_{\max}(g_X^{-1}h_T)$")
    plt.ylabel("target-region RMSE")
    plt.title("Geometry warning versus task failure")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "geometry_warning_vs_target_rmse.png", dpi=180)
    plt.close()

    return df


if __name__ == "__main__":
    summary = run_all(output_dir="figures")
    print(summary)
