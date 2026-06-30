"""Regression metrics for drug-target affinity prediction.

Mirrors the metrics used by DeepDTAGen (see ../utils.py) so the simplified
models are evaluated on exactly the same footing as the baseline.
"""
import numpy as np
from math import sqrt
from scipy import stats


def mse(y, f):
    return ((y - f) ** 2).mean(axis=0)


def rmse(y, f):
    return sqrt(((y - f) ** 2).mean(axis=0))


def pearson(y, f):
    return np.corrcoef(y, f)[0, 1]


def spearman(y, f):
    return stats.spearmanr(y, f)[0]


def get_cindex(Y, P):
    """Concordance index: probability that predicted order matches true order."""
    P = P[:, np.newaxis] - P
    P = np.float32(P == 0) * 0.5 + np.float32(P > 0)
    Y = Y[:, np.newaxis] - Y
    Y = np.tril(np.float32(Y > 0), 0)
    P_sum = np.sum(P * Y)
    Y_sum = np.sum(Y)
    return 0.0 if Y_sum == 0 else P_sum / Y_sum


def _r_squared_error(y_obs, y_pred):
    y_obs, y_pred = np.array(y_obs), np.array(y_pred)
    y_obs_mean = np.mean(y_obs)
    y_pred_mean = np.mean(y_pred)
    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean)) ** 2
    y_obs_sq = sum((y_obs - y_obs_mean) ** 2)
    y_pred_sq = sum((y_pred - y_pred_mean) ** 2)
    return mult / float(y_obs_sq * y_pred_sq)


def _squared_error_zero(y_obs, y_pred):
    y_obs, y_pred = np.array(y_obs), np.array(y_pred)
    k = sum(y_obs * y_pred) / float(sum(y_pred * y_pred))
    y_obs_mean = np.mean(y_obs)
    upp = sum((y_obs - (k * y_pred)) ** 2)
    down = sum((y_obs - y_obs_mean) ** 2)
    return 1 - (upp / float(down))


def get_rm2(ys_orig, ys_line):
    r2 = _r_squared_error(ys_orig, ys_line)
    r02 = _squared_error_zero(ys_orig, ys_line)
    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))


def all_metrics(G, P):
    """Return the standard DTA metric bundle as a dict."""
    return {
        "MSE": float(mse(G, P)),
        "RMSE": float(rmse(G, P)),
        "CI": float(get_cindex(G, P)),
        "rm2": float(get_rm2(G, P)),
        "Pearson": float(pearson(G, P)),
        "Spearman": float(spearman(G, P)),
    }
