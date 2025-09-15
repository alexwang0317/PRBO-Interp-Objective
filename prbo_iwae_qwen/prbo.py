# log-sum-exp, IWAE curve, etc.

# prbo.py
import math, numpy as np

def logsumexp(a, axis=None):
    a = np.asarray(a, dtype=float)
    a_max = np.max(a, axis=axis, keepdims=True)
    return np.log(np.sum(np.exp(a - a_max), axis=axis)) + np.squeeze(a_max, axis=axis)

def iwae_curve(logw: np.ndarray, ks, resamples: int = 100, seed: int = 0):
    rng = np.random.default_rng(seed)
    curves = {}
    n = len(logw)
    for k in ks:
        vals = []
        for _ in range(resamples):
            idx = rng.integers(0, n, size=k)
            vals.append(logsumexp(logw[idx]) - math.log(k))
        curves[k] = float(np.mean(vals))
    return curves

def prbo_expectation(logw: np.ndarray):
    """
    Calculate simple PRBO expectation: E[log p - log q + s]
    
    Args:
        logw: Array of log weights (log p - log q + s) for each sample
    
    Returns:
        float: Simple expectation of the log weights
    """
    finite_logw = logw[np.isfinite(logw)]
    if len(finite_logw) == 0:
        return float('-inf')
    return float(np.mean(finite_logw))

