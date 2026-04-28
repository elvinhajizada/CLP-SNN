"""
Utility helpers used by the streaming classifiers.
Adapted from Tyler Hayes' Embedded-CL (https://github.com/tyler-hayes/Embedded-CL).
"""
import numpy as np


class CMA(object):
    """Continual moving average for tracking loss updates."""

    def __init__(self):
        self.N = 0
        self.avg = 0.0

    def update(self, X):
        self.avg = (X + self.N * self.avg) / (self.N + 1)
        self.N = self.N + 1


def randint(max_val, num_samples):
    """Return num_samples unique random integers in range(max_val)."""
    rand_vals = {}
    _num_samples = min(max_val, num_samples)
    while True:
        _rand_vals = np.random.randint(0, max_val, num_samples)
        for r in _rand_vals:
            rand_vals[r] = r
            if len(rand_vals) >= _num_samples:
                break
        if len(rand_vals) >= _num_samples:
            break
    return rand_vals.keys()
