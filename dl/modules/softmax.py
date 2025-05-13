import numpy as np


# refactor to module
def softmax(X):
    m = np.max(X, axis=1)[:, None]
    sm_numerators = np.exp(X - m)
    sm_denominators = np.sum(sm_numerators, axis=1)[:, None]
    return sm_numerators / sm_denominators
