import numpy as np


# incomplete
def softmax(X):
    m = np.max(X, axis=1, keepdims=True)
    sm_numerators = np.exp(X - m)
    sm_denominators = np.sum(sm_numerators, axis=1, keepdims=True)
    return sm_numerators / sm_denominators
