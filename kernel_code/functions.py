import numpy as np

def sigmoid(t):
    return 1 / (1 + np.exp(-t))

def softmax(scores):
    exp_scores = np.exp(scores - scores.max())
    softmax = exp_scores / exp_scores.sum()
    return softmax