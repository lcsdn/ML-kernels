import numpy as np

from .functions import sigmoid

class LinearClassifier:
    """General class for a linear classifier."""
    method = "linear"
    
    def __init__(self, reg_param=1e-5, fit_intercept=True):
        self.reg_param = reg_param
        self.fit_intercept = fit_intercept

    def augment(self, X):
        n = len(X)
        if self.fit_intercept:
            X = np.hstack([np.ones((n, 1)), X])
        return X
    
    def predict(self, X):
        X = self.augment(X)
        scores = X.dot(self.coef)
        y_pred = (scores >= self.threshold).astype(int)
        return y_pred
        
class RidgeRegressionClassifier(LinearClassifier):
    """
    Ridge regression classifier, solve a linear system to find the parameters w
    minimising :

    1/n ||y - Xw||_2^2 + lambda * ||w||_2^2
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = 0.5
    
    def fit(self, X, y):
        X = self.augment(X)
        n, p = X.shape
        hessian = X.T @ X + self.reg_param * np.eye(p)
        self.coef = np.linalg.solve(hessian, X.T.dot(y))
        return self
        
# TODO don't forger 1/2 in reg
class LogisticRegression(LinearClassifier):
    """
    Logistic regression classifier, use gradient descent to find the parameters w
    minimising :

    1/n sum (y_i log(sigmoid(Xw)) + (1-y_i) log(1-sigmoid(Xw))) + lambda * 1/2 ||w||_2^2
    """
    def __init__(self, max_iter=10000, stop_threshold=1e-5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_iter = max_iter
        self.stop_threshold = stop_threshold
        self.threshold = 0
    
    def compute_derivatives(self, X, y):
        """Compute the gradient and hessian matrix of the loss for current parameters."""
        n, p = X.shape
        probas = sigmoid(X.dot(self.coef))
        difference = probas - y
        gradient = (difference.reshape(-1, 1) * X).mean(axis=0)
        gradient += self.reg_param * self.coef
        sigma = np.diag(probas * (1-probas))
        hessian = X.T @ sigma @ X / n
        hessian += self.reg_param * np.eye(p)
        return gradient, hessian
    
    def fit(self, X, y):
        X = self.augment(X)
        self.coef = np.zeros(X.shape[1])
        
        for t in range(self.max_iter):
            old_coef = self.coef.copy()
            gradient, hessian = self.compute_derivatives(X, y)
            self.coef -= np.linalg.solve(hessian, gradient)
            if np.linalg.norm(self.coef - old_coef) < self.stop_threshold:
                break
        self.last_iter = t
        return self