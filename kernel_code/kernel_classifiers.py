import numpy as np
import cvxopt

from .functions import sigmoid
from .preprocessing import zeros_to_minus_ones

class KernelClassifier:
    method = "kernel"
    
    def __init__(self, kernel, reg_param=1e-5):
        self.kernel = kernel
        self.reg_param = reg_param
    
    def fit(self, X, y, K=None):
        self.X = X
        if K is None:
            K = self.kernel.pairwise_matrix(X)
        self.coef = self._estimate_coef(K, y)
        return self
    
    def predict(self, X_new, K=None):
        if K is None:
            K = self.kernel.pairwise_matrix(X_new, self.X)
        scores = K.dot(self.coef)
        y_pred = (scores >= self.threshold).astype(int)
        return y_pred

class KernelRidgeRegressionClassifier(KernelClassifier):
    """
    Ridge regression classifier, solve a linear system to find the parameters alpha
    minimising :

    1/n ||y - Kalpha||_2^2 + lambda * alpha^T K alpha
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = 0
        
    def _estimate_coef(self, K, y):
        n = len(K)
        hessian = K + n * self.reg_param * np.eye(n)
        coef = np.linalg.solve(hessian, y)
        return coef

class KernelLogisticRegression(KernelClassifier):
    """
    Logistic regression classifier, use gradient descent to find the parameters alpha
    minimising :

    1/n sum (y_i log(sigmoid(K_i^T alpha)) + (1-y_i) log(1-sigmoid(K_i^T alpha))) + lambda * 1/2 alpha^T K alpha
    """
    def __init__(self, max_iter=10000, stop_threshold=1e-5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_iter = max_iter
        self.stop_threshold = stop_threshold
        self.threshold = 0
        
    def _estimate_coef(self, K, y):
        n = len(K)
        coef = np.zeros(n)
        step = 1 / (np.linalg.norm(K, 2)**2 * (1/n + 1/4))
        
        for t in range(self.max_iter):
            old_coef = coef.copy()
            probas = sigmoid(K.dot(coef))
            gradient = K.dot((probas - y) / n + coef)
            coef -= step * gradient
            if np.linalg.norm(coef - old_coef) < self.stop_threshold:
                break
        return coef

class KernelSVM(KernelClassifier): #TODO
    """
    Use quadratic programming to minimise:
    TODO
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = 0
        
    def _estimate_coef(self, K, y):
        n = len(K)
        C = 1 / (2 * self.reg_param * n)
        y = zeros_to_minus_ones(y).astype(float)
        
        P = cvxopt.matrix(K)
        q = cvxopt.matrix(-y)
        if C == np.inf:
            G = - np.diag(y)
            h = np.zeros(n)
        else:
            G = np.vstack([-np.diag(y), np.diag(y)])
            h = np.hstack([np.zeros(n), C*np.ones(n)])
        G = cvxopt.matrix(G)
        h = cvxopt.matrix(h)

        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P, q, G, h)
        coef = np.array(solution['x']).flatten()
        return coef