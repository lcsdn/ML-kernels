from numpy import ndarray

class Normalise:
    """
    Normalise a data matrix by substracting by its mean and dividing by its
    standard deviation.
    """
    def fit(self, X: ndarray):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        return self
    
    def transform(self, X: ndarray) -> ndarray:
        X_transformed = (X - self.mean) / self.std
        return X_transformed

def zeros_to_minus_ones(y: ndarray) -> ndarray:
    """
    Change labels of negative class from 0 to -1.
    """
    y_new = y.copy()
    y_new[y_new == 0] = -1
    return y_new