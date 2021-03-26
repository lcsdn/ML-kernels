from numpy import ndarray

def accuracy(y_true: ndarray, y_pred: ndarray) -> float:
    """Compute the proportion of accurately predicted labels."""
    acc = (y_true == y_pred).astype(int).mean()
    return acc