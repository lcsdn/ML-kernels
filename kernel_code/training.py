import numpy as np
from tqdm import tqdm

from .metrics import accuracy

def train_model(ModelClass, X, y, **kwargs):
    """Initialise and train a model on labeled data."""
    model = ModelClass(**kwargs)
    model.fit(X, y)
    y_pred = model.predict(X)
    return model, accuracy(y, y_pred)
    
def validate_model(model, X_val, y_val):
    """Validate a model on validation data."""
    y_pred = model.predict(X_val)
    return accuracy(y_val, y_pred)

def train_and_validate_model(ModelClass, X, y, kwargs_model={}, kwargs_split={}):
    """
    Given a model class and labeled data, train and validate the model
    on the data.
    """
    X_train, y_train, X_val, y_val = train_val_split(X, y, **kwargs_split)
    model, train_acc = train_model(ModelClass, X_train, y_train, **kwargs_model)
    val_acc = validate_model(model, X_val, y_val)
    return model, train_acc, val_acc

def train_val_split(X, y, train_proportion=0.8, random_state=None):
    """
    Split labeled data into a training and validation set, with given proportion of
    training samples.
    """
    if random_state is not None:
        np.random.seed(random_state)
    n = len(X)
    
    # Split the data
    train_choice = np.random.choice(n, size=int(train_proportion*n), replace=False)
    train_indices = np.zeros(n, dtype=bool)
    train_indices[train_choice] = True
    
    # Return the split
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[~train_indices], y[~train_indices]
    return X_train, y_train, X_val, y_val
    
def stratified_binary_k_fold_split(X, y, k=3, random_state=None):
    """
    Python generator.
    
    Split labeled data into k folds such that the distribution of the labels
    in each fold closely matches the distribution of the labels in the data.
    """
    if random_state is not None:
        np.random.seed(random_state)
    assert k >= 2
    n = len(X)
    indices_neg = np.where(y == 0)[0]
    indices_pos = np.where(y == 1)[0]
    
    # Shuffle the data
    np.random.shuffle(indices_neg)
    np.random.shuffle(indices_pos)
    
    # Select the folds
    folds_neg = np.array_split(indices_neg, k)
    folds_pos = np.array_split(indices_pos, k)
    folds = [np.hstack([folds_neg[i], folds_pos[i]]) for i in range(k)]
    
    # Return a fold
    for fold in folds:
        train_indices = np.ones(n, dtype=bool)
        train_indices[fold] = False
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[~train_indices], y[~train_indices]
        yield X_train, y_train, X_val, y_val, train_indices

def cross_validate(
        ModelClass,
        X,
        y,
        k=3,
        preprocesser=None,
        random_state=None,
        show_progression=True,
        K=None,
        **kwargs_model
    ):
    """
    Compute an estimation of the predictive accuracy using cross validation.
    """
    train_acc = np.zeros(k)
    val_acc = np.zeros(k)
    best_val_acc = 0
    best_model = None
    folds = stratified_binary_k_fold_split(X, y, k, random_state=random_state)
    
    if ModelClass.method == "kernel" and K is None:
        kernel = kwargs_model.get("kernel")
        K = kernel.pairwise_matrix(X)

    _tqdm = tqdm if show_progression else lambda x: x
    
    for i, (X_train, y_train, X_val, y_val, train_indices) in _tqdm(enumerate(folds)):
        # Preprocessing the data
        if preprocesser is not None:
            preprocess = preprocesser()
            X_train = preprocess.fit(X_train).transform(X_train)
            X_val = preprocess.transform(X_val)
        
        # Model initialisation
        model = ModelClass(**kwargs_model)
        kwargs_train, kwargs_val = {}, {}
        
        # Kernel matrix splitting
        if ModelClass.method == "kernel":
            K_train = K[train_indices][:, train_indices]
            K_val = K[~train_indices][:, train_indices]
            kwargs_train["K"] = K_train
            kwargs_val["K"] = K_val
        
        # Train and predict
        model.fit(X_train, y_train, **kwargs_train)
        y_train_pred = model.predict(X_train, **kwargs_train)
        y_val_pred = model.predict(X_val, **kwargs_val)
        train_acc[i] = accuracy(y_train, y_train_pred)
        val_acc[i] = accuracy(y_val, y_val_pred)
        
        # Save best model
        if val_acc[i] > best_val_acc:
            best_val_acc = val_acc[i]
            best_model = model
            
    return best_model, train_acc.mean(), val_acc.mean()