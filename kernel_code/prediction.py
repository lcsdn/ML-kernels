import pandas as pd
from os import path
import pickle
import yaml

def predictions(
        ModelClass,
        X_train_list,
        y_train_list,
        X_test_list,
        kwargs_models=[{}]*3,
        preprocesser=None
    ):
    """
    Compute the predictions of a model on a list of datasets for specified
    hyperparameters.
    """
    y_test = []
    for i in range(3):
        X_train, y_train, X_test, kwargs_model = X_train_list[i], y_train_list[i], X_test_list[i], kwargs_models[i]
        if preprocesser is not None:
            preprocess = preprocesser()
            X_train = preprocess.fit(X_train).transform(X_train)
            X_test = preprocess.transform(X_test)
        model = ModelClass(**kwargs_model)
        model.fit(X_train, y_train)
        y_test.extend(model.predict(X_test))
    df = pd.DataFrame({
        "Id": range(len(y_test)),
        "Bound": y_test
    })
    return df

def save(name_predictions, folder_predictions, predictions_df, searcher=None):
    predictions_df.to_csv(path.join(folder_predictions, name_predictions+".csv"), index=False)
    if searcher is not None:
        with open(path.join(folder_predictions, name_predictions+".txt"), "w") as f:
            f.write(str(searcher))
        with open(path.join(folder_predictions, name_predictions+".p"), 'wb') as f:
            pickle.dump(searcher.studies, f)