import numpy as np
import pandas as pd
import yaml
from os import path

from .functions import softmax
    
def average_vote_method(predictions_on_set, scores_on_set):
    weighted_predictions = predictions_on_set.mean(axis=0)
    voted_predictions = (weighted_predictions >= 0.5).astype(int)
    return voted_predictions    
    
def weighted_average_vote_method(predictions_on_set, scores_on_set):
    weights = scores_on_set / scores_on_set.sum()
    weighted_predictions = predictions_on_set.dot(weights)
    voted_predictions = (weighted_predictions >= 0.5).astype(int)
    return voted_predictions  
    
def exp_weighted_vote_method(predictions_on_set, scores_on_set):
    breakpoint()
    weights = softmax(scores_on_set)
    weighted_predictions = predictions_on_set.dot(weights)
    voted_predictions = (weighted_predictions >= 0.5).astype(int)
    return voted_predictions

def vote(predictions, scores, vote_method):
    voted_predictions = np.zeros(3000, dtype=int)
    for i in range(3):
        predictions_on_set = predictions[i*1000:(i+1)*1000]
        scores_on_set = scores[i]
        voted_predictions[i*1000:(i+1)*1000] = vote_method(predictions_on_set,
                                                           scores_on_set)
    return voted_predictions

def ensemble_predictions(predictions_folder, filenames, vote_method):
    """
    Given a list of filenames for which there is a csv file storing predictions
    and a yaml file storing accuracy in the given predictions_folder,
    compute a vote using the given vote_method.
    """
    num_predictors = len(filenames)
    predictions = np.zeros((3000, num_predictors), dtype=int)
    scores = np.zeros((3, num_predictors))
    preds_and_logs = {}
    
    for idx, filename in enumerate(filenames):
        filepath = path.join(predictions_folder, filename)
        predictions[:, idx] = pd.read_csv(filepath+'.csv', index_col=0).Bound.values
        with open(filepath+'.yml', 'r') as file:
            logs = yaml.safe_load(file)
        for i in range(3):
            scores[i, idx] = logs[i]['CV_acc']
    
    voted_predictions = vote(predictions, scores, vote_method)
    
    df = pd.DataFrame({
        "Id": range(len(voted_predictions)),
        "Bound": voted_predictions
    })
    
    return df