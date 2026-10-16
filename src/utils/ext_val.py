import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, r2_score, mean_squared_error, roc_auc_score, accuracy_score
from sklearn.mixture import GaussianMixture

import anndata
from sklearn.ensemble import RandomForestClassifier
from utils.labelling import binarize_labels


def external_val_classification(X_test, y_test, trained_model, random_state=4):
    '''
    Scale X_test, predict on X_test using trained model and evaluate predictions
    '''
    scl = StandardScaler()
    samples = X_test.index

    X_test = scl.fit_transform(X_test)
    y_test = binarize_labels(y_test, random_state=random_state)

    y_pred = trained_model.predict(X_test)
    y_prob = trained_model.predict_proba(X_test)[:,1]
    
    results_df = {
            "samples":samples,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "y_true": y_test,
            }

    metrics = {
            "precision_score":precision_score(y_test, y_pred),
            "recall_score":recall_score(y_test, y_pred),
            "f1_score":f1_score(y_test, y_pred),
            "accuracy": accuracy_score(y_test, y_pred), 
            "roc_auc": roc_auc_score(y_test, y_prob)

            }

    return results_df, metrics




def test():
    pass

if __name__ == "__main__":
    test()
