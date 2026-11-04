import pandas as pd 
import numpy as np 
from datasets.Harmonized_Datasets import harmonize
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from utils.labelling import binarize_labels
from compare_published_genesets_for_features import feature_sets

from gseapy import read_gmt
import os
import pickle


mf_ad = harmonize()['Moffitt']
mf_ad.obs['venetoclax_response'] = binarize_labels(mf_ad.obs['ABT-199'])
mf_ad = mf_ad[mf_ad.obs['venetoclax_response'] != -1 ]


with open("../outputs/t11_14_degs.txt", "r") as file:
    feature_set = file.read().split(',')

X = mf_ad.to_df()
y = mf_ad.obs['venetoclax_response']

feature_set = list(set(X.columns) & set(feature_set))
X = X[feature_set]


model = LogisticRegression(random_state=4)
params = {
    'logisticregression__C': np.logspace(-4, 2, 10),
    'logisticregression__l1_ratio': [0, 0.5, 1],
    'logisticregression__max_iter': [10000]
}
pipeline = make_pipeline(StandardScaler(), model)

cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=4
        )

gscv = GridSearchCV(estimator = pipeline, param_grid=params, scoring='roc_auc', cv=cv)
gscv.fit(X, y)

best_model = gscv.best_estimator_
best_score = gscv.best_score_
print(best_score, best_model)

with open('../models/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)


