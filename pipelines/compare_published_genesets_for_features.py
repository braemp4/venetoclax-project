import pandas as pd 
import numpy as np 
from datasets.Harmonized_Datasets import harmonize
from sklearn.linear_model import RidgeClassifier, LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier 
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from utils.labelling import binarize_labels

from gseapy import read_gmt
import os

gmt_dir = '../data/GMT/'
random_state=4

mf_ad = harmonize()['Moffitt']

ml_models = {'RidgeClassifier': RidgeClassifier(random_state=random_state),
             'LogisticRegression': LogisticRegression(random_state=random_state),
             'SGDClassifier': SGDClassifier(random_state=random_state),
             'GaussianNB': GaussianNB(),
             'RandomForestClassifier': RandomForestClassifier(random_state=random_state),
             'LinearSVC': LinearSVC(random_state=random_state), 
             'SVC': SVC(random_state=random_state),
             'HistGradientBoostingClassifier': HistGradientBoostingClassifier(random_state=random_state, max_iter=1000, early_stopping=True, n_iter_no_change=30)

             }

feature_sets = {
        'all': list(mf_ad.to_df().columns),
        'Gupta_six_gene_signature': ['VPREB3', 'CCR7', 'SORL1', 'TRABD2A', 'EFEMP1', 'TDO2'], 
        }

all_published = []
for file_name in os.listdir(gmt_dir):
    #if 'gmt' in file_name and "ZHAN" in file_name and "TARTE" in file_name and "B_CELLl" in file_name:
    if 'gmt' in file_name:
        if "ZHAN" in file_name or "TART" in file_name or "HAY" in file_name or "SHAF" in file_name or "BCL" in file_name or 'chr' in file_name: 
            gmt = read_gmt(gmt_dir + file_name)

            print(gmt.keys())
            
            feature_sets[file_name] = list(gmt.values())[0]
            all_published +=list(gmt.values())[0]

feature_sets['all_published'] = list(set(all_published))


mf_ad.obs['venetoclax_response'] = binarize_labels(mf_ad.obs['ABT-199'])
mf_ad = mf_ad[mf_ad.obs['venetoclax_response'] != -1 ]


results = {} # rows = feature set name, cols = model name
results_with_variability = {}

for feature_set_name in feature_sets.keys():
    
    X = mf_ad.to_df()
    y = mf_ad.obs['venetoclax_response']
         
    feature_set = list( set(feature_sets[feature_set_name]) & set(mf_ad.to_df().columns) )
    print(f'Using: {feature_set_name}')
    print(f'n features: {len(feature_set)}')

     
    X = X[feature_set]

    results[feature_set_name ] = []
    results_with_variability[ feature_set_name ] = []

    for model_name in ml_models.keys():

        model = ml_models[model_name]
        cv = StratifiedKFold(5, shuffle=True, random_state=random_state)
        cv_score  = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
        results[ feature_set_name ].append(cv_score.mean())

        results_with_variability[ feature_set_name ].append(f'{cv_score.mean()} +/- {cv_score.std()}')


print('\n')
results_df = pd.DataFrame(results, index=list(ml_models.keys())) 
results_df.to_csv("../outputs/comparing_published_genesets.csv")


results_with_variability_df = pd.DataFrame(results_with_variability, index=list(ml_models.keys())) 
results_with_variability_df.to_csv("../outputs/comparing_published_genesets_with_variability.csv")
print(results_df)


