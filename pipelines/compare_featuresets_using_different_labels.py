import pandas as pd 
import numpy as np 
from datasets.Harmonized_Datasets import harmonize
from sklearn.linear_model import RidgeClassifier, LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier 
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils.labelling import binarize_labels

from gseapy import read_gmt
import os


gmt_dir = '../data/GMT/'
random_state=4

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
        #'all': list(mf_ad.to_df().columns),
        'Gupta_six_gene_signature': ['VPREB3', 'CCR7', 'SORL1', 'TRABD2A', 'EFEMP1', 'TDO2'], 
        'BCL2_family_ratios': ['BCL2/MCL1', "BCL2/BCL2L1"],
        'Leblay_integrated_single_cell': ['RUNX1', 'BCL2/MCL1', "BCL2/BCL2L1", "IRF1", "IRF4", "PAX5", "ID3", "PRDM1", "TCF4", "XBP1", "CD79A", "MS4A1", "ERN1", "RUNX2", "AP1", "NOXA", "BBC3"],
        
        # combinations
        'BCL2_family_ratios + Gupta_six_gene_signature': ['VPREB3', 'CCR7', 'SORL1', 'TRABD2A', 'EFEMP1', 'TDO2', 'BCL2/MCL1', "BCL2/BCL2L1"],

        # novel data-driven (i.e. t11;14 degs from commpass)
        't11_14_degs': [],
        }

with open("../outputs/t11_14_degs.txt", "r") as file:
    feature_sets['t11_14_degs'] = file.read().split(',')

for file_name in os.listdir(gmt_dir):
    if 'gmt' in file_name:
        if "ZHAN" in file_name or "TART" in file_name or "HAY" in file_name or "SHAF" in file_name or "BCL" in file_name or 'chr' in file_name: 
            gmt = read_gmt(gmt_dir + file_name)            
            feature_sets[file_name] = list(gmt.values())[0]


def main():

    mf_ad = harmonize()['Moffitt']
    feature_sets['all'] = list(mf_ad.var_names)
    mf_ad.obs['venetoclax_response_auc'] = binarize_labels(mf_ad.obs['ABT-199'])
    mf_ad = mf_ad[mf_ad.obs['venetoclax_response_auc'] != -1 ]


    mf_ad.obs['venetoclax_response_calculated_auc'] = binarize_labels(mf_ad.obs['Calculated_AUC'])
    mf_ad = mf_ad[mf_ad.obs['venetoclax_response_calculated_auc'] != -1 ]


    mf_ad.obs['venetoclax_response_calculated_vuc'] = binarize_labels(mf_ad.obs['Calculated_VUC'])
    mf_ad = mf_ad[mf_ad.obs['venetoclax_response_calculated_vuc'] != -1 ]


    mf_ad.obs['venetoclax_response_calculated_vuc_log_conc'] = binarize_labels(mf_ad.obs['Calculated_VUC_Log_Conc'])
    mf_ad = mf_ad[mf_ad.obs['venetoclax_response_calculated_vuc_log_conc'] != -1 ]
    

    results = {} # rows = feature set name, cols = model name
    results_with_variability = {}


    X = mf_ad.to_df()
    cols = ['venetoclax_response_calculated_auc', 'venetoclax_response_calculated_vuc', 'venetoclax_response_calculated_vuc_log_conc']

    for col in cols: 
    
        y = mf_ad.obs[col]

        X['BCL2/MCL1'] = X['BCL2'] / X['MCL1']

        X['BCL2/BCL2L1'] = X['BCL2'] / X['BCL2L1']


        for feature_set_name in feature_sets.keys():
            
                 
            feature_set = list( set(feature_sets[feature_set_name]) & set(X.columns) )
            print(f'Using: {feature_set_name}')
            print(f'n features: {len(feature_set)}')

             
            X_selected = X[feature_set]

            results[feature_set_name ] = []
            results_with_variability[ feature_set_name ] = []

            for model_name in ml_models.keys():
                
                model = ml_models[model_name]

                pipeline = make_pipeline(StandardScaler(),model)

                cv = StratifiedKFold(5, shuffle=True, random_state=random_state)
                cv_score  = cross_val_score(pipeline, X_selected, y, cv=cv, scoring='roc_auc')
                
                results[ feature_set_name ].append(cv_score.mean())

                results_with_variability[ feature_set_name ].append(f'{cv_score.mean():.5f} +/- {cv_score.std():.5f}')


        print('\n')
        results_df = pd.DataFrame(results, index=list(ml_models.keys())) 
        results_df.to_csv(f"../outputs/comparing_published_genesets_{col}.csv")


        results_with_variability_df = pd.DataFrame(results_with_variability, index=list(ml_models.keys())) 
        results_with_variability_df.to_csv(f"../outputs/comparing_published_genesets_with_variability_{col}.csv")
        print(results_df)

if __name__ == "__main__":
    main()
