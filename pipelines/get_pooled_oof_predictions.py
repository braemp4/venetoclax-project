import pandas as pd 
import numpy as np 
from datasets.Harmonized_Datasets import harmonize
from sklearn.linear_model import RidgeClassifier, LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier 
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.utils import resample
from utils.labelling import binarize_labels
import os
from gseapy import read_gmt
from scipy.special import expit

import matplotlib.pyplot as plt 
from compare_published_genesets_for_features import * 

def get_bootstrap_metrics(results_df, alpha=0.05, bootstraps=10000, verbose=True):
    '''
    Calculate bootstrap metrics using pooled out-of-fold predictions. 
    '''
    results_df = results_df.copy().set_index('samples')

    
    metric_statistics= {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': []
    }
    bootstrap_results = {
        'accuracy': (),
        'precision': (),
        'recall': (),
        'f1': (),
        'roc_auc': (),
    }

    # get bootstrap statistics
    for i in range(bootstraps):
        res = resample(results_df)

        y_pred = res['y_pred']
        y_prob = res['y_prob']
        y_true = res['y_true']

        metric_statistics['accuracy'].append(accuracy_score(y_true, y_pred))
        metric_statistics['precision'].append(precision_score(y_true, y_pred))
        metric_statistics['recall'].append(recall_score(y_true, y_pred))
        metric_statistics['f1'].append(f1_score(y_true, y_pred))
        metric_statistics['roc_auc'].append(roc_auc_score(y_true, y_prob))

    # get confidence intervals
    for key in metric_statistics.keys():
        stat = metric_statistics[key]
        stat = sorted(stat)
        
        lower = np.percentile(stat, (1-alpha) / 2)
        upper = np.percentile(stat, alpha+((1-alpha)/2))
        bootstrap_results[key] = (lower, upper)

    if verbose:
        print("ML Classification metric 95% Confidence Infervals: ")
        for key in bootstrap_results.keys():
            lower, upper = bootstrap_results[key]
            print(f"{key.replace('_', ' ').title():<12} [{lower:.3f}, {upper:.3f}]")
            
            
    return bootstrap_results 

def cross_val_classification(X,y_ad, model,n_splits=5, random_state=4, verbose=True):
   

    results = {"samples": [],
                'wes_id': [],
               "y_true": [],
               "y_pred": [],
               "y_prob": [],
               }

    fold_metrics = {
            "precision_score": [],
            "recall_score": [], 
            "f1_score": [],
  
            "accuracy": [],
            "roc_auc": [],
            }

    cv_metrics = {
            "precision_score": '',
            "recall_score": '', 
            "f1_score": '',
            "accuracy": '',
            "roc_auc": '',
            }

    y = y_ad['venetoclax_response']
    

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    n_fold= 0
    for i, (train_index, test_index) in enumerate(skf.split(X,y)):
        if verbose:
            print(f"CV FOLD: {n_fold + 1}")

        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_test, y_test = X.iloc[test_index], y.iloc[test_index]


        # add ground truth values
        results["y_true"] += list(y_test)
        
        # get test samples
        test_samples = X_test.index
        results["samples"] +=list(test_samples)
        wes_ids = y_ad.loc[test_samples]['WES_SampleID']
        results['wes_id'] += list(wes_ids)
       
        # Scale the training data
        scl = StandardScaler()
        X_train = scl.fit_transform(X_train)
        X_test = scl.transform(X_test)

        # train model
        model.fit(X_train, y_train)

        # get fold predictions
        y_pred = model.predict(X_test)
        results["y_pred"] +=list(y_pred)

        # get fold probas
        y_prob = model.decision_function(X_test) if isinstance(model , RidgeClassifier) else  model.predict_proba(X_test)[:,1]
        results["y_prob"] +=list(y_prob)

        n_fold+=1

        fold_metrics["precision_score"].append(precision_score(y_test, y_pred))
        fold_metrics["recall_score"].append(recall_score(y_test, y_pred))
        fold_metrics["f1_score"].append(f1_score(y_test, y_pred))
        fold_metrics["accuracy"].append(accuracy_score(y_test,y_pred))
        fold_metrics["roc_auc"].append(roc_auc_score(y_test, y_prob))
        
     
    for i, metric_name in enumerate(fold_metrics.keys()):
        fold_metric = np.array(fold_metrics[metric_name])
        
        cv_metrics[metric_name] = f'{fold_metric.mean():.5f} +/- {fold_metric.std():.5f}'
        plt.plot([fold_metric.mean() - fold_metric.std(), fold_metric.mean() + fold_metric.std()],[i + 1,i + 1], c='blue')
        plt.plot(fold_metric.mean(),i+1, marker='o', c='blue')

    plt.yticks(np.arange(len(fold_metrics.keys())) + 1, list(fold_metrics.keys()))
    plt.xticks([0, 0.1 ,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.xlabel('Score')
    plt.title("Top model + feature-set combination")
    plt.tight_layout()
    plt.savefig('top-model-feature-set-combination.png')

    return results, cv_metrics

mf_ad = harmonize()['Moffitt']
feature_sets['all'] = list(mf_ad.var_names)

##### UI #####
for i, v in enumerate(ml_models.keys()):
    print(f"{i + 1}) {v}")
model_index = int(input("Select an ML model to test: ")) - 1

print("\n")

for i, v in enumerate(feature_sets):
    print(f"{i + 1}) {v}")
feature_set_index = int(input("Select a feature set to test: ")) - 1

ml_model = ml_models[list(ml_models.keys())[model_index]]
feature_set = feature_sets[list(feature_sets.keys())[feature_set_index]]
##### UI #####

X = mf_ad.to_df()

X['BCL2/MCL1'] = X['BCL2'] / X['MCL1']

X['BCL2/BCL2L1'] = X['BCL2'] / X['BCL2L1']
X = X[list( set(feature_set) & set(X.columns) )]

'''
Just learned something new. We get different scores looking at pooled out-of-fold metrics to average (mean or median) fold metrics. The primary metric should be the average fold predictions +/-. Later on we can try permuation testing of the labels to see if results are by chance. To do permutation test we shuffle the labels n times and the proportion of times we get greater than or equal to our metric is the P value. For CSV to save metric average +/- std, save a list with columns as metric for just one feature set.

'''
mf_ad.obs['venetoclax_response'] = binarize_labels(mf_ad.obs['ABT-199'])
results, cv_metrics = cross_val_classification(X, mf_ad.obs, ml_model) 
results_df = pd.DataFrame(results)


print("Mean fold metrics: ")
print(pd.Series(cv_metrics).T)

#print("Calculating out-of-fold bootstrap metrics: ")
#get_bootstrap_metrics(results_df, bootstraps=10000)

results_df.to_csv("../outputs/predictions_for_wes_analysis.csv")

