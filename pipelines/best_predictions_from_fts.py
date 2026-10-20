import pandas as pd 
import numpy as np 
from datasets.Harmonized_Datasets import harmonize
from sklearn.linear_model import RidgeClassifier, LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier 
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from utils.labelling import binarize_labels

def cross_val_classification(X,y_ad, model,n_splits=5, random_state=4, verbose=True):
   

    results = {"samples": [],
                'wes_id': [],
               "y_true": [],
               "y_pred": [],
               "y_prob": [],
               }

    fold_metrics = {
            "n_fold": [],
            "precision_score": [],
            "recall_score": [], 
            "f1_score": [],
  
            "accuracy": [],
            "roc_auc": [],
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
       
        

        #sele_k = SelectKBest(k=100)

        #top_features = sele_k.fit(X_train, y_train).get_feature_names_out()

        #X_train = X_train[top_features]
        #X_test = X_test[top_features]

        # Scale the training data
#        scl = StandardScaler()
#        X_train = scl.fit_transform(X_train)
#        X_test = scl.transform(X_test)

        # train model
        model.fit(X_train, y_train)

        # get fold predictions
        y_pred = model.predict(X_test)
        results["y_pred"] +=list(y_pred)

        # get fold probas
        y_prob = model.predict_proba(X_test)[:,1]
        results["y_prob"] +=list(y_prob)

        # record fold metrics
        n_fold +=1
        '''
        fold_metrics["n_fold"].append(n_fold)
        fold_metrics["precision_score"].append(precision_score(y_test, y_pred))
        fold_metrics["recall_score"].append(recall_score(y_test, y_pred))
        fold_metrics["f1_score"].append(f1_score(y_test, y_pred))
        fold_metrics["accuracy"].append(accuracy_score(y_test,y_pred))
        fold_metrics["roc_auc"].append(roc_auc_score(y_test, y_prob))
        '''

#    if verbose:
#        print(pd.DataFrame(fold_metrics))

   
    return results, fold_metrics

best_feature_set = ['VPREB3', 'CCR7', 'SORL1', 'TRABD2A', 'EFEMP1', 'TDO2']

mf_ad = harmonize()['Moffitt']
X = mf_ad.to_df()[best_feature_set]

mf_ad.obs['venetoclax_response'] = binarize_labels(mf_ad.obs['ABT-199'])
results, metrics = cross_val_classification(X, mf_ad.obs, LogisticRegression()) 
results_df = pd.DataFrame(results)
results_df.to_csv("../outputs/predictions_for_wes_analysis.csv")

