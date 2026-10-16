import numpy as np 
import pandas as pd 
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, r2_score, mean_squared_error, roc_auc_score, accuracy_score
from sklearn.mixture import GaussianMixture


from datasets import Moffitt
import anndata
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from utils.labelling import binarize_fold_labels,binarize_labels 

def cross_val_classification(X,y, model, binarizing_method="gmm",n_splits=2, random_state=4, select_top_fts=False, k_top_fts=100):
   

    results = {"samples": [],
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

    overall_metrics = {"precision_score":[], "recall_score": [], "f1_score": [], "accuracy": [], "roc_auc":[]}
    
     
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    n_fold= 0
    for i, (train_index, test_index) in enumerate(kf.split(X,y)):
         
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_test, y_test = X.iloc[test_index], y.iloc[test_index]

        if binarizing_method != "tertiles":
            y_test, y_train = binarize_fold_labels(y_test,y_train, random_state=random_state)

        if select_top_fts:

            sele_k = SelectKBest(k=k_top_fts).fit(X_train, y_train)


            X_train = pd.DataFrame(sele_k.transform(X_train), index=X_train.index)
            X_test = pd.DataFrame(sele_k.transform(X_test), index=X_test.index)
    

        # add ground truth values
        results["y_true"] += list(y_test)
        
        # get test samples
        test_samples = X_test.index
        results["samples"] +=list(test_samples)
       
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

        y_prob = model.predict_proba(X_test)[:,1]
        results["y_prob"] +=list(y_prob)

        # record fold metrics
        n_fold +=1
        fold_metrics["n_fold"].append(n_fold)
        fold_metrics["precision_score"].append(precision_score(y_test, y_pred))
        fold_metrics["recall_score"].append(recall_score(y_test, y_pred))
        fold_metrics["f1_score"].append(f1_score(y_test, y_pred))
        fold_metrics["accuracy"].append(accuracy_score(y_test,y_pred))
        fold_metrics["roc_auc"].append(roc_auc_score(y_test, y_prob))



    # get overall metrics
    overall_metrics = {
        "precision_score": precision_score(results["y_true"], results["y_pred"]),
        "recall_score": recall_score(results["y_true"], results["y_pred"]), 
        "f1_score": f1_score(results["y_true"], results["y_pred"]),
        "accuracy": accuracy_score(results["y_true"], results["y_pred"]),
        "roc_auc": roc_auc_score(results["y_true"], results["y_prob"]),
        }
    return results, fold_metrics, overall_metrics



def cross_val_regression(X, y, model, n_splits=2):

    results = {"samples": [],
               "y_true": [],
               "y_pred": [],
               }

    fold_metrics = {
            "n_fold": [],
            "r2_score": [],
            "mse": [], 
            }

    overall_metrics = {"r2_score":[], "mse": []}

    kf = KFold(n_splits=n_splits)
    n_fold= 0
    for i, (train_index, test_index) in enumerate(kf.split(X,y)):

        # split the data
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_test, y_test = X.iloc[test_index], y.iloc[test_index]


        # add ground truth values
        results["y_true"] += list(y_test)
        
        # get test samples
        test_samples = X_test.index
        results["samples"] +=list(test_samples)
       
        # Scale the training data
        scl = StandardScaler()
        X_train = scl.fit_transform(X_train)
        X_test = scl.transform(X_test)
        
        # train model
        model.fit(X_train, y_train)

        # get fold predictions
        y_pred = model.predict(X_test)
        results["y_pred"] +=list(y_pred)


        # record fold metrics
        n_fold +=1
        fold_metrics["n_fold"].append(n_fold)
        fold_metrics["r2_score"].append(r2_score(y_test, y_pred))
        fold_metrics["mse"].append(mean_squared_error(y_test, y_pred))



    # get overall metrics
    overall_metrics = {
        "overall_r2_score": r2_score(results["y_true"], results["y_pred"]),
        "overall_mse": mean_squared_error(results["y_true"], results["y_pred"]), 
        }
    return results, fold_metrics, overall_metrics

def test():
    adata = Moffitt.load_moffitt_ad()
    expr_df = adata.to_df()
    rf_model = RandomForestClassifier(random_state=4)
    ven_res = adata.obs["ABT-199"]
    results, fold_metrics, overall_metrics = cross_val_classification(adata.to_df(), ven_res, model=rf_model)


if __name__ == "__main__":
    test()

