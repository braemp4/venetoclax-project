import numpy as np 
import pandas as pd 
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.feature_selection import SelectKBest
from datasets.Harmonized_Datasets import harmonize
import anndata

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from utils.labelling import binarize_fold_labels

output_dir = "../outputs/cv_moffitt_top_fts_2026_09_08/"
mf_ad = harmonize()["Moffitt"]

X = mf_ad.to_df()
y = mf_ad.obs["ABT-199"]

n_splits = 2
k_features = 100
random_state=4



models = {
        "lr_reg" : LogisticRegression(random_state=random_state),
        "rf_model": RandomForestClassifier(random_state=random_state),
        "dec_tree": DecisionTreeClassifier(random_state=random_state),
        "ada_boost": AdaBoostClassifier(random_state=4),
        "xgboost": XGBClassifier(random_state=4, device="cpu", n_estimators=100, learning_rate=0.01, objective="binary:logistic"),

    } 



for model_name in models.keys():

    model = models[model_name]

    results = {"samples": [],
               "y_true": [],
               "y_pred": [],
               "y_prob": [],
               }

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    n_fold= 0
    for i, (train_index, test_index) in enumerate(kf.split(X,y)):


        print(f"Running CV Fold: {i} ...")

        # split the data
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_test, y_test = X.iloc[test_index], y.iloc[test_index]
        
        y_test, y_train = binarize_fold_labels(y_test,y_train, random_state=random_state)

        # GET TOP FEATURES 
        sele_k = SelectKBest(k=k_features).fit(X_train, y_train)


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


    # save output
    pd.DataFrame(results).to_csv(output_dir + model_name + "_results.csv")
