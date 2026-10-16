import pandas as pd 
import numpy as np

from datasets import Harmonized_Datasets 
from feature_selection.select_from_method import select_from
from feature_selection.top_ft_anova_from_sens import binarize_labels
from utils.cross_val import cross_val_classification
from utils.ext_val import external_val_classification 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import datetime
from experiment_tracking.exp_tracker import track


RANDOM_STATE=4

# needs to be form utils .....
def get_ft_eng_dir_suffix(params):
    if params['method'] ==  "t11_14_gene_modules":
        return "_k" + str(params["n_clusters"])
    elif params['method'] == "t11_14_degs":
        return ""

models = {"lr_reg" : LogisticRegression(random_state=RANDOM_STATE)} # new addition!


general_params = {"title": "t11_14_degs",

                  "date_timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                  "random_state": RANDOM_STATE,
                  "ml_model": models["lr_reg"],
                  "cv_splits": 2 # loo cv 
                  }

feature_engineering_params = {
                    "method": "t11_14_degs",
                    "lfc_threshold": 2,
        #             'n_clusters': 500,
        #             "random_state": general_params['random_state'],
        #             "n_init":1, # going to leave this as 1 for now
        #             'cache_feature_transformation': True,
                 }

# outdir for feature engineering
print("Start feature engineering")


datasets = Harmonized_Datasets.harmonize()
feature_engineering_params["output_dir"] = feature_engineering_params["method"] + get_ft_eng_dir_suffix(feature_engineering_params)

X_train, X_test, y_train, y_test = select_from(datasets,**feature_engineering_params)

# cross validation
# for leave one out cv make n_splits = # of samples
#cv_res, cv_fold_metrics, cv_overall_metrics = cross_val_classification(moffit_ad.to_df(), moffit_ad.obs["ABT-199"], lr_model, random_state=params["random_state"], n_splits=len(moffit_ad.to_df()))

print(type(general_params['ml_model']))
#print("Cross validation")
cv_res, cv_fold_metrics, cv_overall_metrics = cross_val_classification(X=X_train, y=y_train, model=general_params['ml_model'], random_state=general_params['random_state'], n_splits=general_params["cv_splits"])

# train model print("Train model") general_params["ml_model"].fit(X_train, binarize_labels(y_train, random_state=general_params['random_state'])) test on external dataset
print("External validation")
results, metrics = external_val_classification(X_test,y_test, general_params["ml_model"], random_state=general_params['random_state'])

print("Cross-validation Results: ")
print(pd.Series(cv_overall_metrics))
print(pd.DataFrame(cv_fold_metrics))


print("External-validation Results: ")
print(pd.Series(metrics))

track(general_params, feature_engineering_params, cv_results=cv_res, cv_per_fold_metrics = cv_fold_metrics, cv_overall_metrics=cv_overall_metrics, ev_results = results, ev_metrics = metrics, ml_model=general_params["ml_model"])


