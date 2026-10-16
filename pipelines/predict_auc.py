import pandas as pd 
import numpy as np

from datasets import Harmonized_Datasets 
from feature_selection.select_from_method import select_from
from utils.cross_val import cross_val_regression
from utils.ext_val import external_val_classification
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

random_state=4
rf_model = RandomForestRegressor(random_state=random_state)

params = {'n_clusters': 500, "random_state": 4, 'run_enrich': False}

# I forgot that feature selection needs to also transform the external dataset
moffit_ad = select_from("t11_14_gene_modules", **params)

# cross validation
cv_res, cv_fold_metrics, cv_overall_metrics = cross_val_regression(moffit_ad.to_df(), moffit_ad.obs["ABT-199"], rf_model)

print(cv_overall_metrics)
#print(pd.Series(cv_overall_metrics))
#print(cv_res, cv_fold_metrics, cv_overall_metrics)

# train model
rf_model.fit(moffit_ad.to_df(), moffit_ad.obs["ABT-199"])






