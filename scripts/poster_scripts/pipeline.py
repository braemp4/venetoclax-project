import pandas as pd
import numpy as np
from figs_output import fig_output
from datasets import Harmonized_Datasets
from datasets.Harmonized_Datasets import harmonize
from utils import cross_val, ext_val, labelling
from sklearn.feature_selection import SelectKBest
from feature_selection import select_from_method, top_ft_anova_from_sens 
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier 
from sklearn.linear_model import LogisticRegression
import pickle

from gmm_auc_cutoffs import cutoff_moderate_auc 

label_method = 'tertiles'
add_cutoffs = False
random_state = 4

models = {
        "lr_reg" : LogisticRegression(random_state=random_state),
        "rf_model": RandomForestClassifier(random_state=random_state),
        "dec_tree": DecisionTreeClassifier(random_state=random_state),
        "ada_boost": AdaBoostClassifier(random_state=4),
        "xgboost": XGBClassifier(random_state=4, device="cpu"),

    } 


feature_engineering_params = {
                    "method": "t11_14_degs",
                    "lfc_threshold": 2,
        #             'n_clusters': 500,
        #             "random_state": general_params['random_state'],
        #             "n_init":1, # going to leave this as 1 for now
        #             'cache_feature_transformation': True,
                 }

datasets = Harmonized_Datasets.harmonize()
for model_name in models.keys():
    
    model = models[model_name]
    mf_ad, dc_ad = datasets["Moffitt"], datasets["Decampos"]
    
    if add_cutoffs: 
        mf_ad = cutoff_moderate_auc(mf_ad, "Moffitt")
        dc_ad = cutoff_moderate_auc(dc_ad, "Decampos")


    # this for  before cross val only because we are using tertile labels
    mf_ad.obs["ven_response"] = labelling.binarize_labels(mf_ad.obs['ABT-199'], method=label_method)
    dc_ad.obs["ven_response"] = labelling.binarize_labels(dc_ad.obs['ABT-199'], method=label_method)
    if label_method == 'tertiles': 
        mf_ad = mf_ad[mf_ad.obs['ven_response'] != -1]
        dc_ad = dc_ad[dc_ad.obs['ven_response'] != -1]

    
    #top_genes = select_from_method.select_from(harmonize(),**feature_engineering_params)

    #mf_ad = mf_ad[:,top_genes]
    #dc_ad = dc_ad[:,top_genes]

    cv_results, cv_fold_metrics, cv_overall_metrics = cross_val.cross_val_classification(mf_ad.to_df(), mf_ad.obs['ven_response'], model, random_state=random_state, n_splits=2, select_top_fts=True, binarizing_method='tertiles')

    cv_results = pd.DataFrame(cv_results)
    cv_overall_metrics = pd.Series(cv_overall_metrics)
 
    
    X_train, X_test, y_train, y_test = mf_ad.to_df(), dc_ad.to_df(), mf_ad.obs['ven_response'], dc_ad.obs['ven_response']

    sele_k = SelectKBest(k=100)

    top_fts = sele_k.fit(X_train, y_train).get_feature_names_out()
    X_train = X_train[top_fts] 
    X_test =  X_test[top_fts]   

    top_fts_df = pd.DataFrame({'feature': list(sele_k.feature_names_in_), "score":list(sele_k.scores_)})
    #top_fts_df.to_csv(fig_output + "pipeline_output/top_anova_fts.csv")


    model.fit(X_train, y_train)

    with open(fig_output + "pipeline_output/" + "models/" + f"{model_name}.pkl", 'wb') as file:
        pickle.dump(model,file)

    ev_results, ev_metrics = ext_val.external_val_classification(X_test, y_test, model)

     
    ev_results = pd.DataFrame(ev_results)
    ev_metrics = pd.Series(ev_metrics)

    

    # save results
    cv_results.to_csv(f"{fig_output}/pipeline_output/{model_name}_cv_results.csv")
    cv_overall_metrics.to_csv(f"{fig_output}/pipeline_output/{model_name}_cv_metrics.csv") 
    ev_results.to_csv(f"{fig_output}/pipeline_output/{model_name}_ev_results.csv")
    ev_metrics.to_csv(f"{fig_output}/pipeline_output/{model_name}_ev_metrics.csv")
    


