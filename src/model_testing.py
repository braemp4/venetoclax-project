import pandas as pd
import numpy as np
from datasets.Harmonized_Datasets import harmonize
from datasets.Moffitt import load_moffitt_duplicates_ad
from utils.labelling import binarize_labels

def external_validation(model):

    adata = harmonize()['Decampos']

    adata.obs['venetoclax_response'] = binarize_labels(adata.obs['ABT-199'])
    adata = adata[adata.obs['venetoclax_response']!=-1]

    X = adata.to_df()[model.feature_names_in_]
    y_true = adata.obs['venetoclax_response']

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:,1]
    return {'samples': adata.obs_names, 'y_pred': y_pred, 'y_prob':y_prob, 'y_true': y_true}

    
def test_on_moderate_responders(model):

    adata = harmonize()['Moffitt']

    adata.obs['venetoclax_response'] = binarize_labels(adata.obs['ABT-199'])

    X = adata.to_df()[model.feature_names_in_]

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:,1]
    return {'samples': adata.obs_names,'y_pred': y_pred, 'y_prob':y_prob}



def test_on_duplicates(model):
    mf_ad = harmonize()["Moffitt"]
    duplicates_ad = load_moffitt_duplicates_ad()

    duplicates_ad.obs['venetoclax_response'] = binarize_labels(duplicates_ad.obs['ABT-199'])

    duplicates_ad = duplicates_ad[duplicates_ad.obs['venetoclax_response']!=-1]

    X, y = duplicates_ad.to_df()[model.feature_names_in_], duplicates_ad.obs['venetoclax_response']

    y_true = y
    y_pred  = model.predict(X)
    y_prob  = model.predict_proba(X)[:,1]



    print(duplicates_ad.obs.columns)
    return {'samples': duplicates_ad.obs_names, 'ex_vivo_id': duplicates_ad.obs['ExVivo_SampleID'],'y_pred':y_pred , 'y_prob':y_prob, 'y_true':y_true, 'duplicate_exvivo':duplicates_ad.obs['duplicate_exvivo'], 'duplicate_rnaseq':duplicates_ad.obs['duplicate_rnaseq']}

def test_on_duplicate_moderate_responders(model):
    mf_ad = harmonize()["Moffitt"]
    duplicates_ad = load_moffitt_duplicates_ad()

    duplicates_ad.obs['venetoclax_response'] = binarize_labels(duplicates_ad.obs['ABT-199'])

    duplicates_ad = duplicates_ad[duplicates_ad.obs['venetoclax_response']==-1]

    X, y = duplicates_ad.to_df()[model.feature_names_in_], duplicates_ad.obs['venetoclax_response']

    y_true = y
    y_pred  = model.predict(X)
    y_prob  = model.predict_proba(X)[:,1]



    print(duplicates_ad.obs.columns)

    return {'samples': duplicates_ad.obs_names,'y_pred':y_pred , 'y_prob':y_prob, 'y_true':y_true, 'duplicate_exvivo':duplicates_ad.obs['duplicate_exvivo'], 'duplicate_rnaseq':duplicates_ad.obs['duplicate_rnaseq']}

