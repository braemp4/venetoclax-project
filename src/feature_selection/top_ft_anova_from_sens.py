import numpy as np 
import pandas as pd 
from sklearn.mixture import GaussianMixture
from sklearn.feature_selection import SelectKBest

from datasets import Harmonized_Datasets
import anndata as ad
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
from utils.labelling import binarize_labels


def get_features(datasets,output_dir="",binarizing_method = "gmm", n_features = 500, random_state= 4, cache_feature_transformation=False):
        '''
        Select best features using anova F value. Will product an inflated assessment in downstream cross validation using the moffitt training data due to data leakage so the main evaluation to be considered for this feature selection method is using an external test set (i.e. the DeCampos dataset).

        Note to self: Using the default k = 500 features, BCL2 is one of the top informative genes.
        '''
        

        top_fts_output = Path(__file__).resolve().parents[2] / "outputs" / "top_ft_sens" / output_dir 

        if not top_fts_output.is_file():
            top_fts_output.mkdir(exist_ok=True)

        commpass_ad, moffitt_ad, decampos_ad = datasets["Commpass"], datasets["Moffitt"], datasets["Decampos"]
        
        moffitt_ad = datasets["Moffitt"]
        decampos_ad = datasets["Decampos"]
        
        y = moffitt_ad.obs["ABT-199"]

         
        y = np.array(binarize_labels(y, method=binarizing_method, random_state=random_state))
        
        sele_k = SelectKBest(k=n_features)

        top_features = sele_k.fit(moffitt_ad.to_df(), y).get_feature_names_out()
        
        moffitt_ad = moffitt_ad[:,top_features]
        decampos_ad = decampos_ad[:,top_features]

        
        if cache_feature_transformation:
            results_df = pd.DataFrame([sele_k.feature_names_in_, sele_k.scores_, sele_k.pvalues_], index=["feature", "score", "pvalue"]).transpose().set_index('feature').sort_values(by='score', ascending=False)
            results_df.to_csv(top_fts_output / "results.h5ad")
            
            print(results_df.head())


        return moffitt_ad.to_df(), decampos_ad.to_df(), moffitt_ad.obs["ABT-199"], decampos_ad.obs["ABT-199"]


def test():
    moffitt_ad, dc_ad = get_features()
    print(moffitt_ad.to_df()["BCL2"])
if __name__ == "__main__":
    test()

