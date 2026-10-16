import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datasets.Harmonized_Datasets import harmonize

from inmoose.pycombat import pycombat_seq
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats


def get_features(datasets,lfc_threshold = 1, cache_feature_transformation = False, output_dir=""):
    '''
    Commpass data should already be batch corrected!
    '''
    
     
     
    feature_sets_folder = Path(__file__).resolve().parents[2] / "data" / "Feature_Sets/"
    feature_set_path = Path(feature_sets_folder / "CoMMpass_t11_14_DEGS.txt")
    deseq_output = Path(__file__).resolve().parents[2] / "outputs" / "t11_14_degs" / output_dir
    
    results_df = pd.DataFrame({})

    commpass_ad, moffitt_ad, decampos_ad = datasets["Commpass"], datasets["Moffitt"], datasets["Decampos"]

    if Path(deseq_output / 'results.csv').is_file():
        try:
            print("DEGs file found")
            results_df = pd.read_csv(deseq_output / "results.csv")
            results_df = results_df.set_index("Unnamed: 0")
        except FileNotFoundError as E:
            print("DEGs File read error!")

    else:
        # check if differential expression
        Path(deseq_output).mkdir(parents=True, exist_ok=True)
        print("Getting DEGs ...")
        
        
        # we shouldnt need to use the training data as an arg since we can grab it from here
        # remove nan values
        commpass_ad = commpass_ad[commpass_ad.obs["Trans_1114"].notna()]
        commpass_ad = commpass_ad[commpass_ad.obs["Country_Skerget"].notna()]

        expr_df = pd.DataFrame(commpass_ad.layers["raw_counts"], columns=np.array(commpass_ad.var_names))
        expr_df = expr_df.astype(int) 
        
        # run deseq
        t11_14_labels = commpass_ad.obs["Trans_1114"]
        
        inference = DefaultInference(n_cpus=8)
        

        dds = DeseqDataSet(counts = expr_df, metadata=pd.DataFrame(commpass_ad.obs), design="~Trans_1114", refit_cooks=True, inference=inference) 
        dds.deseq2()

        ds_res = DeseqStats(dds, contrast=["Trans_1114", "Detected", "Not Detected"], inference=inference)
        ds_res.summary()        


        results_df = ds_res.results_df
        results_df.to_csv(deseq_output / 'results.csv')
    
    
    results_df = results_df.sort_values(by="padj", ascending=False)
    results_df = results_df[(abs(results_df['log2FoldChange']) >= lfc_threshold)]
    

    expr_df = moffitt_ad.to_df()[results_df.index] # if breaks, check index in the "else" case

    moffitt_ad = moffitt_ad[:, results_df.index]
    decampos_ad = decampos_ad[:, results_df.index]
    
    
    return results_df.index
    
    
def test():
    mf_adata, dc_adata = get_features()
    print(dc_adata.to_df())


if __name__ == "__main__":
    test()
