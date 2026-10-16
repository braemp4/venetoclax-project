import pandas as pd 
import numpy as np
import anndata as ad 
from datasets import Moffitt, Commpass, DeCampos 

def harmonize():
    
    # LOAD ADATA FILES
    moffitt_ad= Moffitt.load_moffitt_ad()
    commpass_ad = Commpass.load_commpass_ad()
    decampos_ad = DeCampos.load_decampos_ad()

    #print("HI")    
    # GET INTERSECTION OF FEATURE NAMES
    harmonized_geneset = list(set(moffitt_ad.var_names) & set(commpass_ad.var_names) & set(decampos_ad.var_names))
    print(f"Number of Harmonized Genes: {len(harmonized_geneset)}")
    
    # Subset each geneset

    moffitt_ad, commpass_ad, decampos_ad = moffitt_ad[:,harmonized_geneset], commpass_ad[:,harmonized_geneset], decampos_ad[:,harmonized_geneset]

    return {"Moffitt": moffitt_ad,
            "Commpass": commpass_ad,
            "Decampos": decampos_ad,
            }


def test():
    datasets = harmonize()
    
    moffitt_ad = datasets["Moffitt"]
    expr_df = moffitt_ad.to_df()

    for key in datasets.keys():
        print(key)
        expr_df = datasets[key].to_df()
        print(expr_df)


if __name__ == "__main__":
    test()








