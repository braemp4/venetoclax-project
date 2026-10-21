from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
from datasets.Harmonized_Datasets import harmonize

import numpy as np 
import pandas as pd 


def map_de_level(a):
    log2FoldChange, nlog10 = a 

    if abs(log2FoldChange) < 1 or nlog10 < 2:
        return "Low DE"
    if abs(log2FoldChange) >= 1 and nlog10 > 2:
        if log2FoldChange >=1:
            return "Overexpressed"
        elif log2FoldChange <=-1:
            return "Underexpressed"
    return "Low DE"
# use approach from original pipeline i.e. same significance thresholds

# run deseq for feature selection
cm_ad = harmonize()['Commpass']

cm_ad = cm_ad[cm_ad.obs['Trans_1114'].notna()]
cm_ad = cm_ad[cm_ad.obs['Country_Skerget'].notna()]

expr_df = pd.DataFrame(cm_ad.layers['raw_counts'], columns=np.array(cm_ad.var_names))



inference = DefaultInference(n_cpus=8)
    

dds = DeseqDataSet(counts = expr_df, metadata=pd.DataFrame(cm_ad.obs), design="~Trans_1114", refit_cooks=True, inference=inference) 
dds.deseq2()

ds_res = DeseqStats(dds, contrast=["Trans_1114", "Detected", "Not Detected"], inference=inference)
ds_res.summary()        

ds_res = ds_res.results_df


ds_res= ds_res.sort_values(by="padj", ascending=False)

#dc.pl.volcano(data=ds_res, x="log2FoldChange", y="padj", top=30)
#plt.show()

ds_res = ds_res[ds_res["baseMean"] >=10]

sigs = ds_res[(ds_res["padj"] < 0.05) & (abs(ds_res["log2FoldChange"]))>0.5]
sigs["nlog10"] = -np.log10(sigs.padj)
sigs["de_level"] = sigs[["log2FoldChange", "nlog10"]].apply(map_de_level, axis=1)
de_genes = list(sigs[sigs["de_level"] != "Low DE"].index)

with open("../outputs/t11_14_degs.txt","w") as file:
    for gene in de_genes:
        file.write(f"{gene},")




