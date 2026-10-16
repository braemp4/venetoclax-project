import anndata as ad 
import scanpy as sc 

import os
import pandas as pd 
import numpy as np


atacs = {}
atac_path = '../../data/Leblay_Single_Cell/ATACseq/peak_matrices/'
for name in os.listdir(atac_path):
    atac = sc.read_10x_h5(atac_path + name, gex_only=False)
    treatment_status_str = name[17:21].replace('_', "")
    treatment_status = [treatment_status_str for i in range(len(atac.obs))]
    atac.obs['treatment_status'] = treatment_status

    atacs[name[11:16]] = atac
    
atac = ad.concat(atacs, join='outer', label='sample')
atac.write_h5ad(  '../../data/Leblay_Single_Cell/ATACseq/scATACseq.h5ad'  )
