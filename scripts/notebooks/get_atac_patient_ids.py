import anndata as ad 
import scanpy as sc 

import os
import pandas as pd 
import numpy as np


ids = []
atac_path = '../../data/Leblay_Single_Cell/ATACseq/peak_matrices/'
for name in os.listdir(atac_path):
    ids.append(name[11:16]) 

with open(atac_path + "atac_patient_ids", 'w') as f:
    for line in ids:
        f.write(line + " ")
