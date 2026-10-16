import pandas as pd 
import numpy as np
from pathlib import Path
from inmoose.pycombat import pycombat_norm


# THIS NEEDS TO BE RUN BEFORE LOADING COMMPASS - OR KEEP THIS FILE IN DATA

commpass_path = Path(__file__).resolve().parents[1] / "data" / "CoMMpass" 

expr_df = pd.read_csv( commpass_path /  "CPM_raw.csv").set_index("gene_name").transpose()
metadata = pd.read_csv(commpass_path/ "patient_clinical_data.csv", sep="\t").set_index("PUBLIC_ID")

print(expr_df)
intersected_indices = list(set(expr_df.index) & set(metadata.index))
expr_df = expr_df.loc[intersected_indices]
metadata = metadata.loc[intersected_indices]

expr_df = pd.DataFrame(np.log2(expr_df.astype(float)+1e-3), columns=expr_df.columns, index=expr_df.index)
print(expr_df)

# batch correct
print("Running Batch Correction ...")
batch = metadata["Country_Skerget"]
expr_df = pycombat_norm(expr_df.transpose(), batch).transpose()
print(expr_df)

expr_df.to_csv(commpass_path / "log2CPM_batch_corrected.csv")


