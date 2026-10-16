import pandas as pd 
import numpy as np
from pathlib import Path
from inmoose.pycombat import pycombat_seq


# THIS NEEDS TO BE RUN BEFORE LOADING COMMPASS - OR KEEP THIS FILE IN DATA

commpass_path = Path(__file__).resolve().parents[1] / "data" / "CoMMpass" 

expr_df = pd.read_csv( commpass_path /  "1stLine_Raw_ENSG.csv").set_index("Unnamed: 0").transpose()
metadata = pd.read_csv(commpass_path/ "patient_clinical_data.csv", sep="\t").set_index("PUBLIC_ID")

intersected_indices = list(set(expr_df.index) & set(metadata.index))
expr_df = expr_df.loc[intersected_indices]
metadata = metadata.loc[intersected_indices]

expr_df = expr_df.astype(int)

print(expr_df)

# batch correct
print("Running Batch Correction ...")
batch = metadata["Country_Skerget"]
expr_df = pycombat_seq(expr_df.transpose(), batch).transpose()
print(expr_df)

expr_df.to_csv(commpass_path / "raw_batch_corrected.csv")

