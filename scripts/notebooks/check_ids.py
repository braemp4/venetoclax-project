import pandas as pandas 
import numpy as np 
from datasets.Harmonized_Datasets import harmonize

moffitt_ad = harmonize()["Moffitt"]
#print(moffitt_ad.obs)

df = moffitt_ad.obs.set_index('patientID')

#print(df[df['t11_14'].notna()]['t11_14'])

with open("scripts/notebooks/check_some_ids.txt","r") as file:
    ids = file.readlines()

print(df.index)
for idx in ids:
    #print(idx)
    if idx in df.index:
        print(idx)



