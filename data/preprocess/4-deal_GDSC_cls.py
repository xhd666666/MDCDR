import os

import numpy as np
import pandas as pd
import pubchempy as pcp

# https://www.cancerrxgene.org/compounds -> select All and Export: CSV
drug_list = pd.read_csv('Drug_list.csv')
pubchem_col = 'PubCHEM'
# Convert to numeric type (non-numeric values to NaN)
drug_list[pubchem_col] = pd.to_numeric(drug_list[pubchem_col], errors='coerce')
# Filter rows with NaN values
drug_list_filtered = drug_list.dropna(subset=[pubchem_col])
# Convert to integer type
drug_list_filtered[pubchem_col] = drug_list_filtered[pubchem_col].astype(int)
# Deduplicate by PubCHEM column, retain only the first row for each value
drug_list_final = drug_list_filtered.drop_duplicates(subset=[pubchem_col], keep='first')
name_to_pubchem_dict = drug_list_final.set_index('Name')[pubchem_col].to_dict()

# https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources/Home.html -> TableS5C.xlsx
gdsc_raw = pd.read_excel('TableS5C.xlsx', index_col=0, header=0)
gdsc_raw.columns = [name_to_pubchem_dict.get(str(col).strip(), col) for col in gdsc_raw.columns]
gdsc_raw = gdsc_raw.loc[:, gdsc_raw.columns.map(lambda x: isinstance(x, int))]

print(gdsc_raw)
gdsc_raw.index.name = None
gdsc_raw.to_csv('GDSC.csv', index=True, encoding='utf-8')

smiles_gdsc2 = []
for drug in gdsc_raw.columns:
    d = pcp.get_properties(properties=['SMILES', 'ConnectivitySMILES', 'InChIKey'],
    identifier=drug,
    namespace='cid')[0]
    smiles_gdsc2.append([drug, d['SMILES'], d['ConnectivitySMILES']])
    print(drug)
smiles_df = pd.DataFrame(
    smiles_gdsc2,
    columns=['pubchem_id', 'isomeric_smiles', 'canonical_smiles']  # Strictly match the required column names
)
smiles_df.to_csv('smiles_GDSC.csv', index=False, encoding='utf-8')
print("SMILES data has been saved as smiles_GDSC.csv")