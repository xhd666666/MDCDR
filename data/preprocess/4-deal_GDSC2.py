import os

import numpy as np
import pandas as pd
import pubchempy as pcp

def preprocess_drug_response(pancancer_gdsc, index='CELL_LINE_NAME', columns='Drug Name', ic_values='LN_IC50', auc_values='AUC'):
    # 1. Deduplication: If a cell line-drug pair has multiple test records, retain the last one (arbitrary rule)
    pancancer_gdsc = pancancer_gdsc.sort_values(by=[index, columns]).drop_duplicates(subset=[index, columns], keep='last') # if a drug was tested more than once on the same cell line, we take the last one (arbitrary decision)
    # 2. Reshape to matrix: Rows=cell lines, Columns=drugs, Values=LN_IC50 (log of half-maximal inhibitory concentration, core drug response indicator)
    pancancer_ic_gdsc = pancancer_gdsc.pivot(index=index, columns=columns, values=ic_values).sort_index().sort_index(axis=1)
    # 3. If AUC (Area Under the Curve, auxiliary drug response indicator) is required, generate AUC matrix simultaneously
    if auc_values is not None:
        pancancer_auc_gdsc = pancancer_gdsc.pivot(index=index, columns=columns, values=auc_values).sort_index().sort_index(axis=1)
        return pancancer_ic_gdsc, pancancer_auc_gdsc
    return pancancer_ic_gdsc

# https://www.cancerrxgene.org/compounds -> select All and Export: CSV
drug_list = pd.read_csv('Drug_list.csv')
drugID_to_pubchemID_dict_raw = drug_list[['Drug Id', 'Name', 'PubCHEM', ' Datasets']].set_index(['Drug Id', 'Name', ' Datasets']).T.to_dict('records')[0]

# Clean mapping data: Filter invalid values and format valid values
drugID_to_pubchemID_dict = {}
for k, v in drugID_to_pubchemID_dict_raw.items():
    if v == v and v != 'none' and v != 'several': # some drugs do not have a PubChem ID listed in Drug_list.csv
        v2 = v.split(',')[0] # sometimes, multiple PubChem IDs are listed, of which we just take the first
        drugID_to_pubchemID_dict[k] = int(v2)
    else: # if v is 'none' or 'several', make them nan so that they will be dropped along with the original nans later
        v2 = np.nan
        drugID_to_pubchemID_dict[k] = v2

# Check data consistency: Print cases where the same drug name has different PubChem IDs (exclude GDSC1)
oldk = [None, None]
oldv = None
for k, v in drugID_to_pubchemID_dict.items():
    if k[1] == oldk[1]:
        if oldv != v and oldk[2] != 'GDSC1':
            print(oldk, oldv, k, v)
    oldk = k
    oldv = v

# Filter mapping relationships belonging only to GDSC2 dataset (Key: Drug Id, Value: PubChem ID)
drugID_to_pubchemID_dict_GDSC2 = {k[0]: v for k, v in drugID_to_pubchemID_dict.items() if k[2] == 'GDSC2'}
# print(drugID_to_pubchemID_dict_GDSC2)

pancancer_gdsc2_raw = pd.read_csv('GDSC2_fitted_dose_response_27Oct23.csv')
pancancer_gdsc2_pubchem = pancancer_gdsc2_raw.copy()
pancancer_gdsc2_pubchem['PubChem ID'] = pancancer_gdsc2_pubchem['DRUG_ID']
pancancer_gdsc2_pubchem = pancancer_gdsc2_pubchem.replace({'PubChem ID': drugID_to_pubchemID_dict_GDSC2}).dropna()
pancancer_gdsc2_pubchem['PubChem ID'] = [int(pubchem_id) for pubchem_id in pancancer_gdsc2_pubchem['PubChem ID']]
pancancer_ic_pubchem_gdsc2, pancancer_auc_pubchem_gdsc2 = preprocess_drug_response(pancancer_gdsc2_pubchem, columns='PubChem ID')

pubchemID_to_name_dict_GDSC2 = {v: k[1] for k, v in drugID_to_pubchemID_dict.items() if k[2] == 'GDSC2'}
name_to_pubchemID_dict_GDSC2 = {k[1]: v for k, v in drugID_to_pubchemID_dict.items() if k[2] == 'GDSC2'}
smiles_gdsc2 = []
for drug in pancancer_ic_pubchem_gdsc2.columns:
    # d = pcp.Compound.from_cid(drug)
    d = pcp.get_properties(properties=['SMILES', 'ConnectivitySMILES', 'InChIKey'],
    identifier=drug,  # e.g., CID of Aspirin
    namespace='cid')[0]
    name = pubchemID_to_name_dict_GDSC2[drug]
    smiles_gdsc2.append([name, drug, d['SMILES'], d['ConnectivitySMILES']])
    print(name)
smiles_df = pd.DataFrame(
    smiles_gdsc2,
    columns=['name', 'pubchem_id', 'isomeric_smiles', 'canonical_smiles']  # Strictly match the required column names
)
smiles_df.to_csv('smiles_gdsc2.csv', index=False, encoding='utf-8')
print("SMILES data has been saved as smiles_gdsc2.csv")

print(pancancer_ic_pubchem_gdsc2)
pancancer_ic_pubchem_gdsc2.index.name = None
pancancer_ic_pubchem_gdsc2.to_csv('GDSC2.csv', index=True, encoding='utf-8')