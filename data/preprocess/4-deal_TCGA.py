import os

import numpy as np
import pandas as pd
import pubchempy as pcp

# Zijian Ding, Songpeng Zu, and Jin Gu. Evaluating the molecule-based prediction of clinical drug responses in cancer. Bioinformatics, 32(19):2891–2895, 2016.
# bioinfo16_supplementary_tables.xlsx -> Table S2
# data -> TCGA -> bioinfo16_supplementary_tables.xlsx

tcga_raw = pd.read_excel('TCGA/bioinfo16_supplementary_tables.xlsx', header=0)
pivot_df = pd.pivot_table(
    tcga_raw,
    index='bcr_patient_barcode',    # Rows: Cancer names
    columns='drug_name',            # Columns: Drug names
    values='measure_of_response',   # Values: Response values
    aggfunc='first',                # For duplicate (cancer, drug) pairs, take the first value
    fill_value=np.nan               # Fill NaN for missing corresponding values
)
pivot_df = pivot_df.reset_index()
pivot_df.columns = [''] + list(pivot_df.columns[1:])
pivot_df.to_excel('TCGA/TCGA.xlsx', index=False)

pivot_df = pd.read_excel('TCGA/TCGA.xlsx', header=0, index_col=0)
column_mapping = {}
for drug in pivot_df.columns:
    results = pcp.get_compounds(drug, 'name')
    if results:
        column_mapping[drug] = str(results[0].cid)
        print(str(results[0].cid))
    else:
        print("cannot find")
pivot_df = pivot_df[list(column_mapping.keys())].rename(columns=column_mapping)
pivot_df = pivot_df.loc[:, ~pivot_df.columns.duplicated()]
pivot_df = pivot_df.reset_index()
pivot_df.columns = [''] + list(pivot_df.columns[1:])
pivot_df.to_excel('TCGA/TCGA.xlsx', index=False)

pivot_df = pd.read_excel('TCGA/TCGA.xlsx', header=0, index_col=0)
gdsc = pd.read_csv('GDSC.csv', index_col=0, header=0)
drug_list = list(set(list(pivot_df.columns) + list(gdsc.columns)))
smiles_gdsc2 = []
for drug in drug_list:
    print(drug)
    d = pcp.get_properties(properties=['SMILES', 'ConnectivitySMILES', 'InChIKey'],
    identifier=drug,
    namespace='cid')[0]
    smiles_gdsc2.append([drug, d['SMILES'], d['ConnectivitySMILES']])
smiles_df = pd.DataFrame(
    smiles_gdsc2,
    columns=['pubchem_id', 'isomeric_smiles', 'canonical_smiles']
)
smiles_df.to_csv('smiles_TCGA.csv', index=False, encoding='utf-8')
print("SMILES data has been saved as smiles_TCGA.csv")