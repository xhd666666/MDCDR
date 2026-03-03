import numpy as np
import pandas as pd

# https://cellmodelpassports.sanger.ac.uk/downloads -> Expression Data -> under RNA-Seq, click View all versions -> zip contains multiple files, of which we chose rnaseq_tpm_20220624.csv because DepMap Public also uses TPM (but choosing any other file is of course also possible)
exp_tpm_raw = pd.read_csv('EXP/rnaseq_tpm_20220624.csv', low_memory=False) # 'low_memory=False' to remove warnings

exp_tpm_raw2 = exp_tpm_raw.T
exp_tpm_raw2.index = exp_tpm_raw2.iloc[:, 0]
exp_tpm_raw2 = exp_tpm_raw2.drop(exp_tpm_raw2.columns[1], axis=1)
exp_tpm_raw2.columns = exp_tpm_raw2.iloc[1]
exp_tpm_raw2 = exp_tpm_raw2.drop(exp_tpm_raw2.index[0])
exp_tpm_raw2 = exp_tpm_raw2.drop(exp_tpm_raw2.columns[0], axis=1)
exp_tpm_raw2 = exp_tpm_raw2.drop(exp_tpm_raw2.index[0])
exp_tpm_raw2 = exp_tpm_raw2.drop(exp_tpm_raw2.columns[0], axis=1)
exp_tpm_raw2 = exp_tpm_raw2.dropna(axis=1)
exp_tpm_raw2 = exp_tpm_raw2.astype(float)
exp_tpm_raw2 = exp_tpm_raw2.sort_index().sort_index(axis=1)
exp_tpm_raw2 = exp_tpm_raw2.loc[:, ~exp_tpm_raw2.columns.duplicated()] # EEF1AKNMT and SEPTIN4 are duplicated columns, we remove them

# because DepMap Public also calculates log2(TPM+1) and we want to compare them
# after running this notebook cell, both have a similar range of values (0 to 17 and 0 to 19)
exp_tpm = np.log2(exp_tpm_raw2 + 1)
exp_tpm.index.name = None
print(exp_tpm.shape)

exp_tpm.to_csv('EXP/expression.csv', index=True, encoding='utf-8')