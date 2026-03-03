import pandas as pd

# https://cellmodelpassports.sanger.ac.uk/downloads -> Copy Number Data -> under Copy Number (SNP6), click View all versions -> zip contains two files: cnv_gistic_20191101.csv and cnv_abs_copy_number_picnic_20191101.csv
# we use cnv_gistic_20191101.csv instead of cnv_abs_copy_number_picnic_20191101.csv because the latter has positive real values, and the numbers have different meanings, see cnv_summary_20230303.csv: 2 is neutral, but sometimes loss, 4 is also neutral, 3 is gain or loss; gistics -2, -1, 0, 1, and 2 are way easier to interpret
# (to get cnv_summary_20230303.csv: https://cellmodelpassports.sanger.ac.uk/downloads -> Copy Number Data -> under CNV Summary, click View all versions)
cnv_all_raw = pd.read_csv('CNV/cnv_gistic_20191101.csv', low_memory=False)

cnv_all_raw2 = cnv_all_raw.T
cnv_all_raw2 = cnv_all_raw2.drop(cnv_all_raw2.index[0])
cnv_all_raw2.index = cnv_all_raw2.iloc[:, 0]
cnv_all_raw2 = cnv_all_raw2.drop(cnv_all_raw2.columns[1], axis=1)
cnv_all_raw2 = cnv_all_raw2.drop(cnv_all_raw2.columns[0], axis=1)
cnv_all_raw2.columns = cnv_all_raw2.iloc[0]
cnv_all_raw2 = cnv_all_raw2.drop(cnv_all_raw2.index[0])
cnv_all_raw2.index = [str(s) for s in cnv_all_raw2.index]
cnv_all = cnv_all_raw2.sort_index().sort_index(axis=1)
cnv_all = cnv_all.loc[:, (cnv_all != 0).any(axis=0)] # drop zero-only columns (but our cnv version has none)
cnv_all = cnv_all.loc[(cnv_all != 0).any(axis=1)] # drop zero-only rows (but our cnv version has none)
cnv_all = cnv_all.dropna(axis=1) # remove nan columns (removing rows would remove all rows)
cnv_all.index.name = None
print(cnv_all.shape)

cnv_all.to_csv('CNV/copy_number.csv', index=True, encoding='utf-8')