
import pandas as pd

# https://cellmodelpassports.sanger.ac.uk/downloads -> Model Annotation -> under Model List, click View all versions
model_list = pd.read_csv('Bulk_Cell_line_Genomic_Data/model_list_20240110.csv')
# https://cellmodelpassports.sanger.ac.uk/downloads -> Mutation Data -> under Mutations Summary, click View all versions
mut_all_raw = pd.read_csv('MUT/mutations_all_20230202.csv')

mut_all_raw2 = mut_all_raw.copy()
mut_all_raw2['is_mutated'] = 1 # all rows stand for a mutation, so we add a new all-1 column 'is_mutated'
mut_all_raw2 = mut_all_raw2.drop_duplicates(subset=['gene_symbol', 'model_id'], keep='first') # if one combination has multiple mutations, treat as one mutation
mut_all = mut_all_raw2.pivot(index='model_id', columns='gene_symbol', values='is_mutated')
mut_all = mut_all.fillna(0).astype(int)
model_list_dict = dict(zip(model_list['model_id'], model_list['model_name']))
mut_all = mut_all.rename(index=model_list_dict)
mut_all = mut_all.sort_index().sort_index(axis=1)
mut_all = mut_all.loc[:, (mut_all != 0).any(axis=0)] # drop zero-only columns (but our mut version has none because pivot)
mut_all = mut_all.loc[(mut_all != 0).any(axis=1)] # drop zero-only rows (but our mut version has none because pivot)
mut_all.index.name = None
print(mut_all.shape)

mut_all.to_csv('MUT/mutation.csv', index=True, encoding='utf-8')