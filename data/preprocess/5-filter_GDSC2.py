import pandas as pd

# COSMIC genes source: https://cancer.sanger.ac.uk/cosmic

mutation = pd.read_csv('MUT/mutation.csv', index_col=0, header=0)
copy_number = pd.read_csv('CNV/copy_number.csv', index_col=0, header=0)
expression = pd.read_csv('EXP/expression.csv', index_col=0, header=0)
gdsc = pd.read_csv('GDSC2.csv', index_col=0, header=0)

mutation_cell_lines = set(mutation.index)
copy_cell_lines = set(copy_number.index)
expression_cell_lines = set(expression.index)
gdsc_cell_lines = set(gdsc.index)
common_cell_lines = list(mutation_cell_lines & copy_cell_lines & expression_cell_lines & gdsc_cell_lines)
print(f"Number of cell lines in mutation file: {len(mutation_cell_lines)}")
print(f"Number of cell lines in copy_number file: {len(copy_cell_lines)}")
print(f"Number of cell lines in expression file: {len(expression_cell_lines)}")
print(f"Number of cell lines in GDSC file: {len(gdsc_cell_lines)}")
print(f"Number of common cell lines across four files: {len(common_cell_lines)}")

mutation_genes = set(mutation.columns)
copy_genes = set(copy_number.columns)
expression_genes = set(expression.columns)
cosmic_genes = pd.read_table('748genes_cosmic.csv', sep=',')
filter_genes = set(cosmic_genes['Gene Symbol'].values.tolist())
common_genes = list(mutation_genes & copy_genes & expression_genes & filter_genes)
print(f"Number of genes in mutation file: {len(mutation_genes)}")
print(f"Number of genes in copy_number file: {len(copy_genes)}")
print(f"Number of genes in expression file: {len(expression_genes)}")
print(f"Number of common genes across three files: {len(common_genes)}")

mutation_filtered = mutation.loc[common_cell_lines, common_genes]
copy_number_filtered = copy_number.loc[common_cell_lines, common_genes]
expression_filtered = expression.loc[common_cell_lines, common_genes]

mutation_filtered.index.name = None
mutation_filtered.to_csv('MUT/filter_mutation.csv', index=True, encoding='utf-8')
copy_number_filtered.index.name = None
copy_number_filtered.to_csv('CNV/filter_copy_number.csv', index=True, encoding='utf-8')
expression_filtered.index.name = None
expression_filtered.to_csv('EXP/filter_expression.csv', index=True, encoding='utf-8')

gdsc_filtered = gdsc.loc[common_cell_lines]
gdsc_filtered.index.name = None
gdsc_filtered.to_csv('GDSC2.csv', index=True, encoding='utf-8')