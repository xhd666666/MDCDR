import pandas as pd

# COSMIC genes source: https://cancer.sanger.ac.uk/cosmic

expression = pd.read_csv('EXP/filter_expression.csv', index_col=0, header=0)
gdsc = pd.read_csv('GDSC.csv', index_col=0, header=0)

expression_cell_lines = set(expression.index)
gdsc_cell_lines = set(gdsc.index)
common_cell_lines = list(expression_cell_lines & gdsc_cell_lines)
print(f"Number of cell lines in expression file: {len(expression_cell_lines)}")
print(f"Number of cell lines in GDSC file: {len(gdsc_cell_lines)}")
print(f"Number of common cell lines across two files: {len(common_cell_lines)}")

gdsc_filtered = gdsc.loc[common_cell_lines]
gdsc_filtered.index.name = None
gdsc_filtered.to_csv('GDSC.csv', index=True, encoding='utf-8')