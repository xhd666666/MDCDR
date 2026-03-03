import pandas as pd

# https://xenabrowser.net/datapages/  -> select GDC TCGA ... -> gene expression RNAseq -> STAR - TPM -> download
gdsc_expression = pd.read_csv('EXP/expression.csv', index_col=0, header=0)
tcga_expression = pd.read_csv('EXP/tcga_gene_exprs_tpm.csv', sep=',', header=0, index_col=[0])
tcga_expression = tcga_expression.T
tcga = pd.read_excel('TCGA/TCGA.xlsx', header=0, index_col=0)

# Example: "TCGA-OR-A5K2-01A" -> "TCGA-OR-A5K2"
tcga_expression.index = tcga_expression.index.str.split('-').str[:3].str.join('-')
duplicated_patients = tcga_expression.index[tcga_expression.index.duplicated(keep=False)]
if len(duplicated_patients) > 0:
    original_count = tcga_expression.shape[0]
    print(f"⚠️ Detected {duplicated_patients.nunique()} patients with multiple samples (total {len(duplicated_patients)} duplicated rows)")
    print(f"   Example duplicated patients: {duplicated_patients.unique()[:5].tolist()}")
    print(f"Number of rows before deduplication: {original_count}")

    # Retain the first occurrence of each patient (in original order)
    # Method: Reset index -> Deduplicate by patient ID (keep first) -> Restore index
    tcga_expression = tcga_expression.reset_index()
    tcga_expression = tcga_expression.drop_duplicates(subset='index', keep='first')
    tcga_expression = tcga_expression.set_index('index')

    print(f"Number of rows after deduplication: {tcga_expression.shape[0]} (removed {original_count - tcga_expression.shape[0]} rows)")
    print(f"✅ Patient-level deduplication completed, all row names are unique")
else:
    print("✅ No duplicated patient IDs, no row deduplication needed")

expression_cell_lines = set(tcga_expression.index)
tcga_cell_lines = set(tcga.index)
common_cell_lines = list(expression_cell_lines & tcga_cell_lines)
print(f"Number of cell lines in expression file: {len(expression_cell_lines)}")
print(f"Number of cell lines in TCGA file: {len(tcga_cell_lines)}")
print(f"Number of common cell lines across two files: {len(common_cell_lines)}")

gdsc_expression_genes = set(gdsc_expression.columns)
tcga_expression_genes = set(tcga_expression.columns)
cosmic_genes = pd.read_table('748genes_cosmic.csv', sep=',')
filter_genes = set(cosmic_genes['Gene Symbol'].values.tolist())
common_genes = list(gdsc_expression_genes & tcga_expression_genes & filter_genes)
print(f"Number of genes in gdsc_expression file: {len(gdsc_expression_genes)}")
print(f"Number of genes in tcga_expression file: {len(tcga_expression_genes)}")
print(f"Number of common genes: {len(common_genes)}")

gdsc_expression_filtered = gdsc_expression.loc[:, common_genes]
tcga_expression_filtered = tcga_expression.loc[:, common_genes]

duplicated_cols = tcga_expression_filtered.columns[tcga_expression_filtered.columns.duplicated()].tolist()
if duplicated_cols:
    print(f"\n⚠️ Detected {len(duplicated_cols)} duplicated gene columns, e.g.: {duplicated_cols[:5]}")
    print(f"Shape of tcga_expression_filtered before deduplication: {tcga_expression_filtered.shape}")

    # Retain the first occurrence of columns
    tcga_expression_filtered = tcga_expression_filtered.loc[:,
                               ~tcga_expression_filtered.columns.duplicated(keep='first')]

    print(f"Shape of tcga_expression_filtered after deduplication: {tcga_expression_filtered.shape}")
    print(f"Actual number of unique genes retained: {tcga_expression_filtered.shape[1]}")
else:
    print("\n✅ No duplicated gene columns in tcga_expression_filtered")

gdsc_expression_filtered.index.name = None
gdsc_expression_filtered.to_csv('EXP/gdsc_expression_filtered.csv', index=True, encoding='utf-8')
tcga_expression_filtered.index.name = None
tcga_expression_filtered.to_csv('EXP/tcga_expression_filtered.csv', index=True, encoding='utf-8')

tcga_filtered = tcga.loc[common_cell_lines]
tcga_filtered.index.name = None
tcga_filtered.to_csv('TCGA_filter.csv', index=True, encoding='utf-8')