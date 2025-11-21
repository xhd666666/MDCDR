from utils import *

gdsc_expression_file_path = 'data/GDSC/gdsc_gene_selected_expr.csv'
gdsc_pathway_file_path = 'data/GDSC/gdsc_pathway.csv'

tcga_expression_file_path = 'data/TCGA/tcga_gene_selected_expr.csv'
tcga_pathway_file_path = 'data/TCGA/tcga_pathway.csv'

ppi_feature_file_path = 'data/STRING/ppi_features_512.tsv'
drugs_pubchem_smiles_file_path = 'data/drugs_pubchem_smiles.csv'

drug_dict  = load_drug_dict(drugs_pubchem_smiles_file_path)
np.save('data/drug_feature_graph.npy', drug_dict)
print(len(drug_dict))

gdsc_cell_dict = load_cell_dict(gdsc_expression_file_path, ppi_feature_file_path, gdsc_pathway_file_path, 'GDSC')
print(gdsc_cell_dict)
np.save('data/GDSC/cell_feature.npy', gdsc_cell_dict)

tcga_cell_dict = load_cell_dict(tcga_expression_file_path, ppi_feature_file_path, tcga_pathway_file_path, 'TCGA')
print(tcga_cell_dict)
np.save('data/TCGA/cell_feature.npy', tcga_cell_dict)

edge_index = load_edge_index()
cluster_predefine = get_predefine_cluster(edge_index)
torch.save(cluster_predefine, 'data/cluster_predefine.pth')
