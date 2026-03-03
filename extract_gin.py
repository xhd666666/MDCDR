import hashlib
import os
import pickle

import dgl
import pandas as pd
import torch
from dgllife.model import load_pretrained
from dgllife.utils import mol_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer
from rdkit import Chem


def smiles2graph(smile, model):
    """
    Convert SMILES to molecular graph and extract features using the passed model
    """
    model.eval()  # Ensure the model is in evaluation mode
    mol = Chem.MolFromSmiles(smile)
    graph = mol_to_bigraph(mol,
                           node_featurizer=PretrainAtomFeaturizer(),
                           edge_featurizer=PretrainBondFeaturizer(),
                           add_self_loop=True)
    bg = dgl.batch([graph])
    node_feats_list = [
        bg.ndata['atomic_number'].long(),
        bg.ndata['chirality_type'].long()
    ]
    edge_feats_list = [
        bg.edata['bond_type'].long(),
        bg.edata['bond_direction_type'].long()
    ]
    with torch.no_grad():
        node_embeddings = model(bg, node_feats_list, edge_feats_list)
        # molecular_embedding = torch.mean(node_embeddings, dim=0)
    return node_embeddings

# ===== Main Program =====
save_dir = 'data/gin'
os.makedirs(save_dir, exist_ok=True)

# Load GIN model once for reuse
model_name = 'gin_supervised_contextpred'
pretrained_model = load_pretrained(model_name)
pretrained_model.eval()

i = 0
# drugs_pubchem_smiles = pd.read_csv('../data/smiles_gdsc2.csv', sep=',')
drugs_pubchem_smiles = pd.read_csv('../data/smiles_TCGA.csv', sep=',')
for idx in drugs_pubchem_smiles.index:
    print(f'Processing: {i + 1}/{len(drugs_pubchem_smiles)}')
    pubchem = str(int(drugs_pubchem_smiles.loc[idx, 'pubchem_id']))
    smiles = drugs_pubchem_smiles.loc[idx, 'isomeric_smiles']

    i += 1
    feature_path = os.path.join(save_dir, f"{pubchem}.pkl")

    if os.path.exists(feature_path):
        print(f'Skipping already processed sequence: {smiles}...')
        continue

    feature = smiles2graph(smiles, pretrained_model)  # Pass the shared model

    with open(feature_path, 'wb') as f:
        pickle.dump(feature, f)

    print(f'Saved successfully: {feature_path}')