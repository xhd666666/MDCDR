import random

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

seed = 2025
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import argparse

from model import *
from utils import load_drug_dict, load_cell_dict, _collate

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='optimizer weight_decay')
parser.add_argument('--epochs', type=int, default=400,
                    help='the epochs for model')
parser.add_argument('--patience', type=int, default=30,
                    help='patience')
parser.add_argument('--cell_dim', type=int, default=733,
                    help='cell_dim')
parser.add_argument('--pathway_dim', type=int, default=1307,
                    help='pathway_dim')
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')

drug_path = 'data/smiles_TCGA.csv'
gdsc_exp_path = 'data/EXP/gdsc_expression_filtered.csv'
gdsc_pathway_path = 'data/gdsc_pathway.csv'
tcga_exp_path = 'data/EXP/tcga_expression_filtered.csv'
tcga_pathway_path = 'data/tcga_pathway.csv'
cluster_path = 'data/tcga_cluster.pth'

gdsc_response = pd.read_csv('data/response_cls.csv')
tcga_response = pd.read_csv('data/response_tcga.csv')
drug_dict = load_drug_dict(drug_path)
gdsc_cell_dict = load_cell_dict(gdsc_exp_path, gdsc_pathway_path)
tcga_cell_dict = load_cell_dict(tcga_exp_path, tcga_pathway_path)
cell_dict = gdsc_cell_dict | tcga_cell_dict
cluster_predefine = get_predefine_cluster(gdsc_exp_path, device, cluster_path)

label_col = 'res'

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

for fold, (_, _) in enumerate(skf.split(tcga_response, tcga_response[label_col])):
    fold_seed = seed + fold

    train_val_tcga, test_tcga = train_test_split(
        tcga_response,
        test_size=0.8,
        random_state=fold_seed,
        shuffle=True,
        stratify=tcga_response[label_col]
    )

    train_tcga, val_tcga = train_test_split(
        train_val_tcga,
        test_size=0.5,
        random_state=fold_seed,
        shuffle=True,
        stratify=train_val_tcga[label_col]
    )

    train_set = pd.concat([gdsc_response, train_tcga], axis=0, ignore_index=True)

    print(f"\n===== Fold {fold + 1}/5 Data Split Result =====")
    print(f"Total TCGA samples: {len(tcga_response)}")
    print(
        f"TCGA training set (10%): {len(train_tcga)} samples | positive/negative ratio: {train_tcga[label_col].value_counts(normalize=True).round(3)}")
    print(
        f"TCGA validation set (10%): {len(val_tcga)} samples | positive/negative ratio: {val_tcga[label_col].value_counts(normalize=True).round(3)}")
    print(
        f"TCGA test set (80%): {len(test_tcga)} samples | positive/negative ratio: {test_tcga[label_col].value_counts(normalize=True).round(3)}")
    print(f"Final training set (GDSC + TCGA training set): {len(train_set)} samples")

    train_dataset = MyDataset(drug_dict, cell_dict, train_set)
    val_dataset = MyDataset(drug_dict, cell_dict, val_tcga)
    test_dataset = MyDataset(drug_dict, cell_dict, test_tcga)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=True, collate_fn=_collate)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=True, collate_fn=_collate)

    model = ModelUtil(device=device, batch_size=args.batch_size,
                      lr=args.lr, weight_decay=args.weight_decay,
                      is_regression=False, cluster_predefine=cluster_predefine,
                      cell_dim=args.cell_dim, pathway_dim=args.pathway_dim)

    train(args, model, train_loader, val_loader, test_loader, fold=fold + 1, mode='TCGA')