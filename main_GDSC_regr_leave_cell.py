import random

import numpy as np
import torch

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
from utils import _collate_regr, load_drug_dict, load_cell_dict

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
parser.add_argument('--cell_dim', type=int, default=688,
                    help='cell_dim')
parser.add_argument('--pathway_dim', type=int, default=1285,
                    help='pathway_dim')
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')

drug_path = 'data/smiles_gdsc2.csv'
exp_path = 'data/EXP/filter_expression.csv'
pathway_path = 'data/gdsc_regr_pathway.csv'
cluster_path = 'data/gdsc_cluster.pth'

gdsc_response = pd.read_csv('data/response.csv')
drug_dict = load_drug_dict(drug_path)
cell_dict = load_cell_dict(exp_path, pathway_path)
cluster_predefine = get_predefine_cluster(exp_path, device, cluster_path)

mode = 'cell'
seed_list = [1, 2, 3, 4, 5]
for seed in seed_list:
    train_gdsc, val_gdsc, test_gdsc = load_leave_hybrid(df=gdsc_response, col_name=mode,
                                                 random_state=seed)

    print(f"Mode {mode}-{seed} data distribution:")
    print(
        f"  Training set: {len(train_gdsc)} samples | Validation set: {len(val_gdsc)} samples | Test set: {len(test_gdsc)} samples")

    train_dataset = MyDataset(drug_dict, cell_dict, train_gdsc)
    val_dataset = MyDataset(drug_dict, cell_dict, val_gdsc)
    test_dataset = MyDataset(drug_dict, cell_dict, test_gdsc)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=_collate_regr)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=True, collate_fn=_collate_regr)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=True, collate_fn=_collate_regr)

    model = ModelUtil(device=device, batch_size=args.batch_size,
                      lr=args.lr, weight_decay=args.weight_decay,
                      is_regression=True, cluster_predefine=cluster_predefine,
                      cell_dim=args.cell_dim, pathway_dim=args.pathway_dim)
    train_regr(args, model, train_loader, val_loader, test_loader, seed, mode = mode)