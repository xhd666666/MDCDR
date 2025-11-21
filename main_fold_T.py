import os
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
from utils import _collate

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='the learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='optimizer weight_decay')
parser.add_argument('--epochs', type=int, default=200,
                    help='the epochs for model')
parser.add_argument('--patience', type=int, default=40,
                    help='early stop')

args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')

gdsc_response = pd.read_csv('data/GDSC/response.csv')
tcga_response = pd.read_csv('data/TCGA/response.csv')
drug_dict = np.load('data/drug_feature_graph.npy', allow_pickle=True).item()
gdsc_cell_dict = np.load('data/GDSC/cell_feature.npy', allow_pickle=True).item()
tcga_cell_dict = np.load('data/TCGA/cell_feature.npy', allow_pickle=True).item()
cell_dict = gdsc_cell_dict | tcga_cell_dict
cluster_predefine = torch.load('data/cluster_predefine.pth')
cluster_predefine = {key: tensor.to(device) for key, tensor in cluster_predefine.items()}

kf = KFold(n_splits=10, shuffle=True, random_state=seed)
for fold, (train_index, test_index) in enumerate(kf.split(tcga_response)):
    train_tcga = tcga_response.iloc[test_index]
    test_tcga = tcga_response.iloc[train_index]
    train_response = pd.concat([gdsc_response, train_tcga], axis=0, ignore_index=True)
    test_response = test_tcga
    train_set = train_response
    test_set = test_response

    train_dataset = MyDataset(drug_dict, cell_dict, train_set)
    test_dataset = MyDataset(drug_dict, cell_dict, test_set)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=_collate)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=_collate)

    model = ModelUtil(device=device, batch_size=args.batch_size,
                      lr=args.lr, weight_decay=args.weight_decay,
                      cluster_predefine=cluster_predefine)
    train(args, model, train_loader, test_loader, fold, mode='TCGA')

