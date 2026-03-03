import torch.nn
from torch import optim
from torch import nn
from torch.nn.utils import weight_norm
from torch_geometric.nn import GATConv, global_mean_pool
from tqdm import tqdm

from base_model.BAN import BANLayer
from base_model.Fusion import ContrastFusion
from base_model.LOSS import ContrastLoss
from utils import *

use_print = True

class GNN_drug(nn.Module):
    def __init__(self, num_features_mol=78, output_dim=128, dropout=0.2):
        super(GNN_drug, self).__init__()
        self.mol_conv = nn.ModuleList([])
        self.mol_conv.append(GATConv(num_features_mol, num_features_mol * 4, heads=2, dropout=dropout, concat=False))
        self.mol_conv.append(
            GATConv(num_features_mol * 4, num_features_mol * 4, heads=2, dropout=dropout, concat=False))
        self.mol_conv.append(
            GATConv(num_features_mol * 4, num_features_mol * 4, heads=2, dropout=dropout, concat=False))
        self.mol_out_feats = num_features_mol * 4
        self.mol_seq_fc1 = nn.Linear(num_features_mol * 4, num_features_mol * 4)
        self.mol_seq_fc2 = nn.Linear(num_features_mol * 4, num_features_mol * 4)
        self.mol_bias = nn.Parameter(torch.rand(1, num_features_mol * 4))
        torch.nn.init.uniform_(self.mol_bias, a=-0.2, b=0.2)
        self.mol_fc_g1 = torch.nn.Linear(num_features_mol * 4, 1024)
        self.mol_fc_g2 = torch.nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, mol_x, mol_edge_index, mol_batch):
        mol_n = mol_x.size(0)
        for i in range(len(self.mol_conv)):
            x = self.mol_conv[i](mol_x, mol_edge_index)
            if i < len(self.mol_conv) - 1:
                x = self.relu(x)
            if i == 0:
                mol_x = x
                continue
            mol_z = torch.sigmoid(
                self.mol_seq_fc1(x) + self.mol_seq_fc2(mol_x) + self.mol_bias.expand(mol_n, self.mol_out_feats))
            mol_x = mol_z * x + (1 - mol_z) * mol_x

        x = global_mean_pool(mol_x, mol_batch)
        # flatten
        x = self.relu(self.mol_fc_g1(x))
        x = self.dropout(x)
        x = self.mol_fc_g2(x)
        x = self.dropout(x)
        return x

class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Classifier, self).__init__()
        self.mlp = nn.Sequential(
                    nn.Linear(in_dim, 128),
                    nn.ELU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, 64),
                    nn.ELU(),
                    nn.Dropout(0.1),
                    nn.Linear(64, out_dim)
                )

    def forward(self, x):
        x = self.mlp(x)
        return x

class ContextEncoder(nn.Module):
    def __init__(self, cell_dim):
        super(ContextEncoder, self).__init__()
        self.morgan_encoder = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        self.espf_encoder = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        self.pubchem_encoder = nn.Sequential(
            nn.Linear(881, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(512, 128),
        )

        self.cell_encoder = nn.Sequential(
            nn.Linear(cell_dim, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.ban = weight_norm(BANLayer(v_dim=128, q_dim=128, h_dim=256), name='h_mat')

    def forward(self, cell_feature, drug):
        # -----cell line representation
        x_cell = []
        x_expr = self.cell_encoder(cell_feature)
        x_cell.append(x_expr)

        # -----drug representation
        x_drug = []
        x1_drug = self.morgan_encoder(drug.morgan)
        x_drug.append(x1_drug)
        x2_drug = self.espf_encoder(drug.espf)
        x_drug.append(x2_drug)
        x3_drug = self.pubchem_encoder(drug.pubchem_fp)
        x_drug.append(x3_drug)

        x_drug = torch.stack(x_drug, dim=1)
        x_cell = torch.stack(x_cell, dim=1)
        f, att = self.ban(x_drug, x_cell)

        return f

class GNN_cell(torch.nn.Module):
    def __init__(self, num_feature, layer_cell, dim_cell, cluster_predefine):
        super(GNN_cell, self).__init__()
        self.num_feature = num_feature
        self.layer_cell = layer_cell
        self.dim_cell = dim_cell
        self.cluster_predefine = cluster_predefine
        self.final_node = len(self.cluster_predefine[self.layer_cell - 1].unique())
        self.convs_cell = torch.nn.ModuleList()
        self.bns_cell = torch.nn.ModuleList()

        for i in range(self.layer_cell):
            if i:
                # 8 8
                conv = GATConv(self.dim_cell, self.dim_cell)
            else:
                # 1 8
                conv = GATConv(self.num_feature, self.dim_cell)
            bn = torch.nn.BatchNorm1d(self.dim_cell, affine=False)  # True or False

            self.convs_cell.append(conv)
            self.bns_cell.append(bn)

    def forward(self, cell):
        original_cell = cell.clone()
        for i in range(self.layer_cell):
            original_cell.x = F.relu(self.convs_cell[i](original_cell.x, original_cell.edge_index))
            num_node = int(original_cell.x.size(0) / original_cell.num_graphs)
            cluster = torch.cat([self.cluster_predefine[i] + j * num_node for j in range(original_cell.num_graphs)])
            original_cell = max_pool(cluster, original_cell, transform=None)
            original_cell.x = self.bns_cell[i](original_cell.x)

        # torch.Size([256, 1848])
        node_representation = original_cell.x.reshape(-1, self.final_node * self.dim_cell)

        return node_representation

class StructuralEncoder(nn.Module):
    def __init__(self, pathway_dim, cluster_predefine):
        super(StructuralEncoder, self).__init__()
        # -------drug_layer
        self.drug_encoder = GNN_drug()

        # -------gene_expression_layer
        self.pathway = nn.Sequential(
            nn.Linear(pathway_dim, 1024),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

        self.gat = GNN_cell(1, 3, 8, cluster_predefine)

        self.gat_emb = nn.Sequential(
            nn.Linear(8 * self.gat.final_node, 1024),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        self.ban = weight_norm(BANLayer(v_dim=128, q_dim=128), name='h_mat')

    def forward(self, drug, cell):
        x_drug = self.drug_encoder(drug.x, drug.edge_index, drug.batch)

        x_cell = []
        x_path = self.pathway(cell.pathway)
        x_cell.append(x_path)
        x2_cell = self.gat(cell)
        x2_cell = self.gat_emb(x2_cell)
        x_cell.append(x2_cell)

        x_cell = torch.stack(x_cell, dim=1)
        x_drug = x_drug.unsqueeze(1)
        f, att = self.ban(x_drug, x_cell)

        return f

class MyModel(nn.Module):
    def __init__(self, cluster_predefine, cell_dim = 688, pathway_dim = 1307, is_regression = False, mask_rate=0.15):
        super(MyModel, self).__init__()
        self.is_regression = is_regression
        self.mask_rate = mask_rate
        self.fusion = ContrastFusion()
        self.contrast_loss = ContrastLoss(256)
        self.structural_encoder_share = StructuralEncoder(pathway_dim, cluster_predefine)
        self.structural_encoder_private = StructuralEncoder(pathway_dim, cluster_predefine)
        self.context_encoder_share = ContextEncoder(cell_dim)
        self.context_encoder_private = ContextEncoder(cell_dim)
        self.context_encoder = nn.Sequential(
            nn.Linear(256, 256),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(256, 256)
        )
        self.context_decoder = nn.Sequential(
            nn.Linear(256, 256),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(256, 256)
        )
        self.structural_encoder = nn.Sequential(
            nn.Linear(256, 256),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(256, 256)
        )
        self.structural_decoder = nn.Sequential(
            nn.Linear(256, 256),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(256, 256)
        )

        self.fc_cls_out_f = Classifier(512, 2)
        self.fc_cls_out_c = Classifier(512, 2)
        self.fc_cls_out_s = Classifier(512, 2)

        self.fc_reg_out_f = Classifier(512, 1)
        self.fc_reg_out_c = Classifier(512, 1)
        self.fc_reg_out_s = Classifier(512, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, context_input, structure_input):

        context_shared = self.context_encoder_share(*context_input)
        context_private = self.context_encoder_private(*context_input)
        context_mask = generate_mask(context_private, self.mask_rate)
        context_masked = context_private * context_mask
        context_encoded = self.context_encoder(context_masked)
        context_decoded = self.context_decoder(context_encoded)

        structure_shared = self.structural_encoder_share(*structure_input)
        structure_private = self.structural_encoder_private(*structure_input)
        structure_mask = generate_mask(structure_private, self.mask_rate)
        structure_masked = structure_private * structure_mask
        structure_encoded = self.structural_encoder(structure_masked)
        structure_decoded = self.structural_decoder(structure_encoded)

        cl_loss = self.contrast_loss(context_shared, structure_shared, context_private, structure_private)

        # Context feature prediction: shared + private
        context_pred_feature = torch.cat([context_shared, context_private], dim=1)

        # Structural feature prediction: shared + private
        structure_pred_feature = torch.cat([structure_shared, structure_private], dim=1)
        fusion_feature = self.fusion(structure_pred_feature, context_pred_feature)

        if self.is_regression:
            context_cdr = self.fc_reg_out_c(context_pred_feature)
            structure_cdr = self.fc_reg_out_s(structure_pred_feature)
            cdr = self.fc_reg_out_f(fusion_feature)
        else:
            context_cdr = self.fc_cls_out_c(context_pred_feature)
            structure_cdr = self.fc_cls_out_s(structure_pred_feature)
            cdr = self.fc_cls_out_f(fusion_feature)

        return cdr, context_cdr, structure_cdr, cl_loss, context_shared, context_private, structure_shared, structure_private, context_decoded, structure_decoded, context_mask, structure_mask

class ModelUtil:
    def __init__(self, device, lr, weight_decay, batch_size, cluster_predefine, cell_dim, pathway_dim, is_regression = False):
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.is_regression = is_regression
        self.model = MyModel(is_regression=self.is_regression, cluster_predefine=cluster_predefine,
                             cell_dim=cell_dim, pathway_dim=pathway_dim).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.l1 = 1
        self.l2 = 1
        self.l3 = 1.25
        self.l4 = 0.5
        self.l5 = 0.5
        self.l6 = 0.5

    def train(self, train_loader):
        self.model.train()
        for idx, data in enumerate(tqdm(train_loader, desc='training', total=len(train_loader))):
            drug, cell, label, pubchem, depmap = data
            drug = drug.to(self.device)
            cell = cell.to(self.device)
            label = label.to(self.device)

            context_input = [cell.gene_expression, drug]
            structure_input = [drug, cell]
            fusion_cdr, context_cdr, structure_cdr, cl_loss, context_shared, context_private, structure_shared, structure_private, context_decoded, structure_decoded, context_mask, structure_mask = self.model(
                context_input, structure_input)

            if self.is_regression:
                fusion_cdr = fusion_cdr.squeeze()
                context_cdr = context_cdr.squeeze()
                structure_cdr = structure_cdr.squeeze()
                loss1 = F.mse_loss(fusion_cdr, label)
                loss2 = F.mse_loss(context_cdr, label)
                loss3 = F.mse_loss(structure_cdr, label)
            else:
                loss1 = F.cross_entropy(fusion_cdr, label)
                loss2 = F.cross_entropy(context_cdr, label)
                loss3 = F.cross_entropy(structure_cdr, label)
            loss4 = cl_loss

            context_mask_loss = calc_masked_loss(context_decoded, context_private, context_mask)
            struct_mask_loss = calc_masked_loss(structure_decoded, structure_private, structure_mask)
            loss5 = context_mask_loss + struct_mask_loss

            loss = (loss1 * self.l1 + loss2 * self.l2 + loss3 * self.l3 +
                    loss4 * self.l4 + loss5 * self.l5)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def validate(self, test_loader):
        y_true = []
        y_pred = []
        y1_pred = []
        y2_pred = []
        pubchems = []
        depmaps = []
        self.model.eval()
        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                drug, cell, label, pubchem, depmap = data
                drug = drug.to(self.device)
                cell = cell.to(self.device)
                label = label.to(self.device)

                context_input = [cell.gene_expression, drug]
                structure_input = [drug, cell]
                fusion_cdr, context_cdr, structure_cdr, cl_loss, context_shared, context_private, structure_shared, structure_private, context_decoded, structure_decoded, context_mask, structure_mask = self.model(
                    context_input, structure_input)
                if self.is_regression:
                    y_true.append(label)
                    y_pred.append(fusion_cdr)
                    y1_pred.append(context_cdr)
                    y2_pred.append(structure_cdr)
                    pubchems = pubchems + pubchem
                    depmaps = depmaps + depmap
                else:
                    ys1 = F.softmax(fusion_cdr, 1)
                    ys2 = F.softmax(context_cdr, 1)
                    ys3 = F.softmax(structure_cdr, 1)
                    y_true.append(label)
                    y_pred.append(ys1)
                    y1_pred.append(ys2)
                    y2_pred.append(ys3)

        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)
        y1_pred = torch.cat(y1_pred, dim=0)
        y2_pred = torch.cat(y2_pred, dim=0)
        y_true = y_true.to('cpu')
        y_pred = y_pred.to('cpu')
        y1_pred = y1_pred.to('cpu')
        y2_pred = y2_pred.to('cpu')
        if self.is_regression:
            y_true = y_true.squeeze()
            y_pred = y_pred.squeeze()
            y1_pred = y1_pred.squeeze()
            y2_pred = y2_pred.squeeze()
            mse, rmse, mae, r2, pearson, pearson_p_value, spearman, spearman_p_value = eval_predict(y_true, y_pred)
            row1 = [mse, rmse, mae, r2, pearson, spearman]
            final_mse = mse
            mse, rmse, mae, r2, pearson, pearson_p_value, spearman, spearman_p_value = eval_predict(y_true, y1_pred)
            row2 = [mse, rmse, mae, r2, pearson, spearman]
            mse, rmse, mae, r2, pearson, pearson_p_value, spearman, spearman_p_value = eval_predict(y_true, y2_pred)
            row3 = [mse, rmse, mae, r2, pearson, spearman]
            return row1, row2, row3, final_mse
        auc, aupr, precision, recall, f1, acc = metrics_cls(y_true=y_true, y_predict=y_pred)
        row1 = [auc, aupr, precision, recall, f1, acc]
        final_auc = auc
        auc, aupr, precision, recall, f1, acc = metrics_cls(y_true=y_true, y_predict=y1_pred)
        row2 = [auc, aupr, precision, recall, f1, acc]
        auc, aupr, precision, recall, f1, acc = metrics_cls(y_true=y_true, y_predict=y2_pred)
        row3 = [auc, aupr, precision, recall, f1, acc]
        return row1, row2, row3, final_auc

    def save_model(self, mode, fold):
        torch.save(self.model.state_dict(), mode + '_model_' + str(fold) + '.pth')
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
