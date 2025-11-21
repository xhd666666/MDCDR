import math

import torch
from torch import nn
from torch_geometric.nn import global_mean_pool, GATConv, max_pool
import torch.nn.functional as F

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

# EGNN
class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    re
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr

class EGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, act_fn=nn.SiLU(), n_layers=4, residual=True, attention=False, normalize=False, tanh=False):
        """

        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param in_edge_nf: Number of features for the edge features
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                        phi_x(m_ij) which definitely improves in stability, but it may decrease in accuracy.
                        We didn't use it in our paper.
        """

        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                act_fn=act_fn, residual=residual, attention=attention,
                                                normalize=normalize, tanh=tanh))

    def forward(self, h, x, edges, edge_attr, batch):
        h = self.embedding_in(h)
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr)
        h = self.embedding_out(h)
        h = global_mean_pool(h, batch)
        return h, x

def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result

def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


