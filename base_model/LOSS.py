import torch
import torch.nn.functional as F
from torch import nn

class ContrastLoss(nn.Module):
    def __init__(self, hidden_dim, tau=0.5):
        super(ContrastLoss, self).__init__()
        self.tau = tau

    def forward(self, context_shared, structure_shared, context_private, structure_private):
        loss1 = self.InfoNCE(context_shared, structure_shared, 0.6)
        loss2 = self.negative_InfoNCE(context_private, structure_private, 1)
        loss3_1 = self.negative_InfoNCE(context_shared, context_private, 0.8)
        loss3_2 = self.negative_InfoNCE(structure_shared, structure_private, 0.8)
        loss3 = (loss3_1 + loss3_2) / 2
        return loss1 + loss2 + loss3

    def InfoNCE(self, context_shared, structure_shared, temperature=0.5):
        batch_size = context_shared.size(0)
        context_norm = F.normalize(context_shared, p=2, dim=1)
        struct_norm = F.normalize(structure_shared, p=2, dim=1)
        similarity_matrix = torch.matmul(context_norm, struct_norm.t()) / temperature
        labels = torch.arange(batch_size).to(context_shared.device)
        loss_i = F.cross_entropy(similarity_matrix, labels)
        loss_j = F.cross_entropy(similarity_matrix.t(), labels)

        return (loss_i + loss_j) / 2

    def negative_InfoNCE(self, context_private, structure_private, temperature=0.5):
        batch_size = context_private.size(0)
        context_norm = F.normalize(context_private, p=2, dim=1)
        struct_norm = F.normalize(structure_private, p=2, dim=1)
        similarity_matrix = torch.matmul(context_norm, struct_norm.t()) / temperature
        labels = torch.arange(batch_size).to(context_private.device)
        negative_similarity = -similarity_matrix
        loss_i = F.cross_entropy(negative_similarity, labels)
        loss_j = F.cross_entropy(negative_similarity.t(), labels)

        return (loss_i + loss_j) / 2


