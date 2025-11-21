from torch import nn


class ContrastFusion(nn.Module):
    def __init__(self, dropout=0.05):
        super(ContrastFusion, self).__init__()
        self.so_L = nn.Sigmoid()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, lm_fea, sty_fea):
        lm_att = self.so_L(lm_fea)
        fus_fea = lm_fea + sty_fea * lm_att
        return fus_fea
