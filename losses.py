import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sigmoid(tensor, temp=1.0):
    """ temperature controlled sigmoid
    takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
    """
    exponent = -tensor / temp
    # clamp the input tensor for stability
    exponent = torch.clamp(exponent, min=-50, max=50)
    y = 1.0 / (1.0 + torch.exp(exponent))
    return y

class SMRRLoss(torch.nn.Module):
    def __init__(self,anneal):
        super(SMRRLoss, self).__init__()
        self.anneal = anneal

    def forward(self, preds, labels):
        batch_size,num_items = preds.size()[0],preds.size()[1]
        label_value = torch.diag(preds[:, labels].squeeze(-1)) # (batch_size,)
        sim_diff = preds - label_value.squeeze(-1)[:,None] # (batch_size,num_items)
        sim_sg = sigmoid(sim_diff,temp=self.anneal)  # (batch_size,num_items)
        sim_all_rk = torch.sum(sim_sg, dim=-1) + 1 # (batch_size,)
        # k = int(torch.sum(sim_all_rk) / batch_size)
        k = 30000
        mrr_loss = torch.sum((k * (1 - 1 / sim_all_rk) + sim_all_rk * torch.log(sim_all_rk))/(sim_all_rk + k))
        return mrr_loss