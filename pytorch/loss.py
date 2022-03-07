
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def nll_loss(output, target):
    '''Negative likelihood loss. The output should be obtained using F.log_softmax(x).

    Args:
      output: (N, classes_num)
      target: (N, classes_num)
    '''

    loss = - torch.mean(target * output)

    return loss

def discrepancy(x1,x2):

    return torch.mean(torch.abs(F.softmax(x1,dim = -1) - F.softmax(x2,dim = -1)))

def klv(feature1,feature2):
    criterion = nn.KLDivLoss()
    return torch.mean(criterion(F.log_softmax(feature1,dim = -1),F.softmax(feature2,dim = -1)))

