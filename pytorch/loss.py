
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

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):


    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)  # 合并在一起

    total0 = total.unsqueeze(0).expand(int(total.size(0)),
                                       int(total.size(0)),
                                       int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)),
                                       int(total.size(0)),
                                       int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2) 

    
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
                  bandwidth_temp in bandwidth_list]

    return sum(kernel_val)  

def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    source = source.flatten(start_dim = 1)
    target = target.flatten(start_dim = 1)
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul,
                              kernel_num=kernel_num,
                              fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]  # Source<->Source
    YY = kernels[batch_size:, batch_size:]  # Target<->Target
    XY = kernels[:batch_size, batch_size:]  # Source<->Target
    YX = kernels[batch_size:, :batch_size]  # Target<->Source
    loss = torch.mean(XX + YY - XY - YX)  
    return loss
