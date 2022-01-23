import math
from torch.nn.utils import spectral_norm
import torch
import torch.nn as nn
import torch.nn.functional as F

def init_layer(layer, nonlinearity='leaky_relu'):
    """Initialize a Linear or Convolutional layer. """
    nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

def init_bn(bn):
    """Initialize a Batchnorm layer. """

    bn.bias.data.fill_(0.)
    bn.running_mean.data.fill_(0.)
    bn.weight.data.fill_(1.)
    bn.running_var.data.fill_(1.)

def deltas(X_in):

    X_out = (X_in[:, 2:, :] - X_in[:, :-2, :])/10.0
    X_out = X_out[:, 1:-1, :] + (X_in[:, 4:, :] - X_in[:, :-4, :])/5.0


    return X_out

class Encoder_ConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1)
                 ):

        super(Encoder_ConvBlock, self).__init__()


        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=self.kernel_size,
                               stride=self.stride,
                               padding=self.padding, bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=self.kernel_size,
                               stride=self.stride,
                               padding=self.padding, bias=False)


        self.selayer = SELayer(out_channels,reduction=16)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)


        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_layer(self.conv2)


    def forward(self, input, pool_size=(2, 2),act = 'relu'):

        x = input

        if act =='relu':
            x = F.relu_(self.bn1(self.conv1(x)))
            x = F.relu_(self.bn2(self.conv2(x)))

        elif act =='leaky_relu':
            x = F.leaky_relu_(self.bn1(self.conv1(x)))
            x = F.leaky_relu_(self.bn2(self.conv2(x)))

        x = self.selayer(x)
        x = F.avg_pool2d(x, kernel_size=pool_size)

        return x

class FeatureS(nn.Module):
    def __init__(self,in_domain_classes_num=10, activation='logsoftmax'):
        super(FeatureS, self).__init__()

        self.encoder1 = Cnn_8layers_Encoder(inchannels=2)
        self.encoder2 = Cnn_8layers_Encoder(inchannels=1)

        self.discribe1 = discribe(10, channel=256)
        self.discribe2 = discribe(10, channel=256)

    def forward(self,x,return_feature = False):


        x1 = torch.cat([deltas(x)[:, None, 2:-2, :],deltas(deltas(x),)[:,None,:,:]],dim=1)

        x2 = x[:, None, 4:-4, :]

        hidden11,hidden12,hidden13 = self.encoder1(x1)
        hidden21,hidden22,hidden23 = self.encoder2(x2)

        x1 = self.discribe1(hidden13)
        x2 = self.discribe2(hidden23)


        if return_feature == True :

            return hidden11 + hidden21,hidden12+hidden22,hidden13+hidden23

        if return_feature == False :


             output = x1 + x2

             return output, x1,x2

class discribe(nn.Module):

    def __init__(self, classes_num, channel):
        super(discribe, self).__init__()

        self.classes_num = classes_num
        self.channel = channel
        self.fc1 = nn.Linear(self.channel, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc1)

    def forward(self, input):

        x = input
        '''
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))

        x = x.view(-1, self.channel)
        '''
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)

        x = self.fc1(x1)

        x = F.log_softmax(x,dim=-1)

        return x

class Cnn_8layers_Encoder(nn.Module):
    def __init__(self,inchannels = 1):
        super(Cnn_8layers_Encoder,self).__init__()

        self.conv1 = Encoder_ConvBlock(in_channels=inchannels,
                                             out_channels=64,
                                             kernel_size=(5, 5),
                                             stride=(1, 1),
                                             padding=(2, 2))
        self.conv2 = Encoder_ConvBlock(in_channels=64,
                                             out_channels=128,
                                             kernel_size=(3, 3),
                                             stride=(1, 1),
                                             padding=(1, 1))
        self.conv3 = Encoder_ConvBlock(in_channels=128,
                                             out_channels=256,
                                             kernel_size=(3, 3),
                                             stride=(1, 1),
                                             padding=(1, 1))

    def forward(self, input):

        x = input

        x1 = self.conv1(x, pool_size=(4, 4),act='relu')
        x2 = self.conv2(x1, pool_size=(4 ,4),act='relu')
        x3 = self.conv3(x2, pool_size=(2, 2),act='relu')

        return x1,x2,x3
