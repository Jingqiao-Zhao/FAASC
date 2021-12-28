import math
from torch.nn.utils import spectral_norm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from einops import rearrange

#from BorderDet.cvpods.layers import *


class Dc_Conv_Block(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1)
                 ):

        super(Dc_Conv_Block, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv1 = ModulatedDeformConvWithOff(in_channels=self.in_channels, out_channels=self.out_channels,
                                   kernel_size=3, stride=1, padding=1,
                                   dilation=1, deformable_groups=1)
        self.conv2 = ModulatedDeformConvWithOff(in_channels=self.out_channels, out_channels=self.out_channels,
                                   kernel_size=3, stride=1, padding=1,
                                   dilation=1, deformable_groups=1)

        #self.atten = Attention(dim = dim)
        self.selayer = SELayer(out_channels,reduction=16)
        #self.att = CBAMBlock(channel= out_channels,reduction = 16 )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)



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



def init_layer(layer, nonlinearity='leaky_relu'):
    """Initialize a Linear or Convolutional layer. """
    nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

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

class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Attention(nn.Module):
    def __init__(self, dim, heads=1, feature=0):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)
        self.dim = dim
        self.to_qkv = nn.Linear(self.dim, self.dim * 3, bias=True)  # Wq,Wk,Wv for each vector, thats why *3
        self.feature = feature
        self.pos_embedding = nn.Parameter(torch.empty(1, 1, self.feature, self.feature))
        self.nn1 = nn.Linear(self.dim, self.dim)
        self.sigmoid = nn.Sigmoid()


    def forward(self, input):
        x = input

        x = torch.mean(x,dim=2)

        x = rearrange(x,'b t f -> b f t')

        qkv = self.to_qkv(x)

        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)  # split into multi head attentions

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        dots += self.pos_embedding

        attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper

        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # product of v times whatever inside softmax

        out = rearrange(out, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block

        out = self.nn1(out)

        out = rearrange(out, 'b f t -> b t f')

        out = input*out[:,:,None,:]

        return out

class CBAMBlock(nn.Module):
    def __init__(self, channel, reduction=2):
        super(CBAMBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel // reduction), bias=False),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel // reduction), channel, bias=False),
                                                )
        self.sigmoid = nn.Sigmoid()

        self.spatial_excitation = nn.Sequential(nn.Conv2d(2, 1, kernel_size=7,
                                                          stride=1, padding=3, bias=False),
                                                )

    def forward(self, x):
        bahs, chs, _, _ = x.size()

        # Returns a new tensor with the same data as the self tensor but of a different size.
        chn_avg = self.avg_pool(x).view(bahs, chs)
        chn_avg = self.channel_excitation(chn_avg).view(bahs, chs, 1, 1)
        chn_max = self.max_pool(x).view(bahs, chs)
        chn_max = self.channel_excitation(chn_max).view(bahs, chs, 1, 1)
        chn_add = chn_avg + chn_max
        chn_add = self.sigmoid(chn_add)

        chn_cbam = torch.mul(x, chn_add)

        avg_out = torch.mean(chn_cbam, dim=1, keepdim=True)
        max_out, _ = torch.max(chn_cbam, dim=1, keepdim=True)
        cat = torch.cat([avg_out, max_out], dim=1)
        spa_add = self.spatial_excitation(cat)
        spa_add = self.sigmoid(spa_add)

        spa_cbam = torch.mul(chn_cbam, spa_add)

        return spa_cbam

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

        #self.atten = Attention(dim = dim)
        self.selayer = SELayer(out_channels,reduction=16)
        #self.att = CBAMBlock(channel= out_channels,reduction = 16 )

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
        #x = self.atten(x)
        x = F.avg_pool2d(x, kernel_size=pool_size)

        return x

class Encoder_ConvBlock_Attention(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 dim = 0,feature = 0,attn=True):

        super(Encoder_ConvBlock_Attention, self).__init__()


        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dim = dim
        self.atten = attn
        self.feature = feature
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

        self.atten = Attention(dim = self.dim,feature = feature)
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
        #x = self.selayer(x)


        #x = self.atten(x)

        x = F.avg_pool2d(x, kernel_size=pool_size)

        return x

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

class Cnn_4layers_Encoder(nn.Module):
    def __init__(self,inchannels = 1):
        super(Cnn_4layers_Encoder, self).__init__()


        self.conv_block1 = Encoder_ConvBlock(in_channels=inchannels,
                                             out_channels=64,
                                             kernel_size=(5, 5),
                                             stride=(1, 1),
                                             padding=(2, 2))

        self.conv_block2 = Encoder_ConvBlock(in_channels=64,
                                             out_channels=128,
                                             kernel_size=(3, 3),
                                             stride=(1, 1),
                                             padding=(1, 1))

        self.conv_block3 = Encoder_ConvBlock(in_channels=128,
                                             out_channels=256,
                                             kernel_size=(3, 3),
                                             stride=(1, 1),
                                             padding=(1, 1))
        '''
        self.conv_block4 = Encoder_ConvBlock(in_channels=inchannels,
                                             out_channels=64,
                                             kernel_size=(5, 5),
                                             stride=(1, 1),
                                             padding=(2, 2))

        self.conv_block5 = Encoder_ConvBlock(in_channels=64,
                                             out_channels=128,
                                             kernel_size=(3, 3),
                                             stride=(1, 1),
                                             padding=(1, 1))

        self.conv_block6 = Encoder_ConvBlock(in_channels=128,
                                             out_channels=256,
                                             kernel_size=(3, 3),
                                             stride=(1, 1),
                                             padding=(1, 1))
        '''
    def forward(self, input):

        #x = input[:,:,:,:48]
        #y = input[:, :, :,16:]

        x = input
        x = self.conv_block1(x, pool_size=(4, 4),act='relu')
        x = self.conv_block2(x, pool_size=(4, 4),act='relu')
        x = self.conv_block3(x, pool_size=(2, 2),act='relu')
        '''
        y = self.conv_block1(y, pool_size=(4, 4),act='leaky_relu')
        y = self.conv_block2(y, pool_size=(4, 4),act='leaky_relu')
        y = self.conv_block3(y, pool_size=(2, 2),act='leaky_relu')
        '''

        #output = torch.cat([x,y],dim = 1)
        output = x
        return output

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

class Cnn_Encoder_Attention(nn.Module):
    def __init__(self,inchannels = 1):
        super(Cnn_Encoder_Attention,self).__init__()

        self.conv1 = Encoder_ConvBlock_Attention(in_channels=inchannels,
                                             out_channels=32,
                                             kernel_size=(5, 5),
                                             stride=(1, 1),
                                             padding=(2, 2),
                                             dim = 32,feature = 64,attn=False)
        self.conv2 = Encoder_ConvBlock_Attention(in_channels=32,
                                             out_channels=64,
                                             kernel_size=(3, 3),
                                             stride=(1, 1),
                                             padding=(1, 1),
                                             dim = 64,feature = 16,attn=False)
        self.conv3 = Encoder_ConvBlock_Attention(in_channels=64,
                                             out_channels=128,
                                             kernel_size=(3, 3),
                                             stride=(1, 1),
                                             padding=(1, 1),
                                             dim = 128,feature = 4,attn=True)

    def forward(self, input):

        x = input

        x1 = self.conv1(x, pool_size=(4, 4),act='relu')
        x2 = self.conv2(x1, pool_size=(4 ,4),act='relu')
        x3 = self.conv3(x2, pool_size=(4, 4),act='relu')

        return x1,x2,x3

class Dense_layer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Dense_layer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Conv2d(in_channels=self.in_channels,
                              out_channels=self.out_channels,
                              kernel_size=1,
                              stride=1,
                              padding=0)

        self.bn = nn.BatchNorm2d(self.out_channels)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv)
        init_bn(self.bn)

    def forward(self, input,softmax=True):

        x = self.bn(self.conv(input))
        x = F.relu_(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))

        x = x.view(-1, self.out_channels)

        x = F.log_softmax(x, dim=-1)

        return x

class FeatureT(nn.Module):
    def __init__(self,in_domain_classes_num=10, activation='logsoftmax'):
        super(FeatureT, self).__init__()

        self.encoder1 = Cnn_8layers_Encoder(inchannels=3)

        self.discribe1 = discribe(10, channel=512)

    def forward(self,x,return_feature = False):

        x1 = torch.cat([x[:, None, 4:-4, :],deltas(x)[:, None, 2:-2, :],deltas(deltas(x),)[:,None,:,:]],dim=1)

        hidden11,hidden12,hidden13 = self.encoder1(x1)

        x1 = self.discribe1(hidden13)

        if return_feature == True :

            return hidden11 ,hidden12,hidden13

        if return_feature == False :


             output = x1 + x1

             return output, x1,x1

class Feature_Attention(nn.Module):
    def __init__(self,in_domain_classes_num=10, activation='logsoftmax'):
        super(Feature_Attention, self).__init__()

        self.encoder1 = Cnn_Encoder_Attention(inchannels=2)
        self.encoder2 = Cnn_Encoder_Attention(inchannels=1)

        self.discribe1 = discribe(10, channel=128)
        self.discribe2 = discribe(10, channel=128)

    def forward(self,x,return_feature = False):

        x1 = torch.cat([deltas(x)[:, None, 2:-2, :],deltas(deltas(x),)[:,None,:,:]],dim=1)
        x2 = x[:, None, 4:-4, :]

        hidden11, hidden12, hidden13 = self.encoder1(x1)
        hidden21, hidden22, hidden23 = self.encoder2(x2)

        if return_feature == True :

            return hidden11+hidden21,hidden12+hidden22,hidden13+hidden23

        if return_feature == False :

             x1 = self.discribe1(hidden13)
             x2 = self.discribe2(hidden23)

             output = x1+x2

             return output, x1,x2

class Feature_focal(nn.Module):
    def __init__(self,in_domain_classes_num=10, activation='logsoftmax'):
        super(Feature_focal, self).__init__()



        self.encoder1 = Cnn_8layers_Encoder(inchannels=2)
        self.encoder2 = Cnn_8layers_Encoder(inchannels=1)


        self.discribe1 = discribe(10, channel=256)
        self.discribe2 = discribe(10, channel=256)


    def forward(self,x,return_feature = False):


        x1 = torch.cat([deltas(x)[:, None, 2:-2, :],deltas(deltas(x),)[:,None,:,:]],dim=1)

        x2 = x[:, None, 4:-4, :]

        hidden11,hidden12,hidden13 = self.encoder1(x1)
        hidden21,hidden22,hidden23 = self.encoder2(x2)

        x1 = self.discribe1(hidden13,softmax=False)
        x2 = self.discribe2(hidden23,softmax=False)


        if return_feature == True :

            return hidden11+hidden21,hidden12+hidden22,hidden13+hidden23

        if return_feature == False :

             output = x1 + x2

             return output, x1,x2

class Cnn_9layers_AvgPooling(nn.Module):

    def __init__(self, classes_num=10):
        super(Cnn_9layers_AvgPooling, self).__init__()



        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc)

    def forward(self, input,return_feature=False):
        '''
        Input: (batch_size, times_steps, freq_bins)'''

        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        feature = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        if return_feature==True:
            return feature
        elif return_feature == False:
            x = nn.functional.adaptive_avg_pool2d(feature, (1, 1))

            x = x.view(-1, 512)

            x = self.fc(x)

            output = F.log_softmax(x, dim=-1)

            return output

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception('Incorrect argument!')

        return x

class Cnn_8layers_Encoder_Dc(nn.Module):
    def __init__(self, inchannels=1):
        super(Cnn_8layers_Encoder_Dc, self).__init__()

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
        self.conv3 = Dc_Conv_Block(in_channels=128,
                                       out_channels=256,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

    def forward(self, input):
        x = input

        x1 = self.conv1(x, pool_size=(4, 4), act='relu')
        x2 = self.conv2(x1, pool_size=(4, 4), act='relu')
        x3 = self.conv3(x2, pool_size=(2, 2), act='relu')

        return x1, x2, x3

class FeatureDC(nn.Module):
    def __init__(self, in_domain_classes_num=10, activation='logsoftmax'):
        super(FeatureDC, self).__init__()

        self.encoder1 = Cnn_8layers_Encoder_Dc(inchannels=2)
        self.encoder2 = Cnn_8layers_Encoder_Dc(inchannels=1)

        self.discribe1 = discribe(10, channel=256)
        self.discribe2 = discribe(10, channel=256)

    def forward(self, x, return_feature=False):

        x1 = torch.cat([deltas(x)[:, None, 2:-2, :], deltas(deltas(x), )[:, None, :, :]], dim=1)

        x2 = x[:, None, 4:-4, :]

        hidden11, hidden12, hidden13 = self.encoder1(x1)
        hidden21, hidden22, hidden23 = self.encoder2(x2)

        x1 = self.discribe1(hidden13)
        x2 = self.discribe2(hidden23)

        if return_feature == True:
            return hidden11 + hidden21, hidden12 + hidden22, hidden13 + hidden23

        if return_feature == False:
            output = x1 + x2

            return output, x1, x2


























































































