import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from measure import *
from vgg import Vgg16

class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-10 #

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])  #算出总共求了多少次差
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        # x[:,:,1:,:]-x[:,:,:h_x-1,:]就是对原图进行错位，分成两张像素位置差1的图片，第一张图片
        # 从像素点1开始（原图从0开始），到最后一个像素点，第二张图片从像素点0开始，到倒数第二个
        # 像素点，这样就实现了对原图进行错位，分成两张图的操作，做差之后就是原图中每个像素点与相
        # 邻的下一个像素点的差。
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

    def tv_loss(Y_hat):
        return  (F.l1_loss(Y_hat[:, :, 1:, :], Y_hat[:, :, :-1, :]) +
                  F.l1_loss(Y_hat[:, :, :, 1:], Y_hat[:, :, :, :-1]))

class content_loss(torch.nn.Module):
# referred from https://github.com/dxyang/StyleTransfer/blob/master/style.py
    def __init__(self):
        super(content_loss, self).__init__()
        # dtype = torch.cuda.FloatTensor
        self.vgg = Vgg16().cuda()
        self.CONTENT_WEIGHT = 1e-2 #
        self.loss_mse = torch.nn.MSELoss()
    def forward(self, X, Y):
        x = torch.cat((X, X, X), dim=1)
        y = torch.cat((Y, Y, Y), dim=1)
        x_features = self.vgg(x)
        y_features = self.vgg(y)
        recon = x_features[2]# 1
        recon_hat = y_features[2]# 1
        content_loss = self.CONTENT_WEIGHT*self.loss_mse(recon_hat, recon)
        loss = 0
        loss += content_loss.item()
        return loss

class style_loss(torch.nn.Module):
    def __init__(self):
        super(style_loss, self).__init__()
        # dtype = torch.cuda.FloatTensor
        self.vgg = Vgg16().cuda()
        self.STYLE_WEIGHT = 1e0 #
        self.loss_mse = torch.nn.MSELoss()
    def gram_matrix(self,input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)
        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
        G = torch.mm(features, features.t())  # compute the gram product
        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)
    def forward(self, X, Y):
        x = torch.cat((X, X, X), dim=1)
        y = torch.cat((Y, Y, Y), dim=1)
        # x_features = self.vgg(x)
        y_features = self.vgg(y)#y is transferred image
        style_features = self.vgg(x)#x is style figure
        style_gram = [self.gram_matrix(fmap) for fmap in style_features]
        y_hat_gram = [self.gram_matrix(fmap) for fmap in y_features]
        style_loss = 0.0
        for j in range(4):
            style_loss += self.loss_mse(y_hat_gram[j], style_gram[j])#[:len(x)]
        style_loss = self.STYLE_WEIGHT * style_loss
        loss = 0
        loss += style_loss
        return loss

class Frobenius_loss(torch.nn.Module):
    def __init__(self):
        super(Frobenius_loss, self).__init__()

    def gram_matrix(self,input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)
        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
        G = torch.mm(features, features.t())  # compute the gram product
        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

    def forward(self, X, Y):
        x_gram = self.gram_matrix(X)
        # print(x_gram)
        y_gram = self.gram_matrix(Y)
        # print(x_gram)
        diff = torch.add(x_gram, -y_gram)
        loss = torch.norm(diff, p='fro')
        # loss = torch.linalg.matrix_norm(diff, p='fro')
        return loss


if __name__ == '__main__':
    m = Frobenius_loss()
    # m = style_loss()
    # m = content_loss()
    input1 = torch.randn(2, 1, 64, 64)
    input2 = torch.randn(2, 1, 64, 64)
    # x = torch.cat((input1, input1, input1), dim=1)
    # output = len(x)
    output = m(input1,input2)
    print(output.shape)
