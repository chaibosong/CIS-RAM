import torch

from models import VGG_mm
from models.ResNet import *
from models.VGG import *
from models.MLP import *
from models.DeepSVDD import *
from models.ResNet import *
from models.VGG_mm import *
from models.DeepSVDD_mm import *
from models.HNet import *


# 跨模态自注意力模块
class crossAtnBlock(nn.Module):
    def __init__(self, in_channels=1024, inter_channels=None, mode='concatenate',
                 dimension=2, bn_layer=True):
        """Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        super(crossAtnBlock, self).__init__()

        assert dimension in [1, 2, 3]

        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.W_z = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                bn(self.in_channels)
            )
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                nn.ReLU()
            )

    def forward(self, branch_one, branch_two):
        # img:  b 1024 14 14
        # hrrp: b 1024
        # img for g and theta
        # hrrp for phi
        """
        args
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """

        batch_size = branch_one.size(0)

        # (N, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
        g_x = self.g(branch_one).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = branch_one.view(batch_size, self.in_channels, -1)
            phi_x = branch_two.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(branch_one).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(branch_two).view(batch_size, self.inter_channels, -1)
            # theta_x = theta_x.permute(0, 2, 1)
            phi_x = phi_x.permute(0, 2, 1)
            f = torch.matmul(phi_x, theta_x)

        elif self.mode == "concatenate":
            theta_x = self.theta(branch_one).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(branch_two).view(batch_size, self.inter_channels, 1, -1)

            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)

            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))

        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1)  # number of position in x
            f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)

        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *branch_one.size()[2:])

        W_y = self.W_z(y)

        # residual connection
        z = W_y + branch_one

        return z


class MultiModel(nn.Module):
    def __init__(self, model1="resnet50", model2="hnet", num_classes=2, loadPretrain=0):
        super(MultiModel, self).__init__()
        self.num_classes = num_classes

        self.model1 = model1
        self.model2 = model2

        if self.model1 == "resnet50":
            self.base1 = resnet50(pretrained=(loadPretrain == 1), num_classes=self.num_classes)[0]
        elif self.model1 == "vgg16":
            self.base1 = VGG_mm.vgg16(pretrained=(loadPretrain == 1), number_classes=self.num_classes)
        elif self.model1 == "resnet18":
            self.base1 = resnet18(pretrained=(loadPretrain == 1), num_classes=self.num_classes)

        if self.model2 == "hnet":
            self.base2 = HNet(self.num_classes)
        elif self.model2 == "deepsvdd":
            self.base2 = DeepSVDDNetwork_mm(self.num_classes)
        elif self.model2 == "mlp":
            self.base2 = MLP(self.num_classes)

        # 特征融合模块
        self.crossAttn = crossAtnBlock(mode='concatenate')

        # 深层特征提取模块
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(3136, 2048, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(2048, self.num_classes, bias=False)
            # nn.ReLU(inplace=True),
            # nn.Linear(16, self.num_classes)
        )

        self.relu = nn.ReLU()

    def forward(self, img, hrrp):

        # 提取红外图像特征
        feature_img = self.base1(img)
        # print(feature_img.shape)

        # 提取雷达信号特征
        feature_hrrp = self.base2(hrrp)
        # print(feature_hrrp.shape)

        feature_hrrp = feature_hrrp.view(feature_hrrp.size(0), feature_hrrp.size(1), 1, 1)

        feature_hrrp = feature_hrrp.repeat(1, 1, feature_img.size(2), feature_img.size(3))

        # 将红外图像特征和雷达特征输入到融合模块进行特征融合
        out1 = self.crossAttn(feature_img, feature_hrrp)
        out1 = self.relu(out1)

        out2 = self.crossAttn(feature_hrrp, feature_img)
        out2 = self.relu(out2)

        out = torch.cat((out1, out2), dim=1)
        out = self.extractor(out)
        # print('out.shape: ', out.shape)

        out = out.view(out.size()[0], -1)
        out = self.relu(out)
        # print('out.shape', out.shape)

        # 送入分类器
        cls = self.classifier(out)

        return cls


if __name__ == '__main__':
    x1 = torch.rand(1, 3, 224, 224)
    x2 = torch.rand(1, 128)

    model = MultiModel(num_classes=2)

    y = model(x1, x2)

    print(y)
    print(y.shape)