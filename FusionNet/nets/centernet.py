import math

import torch.nn as nn
import torch.quantization
from torch import nn

from FusionNet.nets.hourglass import *
from FusionNet.nets.resnet50 import resnet50, resnet50_Decoder, resnet50_Head
from FusionNet.nets.hnet import *
from models.HNet_base import HNet_base

nums = 0
try:
    with open('model_data/my_classes.txt') as f:
        for i in f.readlines():
            nums += 1
except:
    nums = 2
    print()


# hnet_base = HNet_base(nums)
# hnet_base.load_state_dict(torch.load('./weights/hrrps_models/model_hnet.pth'),strict=False)

# hnet = HNet(nums)
# pretrained_dict = torch.load('./weights/hrrps_models/model_hnet.pth',strict=False)
# model_dict = hnet.state_dict()
#
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict
#                    and v.size() == model_dict[k].size()}
#
# model_dict.update(pretrained_dict)
# hnet.load_state_dict(model_dict)


# def draw_pk_map(xys, hrrps, length=4):
#     xys = xys.cpu().numpy()
#
#     pk_map = np.ones((xys.shape[0], 64, 128, 128), dtype=np.float32)
#
#     for i, xy in enumerate(xys):
#         for j, x_y in enumerate(xy):
#             if x_y[0] != -1:
#                 x, y = x_y
#                 x = int(x) // 4
#                 y = int(y) // 4
#
#                 hrrp_data = hrrps[i][j].type(torch.FloatTensor)
#                 fea_hrrp = np.float32(hnet(hrrp_data).detach().numpy())
#
#                 min_x = max(x - length, 0)
#                 max_x = min(x + length, 127)
#
#                 min_y = max(y - length, 0)
#                 max_y = min(y + length, 127)
#
#                 for x in range(min_x, max_x):
#                     for y in range(min_y, max_y):
#                         pk_map[i, :, x, y] = fea_hrrp
#
#     return pk_map


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


from torch.nn import Softmax


# 跨模态注意力模块
# class CC_module(nn.Module):
#     def __init__(self, in_dim, in_dim2):
#         super(CC_module, self).__init__()
#         self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_channels=in_dim2, out_channels=in_dim2, kernel_size=1)
#         self.softmax = Softmax(dim=3)
#         self.INF = INF
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#     def forward(self, x, x2):
#         m_batchsize, _, height, width = x.size()
#         proj_query = self.query_conv(x)
#         proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
#                                                                                                                  1)
#         proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
#                                                                                                                  1)
#         proj_key = self.key_conv(x)
#         proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
#         proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
#
#         proj_value = self.value_conv(x2)
#         proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
#         proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
#         energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width,
#                                                                                                      height,
#                                                                                                      height).permute(0,
#                                                                                                                      2,
#                                                                                                                      1,
#                                                                                                                      3)
#         energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
#         concate = self.softmax(torch.cat([energy_H, energy_W], 3))
#
#         att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
#         # print(concate)
#         # print(att_H)
#         att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
#         out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
#         out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
#         # print(out_H.size(),out_W.size())
#         return self.gamma * (out_H + out_W) + x

# class CenterNet_Resnet50(nn.Module):
#     def __init__(self, num_classes=20, pretrained=False):
#         super(CenterNet_Resnet50, self).__init__()
#         self.pretrained = pretrained
#         # 512,512,3 -> 16,16,2048
#         self.backbone = resnet50(pretrained=pretrained)
#         # 16,16,2048 -> 128,128,64
#         self.decoder = resnet50_Decoder(2048)
#         # -----------------------------------------------------------------#
#         #   对获取到的特征进行上采样，进行分类预测和回归预测
#         #   128, 128, 64 -> 128, 128, 64 -> 128, 128, num_classes
#         #                -> 128, 128, 64 -> 128, 128, 2
#         #                -> 128, 128, 64 -> 128, 128, 2
#         # -----------------------------------------------------------------#
#         self.head = resnet50_Head(channel=64, num_classes=num_classes)
#
#         self._init_weights()
#
#     def freeze_backbone(self):
#         for param in self.backbone.parameters():
#             param.requires_grad = False
#
#     def unfreeze_backbone(self):
#         for param in self.backbone.parameters():
#             param.requires_grad = True
#
#     def _init_weights(self):
#         if not self.pretrained:
#             for m in self.modules():
#                 if isinstance(m, nn.Conv2d):
#                     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                     m.weight.data.normal_(0, math.sqrt(2. / n))
#                 elif isinstance(m, nn.BatchNorm2d):
#                     m.weight.data.fill_(1)
#                     m.bias.data.zero_()
#
#         self.head.cls_head[-1].weight.data.fill_(0)
#         self.head.cls_head[-1].bias.data.fill_(-2.19)
#
#     def forward(self, img, xys, hrrps=None):
#         feat = self.backbone(img)
#         return self.head(self.decoder(feat))
def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        heatmap[y - top: y + bottom, x - left: x + right] = masked_heatmap
    return heatmap


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        out = self.avg_pool(x).view(b, c)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out).view(b, c, 1, 1)
        out = x * out.expand_as(x)
        return out


class ChannelAttention(nn.Module):  # Channel attention module
    def __init__(self, channels, ratio=16):  # r: reduction ratio=16
        super(ChannelAttention, self).__init__()

        hidden_channels = channels // ratio
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # global avg pool
        self.maxpool = nn.AdaptiveMaxPool2d(1)  # global max pool
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1, 1, 0, bias=False),  # 1x1conv代替全连接，根据原文公式没有偏置项
            nn.ReLU(inplace=True),  # relu
            nn.Conv2d(hidden_channels, channels, 1, 1, 0, bias=False)  # 1x1conv代替全连接，根据原文公式没有偏置项
        )
        self.sigmoid = nn.Sigmoid()  # sigmoid

    def forward(self, x):
        x_avg = self.avgpool(x)
        x_max = self.maxpool(x)
        return self.sigmoid(
            self.mlp(x_avg) + self.mlp(x_max)
        )  # Mc(F) = σ(MLP(AvgPool(F))+MLP(MaxPool(F)))= σ(W1(W0(Fcavg))+W1(W0(Fcmax)))，对应原文公式(2)


class SpatialAttention(nn.Module):  # Spatial attention module
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, 7, 1, 3, bias=False)  # 7x7conv
        self.sigmoid = nn.Sigmoid()  # sigmoid

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)  # 在通道维度上进行avgpool，(B,C,H,W)->(B,1,H,W)
        x_max = torch.max(x, dim=1, keepdim=True)[0]  # 在通道维度上进行maxpool，(B,C,H,W)->(B,1,H,W)
        return self.sigmoid(
            self.conv(torch.cat([x_avg, x_max],dim=1))
        )  # Ms(F) = σ(f7×7([AvgP ool(F);MaxPool(F)])) = σ(f7×7([Fsavg;Fsmax]))，对应原文公式(3)


class CBAM(nn.Module):  # Convolutional Block Attention Module
    def __init__(self, channels, ratio=16):
        super(CBAM, self).__init__()

        self.channel_attention = ChannelAttention(channels, ratio)  # Channel attention module
        self.spatial_attention = SpatialAttention()  # Spatial attention module

    def forward(self, x):
        f1 = self.channel_attention(x) * x  # F0 = Mc(F)⊗F，对应原文公式(1)
        f2 = self.spatial_attention(f1) * f1  # F00 = Ms(F0)⊗F0，对应原文公式(1)
        return f2


class CenterNet_Resnet50(nn.Module):
    def __init__(self, num_classes=20, pretrained=False):
        super(CenterNet_Resnet50, self).__init__()
        self.pretrained = pretrained
        # 512,512,3 -> 16,16,2048
        self.backbone = resnet50(pretrained=pretrained)
        # 16,16,2048 -> 128,128,64
        self.decoder = resnet50_Decoder(2048)
        # -----------------------------------------------------------------#
        #   对获取到的特征进行上采样，进行分类预测和回归预测
        #   128, 128, 64 -> 128, 128, 64 -> 128, 128, num_classes
        #                -> 128, 128, 64 -> 128, 128, 2
        #                -> 128, 128, 64 -> 128, 128, 2
        # -----------------------------------------------------------------#
        self.head = resnet50_Head(channel=64, num_classes=num_classes)
        self.sharpe_c = nn.Conv2d(128, 64, 1)
        self._init_weights()
        self.num_classes = num_classes
        self.relu = nn.ReLU()
        self.i = 1
        # new method
        self.cbma = CBAM(channels=2)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def _init_weights(self):
        if not self.pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

        self.head.cls_head[-1].weight.data.fill_(0)
        self.head.cls_head[-1].bias.data.fill_(-2.19)

    # def __init__(self, num_classes=20, pretrained=False):
    #     super(CenterNet_Resnet50, self).__init__()
    #     self.pretrained = pretrained
    #     # 512,512,3 -> 16,16,2048
    #     self.backbone = resnet50(pretrained=pretrained)
    #     # 16,16,2048 -> 128,128,64
    #     self.decoder = resnet50_Decoder(2048)
    #     self.sharpe_c = nn.Conv2d(128, 64, 1)
    #     # -----------------------------------------------------------------#
    #     #   对获取到的特征进行上采样，进行分类预测和回归预测
    #     #   128, 128, 64 -> 128, 128, 64 -> 128, 128, num_classes
    #     #                -> 128, 128, 64 -> 128, 128, 2
    #     #                -> 128, 128, 64 -> 128, 128, 2
    #     # -----------------------------------------------------------------#
    #     self.head = resnet50_Head(channel=64, num_classes=num_classes)
    #
    #     self.hnet = HNet()
    #
    #     self.conv = nn.Sequential(
    #         nn.BatchNorm2d(128),
    #         nn.ReLU(inplace=True),
    #         nn.Conv2d(128, 64, 1, stride=1, bias=False),
    #         nn.BatchNorm2d(64),
    #         nn.ReLU(inplace=True)
    #     )
    #     # self.attn = CC_module(64, 64)
    #
    #     self.extractor = nn.Sequential(
    #         nn.BatchNorm2d(128),
    #         nn.ReLU(inplace=True),
    #         nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
    #         nn.BatchNorm2d(64),
    #         nn.ReLU(inplace=True)
    #     )
    #
    #     self._init_weights()
    #
    # def freeze_backbone(self):
    #     for param in self.backbone.parameters():
    #         param.requires_grad = False
    #
    # def unfreeze_backbone(self):
    #     for param in self.backbone.parameters():
    #         param.requires_grad = True
    #
    # def _init_weights(self):
    #     if not self.pretrained:
    #         for m in self.modules():
    #             if isinstance(m, nn.Conv2d):
    #                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #                 m.weight.data.normal_(0, math.sqrt(2. / n))
    #             elif isinstance(m, nn.BatchNorm2d):
    #                 m.weight.data.fill_(1)
    #                 m.bias.data.zero_()
    #
    #     self.head.cls_head[-1].weight.data.fill_(0)
    #     self.head.cls_head[-1].bias.data.fill_(-2.19)

    # ---------------------old method-----------------------------------
    def forward(self, img, xys, hrrps=None):
        nums = 0
        with open('model_data/my_classes.txt') as f:
            for i in f.readlines():
                nums += 1

        hnet_base = HNet_base(nums)
        hnet_base.load_state_dict(torch.load('./weights/hrrps_models/model_hnet.pth'), strict=False)
        hrrp_data = hrrps.type(torch.FloatTensor).reshape(-1, 128)
        fea_hrrp = np.argmax(np.float32(hnet_base(hrrp_data).detach().reshape(-1, 5, 2).numpy()), axis=2)
        xys = torch.clip(xys / 4, 0, 127)
        feat = self.backbone(img)
        hm, wh, offset = self.head(self.decoder(feat))

        hrrps_hm = []
        for i, im in enumerate(img):
            batch_hm = np.zeros((img.shape[-2] // 4, img.shape[-1] // 4, self.num_classes), dtype=np.float32)
            for j in range(5):
                if xys[i, j, 0] != -1:
                    pred_wh = wh[i].permute(1, 2, 0)[int(xys[i, j, 1].item()), int(xys[i, j, 0].item()), :]
                    radius = gaussian_radius((math.ceil(pred_wh[1].item()), math.ceil(pred_wh[0].item())))
                    radius = max(0, int(radius))
                    cls = fea_hrrp[i, j]
                    batch_hm[:, :, cls] = draw_gaussian(batch_hm[:, :, cls], xys[i, j, :], radius)
            hrrps_hm.append(torch.from_numpy(batch_hm).detach())
        hrrps_hm = torch.stack(hrrps_hm, dim=0).permute(0, 3, 1, 2).cuda()
        # return hm*1+hrrps_hm*0.0005, wh, offset
        return hm * 1 + hrrps_hm * 0.0007, wh, offset

    # ---------------------new method-----------------------------------
    # def forward(self, img, xys, hrrps=None):
    #     nums = 0
    #     with open('model_data/my_classes.txt') as f:
    #         for i in f.readlines():
    #             nums += 1
    #
    #     hnet_base = HNet_base(nums)
    #     hnet_base.load_state_dict(torch.load('./weights/hrrps_models/model_hnet.pth'), strict=False)
    #     hrrp_data = hrrps.type(torch.FloatTensor).reshape(-1, 128)
    #     fea_hrrp = np.argmax(np.float32(hnet_base(hrrp_data).detach().reshape(-1, 5, 2).numpy()), axis=2)
    #     xys = torch.clip(xys / 4, 0, 127)
    #     feat = self.backbone(img)
    #     hm, wh, offset = self.head(self.decoder(feat))
    #
    #     hrrps_hm = []
    #     for i, im in enumerate(img):
    #         batch_hm = np.zeros((img.shape[-2] // 4, img.shape[-1] // 4, self.num_classes), dtype=np.float32)
    #         for j in range(5):
    #             if xys[i, j, 0] != -1:
    #                 pred_wh = wh[i].permute(1, 2, 0)[int(xys[i, j, 1].item()), int(xys[i, j, 0].item()), :]
    #                 radius = gaussian_radius((math.ceil(pred_wh[1].item()), math.ceil(pred_wh[0].item())))
    #                 radius = max(0, int(radius))
    #                 cls = fea_hrrp[i, j]
    #                 batch_hm[:, :, cls] = draw_gaussian(batch_hm[:, :, cls], xys[i, j, :], radius)
    #         hrrps_hm.append(torch.from_numpy(batch_hm).detach())
    #     hrrps_hm = torch.stack(hrrps_hm, dim=0).permute(0, 3, 1, 2).cuda()
    #     # return hm*1+hrrps_hm*0.0005, wh, offset
    #     hm_cbma = self.cbma(hm)
    #     return hm * 1 + hrrps_hm * 0.0007, wh, offset





if __name__ == '__main__':
    hnet = HNet(2)

    model_dict = hnet.state_dict()
    print(model_dict.keys())

    pretrained_hnet_path = "../weights/model_hnet.pth"
    # hnet.load_state_dict(torch.load(pretrained_hnet_path))
    state_dict = torch.load(pretrained_hnet_path)
    print(state_dict.keys())


class CenterNet_HourglassNet(nn.Module):
    def __init__(self, heads, pretrained=False, num_stacks=2, n=5, cnv_dim=256, dims=[256, 256, 384, 384, 384, 512],
                 modules=[2, 2, 2, 2, 2, 4]):
        super(CenterNet_HourglassNet, self).__init__()
        if pretrained:
            raise ValueError("HourglassNet has no pretrained model")

        self.nstack = num_stacks
        self.heads = heads

        curr_dim = dims[0]

        self.pre = nn.Sequential(
            conv2d(7, 3, 128, stride=2),
            residual(3, 128, 256, stride=2)
        )

        self.kps = nn.ModuleList([
            kp_module(
                n, dims, modules
            ) for _ in range(num_stacks)
        ])

        self.cnvs = nn.ModuleList([
            conv2d(3, curr_dim, cnv_dim) for _ in range(num_stacks)
        ])

        self.inters = nn.ModuleList([
            residual(3, curr_dim, curr_dim) for _ in range(num_stacks - 1)
        ])

        self.inters_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(num_stacks - 1)
        ])

        self.cnvs_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(num_stacks - 1)
        ])

        for head in heads.keys():
            if 'hm' in head:
                module = nn.ModuleList([
                    nn.Sequential(
                        conv2d(3, cnv_dim, curr_dim, with_bn=False),
                        nn.Conv2d(curr_dim, heads[head], (1, 1))
                    ) for _ in range(num_stacks)
                ])
                self.__setattr__(head, module)
                for heat in self.__getattr__(head):
                    heat[-1].weight.data.fill_(0)
                    heat[-1].bias.data.fill_(-2.19)
            else:
                module = nn.ModuleList([
                    nn.Sequential(
                        conv2d(3, cnv_dim, curr_dim, with_bn=False),
                        nn.Conv2d(curr_dim, heads[head], (1, 1))
                    ) for _ in range(num_stacks)
                ])
                self.__setattr__(head, module)

        self.relu = nn.ReLU(inplace=True)

    def freeze_backbone(self):
        freeze_list = [self.pre, self.kps]
        for module in freeze_list:
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        freeze_list = [self.pre, self.kps]
        for module in freeze_list:
            for param in module.parameters():
                param.requires_grad = True

    def forward(self, image):
        # print('image shape', image.shape)
        inter = self.pre(image)
        outs = []

        for ind in range(self.nstack):
            kp = self.kps[ind](inter)
            cnv = self.cnvs[ind](kp)

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)

            out = {}
            for head in self.heads:
                out[head] = self.__getattr__(head)[ind](cnv)
            outs.append(out)
        return outs
