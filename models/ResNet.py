import torch.nn as nn
import torch
import math


# 使用resnet50提取红外图像的特征图
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, layers, num_classes=2, model_path=""):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.modelPath = model_path
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stack1 = self.make_stack(64, layers[0])
        self.stack2 = self.make_stack(128, layers[1], stride=2)
        self.stack3 = self.make_stack(256, layers[2], stride=2)
        self.stack4 = self.make_stack(512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Sequential(
            nn.Linear(512 * Bottleneck.expansion, num_classes),
            # nn.Softmax(dim=1)
        )
        # initialize parameters
        self.init_param()

    def init_param(self):
        # The following is initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.shape[0] * m.weight.shape[1]
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def make_stack(self, planes, blocks, stride=1):
        downsample = None
        layers = []

        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )

        layers.append(Bottleneck(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * Bottleneck.expansion
        for i in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.stack1(x)
        x = self.stack2(x)
        x = self.stack3(x)
        x = self.stack4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet50_only(pretrained=False, **kwargs):
    """Constructs a ResNet-50 models.
    Args:
        pretrained (bool): If True, returns a models pre-trained on ImageNet
    """
    model = ResNet([3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(model.modelPath))

    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 models.
    Args:
        pretrained (bool): If True, returns a models pre-trained on ImageNet
    """
    model = ResNet([3, 4, 6, 3], **kwargs)

    if pretrained:
        model.load_state_dict(torch.load(model.modelPath))

    extractor1 = list([model.conv1, model.bn1, model.relu, model.maxpool, model.stack1, model.stack2, model.stack3])

    extractor2 = list([model.stack4, model.avgpool])

    # 用于提取红外图像的特征图
    extractor1 = nn.Sequential(*extractor1)

    # 用于提取两种模态融合后的深层特征
    extractor2 = nn.Sequential(*extractor2)

    return extractor1, extractor2


def resnet18(pretrained=False, **kwargs):
    """ return a ResNet 18 object
    """
    model = ResNet([2, 2, 2, 2], **kwargs)

    if pretrained:
        model.load_state_dict(torch.load(model.modelPath))

    extractor1 = list([model.conv1, model.bn1, model.relu, model.maxpool, model.stack1, model.stack2, model.stack3])

    extractor2 = list([model.stack4, model.avgpool])

    # 用于提取红外图像的特征图
    extractor1 = nn.Sequential(*extractor1)

    # 用于提取两种模态融合后的深层特征
    extractor2 = nn.Sequential(*extractor2)

    return extractor1, extractor2


if __name__ == '__main__':
    x = torch.rand(1, 3, 224, 224)
    y = torch.rand(1, 1024, 14, 14)

    model = resnet50()[0]
    z = model(x)
    print(z)
    print(z.shape)