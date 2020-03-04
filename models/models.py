import torch
import torchvision
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms
from models.mdn import MDN
from yolov3.models import Darknet
from yolov3.utils.utils import non_max_suppression as nms
from yolov3.utils.datasets import pad_to_square, resize


def difference_detector(template, frame, threshold):
    if threshold <= 0:
        return True
    diff = np.mean(np.square(template - frame))
    return diff >= threshold


class YOLOv3():
    def __init__(self, config_path, weight_path, conf_thr=0.5, nms_thr=0.45):
        self.conf_thr = conf_thr
        self.nms_thr = nms_thr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Darknet(config_path).to(self.device)
        self.model.load_darknet_weights(weight_path)
        self.model.eval()

    def forward(self, data):
        """
        output: list of Tensor
            [Tensor([[x1, y1, x2, y2, conf, cls_conf, cls_pred]], ...], ...]
        """
        data = data.to(self.device, torch.float32, non_blocking=True)
        output = self.model(data)
        output = nms(output, self.conf_thr, self.nms_thr)

        return output

    def predict(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = transforms.ToTensor()(image)
        # image, _ = pad_to_square(image, 0)
        # image = resize(image, 416)
        image = image[None, ...]
        output = self.forward(image)

        return output


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def init_op(op):
    if isinstance(op, nn.Conv2d):
        nn.init.normal_(op.weight, std=0.01)
        if op.bias is not None:
            nn.init.constant_(op.bias, 0)
        print('Init {}'.format(op))
    elif isinstance(op, nn.Linear):
        nn.init.normal_(op.weight, std=0.01)
        print('Init {}'.format(op))
    elif isinstance(op, nn.BatchNorm2d):
        nn.init.constant_(op.weight, 1)
        nn.init.constant_(op.bias, 0)
        print('Init {}'.format(op))
    elif isinstance(op, (list, nn.Sequential, nn.Module)):
        for _op in op.children():
            init_op(_op)
    elif isinstance(op, nn.ReLU):
        pass
    else:
        print('Unknown type: {}'.format(op))


class ResNetMDN(nn.Module):
    def __init__(self, training=True):
        super(ResNetMDN, self).__init__()

        self.training = training

        resnet = torchvision.models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )

        # self.layer3 = []
        # in_planes = 128
        # planes = 256
        # self.layer3.append(
        #     BasicBlock(
        #         in_planes,
        #         planes,
        #         stride=1,
        #         downsample=nn.Sequential(
        #             conv1x1(in_planes, planes, stride=1),
        #             nn.BatchNorm2d(planes)
        #         )
        #     )
        # )
        # self.layer3.append(
        #     BasicBlock(
        #         planes,
        #         planes,
        #         stride=1,
        #         downsample=None
        #     )
        # )
        # self.layer3 = nn.Sequential(*self.layer3)
        # self.layer3.load_state_dict(resnet.layer3.state_dict())

        # self.layer4 = nn.Sequential(
        #     nn.Conv2d(256, 128, 3, stride=2, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, 64, 3, stride=2, padding=1),
        #     nn.ReLU(inplace=True)
        # )
        # init_op(self.layer4)

        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
            nn.ReLU(inplace=True)
        )
        init_op(self.decoder)

        self.mdn = MDN(in_features=512 * 13 * 13, out_features=1, num_gaussians=20)

        self.mse_loss = nn.MSELoss()

    def forward(self, x):
        res = self.encoder(x)
        mdn_out = self.mdn(res)

        if self.training:
            out = self.decoder(res)
            out = F.interpolate(out, scale_factor=32, mode='bilinear', align_corners=False)
            return out, mdn_out
        else:
            return mdn_out


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, NL='relu', same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        if NL == 'relu':
            self.relu = nn.ReLU(inplace=True)
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class FC(nn.Module):
    def __init__(self, in_features, out_features, NL='relu'):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        if NL == 'relu':
            self.relu = nn.ReLU(inplace=True)
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class CMTL(nn.Module):
    '''
    Implementation of CNN-based Cascaded Multi-task Learning of High-level Prior and Density
    Estimation for Crowd Counting (Sindagi et al.)
    '''
    def __init__(self, bn=False, num_classes=10):
        super(CMTL, self).__init__()

        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            Conv2d(3,  16, 9, same_padding=True, NL='prelu', bn=bn),
            Conv2d(16, 32, 7, same_padding=True, NL='prelu', bn=bn)
        )

        self.hl_prior_1 = nn.Sequential(
            Conv2d(32, 16, 9, same_padding=True, NL='prelu', bn=bn),
            nn.MaxPool2d(2),
            Conv2d(16, 32, 7, same_padding=True, NL='prelu', bn=bn),
            nn.MaxPool2d(2),
            Conv2d(32, 16, 7, same_padding=True, NL='prelu', bn=bn),
            Conv2d(16, 8,  7, same_padding=True, NL='prelu', bn=bn)
        )

        self.hl_prior_2 = nn.Sequential(
            nn.AdaptiveMaxPool2d((32, 32)),
            Conv2d(8, 4, 1, same_padding=True, NL='prelu', bn=bn)
        )

        self.hl_prior_fc1 = FC(4*1024, 512, NL='prelu')
        self.hl_prior_fc2 = FC(512, 256, NL='prelu')
        self.hl_prior_fc3 = FC(256, self.num_classes, NL='prelu')

        self.de_stage_1 = nn.Sequential(
            Conv2d(32, 20, 7, same_padding=True, NL='prelu', bn=bn),
            nn.MaxPool2d(2),
            Conv2d(20, 40, 5, same_padding=True, NL='prelu', bn=bn),
            nn.MaxPool2d(2),
            Conv2d(40, 20, 5, same_padding=True, NL='prelu', bn=bn),
            Conv2d(20, 10, 5, same_padding=True, NL='prelu', bn=bn)
        )

        self.de_stage_2 = nn.Sequential(
            Conv2d(18, 24, 3, same_padding=True, NL='prelu', bn=bn),
            Conv2d(24, 32, 3, same_padding=True, NL='prelu', bn=bn),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, output_padding=0, bias=True),
            nn.PReLU(),
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1, output_padding=0, bias=True),
            nn.PReLU(),
            Conv2d(8, 1, 1, same_padding=True, NL='relu', bn=bn)
        )

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, im_data):
        x_base = self.base_layer(im_data)
        x_hlp1 = self.hl_prior_1(x_base)
        x_hlp2 = self.hl_prior_2(x_hlp1)
        x_hlp2 = x_hlp2.view(x_hlp2.size()[0], -1)
        x_hlp = self.hl_prior_fc1(x_hlp2)
        x_hlp = F.dropout(x_hlp, training=self.training)
        x_hlp = self.hl_prior_fc2(x_hlp)
        x_hlp = F.dropout(x_hlp, training=self.training)
        x_cls = self.hl_prior_fc3(x_hlp)
        x_den = self.de_stage_1(x_base)
        x_den = torch.cat((x_hlp1, x_den), 1)
        x_den = self.de_stage_2(x_den)
        x_cls = F.softmax(x_cls, dim=1)
        return x_den, x_cls


if __name__ == '__main__':
    x = ResNetMDN()
    print(x.layer3)
