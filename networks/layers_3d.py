import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial
import numpy as np
from .squeeze_and_excitation_3D import ChannelSELayer3D

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def resolve_norm_layer_3d(planes, norm_class, num_groups=8):
    if norm_class.lower() == "batch":
        return nn.BatchNorm3d(planes)
    if norm_class.lower() == "group":
        return nn.GroupNorm(num_groups, planes)
    raise NotImplementedError("norm_class must be batch or group, but {norm_class} was given"
    )


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, groups=1, 
                 norm_class='batch', downsample=None, sd_prob=1,
                 use_se_layer=False, se_reduction_ratio=4):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = resolve_norm_layer_3d(planes, norm_class, groups)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = resolve_norm_layer_3d(planes, norm_class, groups)
        self.downsample = downsample
        self.stride = stride
        self.sd_prob = sd_prob  # stochastic depth probability (1 - never drop) (0 - always drop)
        self.use_se_layer = use_se_layer
        if self.use_se_layer:
            self.se = ChannelSELayer3D(planes, reduction_ratio=se_reduction_ratio)

    def forward(self, x):
        residual = x.clone()

        if self.training:
            k = int(np.random.binomial(1, self.sd_prob, 1))
            if k == 1:  # do not drop
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)

                out = self.conv2(out)
                out = self.bn2(out)

                if self.use_se_layer is True:  # squeeze-and-excitation
                    out = self.se(out)

                if self.downsample is not None:
                    residual = self.downsample(x)

                out += residual
            else:  # stochastic gradient drop layer
                if self.downsample is not None:
                    residual = self.downsample(x)
                out = residual
        else:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.use_se_layer is True:  # squeeze-and-excitation
                out = self.se(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out = self.sd_prob * out + residual
            #out += residual

        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, groups=1,
                 norm_class='batch', downsample=None,
                 use_se_layer=False, se_reduction_ratio=4):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = resolve_norm_layer_3d(planes, norm_class, groups)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = resolve_norm_layer_3d(planes, norm_class, groups)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = resolve_norm_layer_3d(planes * 4, norm_class, groups)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.use_se_layer = use_se_layer
        self.se = ChannelSELayer3D(planes, reduction_ratio=se_reduction_ratio)

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

        if self.use_se_layer:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 ada_pool_out=(1,1,1),
                 norm_class='batch',
                 groups=1,
                 shortcut_type='B',
                 num_classes=400):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = resolve_norm_layer_3d(64, norm_class, groups)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0],
                                       shortcut_type, norm_class, groups)
        self.layer2 = self._make_layer(block, 128, layers[1],
                                       shortcut_type, norm_class, groups, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2],
                                       shortcut_type, norm_class, groups, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3],
                                       shortcut_type, norm_class, groups, stride=2)
        self.avgpool = nn.AdaptiveMaxPool3d(ada_pool_out)
        ada_mul = ada_pool_out[0] * ada_pool_out[1] * ada_pool_out[2]
        self.fc = nn.Linear(ada_mul * 512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, shortcut_type, norm_class='batch', groups=1, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), resolve_norm_layer_3d(planes * block.expansion, norm_class, groups))

        layers = []
        layers.append(block(self.inplanes, planes, stride, groups=groups, norm_class=norm_class, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if x.ndim==4:
            x = x.unsqueeze(1).contiguous()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        
        x = self.layer2(x)
        
        x = self.layer3(x)
     
        x = self.layer4(x)
        

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class Conv3dWS(nn.Conv3d):
    """3D Convolution layer with weight standardization
    https://arxiv.org/abs/1903.10520
    Based on https://github.com/joe-siyuan-qiao/WeightStandardization
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv3dWS, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )
    
    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=(1, 2, 3, 4), keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class ResNet_heatmap(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 norm_class='batch',
                 groups=8,
                 shortcut_type='B',
                 in_channels=1,
                 weight_standardization=False,
                 inplanes=64,
                 wide_factor=1,
                 stochastic_depth_rate=0.0,
                 use_se_layer=False
                ):
        
        #self.inplanes = inplanes
        self.inplanes = int(inplanes * wide_factor)  # extend width if wide network

        super(ResNet_heatmap, self).__init__()

        if weight_standardization:
            self.conv_layer = Conv3dWS
        else:
            self.conv_layer = nn.Conv3d
        
        #self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.conv1 = self.conv_layer(in_channels, self.inplanes, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)

        self.bn1 = resolve_norm_layer_3d(self.inplanes, norm_class, groups)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        self.sd_prob_now = 1.0
        self.sd_prob_delta = stochastic_depth_rate
        self.sd_prob_step = self.sd_prob_delta / (sum(layers) - 1)       

        self.use_se_layer = use_se_layer

        self.layer1 = self._make_layer(block, self.inplanes, layers[0], shortcut_type, norm_class, groups, stride=1)
        # self.inplanes gets updated after each _make_layer call 
        # so we only need to multiply it by 2 in next layers:
        self.layer2 = self._make_layer(block, self.inplanes*2, layers[1], shortcut_type, norm_class, groups, stride=2)
        self.layer3 = self._make_layer(block, self.inplanes*2, layers[2], shortcut_type, norm_class, groups, stride=2)
        self.layer4 = self._make_layer(block, self.inplanes*2, layers[3], shortcut_type, norm_class, groups, stride=2)

    def _make_layer(self, block, planes, blocks, shortcut_type, norm_class='batch', groups=8, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    self.conv_layer(self.inplanes, planes * block.expansion,
                                    kernel_size=1, stride=stride, bias=False),
                    #nn.Conv3d(self.inplanes, planes * block.expansion,
                    #          kernel_size=1, stride=stride, bias=False),
                    resolve_norm_layer_3d(planes * block.expansion, norm_class, groups)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, 
                            groups=groups, norm_class=norm_class,
                            downsample=downsample, 
                            sd_prob=self.sd_prob_now,
                            use_se_layer=self.use_se_layer))
        #layers.append(block(self.inplanes, planes, stride, groups=groups, norm_class=norm_class, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 
                                sd_prob=self.sd_prob_now, 
                                use_se_layer=self.use_se_layer))
            self.sd_prob_now = self.sd_prob_now - self.sd_prob_step
            #layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

def mixup_process(x, target_reweighted, lam):
    indices = np.random.permutation(x.size(0))
    x = x*lam + x[indices]*(1-lam)
    target_shuffled_onehot = target_reweighted[indices]
    target_reweighted = target_reweighted*lam + target_shuffled_onehot*(1-lam)
    return x, target_reweighted

def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


def resnet18_heatmap(resnet_groups, **kwargs):
    if resnet_groups == 0:
        model = ResNet_heatmap(
            BasicBlock,
            [2,2,2,2],
            norm_class='batch',
            **kwargs
        )
    else:
        model = ResNet_heatmap(
            BasicBlock,
            [2,2,2,2],
            norm_class='group',
            groups=resnet_groups,
            weight_standardization=False,  # NOTE temporary
            **kwargs
        )
    return model

def resnet18_bottleneck_heatmap(resnet_groups, **kwargs):
    if resnet_groups == 0:
        model = ResNet_heatmap(
            Bottleneck,
            [2,2,2,2],
            norm_class='batch',
            **kwargs
        )
    else:
        model = ResNet_heatmap(
            Bottleneck,
            [2,2,2,2],
            norm_class='group',
            groups=resnet_groups,
            weight_standardization=False,  # NOTE temporary
            **kwargs
        )
    return model

def resnet18_fc(resnet_groups, num_classes=4, **kwargs):
    model = ResNet(
        BasicBlock,
        [2,2,2,2],
        norm_class='group',
        groups=resnet_groups,
        num_classes=num_classes
    )
    return model


def resnet34_heatmap(resnet_groups, **kwargs):
    model = ResNet_heatmap(
        BasicBlock,
        [3, 4, 6, 3],
        norm_class='group',
        groups=resnet_groups,
        **kwargs
    )
    return model


def resnet36_heatmap(resnet_groups, **kwargs):
    model = ResNet_heatmap(
        Bottleneck,
        [2, 3, 5, 2],
        norm_class='group',
        groups=resnet_groups,
        **kwargs
    )
    return model


def resnet50_heatmap(resnet_groups, **kwargs):
    model = ResNet_heatmap(
        Bottleneck,
        [3, 4, 6, 3],
        norm_class='group',
        groups=resnet_groups,
        **kwargs
    )
    return model


def resnet101_heatmap(resnet_groups, **kwargs):
    model = ResNet_heatmap(
        Bottleneck,
        [3, 4, 23, 3],
        norm_class='group',
        groups=resnet_groups,
        **kwargs
    )
    return model


def get_resnet_heatmap(resnet_size, resnet_groups=16, force_bottleneck=False, **kwargs):
    if resnet_size == 18:
        if force_bottleneck:
            return resnet18_bottleneck_heatmap(resnet_groups, **kwargs)
        else:
            return resnet18_heatmap(resnet_groups, **kwargs)
    elif resnet_size == 34:
        return resnet34_heatmap(resnet_groups, **kwargs)
    elif resnet_size == 36:
        return resnet36_heatmap(resnet_groups, **kwargs)
    elif resnet_size == 50:
        return resnet50_heatmap(resnet_groups, **kwargs)
    elif resnet_size == 101:
        return resnet101_heatmap(resnet_groups, **kwargs)
    else:
        raise ValueError(f"Unknown ResNet size {resnet_size}")
