import torch
import torch.nn as nn
from torchvision.models import resnet50
from networks.resnet_3d_pre_post import MRIResNet3D_wls_right
from networks.layers_3d import resnet18_fc


class MultiSequenceChannels(nn.Module):
    """
    This is an extension of 3D-ResNet that can take multiple input channels
    and different number of output classes.
    """
    def __init__(self, experiment_parameters, in_channels=3, 
                 inplanes=64, num_classes=4, wide_factor=1,
                 use_dropout=False, stochastic_depth_rate=False,
                 use_se_layer=False, return_hidden=False):
        super(MultiSequenceChannels, self).__init__()
        self.experiment_parameters = experiment_parameters
        self.feature_extractor = MRIResNet3D_wls_right(
            feature_extractor_only=True,
            inplanes=inplanes,
            in_channels=in_channels,
            wide_factor=wide_factor,
            stochastic_depth_rate=stochastic_depth_rate,
            use_se_layer=use_se_layer
        )
        self.pool = nn.AdaptiveMaxPool3d(1)
        self.classifier = nn.Linear(
            # NOTE: Below number of inplanes is multiplied by 8 because
            # in the 3D ResNet-18 there are 4 conv layers, so after 2nd, 3rd and 4th layer
            # number of inplanes was multiplied by 8 (inplanesx2x2x2 = inplanesx8)
            int(inplanes * 8 * wide_factor),
            num_classes,
            bias=False
        )
        self.dropout = nn.Dropout(0.25)
        self.use_dropout = use_dropout
        self.return_hidden = return_hidden

    def forward(self, x, return_logits=False):
        # x: (b_s, 3, 190, 448, 448)
        h = self.feature_extractor(x)  # h: (b_s, 512, 12, 14, 14)
        v = self.pool(h)  # v: (b_s, 512, 1, 1, 1)
        v = v.squeeze(-1).squeeze(-1).squeeze(-1)  # remove empty dims; (b_s, 512)
        if self.use_dropout:
            v = self.dropout(v)
        y_hat = self.classifier(v)  # y_hat: (b_s, 4)
        
        if self.return_hidden:
            return y_hat, v
        else:
            return y_hat
        # if return_logits:
        #     return y_hat
        # else:
        #     return torch.sigmoid(y_hat)


class MRIModels(nn.Module):
    def __init__(self, parameters, in_channels=1, 
                 num_classes=4, force_bottleneck=False, inplanes=64):
        super(MRIModels, self).__init__()
        self.parameters = parameters
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.force_bottleneck = force_bottleneck
        self.inplanes = inplanes

        if parameters.input_type == 'three_channel':
            self.in_channels = 3
        if parameters.input_type == 'birads':
            self.num_classes = 5
        if parameters.network_modification == 'resnet18_bottleneck':
            self.force_bottleneck = force_bottleneck
        
        # Resolve which model to choose
        if parameters.architecture == '3d_resnet18':
            model = MRIResNet3D_wls_right(
                resnet_size=18,
                groups=parameters.resnet_groups,
                in_channels=self.in_channels,
                topk=parameters.topk,
                force_bottleneck=force_bottleneck,
                inplanes=self.inplanes
            )
        elif parameters.architecture == '3d_resnet18_fc':
            model = resnet18_fc(
                parameters.resnet_groups,
                num_classes=num_classes
            )
        elif parameters.architecture == '3d_resnet34':
            model = MRIResNet3D_wls_right(
                resnet_size=34,
                groups=parameters.resnet_groups
            )
        elif parameters.architecture == '3d_resnet50':
            model = MRIResNet3D_wls_right(
                resnet_size=50,
                groups=parameters.resnet_groups
            )
        elif parameters.architecture == '3d_resnet101':
            model = MRIResNet3D_wls_right(
                resnet_size=101,
                groups=parameters.resnet_groups
            )
        elif parameters.architecture == '2d_resnet50':
            model = resnet50(
                pretrained=False,
                num_classes=4
            )
        else:
            raise ValueError(f"Unknown architecture {parameters.architecture}")
        
        self.model = model

        # Load weights
        if parameters.weights:
            self.load_weights()
        
    
    def load_weights(self):
        weights = torch.load(self.parameters.weights)
        
        if self.parameters.weights_policy == 'bpe':
            # Pretrained from BI-RADS or BPE
            new_weights = {}
            for k, v in weights.items():
                new_weights[f"resnet.{k}"] = weights[k]
            self.model.load_state_dict(new_weights, strict=False)
        elif self.parameters.weights_policy == 'kinetics':
            new_weights = {}
            for k, v in weights.items():
                if not any(exclusion in k for exclusion in ['num_batches_tracked']):
                    k_ = k.replace("layer", "resnet.layer")
                    k_ = k_.replace("conv1.0", "conv1")
                    k_ = k_.replace("conv2.0", "conv2")
                    new_weights[k_] = weights[k]
            self.model.load_state_dict(new_weights, strict=False)
        
        elif self.parameters.weights_policy == 'new':
            self.model.load_state_dict(weights['model'])
        
        else:
            raise ValueError("Unknown or no weights policy chosen. You need to choose a weights_policy if you are loading weights.")
