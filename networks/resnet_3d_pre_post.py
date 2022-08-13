import torch
import torch.nn as nn
from networks import layers_3d



class MRIResNet3D_wls_right(nn.Module):
    def __init__(self,
                 resnet_size=18,
                 groups=16,
                 in_channels=1,
                 topk=10,
                 return_h=False,
                 force_bottleneck=False,
                 feature_extractor_only=False,
                 inplanes=64,
                 wide_factor=1,
                 stochastic_depth_rate=0.0,
                 use_se_layer=False):
        super(MRIResNet3D_wls_right, self).__init__()
        
        self.topk = topk
        self.return_h = return_h
        self.resnet = layers_3d.get_resnet_heatmap(
            resnet_size,
            resnet_groups=groups,
            in_channels=in_channels,
            force_bottleneck=force_bottleneck,  # force Bottleneck block
            inplanes=inplanes,
            wide_factor=wide_factor,
            stochastic_depth_rate=stochastic_depth_rate,
            use_se_layer=use_se_layer
        )
        self.feature_extractor_only = feature_extractor_only
        self.inplanes=inplanes

        if resnet_size in [18,34]:
            if force_bottleneck:
                self.wsl = nn.Conv3d(2048, 4*2, kernel_size=1)
            else:
                if self.inplanes == 64:
                    self.wsl = nn.Conv3d(512, 4*2, kernel_size=1)
                elif self.inplanes == 16:
                    self.wsl = nn.Conv3d(128, 4*2, kernel_size=1)
                elif self.inplanes == 32:
                    self.wsl = nn.Conv3d(256, 4*2, kernel_size=1)
        elif resnet_size in [36, 50, 101]:
            self.wsl = nn.Conv3d(2048, 4*2, kernel_size=1)

    def forward(self, x, return_logits=False):
        if x.ndim == 4:
            h = x.unsqueeze(1).contiguous()  # (b_s, 1, z, x, y)
        else:
            h = x
        h = self.resnet(h)
        
        if self.feature_extractor_only:
            return h
        
        h = self.wsl(h)  # (b_s, 8, 12, 14, 14)

        y = torch.zeros(h.shape[0], 2, h.shape[2], h.shape[3], h.shape[4])
        y[:,0] = torch.mean(h[:,:4],dim=1)
        y[:,1] = torch.mean(h[:,4:8],dim=1)
        
        if not return_logits:  # by default logits->sigmoid, but if True return logits only
            y = torch.sigmoid(y)
        output = torch.zeros(y.shape[0],4, device=torch.device('cuda'))
        output[:,2] = torch.mean(torch.topk(y[:,0,:,:, :7].contiguous().view(-1), self.topk)[0])
        output[:,3] = torch.mean(torch.topk(y[:,1,:,:, :7].contiguous().view(-1), self.topk)[0])
        output[:,0] = torch.mean(torch.topk(y[:,0,:,:, 7:14].contiguous().view(-1), self.topk)[0])
        output[:,1] = torch.mean(torch.topk(y[:,1,:,:, 7:14].contiguous().view(-1), self.topk)[0])
        
        if self.return_h:
            return output, y
        else:
            return output