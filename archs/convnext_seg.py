import torch
import torch.nn as nn
from collections import OrderedDict
## LayerNorm в UpBlock


class ConvNextSeg(nn.Module):
    def __init__(self, convnext, n_classes):
        super().__init__()
        self.convnext = convnext

        up_blocks = OrderedDict()
        for i, stage in enumerate(convnext.stages[:0:-1]):
            in_ch, out_ch = stage.downsample[-1].in_channels, stage.blocks[-1].mlp.fc2.out_channels
            up_blocks[f'up{i}']= UpBlock(out_ch, in_ch)
        self.up_blocks = nn.Sequential(up_blocks)

        self.head = DSC(self.up_blocks[-1].conv.dsc2.out_channels, n_classes) 
        # Initialize weights using Xavier Normal
        # self._initialize_weights()

    # def _initialize_weights(self):
    #     for name, param in self.up_blocks.named_modules():
    #         if isinstance(param, nn.Conv2d):
    #             nn.init.xavier_normal_(param.weight)
    #             if param.bias is not None:
    #                 nn.init.zeros_(param.bias)

    def forward(self, x):
        return self.head(self.up_blocks(self.convnext.stages(self.convnext.stem(x))))
    
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=4, padding=1, stride=2)
        self.conv = nn.Sequential(OrderedDict([
            ('dsc1', DSC(in_channels // 2, out_channels, depthwise_ch_mult=2)),
            ('norm1', nn.GroupNorm(1, in_channels // 2)),
            ('act1', nn.GELU()),
            ('dsc2', DSC(in_channels // 2, out_channels, depthwise_ch_mult=2)),
            ('norm2', nn.GroupNorm(1, in_channels // 2)),
            ('act2', nn.GELU())
        ]
        ))
                                  
    def forward(self, x):
        return self.conv(self.up(x))


class DSC(nn.Module):
    ''' Just depthwise separable convolution'''
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, depthwise_ch_mult=1):
        super().__init__()
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.depthwise = nn.Conv2d(in_ch, in_ch * depthwise_ch_mult, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch * depthwise_ch_mult, out_ch, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
    
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.bottleneck = nn.Sequential(
            DSC(in_ch=in_channels, out_ch=mid_channels),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            DSC(in_ch=mid_channels, out_ch=out_channels),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
        )
            
    def forward(self, x):
        return self.bottleneck(x)