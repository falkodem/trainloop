import torch
import torch.nn as nn

## LayerNorm в UpBlock


class UNetConvNext(nn.Module):
    def __init__(self, convnext, n_classes):
        super().__init__()
        self.convnext = convnext
        
        convnext_out_ch = self.convnext.stages[-1].blocks[-1].mlp.fc2.out_channels
        self.bottleneck = Bottleneck(convnext_out_ch, convnext_out_ch)
        
        # gather in and out channels sizes from convnext stages and initialize upsample blocks
        self.up_blocks = nn.ModuleList()
        # iterate from the end to the beginning until the first element
        for stage in convnext.stages[:0:-1]:
            in_ch, out_ch = stage.downsample[-1].in_channels, stage.blocks[-1].mlp.fc2.out_channels
            self.up_blocks.append(UpBlock(in_channels=out_ch, out_channels=in_ch))
        self.head = DSC(self.up_blocks[-1].conv[1].out_channels, n_classes) 
        
    def forward(self, x):
        x = self.convnext.stem(x)
        down_outs = []
        for i_stage, stage in enumerate(self.convnext.stages):
            x = stage(x)
            down_outs.append(x)
                                  
        x = self.bottleneck(down_outs[-1])
        for i_up_block, up_block in enumerate(self.up_blocks):
            x = up_block(x, down_outs[-2-i_up_block])
        
        return self.head(x)

class UNetEffNet(nn.Module):
    def __init__(self, model, n_classes):
        super().__init__()
        self.model = model
        
        model_out_ch = model.features[8][0].out_channels
        last_stage_in_ch = model.features[8][0].in_channels
        self.bottleneck = Bottleneck(model_out_ch, last_stage_in_ch)
        
        # gather in and out channels sizes from convnext stages and initialize upsample blocks
        self.up_blocks = nn.ModuleList()
        # iterate from the end to the beginning until the first element
        for i_stage, stage in enumerate(self.model.features[-2:2:-1]):
            in_ch = stage[0].block[0][0].in_channels
            out_ch = stage[-1].block[-1][0].out_channels
            if i_stage in [0,1,3]: # No upscaling for these layers
                do_x2_upscale=False
            else:
                do_x2_upscale=True
            self.up_blocks.append(UpBlockv2(in_channels=out_ch, out_channels=in_ch, do_x2_upscale=do_x2_upscale))
        self.head = DSC(self.up_blocks[-1].conv[1].out_channels, n_classes) 
        
    def forward(self, x):
        x = self.model.features[:3](x)
        down_outs = []
        for i_stage, stage in enumerate(self.model.features[3:]):
            x = stage(x)
            down_outs.append(x)
                                  
        x = self.bottleneck(down_outs[-1])
        for i_up_block, up_block in enumerate(self.up_blocks):
            x = up_block(x, down_outs[-2-i_up_block])
        return self.head(x)


class UNetConvNextv2(nn.Module):
    def __init__(self, convnext, n_classes):
        super().__init__()
        self.convnext = convnext
        
        convnext_out_ch = self.convnext.stages[-1].blocks[-1].mlp.fc2.out_channels
        self.bottleneck = Bottleneck(convnext_out_ch, convnext_out_ch)
        
        # gather in and out channels sizes from convnext stages and initialize upsample blocks
        self.up_blocks = nn.ModuleList()
        # iterate from the end to the beginning until the first element
        for stage in convnext.stages[:0:-1]:
            in_ch, out_ch = stage.downsample[-1].in_channels, stage.blocks[-1].mlp.fc2.out_channels
            self.up_blocks.append(UpBlock(in_channels=out_ch, out_channels=in_ch))
        self.head = DSC(self.up_blocks[-1].conv[1].out_channels, n_classes) 
        
    def forward(self, x):
        x = self.convnext.stem(x)
        down_outs = []
        for i_stage, stage in enumerate(self.convnext.stages):
            x = stage(x)
            down_outs.append(x)
                                  
        x = self.bottleneck(down_outs[-1])
        for i_up_block, up_block in enumerate(self.up_blocks):
            x = up_block(x, down_outs[-2-i_up_block])
        
        return self.head(x)
    
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, do_x2_upscale=True):
        super().__init__()
        self.do_x2_upscale = do_x2_upscale
        if self.do_x2_upscale:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            DSC(in_channels, out_channels, depthwise_ch_mult=2),
            nn.GELU()
        )
                                     
    def forward(self, x1, x2):
        if self.do_x2_upscale:
            x1 = self.up(x1)
        return self.conv(torch.cat([x1, x2], dim=1))


class UpBlockv2(nn.Module):
    def __init__(self, in_channels, out_channels, do_x2_upscale=True):
        super().__init__()
        self.do_x2_upscale = do_x2_upscale
        if self.do_x2_upscale:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(2*in_channels),
            DSC(2*in_channels, out_channels, depthwise_ch_mult=2),
            nn.SiLU()
        )
                                     
    def forward(self, x1, x2):
        if self.do_x2_upscale:
            x1 = self.up(x1)
        return self.conv(torch.cat([x1, x2], dim=1))
           
    
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.bottleneck = nn.Sequential(
            DSC(in_ch=in_channels, out_ch=mid_channels),
            nn.SiLU(),
            DSC(in_ch=mid_channels, out_ch=out_channels),
            nn.SiLU(),
        )
            
    def forward(self, x):
        return self.bottleneck(x)

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
    
