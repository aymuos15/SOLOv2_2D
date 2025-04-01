import torch
import torch.nn as nn
from ..utils.nninit import normal_init

class FeatureBranch(nn.Module):
    """
    Feature branch for SOLOv2 instance segmentation.
    
    This branch generates mask features that encode spatial information.
    It processes multiple feature levels from FPN and fuses them to 
    create the feature map to which instance kernels are applied.
    """
    
    def __init__(self, in_channels, out_channels, start_level, end_level, num_classes, conv_cfg=None, norm_cfg=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.start_level = start_level
        self.end_level = end_level
        self.num_classes = num_classes
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        
        self._init_layers()
        
    def _init_layers(self):
        self.convs_all_levels = nn.ModuleList()
        for i in range(self.start_level, self.end_level + 1):
            convs_per_level = nn.Sequential()
            if i == 0:
                one_conv = nn.Sequential(
                    nn.Conv2d(
                        self.in_channels,
                        self.out_channels,
                        3,
                        padding=1,
                        bias=False),
                    nn.GroupNorm(num_channels=self.out_channels, num_groups=32),
                    nn.ReLU(inplace=False)
                )
                convs_per_level.add_module('conv' + str(i), one_conv)
                self.convs_all_levels.append(convs_per_level)
                continue

            for j in range(i):
                if j == 0:
                    # Special case for level 3 where we add coordinate features
                    chn = self.in_channels+2 if i==3 else self.in_channels
                    one_conv = nn.Sequential(
                        nn.Conv2d(
                            chn,
                            self.out_channels,
                            3,
                            padding=1,
                            bias=False),
                        nn.GroupNorm(num_channels=self.out_channels, num_groups=32),
                        nn.ReLU(inplace=False)
                    )
                    convs_per_level.add_module('conv' + str(j), one_conv)
                    one_upsample = nn.Upsample(
                        scale_factor=2, mode='bilinear', align_corners=False)
                    convs_per_level.add_module(
                        'upsample' + str(j), one_upsample)
                    continue

                one_conv = nn.Sequential(
                    nn.Conv2d(
                        self.out_channels,
                        self.out_channels,
                        3,
                        padding=1,
                        bias=False),
                    nn.GroupNorm(num_channels=self.out_channels, num_groups=32),
                    nn.ReLU(inplace=False)
                )
                convs_per_level.add_module('conv' + str(j), one_conv)
                one_upsample = nn.Upsample(
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=False)
                convs_per_level.add_module('upsample' + str(j), one_upsample)

            self.convs_all_levels.append(convs_per_level)

        self.conv_pred = nn.Sequential(
            nn.Conv2d(
                self.out_channels,
                self.num_classes,
                1,
                padding=0,
                bias=False),
            nn.GroupNorm(num_channels=self.out_channels, num_groups=32),
            nn.ReLU(inplace=True)
        )
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
                
    def forward(self, inputs):
        """
        Forward pass for the feature branch.
        
        Args:
            inputs: List of feature maps from FPN
            
        Returns:
            feature_pred: Processed mask features 
        """
        assert len(inputs) == (self.end_level - self.start_level + 1)
        
        # Process the first level directly
        feature_add_all_level = self.convs_all_levels[0](inputs[0])
        
        # Process and add each subsequent level
        for i in range(1, len(inputs)):
            input_p = inputs[i]
            
            # For level 3, add coordinate features
            if i == 3:
                input_feat = input_p
                x_range = torch.linspace(-1, 1, input_feat.shape[-1], device=input_feat.device)
                y_range = torch.linspace(-1, 1, input_feat.shape[-2], device=input_feat.device)
                y, x = torch.meshgrid(y_range, x_range)
                y = y.expand([input_feat.shape[0], 1, -1, -1])
                x = x.expand([input_feat.shape[0], 1, -1, -1])
                coord_feat = torch.cat([x, y], 1)
                input_p = torch.cat([input_p, coord_feat], 1)
            
            # Process this level and add to accumulated features
            feature_add_all_level = feature_add_all_level + self.convs_all_levels[i](input_p)
        
        # Final prediction layer
        feature_pred = self.conv_pred(feature_add_all_level)
        return feature_pred
