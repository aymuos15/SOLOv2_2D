import torch.nn as nn
import torch.nn.functional as F
from ..utils.nninit import normal_init

class KernelBranch(nn.Module):
    """
    Kernel prediction branch for SOLOv2 instance segmentation.
    
    This branch predicts instance-specific convolution kernels that are used
    to generate instance masks when applied to mask features.
    """
    
    def __init__(self, in_channels, seg_feat_channels, stacked_convs, kernel_out_channels, norm_cfg=None):
        super().__init__()
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.kernel_out_channels = kernel_out_channels
        self.norm_cfg = norm_cfg
        
        self._init_layers()
        
    def _init_layers(self):
        norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
        self.kernel_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            # For kernel branch, we add coordinate features, so input channels are increased by 2
            chn = self.in_channels + 2 if i == 0 else self.seg_feat_channels
            self.kernel_convs.append(nn.Sequential(
                nn.Conv2d(
                    chn,
                    self.seg_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    bias=norm_cfg is None),
                nn.GroupNorm(num_channels=self.seg_feat_channels, num_groups=32),
                nn.ReLU(inplace=True)
            ))
            
        self.solo_kernel = nn.Conv2d(
            self.seg_feat_channels, self.kernel_out_channels, 3, padding=1)
    
    def init_weights(self):
        for m in self.kernel_convs:
            if isinstance(m, nn.Sequential):
                for con in m:
                    if isinstance(con, nn.Conv2d):
                        normal_init(con, std=0.01)
        
        normal_init(self.solo_kernel, std=0.01)
        
    def forward(self, x, grid_size=None):
        """
        Forward pass for kernel branch.
        
        Args:
            x: Input feature map with coordinate features already added
            grid_size: Grid size for downsampling
            
        Returns:
            kernel_pred: Kernel predictions
        """
        # Downsample to grid size if specified
        if grid_size is not None:
            kernel_feat = F.interpolate(x, size=grid_size, mode='bilinear', align_corners=False)
        else:
            kernel_feat = x
            
        # Apply kernel convs
        for kernel_layer in self.kernel_convs:
            kernel_feat = kernel_layer(kernel_feat)
            
        kernel_pred = self.solo_kernel(kernel_feat)
        return kernel_pred
