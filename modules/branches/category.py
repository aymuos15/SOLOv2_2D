import torch.nn as nn
import torch.nn.functional as F
from ..utils.nninit import normal_init, bias_init_with_prob
from ..utils.nms import points_nms

class CategoryBranch(nn.Module):
    """
    Category prediction branch for SOLOv2 instance segmentation.
    
    This branch predicts category scores for potential object instances.
    """
    
    def __init__(self, in_channels, seg_feat_channels, stacked_convs, cate_out_channels, norm_cfg=None):
        super().__init__()
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.cate_out_channels = cate_out_channels
        self.norm_cfg = norm_cfg
        
        self._init_layers()
        
    def _init_layers(self):
        norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
        self.cate_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.seg_feat_channels
            self.cate_convs.append(nn.Sequential(
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
            
        self.solo_cate = nn.Conv2d(
            self.seg_feat_channels, self.cate_out_channels, 3, padding=1)
    
    def init_weights(self):
        for m in self.cate_convs:
            if isinstance(m, nn.Sequential):
                for con in m:
                    if isinstance(con, nn.Conv2d):
                        normal_init(con, std=0.01)
        
        bias_cate = bias_init_with_prob(0.01)
        normal_init(self.solo_cate, std=0.01, bias=bias_cate)
        
    def forward(self, x, grid_size=None, eval=False):
        """
        Forward pass for category branch.
        
        Args:
            x: Input feature map
            grid_size: Grid size for downsampling
            eval: Whether in evaluation mode
            
        Returns:
            cate_pred: Category predictions
        """
        # Downsample to grid size if specified
        if grid_size is not None:
            cate_feat = F.interpolate(x, size=grid_size, mode='bilinear', align_corners=False)
        else:
            cate_feat = x
            
        # Apply category convs
        for cate_layer in self.cate_convs:
            cate_feat = cate_layer(cate_feat)
        cate_pred = self.solo_cate(cate_feat)
        
        # Apply NMS during evaluation
        if eval:
            cate_pred = points_nms(cate_pred.sigmoid(), kernel=2).permute(0, 2, 3, 1)
            
        return cate_pred
