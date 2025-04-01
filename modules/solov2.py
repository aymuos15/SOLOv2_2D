import torch
import torch.nn as nn

from .fpn import FPN
from .backbone import resnet18
from .head import SOLOv2Head

class SOLOV2(nn.Module):
    
    def __init__(self,
                 cfg=None,
                 pretrained=None,
                 mode='train'):
        super(SOLOV2, self).__init__()
        if cfg.backbone.name == 'resnet18':
            self.backbone = resnet18(pretrained=True, loadpath = cfg.backbone.path)
        else:
            raise NotImplementedError
        
        #this set only support resnet18
        self.fpn = FPN(in_channels=[64, 128, 256, 512],out_channels=256,start_level=0,num_outs=5,upsample_cfg=dict(mode='nearest'))

        #this set only support resnet18
        self.bbox_head = SOLOv2Head(num_classes=cfg.num_classes,
                            in_channels=256,
                            seg_feat_channels=256,
                            stacked_convs=2,
                            strides=[8, 8, 16, 32, 32],
                            scale_ranges=((1, 56), (28, 112), (56, 224), (112, 448), (224, 896)),
                            num_grids=[40, 36, 24, 16, 12],
                            ins_out_channels=128,
                            mask_feat_start_level=0,  # Added these parameters
                            mask_feat_end_level=3     # Added these parameters
                        )
        
        self.mode = mode

        self.test_cfg = cfg.test_config.test_cfg

        if self.mode == 'train':
            self.backbone.train(mode=True)
        else:
            self.backbone.train(mode=True)
        
        if pretrained is None:
            self.init_weights() #if first train, use this initweight
        else:
            self.load_weights(pretrained)             #load weight from file
    
    def init_weights(self):
        #fpn
        if isinstance(self.fpn, nn.Sequential):
            for m in self.fpn:
                m.init_weights()
        else:
            self.fpn.init_weights()
        
        # Initialize bbox_head (which now includes mask feature functionality)
        self.bbox_head.init_weights()
    
    def save_weights(self, path):
        """ Saves the model's weights using compression because the file sizes were getting too big. """
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
 
    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        x = self.fpn(x)
        return x

    def forward(self, img, img_meta, return_loss=True, **kwargs):

        if return_loss:
            return self.forward_train(img, img_meta, **kwargs)
        else:
            return self.forward_test(img, img_meta, **kwargs)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        # Extract features from backbone and FPN
        x = self.extract_feat(img)
        
        # Forward through SOLOv2Head, which now includes mask feature generation
        cate_preds, kernel_preds, mask_feat_pred = self.bbox_head(x)
        
        # Calculate losses
        loss_inputs = (cate_preds, kernel_preds, mask_feat_pred, 
                       gt_bboxes, gt_labels, gt_masks, img_metas)

        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def forward_test(self, imgs, img_metas, **kwargs):

        return self.simple_test(imgs[0], img_metas[0], **kwargs)

    
    def simple_test(self, img, img_meta, rescale=False):
        # Extract features from backbone and FPN
        x = self.extract_feat(img)

        # Forward through SOLOv2Head, which now includes mask feature generation
        cate_preds, kernel_preds, mask_feat_pred = self.bbox_head(x, eval=True)
        
        # Generate segmentation result
        seg_inputs = (cate_preds, kernel_preds, mask_feat_pred, img_meta, self.test_cfg, rescale)
        seg_result = self.bbox_head.get_seg(*seg_inputs)
        return seg_result