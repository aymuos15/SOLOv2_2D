import torch
import torch.nn as nn

from .fpn import FPN
from .backbone import resnet18
from .solov2_head import SOLOv2Head
from .mask_feat_head import MaskFeatHead

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
        self.mask_feat_head = MaskFeatHead(in_channels=256,
                            out_channels=128,
                            start_level=0,
                            end_level=3,
                            num_classes=128)
        
        #this set only support resnet18
        self.bbox_head = SOLOv2Head(num_classes=cfg.num_classes,
                            in_channels=256,
                            seg_feat_channels=256,
                            stacked_convs=2,
                            strides=[8, 8, 16, 32, 32],
                            scale_ranges=((1, 56), (28, 112), (56, 224), (112, 448), (224, 896)),
                            num_grids=[40, 36, 24, 16, 12],
                            ins_out_channels=128
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
        
        #mask feature mask
        if isinstance(self.mask_feat_head, nn.Sequential):
            for m in self.mask_feat_head:
                m.init_weights()
        else:
            self.mask_feat_head.init_weights()

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
    
    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs


    def forward(self, img, img_meta, return_loss=True, **kwargs):
        """
        Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
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


        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        mask_feat_pred = self.mask_feat_head(
                x[self.mask_feat_head.
                  start_level:self.mask_feat_head.end_level + 1])
        loss_inputs = outs + (mask_feat_pred, gt_bboxes, gt_labels, gt_masks, img_metas)

        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses
  
    '''
    img_metas context
    'filename': 'data/casia-SPT_val/val/JPEGImages/00238.jpg', 
    'ori_shape': (402, 600, 3), 'img_shape': (448, 669, 3), 
    'pad_shape': (448, 672, 3), 'scale_factor': 1.1144278606965174, 'flip': False, 
    'img_norm_cfg': {'mean': array([123.675, 116.28 , 103.53 ], dtype=float32), 
    'std': array([58.395, 57.12 , 57.375], dtype=float32), 'to_rgb': True}}

    '''

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_meta (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
        """

        return self.simple_test(imgs[0], img_metas[0], **kwargs)

    
    def simple_test(self, img, img_meta, rescale=False):
       
        x = self.extract_feat(img)

        outs = self.bbox_head(x,eval=True)
  
        mask_feat_pred = self.mask_feat_head(
                x[self.mask_feat_head.
                  start_level:self.mask_feat_head.end_level + 1])

        seg_inputs = outs + (mask_feat_pred, img_meta, self.test_cfg, rescale)

        seg_result = self.bbox_head.get_seg(*seg_inputs)
        return seg_result